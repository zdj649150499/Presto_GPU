#include "accel.h"
#include "accelsearch_cmd.h"


#if defined (__GNUC__)
#define inline __inline__
#else
#undef inline
#endif

/*#undef USEMMAP*/

#ifdef USEMMAP
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif

// Use OpenMP
#ifdef _OPENMP
#include <omp.h>
static int openmp_numthreads = 1;
void set_openmp_numthreads(int numthreads)
{
    int maxcpus = omp_get_num_procs();
    openmp_numthreads = (numthreads <= maxcpus) ? numthreads : maxcpus;
    // Make sure we are not dynamically setting the number of threads
    omp_set_dynamic(0);
    omp_set_num_threads(openmp_numthreads);
    printf("Starting the search using %d threads with OpenMP.\n\n",
           openmp_numthreads);
}
#endif

#define NEAREST_INT(x) (int) (x<0 ? x-0.5 : x+0.5)

/* Return 2**n */
#define index_to_twon(n) (1<<n)



cufftHandle plan_pdata_gpu_array[32][32];
cufftHandle plan_tmpdat_gpu_array[32][32];



/* Return x such that 2**x = n */
static inline int twon_to_index(int n)
{
    int x = 0;

    while (n > 1) {
        n >>= 1;
        x++;
    }
    return x;
}


static inline double calc_required_r(double harm_fract, double rfull)
/* Calculate the 'r' you need for subharmonic  */
/* harm_fract = harmnum / numharm if the       */
/* 'r' at the fundamental harmonic is 'rfull'. */
{
    return rint(ACCEL_RDR * rfull * harm_fract) * ACCEL_DR;
}


static inline int calc_required_z(double harm_fract, double zfull)
/* Calculate the 'z' you need for subharmonic  */
/* harm_fract = harmnum / numharm if the       */
/* 'z' at the fundamental harmonic is 'zfull'. */
{
    return NEAREST_INT(ACCEL_RDZ * zfull * harm_fract) * ACCEL_DZ;
}


static inline int calc_required_w(double harm_fract, double wfull)
/* Calculate the maximum 'w' needed for the given subharmonic  */
/* harm_fract = harmnum / numharm if the       */
/* 'w' at the fundamental harmonic is 'wfull'. */
{
    return NEAREST_INT(ACCEL_RDW * wfull * harm_fract) * ACCEL_DW;
}


static inline int index_from_r(double r, double lor)
/* Return an index for a Fourier Freq given an array that */
/* has stepsize ACCEL_DR and low freq 'lor'.              */
{
    return (int) ((r - lor) * ACCEL_RDR + DBLCORRECT);
}


static inline int index_from_z(double z, double loz)
/* Return an index for a Fourier Fdot given an array that */
/* has stepsize ACCEL_DZ and low freq dot 'loz'.              */
{
    return (int) ((z - loz) * ACCEL_RDZ + DBLCORRECT);
}


static inline int index_from_w(double w, double low)
/* Return an index for a Fourier Fdotdot given an array that */
/* has stepsize ACCEL_DW and low freq dotdot 'low'.              */
{
    return (int) ((w - low) * ACCEL_RDW + DBLCORRECT);
}


static void compare_rzw_cands(fourierprops * list, int nlist, char *notes)
{
    int ii, jj, kk;
    char tmp[30];

    for (ii = 0; ii < nlist; ii++) {
        for (jj = 0; jj < nlist; jj++) {
            if (ii == jj)
                continue;
            if (fabs(list[ii].r - list[jj].r) < 15.0 &&
                fabs(list[ii].z - list[jj].z) > 1.0 && list[ii].pow > list[jj].pow) {
                if (strncmp(notes + jj * 20, "                      ", 20) == 0) {
                    sprintf(tmp, "SL? of Cand %d", ii + 1);
                    strncpy(notes + jj * 20, tmp, 20);
                }
                continue;
            }
            for (kk = 1; kk < 61; kk++) {
                if ((fabs(list[ii].r - list[jj].r / kk) < list[jj].rerr * 3) &&
                    (fabs(list[ii].z - list[jj].z / kk) < list[jj].zerr * 2)) {
                    if (strncmp(notes + jj * 20, "                      ", 20) == 0) {
                        sprintf(tmp, "H %d of Cand %d", kk, ii + 1);
                        strncpy(notes + jj * 20, tmp, 20);
                        break;
                    }
                }
            }
        }
    }
}


static int calc_fftlen(int numharm, int harmnum, int max_zfull, int max_wfull, accelobs * obs)
/* The fft length needed to properly process a subharmonic */
{
    int bins_needed, end_effects;
    double harm_fract;

    harm_fract = (double) harmnum / (double) numharm;
    bins_needed = (int) ceil(obs->corr_uselen * harm_fract) + 2;
    end_effects = 2 * ACCEL_NUMBETWEEN *
        w_resp_halfwidth(calc_required_z(harm_fract, max_zfull),
                         calc_required_w(harm_fract, max_wfull), LOWACC);
    return next_good_fftlen(bins_needed + end_effects);

    // return next2_to_n(bins_needed + end_effects);
}


static void init_kernel(int z, int w, int fftlen, kernel * kern)
{
    int numkern;
    fcomplex *tempkern;

    kern->z = z;
    kern->w = w;
    kern->fftlen = fftlen;
    kern->numbetween = ACCEL_NUMBETWEEN;
    kern->kern_half_width = w_resp_halfwidth((double) z, (double) w, LOWACC);
    numkern = 2 * kern->numbetween * kern->kern_half_width;
    kern->numgoodbins = kern->fftlen - numkern;
    kern->data = gen_cvect(kern->fftlen);
    tempkern = gen_w_response(0.0, kern->numbetween, kern->z, kern->w, numkern);
    place_complex_kernel(tempkern, numkern, kern->data, kern->fftlen);
    vect_free(tempkern);
    COMPLEXFFT(kern->data, kern->fftlen, -1);
}


static void free_kernel(kernel * kern)
{
    vect_free(kern->data);
}


kernel **gen_kernmatrix(int numz, int numw) {
    int ii;
    kernel **kerns;
    
    kerns = (kernel **) malloc((size_t) numw * sizeof(kernel *));
    if (!kerns) {
        perror("\nError in 1st malloc() in gen_kernmatrix()");
        printf("\n");
        exit(-1);
    }
    kerns[0] = (kernel *) malloc((size_t) ((numz * numw) * sizeof(kernel)));
    if (!kerns[0]) {
        perror("\nError in 2nd malloc() in gen_kernmatrix()");
        printf("\n");
        exit(-1);
    }
    for (ii = 1; ii < numw; ii++)
        kerns[ii] = kerns[ii - 1] + numz;
    return kerns;
}


static void init_subharminfo(int numharm, int harmnum, int zmax, int wmax, subharminfo * shi, accelobs * obs)
/* Note:  'zmax' is the overall maximum 'z' in the search while
          'wmax' is the overall maximum 'w' in the search       */
{
    int ii, jj, fftlen;
    double harm_fract;

    harm_fract = (double) harmnum / (double) numharm;
    shi->numharm = numharm;
    shi->harmnum = harmnum;
    shi->zmax = calc_required_z(harm_fract, zmax);
    shi->wmax = calc_required_w(harm_fract, wmax);
    shi->numkern_zdim = (shi->zmax / ACCEL_DZ) * 2 + 1;
    shi->numkern_wdim = (shi->wmax / ACCEL_DW) * 2 + 1;
    shi->numkern = shi->numkern_zdim * shi->numkern_wdim;
    // shi->numkern = (shi->zmax / ACCEL_DZ) * 2 + 1;  


    if (numharm > 1 && !obs->cudaP) {
        shi->rinds = (unsigned short *) malloc(obs->corr_uselen * sizeof(unsigned short));
        shi->zinds = (unsigned short *) malloc(obs->corr_uselen * sizeof(unsigned short));
    }
    if (numharm==1 && harmnum==1)
        fftlen = obs->fftlen;
    else
        fftlen = calc_fftlen(numharm, harmnum, zmax, wmax, obs);
    /* Allocate 2D array of kernels, with dimensions being z and w */
    shi->kern = gen_kernmatrix(shi->numkern_zdim, shi->numkern_wdim);
    /* Actually append kernels to each array element */
    for (ii = 0; ii < shi->numkern_wdim; ii++) {
        for (jj = 0; jj < shi->numkern_zdim; jj++) {
            init_kernel(-shi->zmax + jj * ACCEL_DZ,
                        -shi->wmax + ii * ACCEL_DW, fftlen, &shi->kern[ii][jj]);
        }
    }
}


subharminfo **create_subharminfos(accelobs * obs)
{
    int ii, jj, harmtosum;// , fftlen;
    subharminfo **shis;
    
    shis = (subharminfo **) malloc(obs->numharmstages * sizeof(subharminfo *));
    /* Prep the fundamental (actually, the highest harmonic) */
    shis[0] = (subharminfo *) malloc(2 * sizeof(subharminfo));

    init_subharminfo(1, 1, (int) obs->zhi, (int) obs->whi, &shis[0][0], obs);
    // fftlen = obs->fftlen;
    /* Prep the sub-harmonics if needed */
    if (!obs->inmem) {
        for (ii = 1; ii < obs->numharmstages; ii++) {
            harmtosum = index_to_twon(ii);
            shis[ii] = (subharminfo *) malloc(harmtosum * sizeof(subharminfo));
            for (jj = 1; jj < harmtosum; jj += 2) {
                init_subharminfo(harmtosum, jj, (int) obs->zhi,
                                 (int) obs->whi, &shis[ii][jj - 1], obs);
                // fftlen = calc_fftlen(harmtosum, jj, (int) obs->zhi, (int) obs->whi, obs);
            }
        }
    }
    return shis;
}


static void free_subharminfo(subharminfo * shi, accelobs * obs)
{
    int ii, jj;
    
    for (ii = 0; ii < shi->numkern_wdim; ii++) {
        for (jj = 0; jj < shi->numkern_zdim; jj++) {
            free_kernel(&shi->kern[ii][jj]);
        }
    }
    if (shi->numharm > 1 && !obs->cudaP) {
        free(shi->rinds);
        free(shi->zinds);
        // cudaFreeHost(shi->rinds);
        // cudaFreeHost(shi->zinds);
    }
    // else if (shi->numharm > 1 && obs->cudaP) {
    //     cudaFreeHost(shi->rinds);
    //     cudaFreeHost(shi->zinds);
    // }
    free(shi->kern);
}


void free_subharminfos(accelobs * obs, subharminfo ** shis)
{
    int ii, jj, harmtosum;
    
    /* Free the sub-harmonics */
    if (!obs->inmem) {
        for (ii = 1; ii < obs->numharmstages; ii++) {
            harmtosum = index_to_twon(ii);
            for (jj = 1; jj < harmtosum; jj += 2) {
                free_subharminfo(&shis[ii][jj - 1], obs);
            }
            free(shis[ii]);
        }
    }
    /* Free the fundamental */
    free_subharminfo(&shis[0][0], obs);
    free(shis[0]);
    /* Free the container */
    free(shis);
}


static accelcand *create_accelcand(float power, float sigma,
                                   int numharm, double r, double z, double w)
{
    accelcand *obj;
    
    obj = (accelcand *) malloc(sizeof(accelcand));
    obj->power = power;
    obj->sigma = sigma;
    obj->numharm = numharm;
    obj->r = r;
    obj->z = z;
    obj->w = w;
    obj->pows = NULL;
    obj->hirs = NULL;
    obj->hizs = NULL;
    obj->hiws = NULL;
    obj->derivs = NULL;
    return obj;
}

void free_accelcand(gpointer data, gpointer user_data)
{
    user_data = NULL;
    if (((accelcand *) data)->pows) {
        vect_free(((accelcand *) data)->pows);
        vect_free(((accelcand *) data)->hirs);
        vect_free(((accelcand *) data)->hizs);
        vect_free(((accelcand *) data)->hiws);
        free(((accelcand *) data)->derivs);
    }
    free((accelcand *) data);
}


static int compare_accelcand_sigma(gconstpointer ca, gconstpointer cb)
/* Sorts from high to low sigma (ties are sorted by increasing r) */
{
    int result;
    accelcand *a, *b;

    a = (accelcand *) ca;
    b = (accelcand *) cb;
    result = (a->sigma < b->sigma) - (a->sigma > b->sigma);
    if (result)
        return result;
    else
        return (a->power < b->power) - (a->power > b->power);
}


GSList *sort_accelcands(GSList * list)
/* Sort the candidate list by decreasing sigma */
{
    return g_slist_sort(list, compare_accelcand_sigma);
}


static GSList *insert_new_accelcand(GSList * list, float power, float sigma,
                                    int numharm, double rr, double zz, double ww, int *added)
/* Checks the current list to see if there is already */
/* a candidate within ACCEL_CLOSEST_R bins.  If not,  */
/* it adds it to the list in increasing freq order.   */
{
    GSList *tmp_list = list, *prev_list = NULL, *new_list;
    double prev_diff_r = ACCEL_CLOSEST_R + 1.0, next_diff_r;

    *added = 0;
    if (!list) {
        new_list = g_slist_alloc();
        new_list->data =
            (gpointer *) create_accelcand(power, sigma, numharm, rr, zz, ww);
        *added = 1;
        return new_list;
    }

    /* Find the correct position in the list for the candidate */

    while ((tmp_list->next) && (((accelcand *) (tmp_list->data))->r < rr)) {
        prev_list = tmp_list;
        tmp_list = tmp_list->next;
    }
    next_diff_r = fabs(rr - ((accelcand *) (tmp_list->data))->r);
    if (prev_list)
        prev_diff_r = fabs(rr - ((accelcand *) (prev_list->data))->r);

    /* Similar candidate(s) is(are) present */

    if (prev_diff_r < ACCEL_CLOSEST_R) {
        /* Overwrite the prev cand */
        if (((accelcand *) (prev_list->data))->sigma < sigma) {
            free_accelcand(prev_list->data, NULL);
            prev_list->data = (gpointer *) create_accelcand(power, sigma,
                                                            numharm, rr, zz, ww);
            *added = 1;
        }
        if (next_diff_r < ACCEL_CLOSEST_R) {
            if (((accelcand *) (tmp_list->data))->sigma < sigma) {
                free_accelcand(tmp_list->data, NULL);
                if (*added) {
                    /* Remove the next cand */
                    list = g_slist_remove_link(list, tmp_list);
                    g_slist_free_1(tmp_list);
                } else {
                    /* Overwrite the next cand */
                    tmp_list->data = (gpointer *) create_accelcand(power, sigma,
                                                                   numharm, rr, zz, ww);
                    *added = 1;
                }
            }
        }
    } else if (next_diff_r < ACCEL_CLOSEST_R) {
        /* Overwrite the next cand */
        if (((accelcand *) (tmp_list->data))->sigma < sigma) {
            free_accelcand(tmp_list->data, NULL);
            tmp_list->data = (gpointer *) create_accelcand(power, sigma,
                                                           numharm, rr, zz, ww);
            *added = 1;
        }
    } else {                    /* This is a new candidate */
        new_list = g_slist_alloc();
        new_list->data =
            (gpointer *) create_accelcand(power, sigma, numharm, rr, zz, ww);
        *added = 1;
        if (!tmp_list->next &&
            (((accelcand *) (tmp_list->data))->r < (rr - ACCEL_CLOSEST_R))) {
            tmp_list->next = new_list;
            return list;
        }
        if (prev_list) {
            prev_list->next = new_list;
            new_list->next = tmp_list;
        } else {
            new_list->next = list;
            return new_list;
        }
    }
    return list;
}


GSList *eliminate_harmonics(GSList * cands, int *numcands)
/* Eliminate obvious but less-significant harmonically-related candidates */
{
    GSList *currentptr, *otherptr, *toocloseptr;
    accelcand *current_cand, *other_cand;
    int ii, maxharm = 16, numremoved = 0;
    double tooclose = 1.5;

    currentptr = cands;
    while (currentptr->next) {
        current_cand = (accelcand *) (currentptr->data);
        otherptr = currentptr->next;
        do {
            int remove = 0;
            other_cand = (accelcand *) (otherptr->data);
            for (ii = 1; ii <= maxharm; ii++) {
                if (fabs(current_cand->r / ii - other_cand->r) < tooclose) {
                    remove = 1;
                    break;
                }
            }
            if (remove == 0) {
                for (ii = 1; ii <= maxharm; ii++) {
                    if (fabs(current_cand->r * ii - other_cand->r) < tooclose) {
                        remove = 1;
                        break;
                    }
                }
            }
            /* Check a few other common harmonic ratios  */
            /* Hopefully this isn't being overzealous... */
            if (remove == 0 &&
                ((fabs(current_cand->r * 3.0 / 2.0 - other_cand->r) < tooclose) ||
                 (fabs(current_cand->r * 5.0 / 2.0 - other_cand->r) < tooclose) ||
                 (fabs(current_cand->r * 2.0 / 3.0 - other_cand->r) < tooclose) ||
                 (fabs(current_cand->r * 4.0 / 3.0 - other_cand->r) < tooclose) ||
                 (fabs(current_cand->r * 5.0 / 3.0 - other_cand->r) < tooclose) ||
                 (fabs(current_cand->r * 3.0 / 4.0 - other_cand->r) < tooclose) ||
                 (fabs(current_cand->r * 5.0 / 4.0 - other_cand->r) < tooclose) ||
                 (fabs(current_cand->r * 2.0 / 5.0 - other_cand->r) < tooclose) ||
                 (fabs(current_cand->r * 3.0 / 5.0 - other_cand->r) < tooclose) ||
                 (fabs(current_cand->r * 4.0 / 5.0 - other_cand->r) < tooclose) ||
                 (fabs(current_cand->r * 5.0 / 6.0 - other_cand->r) < tooclose) ||
                 (fabs(current_cand->r * 2.0 / 7.0 - other_cand->r) < tooclose) ||
                 (fabs(current_cand->r * 3.0 / 7.0 - other_cand->r) < tooclose) ||
                 (fabs(current_cand->r * 4.0 / 7.0 - other_cand->r) < tooclose) ||
                 (fabs(current_cand->r * 3.0 / 8.0 - other_cand->r) < tooclose) ||
                 (fabs(current_cand->r * 5.0 / 8.0 - other_cand->r) < tooclose) ||
                 (fabs(current_cand->r * 2.0 / 9.0 - other_cand->r) < tooclose) ||
                 (fabs(current_cand->r * 3.0 / 10.0 - other_cand->r) < tooclose) ||
                 (fabs(current_cand->r * 2.0 / 11.0 - other_cand->r) < tooclose) ||
                 (fabs(current_cand->r * 3.0 / 11.0 - other_cand->r) < tooclose) ||
                 (fabs(current_cand->r * 2.0 / 13.0 - other_cand->r) < tooclose) ||
                 (fabs(current_cand->r * 3.0 / 13.0 - other_cand->r) < tooclose) ||
                 (fabs(current_cand->r * 2.0 / 15.0 - other_cand->r) < tooclose))) {
                remove = 1;
            }
            /* Remove the "other" cand */
            if (remove) {
                numremoved++;
                toocloseptr = otherptr;
                otherptr = otherptr->next;
                free_accelcand(other_cand, NULL);
                cands = g_slist_remove_link(cands, toocloseptr);
                g_slist_free_1(toocloseptr);
                *numcands = *numcands - 1;
            } else {
                otherptr = otherptr->next;
            }
        } while (otherptr);
        if (currentptr->next)
            currentptr = currentptr->next;
    }
    if (numremoved) {
        printf("Removed %d likely harmonically related candidates.\n", numremoved);
    }
    return cands;
}


void optimize_accelcand(accelcand * cand, accelobs * obs)
{
    register int ii;
    static double r, z, w;

    register int numharm = cand->numharm;
    register double cand_r = cand->r;
    register double cand_z = cand->z;
    register double cand_w = cand->w;

    

    register long long obs_numbins = obs->numbins;
    register long long obs_lobin = obs->lobin;
    register int obs_numw = obs->numw;

    register int obs_dat_input = obs->dat_input;
    register int obs_mmap_file = obs->mmap_file;
    register int obs_use_harmonic_polishing = obs->use_harmonic_polishing;

    cand->pows = gen_dvect(numharm);
    cand->hirs = gen_dvect(numharm);
    cand->hizs = gen_dvect(numharm);
    cand->hiws = gen_dvect(numharm);
    cand->derivs = (rderivs *) malloc(sizeof(rderivs) * numharm);

    double pows_bk[numharm];
    double hirs_bk[numharm];
    double hizs_bk[numharm];
    double hiws_bk[numharm];
    rderivs derivs_bk[numharm];
    
    
    if (obs_use_harmonic_polishing &&
        (obs_mmap_file || obs_dat_input)) {
        if (obs_numw) {
            max_rzw_arr_harmonics(obs->fft, obs_numbins,
                                  numharm,
                                  cand_r - obs_lobin,
                                  cand_z, cand_w, &r, &z, &w,
                                  cand->derivs, cand->pows);
            
        } else {
            // max_rz_arr_harmonics(obs->fft, obs_numbins,
            //                      numharm,
            //                      cand_r - obs_lobin,
            //                      cand_z, &r, &z,
            //                      cand->derivs, cand->pows);
            max_rz_arr_harmonics(obs->fft, obs_numbins,
                                 numharm,
                                 cand_r - obs_lobin,
                                 cand_z, &r, &z,
                                 derivs_bk, pows_bk);
        }
        for (ii = 0; ii < numharm; ii++) {
            // cand->hirs[ii] = (r + obs_lobin) * (ii + 1);
            // cand->hizs[ii] = z * (ii + 1);
            // cand->hiws[ii] = obs_numw ? w * (ii + 1) : 0.0;
            hirs_bk[ii] = (r + obs_lobin) * (ii + 1);
            hizs_bk[ii] = z * (ii + 1);
            hiws_bk[ii] = obs_numw ? w * (ii + 1) : 0.0;
        }
    } else {
        for (ii = 0; ii < numharm; ii++) {
            if (obs_mmap_file || obs_dat_input) {
                if (obs_numw)
                    // cand->pows[ii] = max_rzw_arr(obs->fft,
                    //                              obs_numbins,
                    //                              cand_r * (ii + 1) - obs_lobin,
                    //                              cand_z * (ii + 1),
                    //                              cand_w * (ii + 1),
                    //                              &(cand->hirs[ii]),
                    //                              &(cand->hizs[ii]),
                    //                              &(cand->hiws[ii]),
                    //                              &(cand->derivs[ii]));
                    pows_bk[ii] = max_rzw_arr(obs->fft,
                                                 obs_numbins,
                                                 cand_r * (ii + 1) - obs_lobin,
                                                 cand_z * (ii + 1),
                                                 cand_w * (ii + 1),
                                                 &(hirs_bk[ii]),
                                                 &(hizs_bk[ii]),
                                                 &(hiws_bk[ii]),
                                                 &(derivs_bk[ii]));
                else
                    // cand->pows[ii] = max_rz_arr(obs->fft,
                    //                             obs_numbins,
                    //                             cand_r * (ii + 1) - obs_lobin,
                    //                             cand_z * (ii + 1),
                    //                             &(cand->hirs[ii]),
                    //                             &(cand->hizs[ii]), 
                    //                             &(cand->derivs[ii]));
                    pows_bk[ii] = max_rz_arr(obs->fft,
                                                obs_numbins,
                                                cand_r * (ii + 1) - obs_lobin,
                                                cand_z * (ii + 1),
                                                &(hirs_bk[ii]),
                                                &(hizs_bk[ii]), 
                                                &(derivs_bk[ii]));
            } else {
                if (obs_numw)
                    // cand->pows[ii] = max_rzw_file(obs->fftfile,
                    //                               cand_r * (ii + 1) - obs_lobin,
                    //                               cand_z * (ii + 1),
                    //                               cand_w * (ii + 1),
                    //                               &(cand->hirs[ii]),
                    //                               &(cand->hizs[ii]),
                    //                               &(cand->hiws[ii]),
                    //                               &(cand->derivs[ii]));
                    pows_bk[ii] = max_rzw_file(obs->fftfile,
                                                  cand_r * (ii + 1) - obs_lobin,
                                                  cand_z * (ii + 1),
                                                  cand_w * (ii + 1),
                                                  &(hirs_bk[ii]),
                                                  &(hizs_bk[ii]),
                                                  &(hiws_bk[ii]),
                                                  &(derivs_bk[ii]));
                else
                    // cand->pows[ii] = max_rz_file(obs->fftfile,
                    //                                 cand_r * (ii + 1) - obs_lobin,
                    //                                 cand_z * (ii + 1),
                    //                                 &(cand->hirs[ii]),
                    //                                 &(cand->hizs[ii]), 
                    //                                 &(cand->derivs[ii]));
                    pows_bk[ii] = max_rz_file(obs->fftfile,
                                                 cand_r * (ii + 1) - obs_lobin,
                                                 cand_z * (ii + 1),
                                                 &(hirs_bk[ii]),
                                                 &(hizs_bk[ii]), 
                                                 &(derivs_bk[ii]));
            }
            // cand->hirs[ii] += obs_lobin;
            hirs_bk[ii] += obs_lobin;
        }
    }
    
    memcpy(cand->pows, pows_bk, sizeof(double)*numharm);
    memcpy(cand->hirs, hirs_bk, sizeof(double)*numharm);
    memcpy(cand->hizs, hizs_bk, sizeof(double)*numharm);
    memcpy(cand->hiws, hiws_bk, sizeof(double)*numharm);
    memcpy(cand->derivs, derivs_bk, sizeof(rderivs)*numharm);

    cand->sigma = candidate_sigma(cand->power, numharm,
                                  obs->numindep[twon_to_index(numharm)]);
}

void optimize_accelcand_list(accelcand *cand, accelobs *obs, int readid)
{
    int ii;
    // static double r, z, w;
    double r, z, w;

    int numharm = cand->numharm;
    double cand_r = cand->r;
    double cand_z = cand->z;
    double cand_w = cand->w;

    

    long long obs_numbins = obs->numbins;
    long long obs_lobin = obs->lobin;
    int obs_numw = obs->numw;

    int obs_dat_input = obs->dat_input;
    int obs_mmap_file = obs->mmap_file;
    int obs_use_harmonic_polishing = obs->use_harmonic_polishing;

    cand->pows = gen_dvect(numharm);
    cand->hirs = gen_dvect(numharm);
    cand->hizs = gen_dvect(numharm);
    cand->hiws = gen_dvect(numharm);
    cand->derivs = (rderivs *) malloc(sizeof(rderivs) * numharm);

    double pows_bk[numharm];
    double hirs_bk[numharm];
    double hizs_bk[numharm];
    double hiws_bk[numharm];
    rderivs derivs_bk[numharm];
    
    
    if (obs_use_harmonic_polishing &&
        (obs_mmap_file || obs_dat_input)) {
        if (obs_numw) {
            max_rzw_arr_harmonics(obs->fftlist[readid], obs_numbins,
                                  numharm,
                                  cand_r - obs_lobin,
                                  cand_z, cand_w, &r, &z, &w,
                                  cand->derivs, cand->pows);
            
        } else {
            // max_rz_arr_harmonics(obs->fft, obs_numbins,
            //                      numharm,
            //                      cand_r - obs_lobin,
            //                      cand_z, &r, &z,
            //                      cand->derivs, cand->pows);
            max_rz_arr_harmonics(obs->fftlist[readid], obs_numbins,
                                 numharm,
                                 cand_r - obs_lobin,
                                 cand_z, &r, &z,
                                 derivs_bk, pows_bk);
        }
        for (ii = 0; ii < numharm; ii++) {
            // cand->hirs[ii] = (r + obs_lobin) * (ii + 1);
            // cand->hizs[ii] = z * (ii + 1);
            // cand->hiws[ii] = obs_numw ? w * (ii + 1) : 0.0;
            hirs_bk[ii] = (r + obs_lobin) * (ii + 1);
            hizs_bk[ii] = z * (ii + 1);
            hiws_bk[ii] = obs_numw ? w * (ii + 1) : 0.0;
        }
    } else {
        for (ii = 0; ii < numharm; ii++) {
            if (obs_mmap_file || obs_dat_input) {
                if (obs_numw)
                    // cand->pows[ii] = max_rzw_arr(obs->fft,
                    //                              obs_numbins,
                    //                              cand_r * (ii + 1) - obs_lobin,
                    //                              cand_z * (ii + 1),
                    //                              cand_w * (ii + 1),
                    //                              &(cand->hirs[ii]),
                    //                              &(cand->hizs[ii]),
                    //                              &(cand->hiws[ii]),
                    //                              &(cand->derivs[ii]));
                    pows_bk[ii] = max_rzw_arr(obs->fftlist[readid],
                                                 obs_numbins,
                                                 cand_r * (ii + 1) - obs_lobin,
                                                 cand_z * (ii + 1),
                                                 cand_w * (ii + 1),
                                                 &(hirs_bk[ii]),
                                                 &(hizs_bk[ii]),
                                                 &(hiws_bk[ii]),
                                                 &(derivs_bk[ii]));
                else
                    // cand->pows[ii] = max_rz_arr(obs->fft,
                    //                             obs_numbins,
                    //                             cand_r * (ii + 1) - obs_lobin,
                    //                             cand_z * (ii + 1),
                    //                             &(cand->hirs[ii]),
                    //                             &(cand->hizs[ii]), 
                    //                             &(cand->derivs[ii]));
                    pows_bk[ii] = max_rz_arr(obs->fftlist[readid],
                                                obs_numbins,
                                                cand_r * (ii + 1) - obs_lobin,
                                                cand_z * (ii + 1),
                                                &(hirs_bk[ii]),
                                                &(hizs_bk[ii]), 
                                                &(derivs_bk[ii]));
            } else {
                if (obs_numw)
                    // cand->pows[ii] = max_rzw_file(obs->fftfile,
                    //                               cand_r * (ii + 1) - obs_lobin,
                    //                               cand_z * (ii + 1),
                    //                               cand_w * (ii + 1),
                    //                               &(cand->hirs[ii]),
                    //                               &(cand->hizs[ii]),
                    //                               &(cand->hiws[ii]),
                    //                               &(cand->derivs[ii]));
                    pows_bk[ii] = max_rzw_file(obs->fftfilelist[readid],
                                                  cand_r * (ii + 1) - obs_lobin,
                                                  cand_z * (ii + 1),
                                                  cand_w * (ii + 1),
                                                  &(hirs_bk[ii]),
                                                  &(hizs_bk[ii]),
                                                  &(hiws_bk[ii]),
                                                  &(derivs_bk[ii]));
                else
                    // cand->pows[ii] = max_rz_file(obs->fftfile,
                    //                                 cand_r * (ii + 1) - obs_lobin,
                    //                                 cand_z * (ii + 1),
                    //                                 &(cand->hirs[ii]),
                    //                                 &(cand->hizs[ii]), 
                    //                                 &(cand->derivs[ii]));
                    pows_bk[ii] = max_rz_file(obs->fftfilelist[readid],
                                                 cand_r * (ii + 1) - obs_lobin,
                                                 cand_z * (ii + 1),
                                                 &(hirs_bk[ii]),
                                                 &(hizs_bk[ii]), 
                                                 &(derivs_bk[ii]));
            }
            // cand->hirs[ii] += obs_lobin;
            hirs_bk[ii] += obs_lobin;
        }
    }
    
    memcpy(cand->pows, pows_bk, sizeof(double)*numharm);
    memcpy(cand->hirs, hirs_bk, sizeof(double)*numharm);
    memcpy(cand->hizs, hizs_bk, sizeof(double)*numharm);
    memcpy(cand->hiws, hiws_bk, sizeof(double)*numharm);
    memcpy(cand->derivs, derivs_bk, sizeof(rderivs)*numharm);

    cand->sigma = candidate_sigma(cand->power, numharm,
                                  obs->numindep[twon_to_index(numharm)]);
}

void optimize_accelcand_list_gpu_noharmpolish(accelcand *cand, accelobs *obs, int readid)
{
    int ii;
    double r, z, w;

    int numharm = cand->numharm;
    double cand_r = cand->r;
    double cand_z = cand->z;
    double cand_w = cand->w;

    

    long long obs_numbins = obs->numbins;
    long long obs_lobin = obs->lobin;
    int obs_numw = obs->numw;

    int obs_dat_input = obs->dat_input;
    int obs_mmap_file = obs->mmap_file;
    int obs_use_harmonic_polishing = obs->use_harmonic_polishing;

    // cand->pows = gen_dvect(numharm);
    // cand->hirs = gen_dvect(numharm);
    // cand->hizs = gen_dvect(numharm);
    // cand->hiws = gen_dvect(numharm);
    // cand->derivs = (rderivs *) malloc(sizeof(rderivs) * numharm);

    double pows_bk[numharm];
    double hirs_bk[numharm];
    double hizs_bk[numharm];
    double hiws_bk[numharm];
    rderivs derivs_bk[numharm];
    
    {
        for (ii = 0; ii < numharm; ii++) {
            if (obs_mmap_file || obs_dat_input) {
                // if (obs_numw)
                //     pows_bk[ii] = max_rzw_arr(obs->fftlist[readid],
                //                                  obs_numbins,
                //                                  cand_r * (ii + 1) - obs_lobin,
                //                                  cand_z * (ii + 1),
                //                                  cand_w * (ii + 1),
                //                                  &(hirs_bk[ii]),
                //                                  &(hizs_bk[ii]),
                //                                  &(hiws_bk[ii]),
                //                                  &(derivs_bk[ii]));
                // else
                    pows_bk[ii] = max_rz_arr(obs->fftlist[readid],
                                                obs_numbins,
                                                cand_r * (ii + 1) - obs_lobin,
                                                cand_z * (ii + 1),
                                                &(hirs_bk[ii]),
                                                &(hizs_bk[ii]), 
                                                &(derivs_bk[ii]));
            } 
            // cand->hirs[ii] += obs_lobin;
            hirs_bk[ii] += obs_lobin;
        }
    }
    
    memcpy(cand->pows, pows_bk, sizeof(double)*numharm);
    memcpy(cand->hirs, hirs_bk, sizeof(double)*numharm);
    memcpy(cand->hizs, hizs_bk, sizeof(double)*numharm);
    memcpy(cand->hiws, hiws_bk, sizeof(double)*numharm);
    memcpy(cand->derivs, derivs_bk, sizeof(rderivs)*numharm);

    cand->sigma = candidate_sigma(cand->power, numharm,
                                  obs->numindep[twon_to_index(numharm)]);
}

static void center_string(char *outstring, char *instring, int width)
{
    int len;

    len = strlen(instring);
    // if (width < len) {
    //     printf("\nwidth < len (%d) in center_string(outstring, '%s', width=%d)\n",
    //            len, instring, width);
    // }
    memset(outstring, ' ', width);
    outstring[width] = '\0';
    if (len >= width) {
        strncpy(outstring, instring, width);
    } else {
        strncpy(outstring + (width - len) / 2, instring, len);
    }
}


static void write_val_with_err(FILE * outfile, double val, double err,
                               int numerr, int width)
{
    int retval;
    char tmpstr[80], ctrstr[80];

    if (numerr == 1)
        retval = nice_output_1(tmpstr, val, err, 0);
    else if (numerr == 2)
        retval = nice_output_2(tmpstr, val, err, 0);
    else
        printf("\numerr = %d is out-of-range (1-2) in write_val_with_err()\n",
               numerr);
    center_string(ctrstr, tmpstr, width);
    fprintf(outfile, "%s  ", ctrstr);
}


void output_fundamentals(fourierprops * props, GSList * list,
                         accelobs * obs, infodata * idata)
{
    double accel = 0.0, accelerr = 0.0, coherent_pow;
    int ii, jj, numcols = 13, numcands, *width, *error;
    int widths[13] = { 4, 5, 6, 8, 4, 16, 15, 15, 15, 11, 11, 15, 20 };
    int errors[13] = { 0, 0, 0, 0, 0, 1, 1, 2, 1, 2, 2, 2, 0 };
    char tmpstr[80], ctrstr[80], *notes;
    accelcand *cand;
    register GSList *listptr;
    rzwerrs errs;
    static char **title;
    static char *titles1[] = { "", "", "Summed", "Coherent", "Num", "Period",
        "Frequency", "FFT 'r'", "Freq Deriv", "FFT 'z'", "FFT 'w'",
        "Accel", ""
    };
    static char *titles2[] = { "Cand", "Sigma", "Power", "Power", "Harm", "(ms)",
        "(Hz)", "(bin)", "(Hz/s)", "(bins)", "(bins)",
        "(m/s^2)", "Notes"
    };

    numcands = g_slist_length(list);
    listptr = list;

    /* Close the old work file and open the cand file */

    if (!obs->dat_input)
        fclose(obs->workfile);  /* Why is this here? -A */
    obs->workfile = chkfopen(obs->accelnm, "w");

    /* Set our candidate notes to all spaces */

    notes = (char *) malloc(numcands * widths[numcols - 1]);
    memset(notes, ' ', numcands * widths[numcols - 1]);

    /* Compare the candidates with the pulsar database */

    if (strncmp(idata->telescope, "None", 4) != 0) {
        if (dms2rad(idata->ra_h, idata->ra_m, idata->ra_s) != 0.0 &&
            hms2rad(idata->dec_d, idata->dec_m, idata->dec_s) != 0.0) {
            for (ii = 0; ii < numcands; ii++) {
                comp_psr_to_cand(props + ii, idata, notes + ii * 20, 0);
            }
        }
    }

    /* Compare the candidates with themselves */

    compare_rzw_cands(props, numcands, notes);

    /* Print the header */

    width = widths;
    title = titles1;
    for (ii = 0; ii < numcols - 1; ii++) {
        if (obs->numw==0 && ii==10) { // Skip jerk parts
            title++;
            width++;
            continue;
        } else {
            center_string(ctrstr, *title++, *width++);
            fprintf(obs->workfile, "%s  ", ctrstr);
        }
    }
    center_string(ctrstr, *title++, *width++);
    fprintf(obs->workfile, "%s\n", ctrstr);

    width = widths;
    title = titles2;
    for (ii = 0; ii < numcols - 1; ii++) {
        if (obs->numw==0 && ii==10) { // Skip jerk parts
            title++;
            width++;
            continue;
        } else {
            center_string(ctrstr, *title++, *width++);
            fprintf(obs->workfile, "%s  ", ctrstr);
        }
    }
    center_string(ctrstr, *title++, *width++);
    fprintf(obs->workfile, "%s\n", ctrstr);

    width = widths;
    for (ii = 0; ii < numcols - 1; ii++) {
        if (obs->numw==0 && ii==10) { // Skip jerk parts
            width++;
            continue;
        } else {
            memset(tmpstr, '-', *width);
            tmpstr[*width++] = '\0';
            fprintf(obs->workfile, "%s--", tmpstr);
        }
    }
    memset(tmpstr, '-', *width++);
    tmpstr[widths[ii]] = '\0';
    fprintf(obs->workfile, "%s\n", tmpstr);

    /* Print the fundamentals */

    for (ii = 0; ii < numcands; ii++) {
        width = widths;
        error = errors;
        cand = (accelcand *) (listptr->data);
        calc_rzwerrs(props + ii, obs->T, &errs);

        {                       /* Calculate the coherently summed power */
            double coherent_r = 0.0, coherent_i = 0.0;
            double phs0, phscorr, amp;
            rderivs harm;

            /* These phase calculations assume the fundamental is best */
            /* Better to irfft them and check the amplitude */
            phs0 = cand->derivs[0].phs;
            for (jj = 0; jj < cand->numharm; jj++) {
                harm = cand->derivs[jj];
                if (obs->nph > 0.0)
                    amp = sqrt(harm.pow / obs->nph);
                else
                    amp = sqrt(harm.pow / harm.locpow);
                phscorr = phs0 - fmod((jj + 1.0) * phs0, TWOPI);
                coherent_r += amp * cos(harm.phs + phscorr);
                coherent_i += amp * sin(harm.phs + phscorr);
            }
            coherent_pow = coherent_r * coherent_r + coherent_i * coherent_i;
        }

        sprintf(tmpstr, "%-4d", ii + 1);
        center_string(ctrstr, tmpstr, *width++);
        error++;
        fprintf(obs->workfile, "%s  ", ctrstr);

        sprintf(tmpstr, "%.2f", cand->sigma);
        center_string(ctrstr, tmpstr, *width++);
        error++;
        fprintf(obs->workfile, "%s  ", ctrstr);

        sprintf(tmpstr, "%.2f", cand->power);
        center_string(ctrstr, tmpstr, *width++);
        error++;
        fprintf(obs->workfile, "%s  ", ctrstr);

        sprintf(tmpstr, "%.2f", coherent_pow);
        center_string(ctrstr, tmpstr, *width++);
        error++;
        fprintf(obs->workfile, "%s  ", ctrstr);

        sprintf(tmpstr, "%d", cand->numharm);
        center_string(ctrstr, tmpstr, *width++);
        error++;
        fprintf(obs->workfile, "%s  ", ctrstr);

        write_val_with_err(obs->workfile, errs.p * 1000.0, errs.perr * 1000.0,
                           *error++, *width++);
        write_val_with_err(obs->workfile, errs.f, errs.ferr, *error++, *width++);
        write_val_with_err(obs->workfile, props[ii].r, props[ii].rerr,
                           *error++, *width++);
        write_val_with_err(obs->workfile, errs.fd, errs.fderr, *error++, *width++);
        write_val_with_err(obs->workfile, props[ii].z, props[ii].zerr,
                           *error++, *width++);
        if (obs->numw) {
            write_val_with_err(obs->workfile, props[ii].w, props[ii].werr,
                               *error++, *width++);
        } else {
            error++;
            width++;
        }
        accel = props[ii].z * SOL / (obs->T * obs->T * errs.f);
        accelerr = props[ii].zerr * SOL / (obs->T * obs->T * errs.f);
        write_val_with_err(obs->workfile, accel, accelerr, *error++, *width++);
        fprintf(obs->workfile, "  %.20s\n", notes + ii * 20);
        fflush(obs->workfile);
        listptr = listptr->next;
    }
    fprintf(obs->workfile, "\n\n");
    free(notes);
}

void output_fundamentals_list(fourierprops *props, GSList *list, 
			 accelobs *obs, infodata *idata, int readid)
{
    double accel = 0.0, accelerr = 0.0, coherent_pow;
    int ii, jj, numcols = 13, numcands, *width, *error;
    int widths[13] = { 4, 5, 6, 8, 4, 16, 15, 15, 15, 11, 11, 15, 20 };
    int errors[13] = { 0, 0, 0, 0, 0, 1, 1, 2, 1, 2, 2, 2, 0 };
    char tmpstr[80], ctrstr[80], *notes;
    accelcand *cand;
    register GSList *listptr;
    rzwerrs errs;
    static char **title;
    static char *titles1[] = { "", "", "Summed", "Coherent", "Num", "Period",
        "Frequency", "FFT 'r'", "Freq Deriv", "FFT 'z'", "FFT 'w'",
        "Accel", ""
    };
    static char *titles2[] = { "Cand", "Sigma", "Power", "Power", "Harm", "(ms)",
        "(Hz)", "(bin)", "(Hz/s)", "(bins)", "(bins)",
        "(m/s^2)", "Notes"
    };

    numcands = g_slist_length(list);
    listptr = list;

    /* Close the old work file and open the cand file */

    if (!obs->dat_input)
        fclose(obs->workfilelist[readid]);  /* Why is this here? -A */
    obs->workfilelist[readid] = chkfopen(obs->accelnmlist[readid], "w");

    /* Set our candidate notes to all spaces */

    notes = (char *) malloc(numcands * widths[numcols - 1]);
    memset(notes, ' ', numcands * widths[numcols - 1]);

    /* Compare the candidates with the pulsar database */

    if (strncmp(idata->telescope, "None", 4) != 0) {
        if (dms2rad(idata->ra_h, idata->ra_m, idata->ra_s) != 0.0 &&
            hms2rad(idata->dec_d, idata->dec_m, idata->dec_s) != 0.0) {
            for (ii = 0; ii < numcands; ii++) {
                comp_psr_to_cand(props + ii, idata, notes + ii * 20, 0);
            }
        }
    }

    /* Compare the candidates with themselves */

    compare_rzw_cands(props, numcands, notes);

    /* Print the header */

    width = widths;
    title = titles1;
    for (ii = 0; ii < numcols - 1; ii++) {
        if (obs->numw==0 && ii==10) { // Skip jerk parts
            title++;
            width++;
            continue;
        } else {
            center_string(ctrstr, *title++, *width++);
            fprintf(obs->workfilelist[readid], "%s  ", ctrstr);
        }
    }
    center_string(ctrstr, *title++, *width++);
    fprintf(obs->workfilelist[readid], "%s\n", ctrstr);

    width = widths;
    title = titles2;
    for (ii = 0; ii < numcols - 1; ii++) {
        if (obs->numw==0 && ii==10) { // Skip jerk parts
            title++;
            width++;
            continue;
        } else {
            center_string(ctrstr, *title++, *width++);
            fprintf(obs->workfilelist[readid], "%s  ", ctrstr);
        }
    }
    center_string(ctrstr, *title++, *width++);
    fprintf(obs->workfilelist[readid], "%s\n", ctrstr);

    width = widths;
    for (ii = 0; ii < numcols - 1; ii++) {
        if (obs->numw==0 && ii==10) { // Skip jerk parts
            width++;
            continue;
        } else {
            memset(tmpstr, '-', *width);
            tmpstr[*width++] = '\0';
            fprintf(obs->workfilelist[readid], "%s--", tmpstr);
        }
    }
    memset(tmpstr, '-', *width++);
    tmpstr[widths[ii]] = '\0';
    fprintf(obs->workfilelist[readid], "%s\n", tmpstr);

    /* Print the fundamentals */

    for (ii = 0; ii < numcands; ii++) {
        width = widths;
        error = errors;
        cand = (accelcand *) (listptr->data);
        calc_rzwerrs(props + ii, obs->T, &errs);

        {                       /* Calculate the coherently summed power */
            double coherent_r = 0.0, coherent_i = 0.0;
            double phs0, phscorr, amp;
            rderivs harm;

            /* These phase calculations assume the fundamental is best */
            /* Better to irfft them and check the amplitude */
            phs0 = cand->derivs[0].phs;
            for (jj = 0; jj < cand->numharm; jj++) {
                harm = cand->derivs[jj];
                if (obs->nph > 0.0)
                    amp = sqrt(harm.pow / obs->nph);
                else
                    amp = sqrt(harm.pow / harm.locpow);
                phscorr = phs0 - fmod((jj + 1.0) * phs0, TWOPI);
                coherent_r += amp * cos(harm.phs + phscorr);
                coherent_i += amp * sin(harm.phs + phscorr);
            }
            coherent_pow = coherent_r * coherent_r + coherent_i * coherent_i;
        }

        sprintf(tmpstr, "%-4d", ii + 1);
        center_string(ctrstr, tmpstr, *width++);
        error++;
        fprintf(obs->workfilelist[readid], "%s  ", ctrstr);

        sprintf(tmpstr, "%.2f", cand->sigma);
        center_string(ctrstr, tmpstr, *width++);
        error++;
        fprintf(obs->workfilelist[readid], "%s  ", ctrstr);

        sprintf(tmpstr, "%.2f", cand->power);
        center_string(ctrstr, tmpstr, *width++);
        error++;
        fprintf(obs->workfilelist[readid], "%s  ", ctrstr);

        sprintf(tmpstr, "%.2f", coherent_pow);
        center_string(ctrstr, tmpstr, *width++);
        error++;
        fprintf(obs->workfilelist[readid], "%s  ", ctrstr);

        sprintf(tmpstr, "%d", cand->numharm);
        center_string(ctrstr, tmpstr, *width++);
        error++;
        fprintf(obs->workfilelist[readid], "%s  ", ctrstr);

        write_val_with_err(obs->workfilelist[readid], errs.p * 1000.0, errs.perr * 1000.0,
                           *error++, *width++);
        write_val_with_err(obs->workfilelist[readid], errs.f, errs.ferr, *error++, *width++);
        write_val_with_err(obs->workfilelist[readid], props[ii].r, props[ii].rerr,
                           *error++, *width++);
        write_val_with_err(obs->workfilelist[readid], errs.fd, errs.fderr, *error++, *width++);
        write_val_with_err(obs->workfilelist[readid], props[ii].z, props[ii].zerr,
                           *error++, *width++);
        if (obs->numw) {
            write_val_with_err(obs->workfilelist[readid], props[ii].w, props[ii].werr,
                               *error++, *width++);
        } else {
            error++;
            width++;
        }
        accel = props[ii].z * SOL / (obs->T * obs->T * errs.f);
        accelerr = props[ii].zerr * SOL / (obs->T * obs->T * errs.f);
        write_val_with_err(obs->workfilelist[readid], accel, accelerr, *error++, *width++);
        fprintf(obs->workfilelist[readid], "  %.20s\n", notes + ii * 20);
        fflush(obs->workfilelist[readid]);
        listptr = listptr->next;
    }
    fprintf(obs->workfilelist[readid], "\n\n");
    free(notes);
}


void output_harmonics(GSList * list, accelobs * obs, infodata * idata)
{
    int ii, jj, numcols = 15, numcands;
    int widths[15] = { 5, 4, 5, 15, 11, 18, 13, 12, 9, 12, 9, 12, 10, 10, 20 };
    int errors[15] = { 0, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 2, 2, 0 };
    char tmpstr[30], ctrstr[30], notes[21], *command;
    accelcand *cand;
    register GSList *listptr;
    fourierprops props;
    rzwerrs errs;
    static char *titles1[] = { "", "", "", "Power /", "Raw",
        "FFT 'r'", "Pred 'r'", "FFT 'z'", "Pred 'z'", "FFT 'w'", "Pred 'w'",
        "Phase", "Centroid", "Purity", ""
    };
    static char *titles2[] = { "Cand", "Harm", "Sigma", "Loc Pow", "Power",
        "(bin)", "(bin)", "(bins)", "(bins)", "(bins)", "(bins)",
        "(rad)", "(0-1)", "<p> = 1", "Notes"
    };

    numcands = g_slist_length(list);
    listptr = list;

    /* Print the header */

    for (ii = 0; ii < numcols - 1; ii++) {
        if (obs->numw==0 && (ii==9 || ii==10)) { // Skip jerk parts
            continue;
        } else {
            center_string(ctrstr, titles1[ii], widths[ii]);
            fprintf(obs->workfile, "%s  ", ctrstr);
        }
    }
    center_string(ctrstr, titles1[ii], widths[ii]);
    fprintf(obs->workfile, "%s\n", ctrstr);
    for (ii = 0; ii < numcols - 1; ii++) {
        if (obs->numw==0 && (ii==9 || ii==10)) { // Skip jerk parts
            continue;
        } else {
            if (obs->nph > 0.0 && ii == 3)  /*  HAAACK!!! */
                center_string(ctrstr, "NumPhot", widths[ii]);
            else
                center_string(ctrstr, titles2[ii], widths[ii]);
            fprintf(obs->workfile, "%s  ", ctrstr);
        }
    }
    center_string(ctrstr, titles2[ii], widths[ii]);
    fprintf(obs->workfile, "%s\n", ctrstr);
    for (ii = 0; ii < numcols - 1; ii++) {
        if (obs->numw==0 && (ii==9 || ii==10)) { // Skip jerk parts
            continue;
        } else {
            memset(tmpstr, '-', widths[ii]);
            tmpstr[widths[ii]] = '\0';
            fprintf(obs->workfile, "%s--", tmpstr);
        }
    }
    memset(tmpstr, '-', widths[ii]);
    tmpstr[widths[ii]] = '\0';
    fprintf(obs->workfile, "%s\n", tmpstr);

    /* Print the harmonics */

    for (ii = 0; ii < numcands; ii++) {
        cand = (accelcand *) (listptr->data);
        for (jj = 0; jj < cand->numharm; jj++) {
            if (obs->nph > 0.0) {
                double tmp_locpow;

                tmp_locpow = cand->derivs[jj].locpow;
                cand->derivs[jj].locpow = obs->nph;
                calc_props(cand->derivs[jj], cand->hirs[jj],
                           cand->hizs[jj], cand->hiws[jj], &props);
                cand->derivs[jj].locpow = tmp_locpow;
            } else {
                calc_props(cand->derivs[jj], cand->hirs[jj],
                           cand->hizs[jj], cand->hiws[jj], &props);
            }
            calc_rzwerrs(&props, obs->T, &errs);
            if (strncmp(idata->telescope, "None", 4) != 0) {
                comp_psr_to_cand(&props, idata, notes, 0);
            } else {
                memset(notes, ' ', 20);
                notes[20] = '\0';
            }
            if (jj == 0)
                sprintf(tmpstr, " %-4d", ii + 1);
            else
                sprintf(tmpstr, "     ");
            center_string(ctrstr, tmpstr, widths[0]);
            fprintf(obs->workfile, "%s  ", ctrstr);
            sprintf(tmpstr, "%-4d", jj + 1);
            center_string(ctrstr, tmpstr, widths[1]);
            fprintf(obs->workfile, "%s  ", ctrstr);
            sprintf(tmpstr, "%.2f", candidate_sigma(props.pow, 1, 1));
            center_string(ctrstr, tmpstr, widths[2]);
            fprintf(obs->workfile, "%s  ", ctrstr);
            write_val_with_err(obs->workfile, props.pow, props.powerr,
                               errors[3], widths[3]);
            sprintf(tmpstr, "%.3g", props.rawpow);
            center_string(ctrstr, tmpstr, widths[4]);
            fprintf(obs->workfile, "%s  ", ctrstr);
            write_val_with_err(obs->workfile, props.r, props.rerr,
                               errors[5], widths[5]);
            sprintf(tmpstr, "%.2f", cand->r * (jj + 1));
            center_string(ctrstr, tmpstr, widths[6]);
            fprintf(obs->workfile, "%s  ", ctrstr);
            write_val_with_err(obs->workfile, props.z, props.zerr,
                               errors[7], widths[7]);
            sprintf(tmpstr, "%.2f", cand->z * (jj + 1));
            center_string(ctrstr, tmpstr, widths[8]);
            fprintf(obs->workfile, "%s  ", ctrstr);
            if (obs->numw) {
                write_val_with_err(obs->workfile, props.w, props.werr,
                                   errors[9], widths[9]);
                sprintf(tmpstr, "%.2f", cand->w * (jj + 1));
                center_string(ctrstr, tmpstr, widths[10]);
                fprintf(obs->workfile, "%s  ", ctrstr);
            }
            write_val_with_err(obs->workfile, props.phs, props.phserr,
                               errors[11], widths[11]);
            write_val_with_err(obs->workfile, props.cen, props.cenerr,
                               errors[12], widths[12]);
            write_val_with_err(obs->workfile, props.pur, props.purerr,
                               errors[13], widths[13]);
            fprintf(obs->workfile, "  %.20s\n", notes);
            fflush(obs->workfile);
        }
        listptr = listptr->next;
    }
    fprintf(obs->workfile, "\n\n");
    fclose(obs->workfile);
    command = malloc(strlen(obs->rootfilenm) + strlen(obs->accelnm) + 20);
    sprintf(command, "cat %s.inf >> %s", obs->rootfilenm, obs->accelnm);
    system(command);
    free(command);
}

void output_harmonics_list(GSList *list, accelobs *obs, infodata *idata, int readid)
{
    int ii, jj, numcols = 15, numcands;
    int widths[15] = { 5, 4, 5, 15, 11, 18, 13, 12, 9, 12, 9, 12, 10, 10, 20 };
    int errors[15] = { 0, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 2, 2, 0 };
    char tmpstr[30], ctrstr[30], notes[21], *command;
    accelcand *cand;
    register GSList *listptr;
    fourierprops props;
    rzwerrs errs;
    static char *titles1[] = { "", "", "", "Power /", "Raw",
        "FFT 'r'", "Pred 'r'", "FFT 'z'", "Pred 'z'", "FFT 'w'", "Pred 'w'",
        "Phase", "Centroid", "Purity", ""
    };
    static char *titles2[] = { "Cand", "Harm", "Sigma", "Loc Pow", "Power",
        "(bin)", "(bin)", "(bins)", "(bins)", "(bins)", "(bins)",
        "(rad)", "(0-1)", "<p> = 1", "Notes"
    };

    numcands = g_slist_length(list);
    listptr = list;

    /* Print the header */

    for (ii = 0; ii < numcols - 1; ii++) {
        if (obs->numw==0 && (ii==9 || ii==10)) { // Skip jerk parts
            continue;
        } else {
            center_string(ctrstr, titles1[ii], widths[ii]);
            fprintf(obs->workfilelist[readid], "%s  ", ctrstr);
        }
    }
    center_string(ctrstr, titles1[ii], widths[ii]);
    fprintf(obs->workfilelist[readid], "%s\n", ctrstr);
    for (ii = 0; ii < numcols - 1; ii++) {
        if (obs->numw==0 && (ii==9 || ii==10)) { // Skip jerk parts
            continue;
        } else {
            if (obs->nph > 0.0 && ii == 3)  /*  HAAACK!!! */
                center_string(ctrstr, "NumPhot", widths[ii]);
            else
                center_string(ctrstr, titles2[ii], widths[ii]);
            fprintf(obs->workfilelist[readid], "%s  ", ctrstr);
        }
    }
    center_string(ctrstr, titles2[ii], widths[ii]);
    fprintf(obs->workfilelist[readid], "%s\n", ctrstr);
    for (ii = 0; ii < numcols - 1; ii++) {
        if (obs->numw==0 && (ii==9 || ii==10)) { // Skip jerk parts
            continue;
        } else {
            memset(tmpstr, '-', widths[ii]);
            tmpstr[widths[ii]] = '\0';
            fprintf(obs->workfilelist[readid], "%s--", tmpstr);
        }
    }
    memset(tmpstr, '-', widths[ii]);
    tmpstr[widths[ii]] = '\0';
    fprintf(obs->workfilelist[readid], "%s\n", tmpstr);

    /* Print the harmonics */

    for (ii = 0; ii < numcands; ii++) {
        cand = (accelcand *) (listptr->data);
        for (jj = 0; jj < cand->numharm; jj++) {
            if (obs->nph > 0.0) {
                double tmp_locpow;

                tmp_locpow = cand->derivs[jj].locpow;
                cand->derivs[jj].locpow = obs->nph;
                calc_props(cand->derivs[jj], cand->hirs[jj],
                           cand->hizs[jj], cand->hiws[jj], &props);
                cand->derivs[jj].locpow = tmp_locpow;
            } else {
                calc_props(cand->derivs[jj], cand->hirs[jj],
                           cand->hizs[jj], cand->hiws[jj], &props);
            }
            calc_rzwerrs(&props, obs->T, &errs);
            if (strncmp(idata->telescope, "None", 4) != 0) {
                comp_psr_to_cand(&props, idata, notes, 0);
            } else {
                memset(notes, ' ', 20);
                notes[20] = '\0';
            }
            if (jj == 0)
                sprintf(tmpstr, " %-4d", ii + 1);
            else
                sprintf(tmpstr, "     ");
            center_string(ctrstr, tmpstr, widths[0]);
            fprintf(obs->workfilelist[readid], "%s  ", ctrstr);
            sprintf(tmpstr, "%-4d", jj + 1);
            center_string(ctrstr, tmpstr, widths[1]);
            fprintf(obs->workfilelist[readid], "%s  ", ctrstr);
            sprintf(tmpstr, "%.2f", candidate_sigma(props.pow, 1, 1));
            center_string(ctrstr, tmpstr, widths[2]);
            fprintf(obs->workfilelist[readid], "%s  ", ctrstr);
            write_val_with_err(obs->workfilelist[readid], props.pow, props.powerr,
                               errors[3], widths[3]);
            sprintf(tmpstr, "%.3g", props.rawpow);
            center_string(ctrstr, tmpstr, widths[4]);
            fprintf(obs->workfilelist[readid], "%s  ", ctrstr);
            write_val_with_err(obs->workfilelist[readid], props.r, props.rerr,
                               errors[5], widths[5]);
            sprintf(tmpstr, "%.2f", cand->r * (jj + 1));
            center_string(ctrstr, tmpstr, widths[6]);
            fprintf(obs->workfilelist[readid], "%s  ", ctrstr);
            write_val_with_err(obs->workfilelist[readid], props.z, props.zerr,
                               errors[7], widths[7]);
            sprintf(tmpstr, "%.2f", cand->z * (jj + 1));
            center_string(ctrstr, tmpstr, widths[8]);
            fprintf(obs->workfilelist[readid], "%s  ", ctrstr);
            if (obs->numw) {
                write_val_with_err(obs->workfilelist[readid], props.w, props.werr,
                                   errors[9], widths[9]);
                sprintf(tmpstr, "%.2f", cand->w * (jj + 1));
                center_string(ctrstr, tmpstr, widths[10]);
                fprintf(obs->workfilelist[readid], "%s  ", ctrstr);
            }
            write_val_with_err(obs->workfilelist[readid], props.phs, props.phserr,
                               errors[11], widths[11]);
            write_val_with_err(obs->workfilelist[readid], props.cen, props.cenerr,
                               errors[12], widths[12]);
            write_val_with_err(obs->workfilelist[readid], props.pur, props.purerr,
                               errors[13], widths[13]);
            fprintf(obs->workfilelist[readid], "  %.20s\n", notes);
            fflush(obs->workfilelist[readid]);
        }
        listptr = listptr->next;
    }
    fprintf(obs->workfilelist[readid], "\n\n");
    fclose(obs->workfilelist[readid]);
    command = malloc(strlen(obs->rootfilenmlist[readid]) + strlen(obs->accelnmlist[readid]) + 20);
    sprintf(command, "cat %s.inf >> %s", obs->rootfilenmlist[readid], obs->accelnmlist[readid]);
    system(command);
    free(command);
}

void print_accelcand(gpointer data, gpointer user_data)
{
    accelcand *obj = (accelcand *) data;

    user_data = NULL;
    printf("sigma: %-7.4f  pow: %-7.2f  harm: %-2d  r: %-14.4f  z: %-10.4f  w: %-10.2f\n",
           obj->sigma, obj->power, obj->numharm, obj->r, obj->z, obj->w);
}


fcomplex *get_fourier_amplitudes(long long lobin, int numbins, accelobs * obs)
{
    if (obs->mmap_file || obs->dat_input) {
        long long ii, offset = 0, firstbin, newnumbins;
        fcomplex *tmpdata = gen_cvect(numbins);
        fcomplex zeros = { 0.0, 0.0 };

        // zero-pad if we try to read before the beginning of the FFT
        if (lobin - obs->lobin < 0) {
            offset = llabs(lobin - obs->lobin);
            for (ii = 0; ii < offset; ii++)
                tmpdata[ii] = zeros;
        }
        firstbin = (lobin - obs->lobin) + offset;
        newnumbins = numbins - offset;

        // zero-pad if we try to read beyond the end of the FFT
        if (firstbin + newnumbins > obs->numbins) {
            long long numpad = firstbin + newnumbins - obs->numbins;
            newnumbins = newnumbins - numpad;
            for (ii = numbins - numpad; ii < numbins; ii++)
                tmpdata[ii] = zeros;
        }
        // Now grab the data we need
        memcpy(tmpdata + offset, obs->fft + firstbin, sizeof(fcomplex) * newnumbins);
        return tmpdata;
    }
     else {
        return read_fcomplex_file(obs->fftfile, lobin - obs->lobin, numbins);
    }
}

void  *get_fourier_amplitudes_gpu(long long lobin, int numbins, accelobs * obs, fcomplex *tmpdata)
{
    static fcomplex zeros = { 0.0, 0.0 };

    if (obs->mmap_file || obs->dat_input) {
        long long ii, offset = 0, firstbin, newnumbins;
        // fcomplex *tmpdata = gen_cvect(numbins);
        
        // zero-pad if we try to read before the beginning of the FFT
        if (lobin - obs->lobin < 0) {
            offset = llabs(lobin - obs->lobin);
            for (ii = 0; ii < offset; ii++)
                tmpdata[ii] = zeros;
        }
        firstbin = (lobin - obs->lobin) + offset;
        newnumbins = numbins - offset;

        // zero-pad if we try to read beyond the end of the FFT
        if (firstbin + newnumbins > obs->numbins) {
            long long numpad = firstbin + newnumbins - obs->numbins;
            newnumbins = newnumbins - numpad;
            for (ii = numbins - numpad; ii < numbins; ii++)
                tmpdata[ii] = zeros;
        }
        // Now grab the data we need
        memcpy(tmpdata + offset, obs->fft + firstbin, sizeof(fcomplex) * newnumbins);
        // return tmpdata;
    }
     else {
        read_fcomplex_file_gpu(obs->fftfile, lobin - obs->lobin, numbins, tmpdata);
    }
}

void  *get_fourier_amplitudes_dat(long long lobin, int numbins, accelobs * obs, fcomplex *tmpdata, long long *offset_bk, long long *firstbin_bk, long long *numpad_bk, long long *newnumbins_bk)
{
    static fcomplex zeros = { 0.0, 0.0 };

    {
        long long ii, offset = 0, firstbin, newnumbins;
        // fcomplex *tmpdata = gen_cvect(numbins);
        
        // zero-pad if we try to read before the beginning of the FFT
        if (lobin - obs->lobin < 0) {
            offset = llabs(lobin - obs->lobin);
            for (ii = 0; ii < offset; ii++)
                tmpdata[ii] = zeros;
        }
        firstbin = (lobin - obs->lobin) + offset;
        newnumbins = numbins - offset;

        *offset_bk = offset;
        *firstbin_bk = firstbin;
        *newnumbins_bk = newnumbins;


        // zero-pad if we try to read beyond the end of the FFT
        if (firstbin + newnumbins > obs->numbins) {
            long long numpad = firstbin + newnumbins - obs->numbins;
            newnumbins = newnumbins - numpad;
            *numpad_bk = numpad;
            *newnumbins_bk = newnumbins;
            for (ii = numbins - numpad; ii < numbins; ii++)
                tmpdata[ii] = zeros;
        }
        else 
            *numpad_bk = 0;
        // Now grab the data we need
        memcpy(tmpdata + offset, obs->fft + firstbin, sizeof(fcomplex) * newnumbins);
        // return tmpdata;
    }
}

void  get_fourier_amplitudes_gpu_1(long long lobin, int numbins, accelobs * obs, fcomplex *tmpdata)
{
    static fcomplex zeros = { 0.0, 0.0 };
    long long ii, offset = 0, firstbin, newnumbins;
    // fcomplex *tmpdata = gen_cvect(numbins);
        
    // zero-pad if we try to read before the beginning of the FFT
    if (lobin - obs->lobin < 0) {
        offset = llabs(lobin - obs->lobin);
        Set_cufftComplex_date_as_zero_gpu(tmpdata, offset);
        // for (ii = 0; ii < offset; ii++)
        //     tmpdata[ii] = zeros;
    }
    firstbin = (lobin - obs->lobin) + offset;
    newnumbins = numbins - offset;

    // zero-pad if we try to read beyond the end of the FFT
    if (firstbin + newnumbins > obs->numbins) {
        long long numpad = firstbin + newnumbins - obs->numbins;
        newnumbins = newnumbins - numpad;
        Set_cufftComplex_date_as_zero_gpu(tmpdata + (numbins - numpad), numpad); 
        // for (ii = numbins - numpad; ii < numbins; ii++)
        //     tmpdata[ii] = zeros;
    }
    // cudaMemcpy(tmpdata + offset, obs->fft_gpu + firstbin, sizeof(fcomplex) * newnumbins, cudaMemcpyDeviceToDevice);
    cudaMemcpy(tmpdata + offset, obs->fft + firstbin, sizeof(fcomplex) * newnumbins, cudaMemcpyHostToDevice);
}

void  *get_fourier_amplitudes_gpu_list(long long lobin, int numbins, accelobs * obs, fcomplex *tmpdata, int readdatanumid)
{
    fcomplex zeros = { 0.0, 0.0 };

    if (obs->mmap_file || obs->dat_input) {
        

        long long ii, offset = 0, firstbin, newnumbins;
        // fcomplex *tmpdata = gen_cvect(numbins);
        
        // zero-pad if we try to read before the beginning of the FFT
        if (lobin - obs->lobin < 0) {
            offset = llabs(lobin - obs->lobin);
            for (ii = 0; ii < offset; ii++)
                tmpdata[ii] = zeros;
        }
        firstbin = (lobin - obs->lobin) + offset;
        newnumbins = numbins - offset;

        // zero-pad if we try to read beyond the end of the FFT
        if (firstbin + newnumbins > obs->numbins) {
            long long numpad = firstbin + newnumbins - obs->numbins;
            newnumbins = newnumbins - numpad;
            for (ii = numbins - numpad; ii < numbins; ii++)
                tmpdata[ii] = zeros;
        }
        // Now grab the data we need
        fcomplex *obsfft;
        obsfft = obs->fftlist[readdatanumid];
        memcpy(tmpdata + offset, obsfft + firstbin, sizeof(fcomplex) * newnumbins);
        // memcpy(tmpdata + offset, obs->fftlist[readdatanumid] + firstbin, sizeof(fcomplex) * newnumbins);
        // return tmpdata;
    }
     else {
        read_fcomplex_file_gpu(obs->fftfilelist[readdatanumid], lobin - obs->lobin, numbins, tmpdata);
    }
}


ffdotpows *subharm_fderivs_vol(int numharm, int harmnum,
                               double fullrlo, double fullrhi,
                               subharminfo * shi, accelobs * obs)
{
    int ii, numdata, fftlen, binoffset;
    long long lobin;
    float powargr, powargi;
    double drlo, drhi, harm_fract;
    fcomplex *data, *pdata;
    fftwf_plan invplan;
    ffdotpows *ffdot = (ffdotpows *) malloc(sizeof(ffdotpows));

    /* Calculate and get the required amplitudes */
    harm_fract = (double) harmnum / (double) numharm;
    drlo = calc_required_r(harm_fract, fullrlo);
    drhi = calc_required_r(harm_fract, fullrhi);
    ffdot->rlo = (long long) floor(drlo);
    ffdot->zlo = calc_required_z(harm_fract, obs->zlo);
    ffdot->wlo = calc_required_w(harm_fract, obs->wlo);
    /* Initialize the lookup indices */
    if (numharm > 1 && !obs->inmem) {
        double rr, subr;
        for (ii = 0; ii < obs->corr_uselen; ii++) {
            rr = fullrlo + ii * ACCEL_DR;
            subr = calc_required_r(harm_fract, rr);
            shi->rinds[ii] = index_from_r(subr, ffdot->rlo);
        }
        double zz, subz;
        for (ii = 0; ii < obs->numz; ii++) {
            zz = obs->zlo + ii * ACCEL_DZ;
            subz = calc_required_z(harm_fract, zz);
            shi->zinds[ii] = index_from_z(subz, ffdot->zlo);
        }
    }
    ffdot->rinds = shi->rinds;
    // The +1 below is important!
    ffdot->numrs = (int) ((ceil(drhi) - floor(drlo))
                          * ACCEL_RDR + DBLCORRECT) + 1;
    if (numharm == 1 && harmnum == 1) {
        ffdot->numrs = obs->corr_uselen;
    } else {
        if (ffdot->numrs % ACCEL_RDR)
            ffdot->numrs = (ffdot->numrs / ACCEL_RDR + 1) * ACCEL_RDR;
    }
    
    ffdot->zinds = shi->zinds;
    ffdot->numzs = shi->numkern_zdim;
    ffdot->numws = shi->numkern_wdim;

    /* Determine the largest kernel halfwidth needed to analyze the current subharmonic */
    /* Verified numerically that, as long as we have symmetric z's and w's, */
    /* shi->kern[0][0].kern_half_width is the maximal halfwidth over the range of w's and z's */
    binoffset = shi->kern[0][0].kern_half_width;
    fftlen = shi->kern[0][0].fftlen;
    lobin = ffdot->rlo - binoffset;
    numdata = fftlen / ACCEL_NUMBETWEEN;
    data = get_fourier_amplitudes(lobin, numdata, obs); // the data: done fft
    //if (!obs->mmap_file && !obs->dat_input && 0)
    //    printf("This is newly malloc'd!\n");

    // Normalize the Fourier amplitudes
    if (obs->nph > 0.0) {
        //  Use freq 0 normalization if requested (i.e. photons)
        double norm = 1.0 / sqrt(obs->nph);
        for (ii = 0; ii < numdata; ii++) {
            data[ii].r *= norm;
            data[ii].i *= norm;
        }
    } else if (obs->norm_type == 0) {
        // default block median normalization
        float *powers;
        double norm;
        powers = gen_fvect(numdata);
        for (ii = 0; ii < numdata; ii++)
            powers[ii] = POWER(data[ii].r, data[ii].i);
        norm = 1.0 / sqrt(median(powers, numdata) / log(2.0));
        vect_free(powers);
        for (ii = 0; ii < numdata; ii++) {
            data[ii].r *= norm;
            data[ii].i *= norm;
        }
    } else {
        // optional running double-tophat local-power normalization
        float *powers, *loc_powers;
        powers = gen_fvect(numdata);
        for (ii = 0; ii < numdata; ii++) {
            powers[ii] = POWER(data[ii].r, data[ii].i);
        }
        loc_powers = corr_loc_pow(powers, numdata);
        for (ii = 0; ii < numdata; ii++) {
            float norm = invsqrtf(loc_powers[ii]);
            data[ii].r *= norm;
            data[ii].i *= norm;
        }
        vect_free(powers);
        vect_free(loc_powers);
    }

    // Prep, spread, and FFT the data
    pdata = gen_cvect(fftlen);

    spread_no_pad(data, fftlen / ACCEL_NUMBETWEEN, pdata, fftlen, ACCEL_NUMBETWEEN);
        // Note COMPLEXFFT is not thread-safe because of wisdom caching
    COMPLEXFFT(pdata, fftlen, -1);

    // Create the output power array
    ffdot->powers = gen_f3Darr(ffdot->numws, ffdot->numzs, ffdot->numrs);

    // Create a plan with temp arrays.  We will reuse the plan
    // with the new-array FFTW execute functions

        fcomplex *tmpdat = gen_cvect(fftlen);
        fcomplex *tmpout = gen_cvect(fftlen);
        // Compute the inverse FFT plan (these are in/out array specific)
        // FFTW planning is *not* thread-safe
        invplan = fftwf_plan_dft_1d(fftlen, (fftwf_complex *) tmpdat,
                                    (fftwf_complex *) tmpout, +1,
                                    FFTW_MEASURE | FFTW_DESTROY_INPUT);
        vect_free(tmpdat);
        vect_free(tmpout);

        // Perform the correlations in a thread-safe manner
        #ifdef _OPENMP
        #pragma omp parallel default(none) shared(pdata,shi,fftlen,binoffset,ffdot,invplan)
        #endif
        {
            const float norm = 1.0 / (fftlen * fftlen);
            const int offset = binoffset * ACCEL_NUMBETWEEN;
            // tmpdat gets overwritten during the correlation
            fcomplex *tmpdat = gen_cvect(fftlen);
            fcomplex *tmpout = gen_cvect(fftlen);
            int jj;
            #ifdef _OPENMP
            // #pragma omp for collapse(2)  Do we want this somehow?
            #pragma omp for
            #endif
            /* Check, should we add the collapse to parallelize numws and numzs loops? */
            for (ii = 0; ii < ffdot->numws; ii++) {
                for (jj = 0; jj < ffdot->numzs; jj++) {
                    int kk;
                    /*input data*/
                    float *fkern = (float *) shi->kern[ii][jj].data;
                    float *fpdata = (float *) pdata;
                    /*output data*/
                    float *fdata = (float *) tmpdat;
                    float *outpows = ffdot->powers[ii][jj];
                    // multiply data and kernel 
                    // (using floats for better vectorization)
                    #if (defined(__GNUC__) || defined(__GNUG__)) &&         \
                        !(defined(__clang__) || defined(__INTEL_COMPILER))
                    #pragma GCC ivdep
                    #endif
                    for (kk = 0; kk < fftlen * 2; kk += 2) {
                        const float dr = fpdata[kk], di = fpdata[kk + 1];
                        const float kr = fkern[kk], ki = fkern[kk + 1];
                        fdata[kk] = dr * kr + di * ki;
                        fdata[kk + 1] = di * kr - dr * ki;
                    }
                    // Do the inverse FFT (tmpdat -> tmpout)
                    fftwf_execute_dft(invplan, (fftwf_complex *) tmpdat,
                                    (fftwf_complex *) tmpout);
                    // Turn the good parts of the result into powers and store
                    // them in the output matrix
                    fdata = (float *) tmpout;
                    #if (defined(__GNUC__) || defined(__GNUG__)) &&         \
                        !(defined(__clang__) || defined(__INTEL_COMPILER))
                    #pragma GCC ivdep
                    #endif
                    for (kk = 0; kk < ffdot->numrs; kk++) {
                        const int ind = 2 * (kk + offset);
                        outpows[kk] = (fdata[ind] * fdata[ind] +
                                    fdata[ind + 1] * fdata[ind + 1]) * norm;
                    }
                }
            }
            vect_free(tmpdat);
            vect_free(tmpout);
        }
    

    // Free data and the spread-data
    vect_free(data);
    vect_free(pdata);
    return ffdot;
}

ffdotpows *ini_subharm_fderivs_vol(int numharm, int harmnum, 
			       double fullrlo, double fullrhi, 
			       subharminfo *shi, accelobs *obs)
{
    double drlo, drhi, harm_fract;
    static ffdotpows *ffdot;
    static int first_time = 1;
    
    if(first_time)
        ffdot = (ffdotpows *) malloc(sizeof(ffdotpows));
    else 
        first_time = 0;

    /* Calculate and get the required amplitudes */
    harm_fract = (double) harmnum / (double) numharm;
    drlo = calc_required_r(harm_fract, fullrlo);
    drhi = calc_required_r(harm_fract, fullrhi);
    ffdot->rlo = (long long) floor(drlo);
    ffdot->zlo = calc_required_z(harm_fract, obs->zlo);
    // The +1 below is important!
    ffdot->numrs = (int) ((ceil(drhi) - floor(drlo))
                        * ACCEL_RDR + DBLCORRECT) + 1;
    if (numharm == 1 && harmnum == 1) {
        ffdot->numrs = obs->corr_uselen;
    } else {
        if (ffdot->numrs % ACCEL_RDR)
            ffdot->numrs = (ffdot->numrs / ACCEL_RDR + 1) * ACCEL_RDR;
    }
    ffdot->numzs = shi->numkern_zdim;
    return ffdot;

}

cufftComplex *cp_kernel_array_to_gpu(subharminfo **subharminfs, int numharmstages, int **offset_array)
//cp kernel arrays contained in **subharminfs to GPU memory *kernel_array_on_gpu
//and store the offset within *kernel_array_on_gpu for every array, offsets stored in **offset_array
//subharminfs: input
//numharmstages: input
//offset_array: output
//kernel_array_on_gpu: output
//
{


	int harm, harmtosum, stage;		
	int ii, jj;
	int kernel_total_size =0 ;
	int numkern, fftlen;
	unsigned int offset_base = 0, offset_tmp;
	cufftComplex *kernel_vect_host;
    cufftComplex *kernel_vect_on_gpu;

	//printf("cp_kernel_array_to_gpu: Check CUDA devices\n");
	
	numkern = subharminfs[0][0].numkern;
	fftlen = subharminfs[0][0].kern[0][0].fftlen;
	offset_array[0][0] = kernel_total_size ; 
	kernel_total_size = kernel_total_size + numkern * fftlen;
	
	for(stage=1; stage<numharmstages; stage++){
		harmtosum = 1 << stage;		
		for (harm = 1; harm < harmtosum; harm += 2) {               					
			numkern = subharminfs[stage][harm-1].numkern;
			fftlen = subharminfs[stage][harm-1].kern[0][0].fftlen;
			offset_array[stage][harm-1] = kernel_total_size ; 
			kernel_total_size = kernel_total_size + numkern * fftlen;
		}
	}
	cudaMalloc((void **)&kernel_vect_on_gpu, sizeof(cufftComplex) * kernel_total_size);
    // CUDA_CHECK(cudaGetLastError());
	kernel_vect_host = (cufftComplex *)malloc( sizeof(cufftComplex) * kernel_total_size);
	//take kernel_vect_host as buffer
	 //stage : 0
	numkern = subharminfs[0][0].numkern;
    fftlen = subharminfs[0][0].kern[0][0].fftlen;
    offset_base = offset_array[0][0];
    for(ii=0; ii<numkern; ii++)
    {
        for(jj=0; jj<fftlen; jj++){   		
            offset_tmp = offset_base + fftlen*ii + jj;
            kernel_vect_host[offset_tmp].x = subharminfs[0][0].kern[0][ii].data[jj].r;
            kernel_vect_host[offset_tmp].y = subharminfs[0][0].kern[0][ii].data[jj].i;
        }
    }

   //other stages
	if (numharmstages > 1) {   
	for(stage=1; stage<numharmstages; stage++){	
        harmtosum = 1 << stage;		
        for(harm = 1; harm < harmtosum; harm += 2) {         

            offset_base = offset_array[stage][harm-1];

            fftlen = subharminfs[stage][harm-1].kern[0][0].fftlen;
            numkern = subharminfs[stage][harm-1].numkern;

            for(ii=0; ii<numkern; ii++){
                for(jj=0; jj<fftlen; jj++){
                    offset_tmp = offset_base + fftlen*ii + jj ;
                    kernel_vect_host[offset_tmp].x = subharminfs[stage][harm-1].kern[0][ii].data[jj].r;
                    kernel_vect_host[offset_tmp].y = subharminfs[stage][harm-1].kern[0][ii].data[jj].i;
                    }
                }
                
            }
        }
	}   
   cudaMemcpy( kernel_vect_on_gpu, kernel_vect_host, sizeof(cufftComplex) * kernel_total_size, cudaMemcpyHostToDevice);
//    CUDA_CHECK(cudaGetLastError());

   free(kernel_vect_host);

   return kernel_vect_on_gpu;
}

void get_rind_zind_gpu(unsigned short *d_rinds_gpu, unsigned short *d_zinds_gpu, unsigned short *rinds_cpu, unsigned short *zinds_cpu, int numharmstages, accelobs obs, double fullrlo)
{
    int ii, kk=0;
    // unsigned short *rinds_cpu;
    // unsigned short *zinds_cpu;
    unsigned short *rinds;
    unsigned short *zinds;
    
    // cudaMallocHost((void**)&rinds_cpu, sizeof(unsigned short)*obs.corr_uselen*(pow(2,numharmstages)-1));
    // cudaMallocHost((void**)&zinds_cpu, sizeof(unsigned short)*obs.numz*(pow(2,numharmstages)-1));


    double rr, subr;
    double drlo, harm_fract;
    int stage;
    int harmtosum, harm;
    
    int corr_uselen= obs.corr_uselen;
    double obszlo = obs.zlo;
    int obsnumz = obs.numz;

    for (stage = 1; stage < numharmstages; stage++)
    {
        harmtosum = 1 << stage;
        for (harm = 1; harm < harmtosum; harm += 2)
        {
            harm_fract = (double) harm / (double) harmtosum;
            drlo = calc_required_r(harm_fract, fullrlo);
            long long rlo = (long long) floor(drlo);
            int zlo = calc_required_z(harm_fract, obszlo);
            rinds = rinds_cpu+kk*corr_uselen;
            zinds = zinds_cpu+kk*obsnumz;
            for (ii = 0; ii < corr_uselen; ii++) {
                rr = fullrlo + ii * ACCEL_DR;
                subr = calc_required_r(harm_fract, rr);
                rinds[ii] = index_from_r(subr, rlo);
            }
            double zz, subz;
            for (ii = 0; ii < obsnumz; ii++) {
                zz = obszlo + ii * ACCEL_DZ;
                subz = calc_required_z(harm_fract, zz);
                zinds[ii] = index_from_z(subz, zlo);
            }
            kk++;
        }
    }

    cudaMemcpy(d_rinds_gpu, rinds_cpu, sizeof(unsigned short)*obs.corr_uselen*(pow(2,numharmstages)-1), cudaMemcpyHostToDevice);
    // CUDA_CHECK(cudaGetLastError());
    cudaMemcpy(d_zinds_gpu, zinds_cpu, sizeof(unsigned short)*obs.numz*(pow(2,numharmstages)-1), cudaMemcpyHostToDevice);
    // CUDA_CHECK(cudaGetLastError());

    // cudaFree(rinds_cpu);
    // cudaFree(zinds_cpu);
}


void *subharm_fderivs_vol_gpu(int numharm, int harmnum, double fullrlo, double fullrhi, subharminfo *shi, accelobs *obs, cufftComplex *fkern_gpu, cufftComplex *pdata_gpu, cufftComplex *tmpdat_gpu, cufftComplex *tmpout_gpu, float *outpows_gpu, float *outpows_gpu_obs, cufftComplex *pdata, int tip, unsigned short *d_zinds_gpu, unsigned short *d_rinds_gpu, unsigned short *zinds_cpu, unsigned short *rinds_cpu, ffdotpows *fundamental, int **offset_array, int stage, cufftComplex *data_gpu, float *powers)
{
    int ii, numdata, fftlen, binoffset;
    long long lobin;
    float powargr, powargi;
    double drlo, drhi, harm_fract;
    static fcomplex *data;
    static float *powers_gpu = NULL;
    static int firsttime = 1;

    harm_fract = (double) harmnum / (double) numharm;
    drlo = calc_required_r(harm_fract, fullrlo);
    drhi = calc_required_r(harm_fract, fullrhi);
    long long rlo = (long long) floor(drlo);
    // register int zlo = calc_required_z(harm_fract, obs->zlo);
    /* Initialize the lookup indices */
    // if (numharm > 1 && !obs->inmem) {
    //     double rr, subr;
    //     for (ii = 0; ii < obs->corr_uselen; ii++) {
    //         rr = fullrlo + ii * ACCEL_DR;
    //         subr = calc_required_r(harm_fract, rr);
    //         rinds_cpu[ii] = index_from_r(subr, rlo);
    //     }
    //     double zz, subz;
    //     for (ii = 0; ii < obs->numz; ii++) {
    //         zz = obs->zlo + ii * ACCEL_DZ;
    //         subz = calc_required_z(harm_fract, zz);
    //         zinds_cpu[ii] = index_from_z(subz, zlo);
    //     }
    // }
    // The +1 below is important!
    int numrs = (int) ((ceil(drhi) - floor(drlo))
                          * ACCEL_RDR + DBLCORRECT) + 1;
    if (numharm == 1 && harmnum == 1) {
        numrs = obs->corr_uselen;
    } else {
        if (numrs % ACCEL_RDR)
            numrs = (numrs / ACCEL_RDR + 1) * ACCEL_RDR;
    }
    int numzs = shi->numkern_zdim;

    /* Determine the largest kernel halfwidth needed to analyze the current subharmonic */
    /* Verified numerically that, as long as we have symmetric z's and w's, */
    /* shi->kern[0][0].kern_half_width is the maximal halfwidth over the range of w's and z's */
    binoffset = shi->kern[0][0].kern_half_width;
    fftlen = shi->kern[0][0].fftlen;
    lobin = rlo - binoffset;
    numdata = fftlen / ACCEL_NUMBETWEEN;

    static double norm_data;

    // if(obs->mmap_file || obs->dat_input)
    // {
    //     float norm_data_tmp;
    //     if (firsttime) 
    //     {
    //         // cudaMalloc(&data, sizeof(fcomplex) * fftlen);
    //         cudaMalloc(&powers_gpu, sizeof(float) * fftlen);
    //         firsttime = 0;
    //     }
    //     get_fourier_amplitudes_gpu_1(lobin, numdata, obs, data_gpu);
    //     // cudaMemcpy(data_gpu, data, sizeof(cufftComplex) * numdata, cudaMemcpyDeviceToDevice);

    //     compute_power_gpu(data_gpu, powers_gpu, numdata);

    //     sort_and_get_median_gpu(powers_gpu, numdata, &norm_data_tmp);
    //     norm_data = 1.0 / sqrt(norm_data_tmp / log(2.0));

    //     spread_no_pad_gpu(data_gpu, fftlen / ACCEL_NUMBETWEEN, pdata_gpu, fftlen, ACCEL_NUMBETWEEN, norm_data);
    // }
    // else
    {
        if(firsttime)
        {
            data = malloc(sizeof(fcomplex)*fftlen);
            firsttime = 0;
        }


        if(obs->dat_input&&obs->numbins<100000000)
        {
            long long offset_bk, firstbin_bk, numpad_bk, newnumbins_bk;
            get_fourier_amplitudes_dat(lobin, numdata, obs, data, &offset_bk, &firstbin_bk, &numpad_bk, &newnumbins_bk);

            for (ii = 0; ii < numdata; ii++)
                powers[ii] = POWER(data[ii].r, data[ii].i);
            norm_data = 1.0 / sqrt(median(powers, numdata) / log(2.0));

            spread_no_pad_gpu_dat(obs->fft_gpu+firstbin_bk, fftlen / ACCEL_NUMBETWEEN, pdata_gpu, fftlen, ACCEL_NUMBETWEEN, norm_data, offset_bk, numpad_bk, newnumbins_bk);
        }
        else 
        {
            // data = get_fourier_amplitudes(lobin, numdata, obs); // the data: done fft
            get_fourier_amplitudes_gpu(lobin, numdata, obs, data);
                    
            for (ii = 0; ii < numdata; ii++)
                    powers[ii] = POWER(data[ii].r, data[ii].i);
            norm_data = 1.0 / sqrt(median(powers, numdata) / log(2.0));

            // 等待所有操作完成
            cudaMemcpy(data_gpu, data, sizeof(cufftComplex) * numdata, cudaMemcpyHostToDevice);
            spread_no_pad_gpu(data_gpu, fftlen / ACCEL_NUMBETWEEN, pdata_gpu, fftlen, ACCEL_NUMBETWEEN, norm_data);
        }

    }

    
            
    // cudaMemcpy(data_gpu, data, sizeof(cufftComplex)*numdata, cudaMemcpyHostToDevice);


    

    

    // Note COMPLEXFFT is not thread-safe because of wisdom caching

    // FFTonGPU(pdata_gpu, pdata_gpu, fftlen, 1, -1, plan_pdata_gpu_array[stage][harmnum-1]);
    cufftExecC2C(plan_pdata_gpu_array[stage][harmnum-1], pdata_gpu, pdata_gpu, -1);


    const float norm = 1.0 / (fftlen * fftlen);
    int offset = binoffset * ACCEL_NUMBETWEEN;
    long long rlen = (obs->highestbin + obs->corr_uselen) * ACCEL_RDR;

    {
        int offs = 0;
        if(numharm ==1 && harmnum ==1)
            offs = 0;
        else if(numharm > 1)
            offs =  offset_array[stage][harmnum-1];
        
        loops_in_GPU_1(pdata_gpu, fkern_gpu+offs ,tmpdat_gpu, fftlen, numzs);
        // FFTonGPU(tmpdat_gpu, tmpout_gpu,fftlen, numzs, +1, plan_tmpdat_gpu_array[stage][harmnum-1]);
        cufftExecC2C(plan_tmpdat_gpu_array[stage][harmnum-1], tmpdat_gpu, tmpout_gpu, 1);

        if(obs->inmem)
            loops_in_GPU_2(tmpout_gpu, outpows_gpu, numrs, numzs,offset, fftlen,norm, outpows_gpu_obs, rlen, rlo*ACCEL_RDR, 1);
        else
        {
            if(numharm == 1 && harmnum == 1)
            {
                loops_in_GPU_2(tmpout_gpu, outpows_gpu, numrs, numzs,offset, fftlen, norm, outpows_gpu_obs, rlen, rlo*ACCEL_RDR, 0);
            }
            if(numharm > 1)
            {
                // cudaMemcpy(d_rinds_gpu, (unsigned short *)rinds_cpu, sizeof(unsigned short)*obs->corr_uselen, cudaMemcpyHostToDevice);
                // cudaMemcpy(d_zinds_gpu, (unsigned short *)zinds_cpu, sizeof(unsigned short)*obs->numz, cudaMemcpyHostToDevice);
                add_subharm_gpu(outpows_gpu, tmpout_gpu, d_rinds_gpu+(int)(((pow(2,stage-1)-1)+harmnum/2)*obs->corr_uselen), d_zinds_gpu+(int)(((pow(2,stage-1)-1)+harmnum/2)*obs->numz), fundamental->numrs, fundamental->numzs, fftlen, numzs, offset, norm);
            }
        }
    }

    // vect_free(data);
}

void *subharm_fderivs_vol_gpu_1(int numharm, int harmnum, double fullrlo, double fullrhi, subharminfo *shi, accelobs *obs, cufftComplex *fkern_gpu, cufftComplex *pdata_gpu, cufftComplex *tmpdat_gpu, cufftComplex *tmpout_gpu, float *outpows_gpu, float *outpows_gpu_obs, cufftComplex *pdata, int tip, unsigned short *d_zinds_gpu, unsigned short *d_rinds_gpu, unsigned short *zinds_cpu, unsigned short *rinds_cpu, ffdotpows *fundamental, int **offset_array, int stage, cufftComplex *data_gpu, float *powers, int readid)
{
    int ii, numdata, fftlen, binoffset;
    long long lobin;
    float powargr, powargi;
    double drlo, drhi, harm_fract;
    static fcomplex *data;
    static int firsttime = 1;

    harm_fract = (double) harmnum / (double) numharm;
    drlo = calc_required_r(harm_fract, fullrlo);
    drhi = calc_required_r(harm_fract, fullrhi);
    long long rlo = (long long) floor(drlo);
    // register int zlo = calc_required_z(harm_fract, obs->zlo);
    /* Initialize the lookup indices */
    // if (numharm > 1 && !obs->inmem) {
    //     double rr, subr;
    //     for (ii = 0; ii < obs->corr_uselen; ii++) {
    //         rr = fullrlo + ii * ACCEL_DR;
    //         subr = calc_required_r(harm_fract, rr);
    //         rinds_cpu[ii] = index_from_r(subr, rlo);
    //     }
    //     double zz, subz;
    //     for (ii = 0; ii < obs->numz; ii++) {
    //         zz = obs->zlo + ii * ACCEL_DZ;
    //         subz = calc_required_z(harm_fract, zz);
    //         zinds_cpu[ii] = index_from_z(subz, zlo);
    //     }
    // }
    // The +1 below is important!
    int numrs = (int) ((ceil(drhi) - floor(drlo))
                          * ACCEL_RDR + DBLCORRECT) + 1;
    if (numharm == 1 && harmnum == 1) {
        numrs = obs->corr_uselen;
    } else {
        if (numrs % ACCEL_RDR)
            numrs = (numrs / ACCEL_RDR + 1) * ACCEL_RDR;
    }
    int numzs = shi->numkern_zdim;

    /* Determine the largest kernel halfwidth needed to analyze the current subharmonic */
    /* Verified numerically that, as long as we have symmetric z's and w's, */
    /* shi->kern[0][0].kern_half_width is the maximal halfwidth over the range of w's and z's */
    binoffset = shi->kern[0][0].kern_half_width;
    fftlen = shi->kern[0][0].fftlen;
    lobin = rlo - binoffset;
    numdata = fftlen / ACCEL_NUMBETWEEN;

    if(firsttime)
    {
        data = (fcomplex *) malloc(sizeof(fcomplex)*fftlen);
        firsttime = 0;
    }
    // data = get_fourier_amplitudes(lobin, numdata, obs); // the data: done fft
    // get_fourier_amplitudes_gpu(lobin, numdata, obs, data);

    
    get_fourier_amplitudes_gpu_list(lobin, numdata, obs, data, readid);
    cudaMemcpy(data_gpu, data, sizeof(cufftComplex) * numdata, cudaMemcpyHostToDevice);
    // CUDA_CHECK(cudaGetLastError());

    // cufftComplex data_cpu_bk[10];
    // cudaMemcpy(data_cpu_bk, data_gpu, sizeof(cufftComplex) * 10, cudaMemcpyDeviceToHost);


    double norm_data;
    for (ii = 0; ii < numdata; ii++)
            powers[ii] = POWER(data[ii].r, data[ii].i);
    norm_data = 1.0 / sqrt(median(powers, numdata) / log(2.0));
    

    // 等待所有操作完成
    spread_no_pad_gpu(data_gpu, fftlen / ACCEL_NUMBETWEEN, pdata_gpu, fftlen, ACCEL_NUMBETWEEN, norm_data);
    

    // Note COMPLEXFFT is not thread-safe because of wisdom caching

    // FFTonGPU(pdata_gpu, pdata_gpu, fftlen, 1, -1, plan_pdata_gpu_array[stage][harmnum-1]);
    cufftExecC2C(plan_pdata_gpu_array[stage][harmnum-1], pdata_gpu, pdata_gpu, -1);
    // CUDA_CHECK(cudaGetLastError());

    const float norm = 1.0 / (fftlen * fftlen);
    int offset = binoffset * ACCEL_NUMBETWEEN;
    long long rlen = (obs->highestbin + obs->corr_uselen) * ACCEL_RDR;

    {
        int offs = 0;
        if(numharm ==1 && harmnum ==1)
            offs = 0;
        else if(numharm > 1)
            offs =  offset_array[stage][harmnum-1];
        
        loops_in_GPU_1(pdata_gpu, fkern_gpu+offs ,tmpdat_gpu, fftlen, numzs);
        
        // FFTonGPU(tmpdat_gpu, tmpout_gpu,fftlen, numzs, +1, plan_tmpdat_gpu_array[stage][harmnum-1]);
        cufftExecC2C(plan_tmpdat_gpu_array[stage][harmnum-1], tmpdat_gpu, tmpout_gpu, 1);
        // CUDA_CHECK(cudaGetLastError());
        
        if(obs->inmem)
        {
            loops_in_GPU_2(tmpout_gpu, outpows_gpu, numrs, numzs,offset, fftlen,norm, outpows_gpu_obs, rlen, rlo*ACCEL_RDR, 1);
        }
        else
        {
            if(numharm == 1 && harmnum == 1)
            {
                loops_in_GPU_2(tmpout_gpu, outpows_gpu, numrs, numzs,offset, fftlen, norm, outpows_gpu_obs, rlen, rlo*ACCEL_RDR, 0);
            }
            if(numharm > 1)
            {
                // cudaMemcpy(d_rinds_gpu, (unsigned short *)rinds_cpu, sizeof(unsigned short)*obs->corr_uselen, cudaMemcpyHostToDevice);
                // cudaMemcpy(d_zinds_gpu, (unsigned short *)zinds_cpu, sizeof(unsigned short)*obs->numz, cudaMemcpyHostToDevice);
                add_subharm_gpu(outpows_gpu, tmpout_gpu, d_rinds_gpu+(int)(((pow(2,stage-1)-1)+harmnum/2)*obs->corr_uselen), d_zinds_gpu+(int)(((pow(2,stage-1)-1)+harmnum/2)*obs->numz), fundamental->numrs, fundamental->numzs, fftlen, numzs,offset, norm);
            }
        }
    }

    // vect_free(data);
}

void *subharm_fderivs_vol_gpu_1_list(int numharm, int harmnum, double fullrlo, double fullrhi, subharminfo *shi, accelobs *obs, cufftComplex *fkern_gpu, cufftComplex *pdata_gpu, cufftComplex *tmpdat_gpu, cufftComplex *tmpout_gpu, float *outpows_gpu, float *outpows_gpu_obs, cufftComplex *pdata, int tip, unsigned short *d_zinds_gpu, unsigned short *d_rinds_gpu, unsigned short *zinds_cpu, unsigned short *rinds_cpu, ffdotpows *fundamental, int **offset_array, int stage, cufftComplex *data_gpu, float *powers, int readdatanum, int fftlen_0, double *norm_data_gpu, int outpows_gpu_xlen, int outpows_gpu_obs_xlen)  // int readid
{ 
    int ii, jj, numdata, fftlen, binoffset;
    long long lobin;
    float powargr, powargi;
    double drlo, drhi, harm_fract;
    static fcomplex *data;
    static int firsttime = 1;

    harm_fract = (double) harmnum / (double) numharm;
    drlo = calc_required_r(harm_fract, fullrlo);
    drhi = calc_required_r(harm_fract, fullrhi);
    long long rlo = (long long) floor(drlo);
    // register int zlo = calc_required_z(harm_fract, obs->zlo);
    /* Initialize the lookup indices */
    // if (numharm > 1 && !obs->inmem) {
    //     double rr, subr;
    //     for (ii = 0; ii < obs->corr_uselen; ii++) {
    //         rr = fullrlo + ii * ACCEL_DR;
    //         subr = calc_required_r(harm_fract, rr);
    //         rinds_cpu[ii] = index_from_r(subr, rlo);
    //     }
    //     double zz, subz;
    //     for (ii = 0; ii < obs->numz; ii++) {
    //         zz = obs->zlo + ii * ACCEL_DZ;
    //         subz = calc_required_z(harm_fract, zz);
    //         zinds_cpu[ii] = index_from_z(subz, zlo);
    //     }
    // }
    // The +1 below is important!
    int numrs = (int) ((ceil(drhi) - floor(drlo))
                          * ACCEL_RDR + DBLCORRECT) + 1;
    if (numharm == 1 && harmnum == 1) {
        numrs = obs->corr_uselen;
    } else {
        if (numrs % ACCEL_RDR)
            numrs = (numrs / ACCEL_RDR + 1) * ACCEL_RDR;
    }
    int numzs = shi->numkern_zdim;

    /* Determine the largest kernel halfwidth needed to analyze the current subharmonic */
    /* Verified numerically that, as long as we have symmetric z's and w's, */
    /* shi->kern[0][0].kern_half_width is the maximal halfwidth over the range of w's and z's */
    binoffset = shi->kern[0][0].kern_half_width;
    fftlen = shi->kern[0][0].fftlen;
    lobin = rlo - binoffset;
    numdata = fftlen / ACCEL_NUMBETWEEN;

    if(firsttime)
    {
        data = (fcomplex *) malloc(sizeof(fcomplex)*fftlen);
        firsttime = 0;
    }

    double norm_data;
    
    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());

    for(ii=0; ii<readdatanum; ii++)
    {
        get_fourier_amplitudes_gpu_list(lobin, numdata, obs, data, ii);
        cudaMemcpy(data_gpu+ii*numdata, data, sizeof(cufftComplex) * numdata, cudaMemcpyHostToDevice);
        // CUDA_CHECK(cudaGetLastError());

        for (jj = 0; jj < numdata; jj++)
            powers[jj] = POWER(data[jj].r, data[jj].i);
        norm_data = 1.0 / sqrt(median(powers, numdata) / log(2.0));
        cudaMemcpy(norm_data_gpu+ii, &norm_data, sizeof(double), cudaMemcpyHostToDevice);
        // cudaDeviceSynchronize();
        // CUDA_CHECK(cudaGetLastError());
    }
    

    // 等待所有操作完成
    // spread_no_pad_gpu(data_gpu, fftlen / ACCEL_NUMBETWEEN, pdata_gpu, fftlen, ACCEL_NUMBETWEEN, norm_data);
    spread_no_pad_gpu_list(data_gpu, numdata, pdata_gpu, fftlen, ACCEL_NUMBETWEEN, readdatanum, norm_data_gpu);
    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());
    
    // Note COMPLEXFFT is not thread-safe because of wisdom caching

    // FFTonGPU(pdata_gpu, pdata_gpu, fftlen, 1, -1, plan_pdata_gpu_array[stage][harmnum-1]);
    cufftExecC2C(plan_pdata_gpu_array[stage][harmnum-1], pdata_gpu, pdata_gpu, -1);
    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());

    // printf("cufftExecC2C %d  %d  %d  %d\n", ii, numdata, fftlen, readdatanum);

    const float norm = 1.0 / (fftlen * fftlen);
    int offset = binoffset * ACCEL_NUMBETWEEN;
    long long rlen = (obs->highestbin + obs->corr_uselen) * ACCEL_RDR;

    {
        int offs = 0;
        if(numharm ==1 && harmnum ==1)
            offs = 0;
        else if(numharm > 1)
            offs =  offset_array[stage][harmnum-1];
        
        // loops_in_GPU_1(pdata_gpu, fkern_gpu+offs ,tmpdat_gpu, fftlen, numzs);
        loops_in_GPU_1_list(pdata_gpu, fkern_gpu+offs ,tmpdat_gpu, fftlen, numzs, readdatanum);
        // cudaDeviceSynchronize();
        // CUDA_CHECK(cudaGetLastError());
        
        // FFTonGPU(tmpdat_gpu, tmpout_gpu,fftlen, numzs, +1, plan_tmpdat_gpu_array[stage][harmnum-1]);
        cufftExecC2C(plan_tmpdat_gpu_array[stage][harmnum-1], tmpdat_gpu, tmpout_gpu, 1);
        // cudaDeviceSynchronize();
        // CUDA_CHECK(cudaGetLastError());
        
        if(obs->inmem)
        {
            // loops_in_GPU_2(tmpout_gpu, outpows_gpu, numrs, numzs,offset, fftlen,norm, outpows_gpu_obs, rlen, rlo*ACCEL_RDR, 1);
            loops_in_GPU_2_list(tmpout_gpu, outpows_gpu, numrs, numzs, offset, fftlen,norm, outpows_gpu_obs, rlen, rlo*ACCEL_RDR, 1, readdatanum, outpows_gpu_xlen, outpows_gpu_obs_xlen);
            // cudaDeviceSynchronize();
            // CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            if(numharm == 1 && harmnum == 1)
            {
                // loops_in_GPU_2(tmpout_gpu, outpows_gpu, numrs, numzs,offset, fftlen, norm, outpows_gpu_obs, rlen, rlo*ACCEL_RDR, 0);
                loops_in_GPU_2_list(tmpout_gpu, outpows_gpu, numrs, numzs, offset, fftlen, norm, outpows_gpu_obs, rlen, rlo*ACCEL_RDR, 0, readdatanum, outpows_gpu_xlen, outpows_gpu_obs_xlen);
                // cudaDeviceSynchronize();
                // CUDA_CHECK(cudaGetLastError());
            }
            if(numharm > 1)
            {
                // cudaMemcpy(d_rinds_gpu, (unsigned short *)rinds_cpu, sizeof(unsigned short)*obs->corr_uselen, cudaMemcpyHostToDevice);
                // cudaMemcpy(d_zinds_gpu, (unsigned short *)zinds_cpu, sizeof(unsigned short)*obs->numz, cudaMemcpyHostToDevice);
                // add_subharm_gpu(outpows_gpu, tmpout_gpu, d_rinds_gpu+(int)(((pow(2,stage-1)-1)+harmnum/2)*obs->corr_uselen), d_zinds_gpu+(int)(((pow(2,stage-1)-1)+harmnum/2)*obs->numz), fundamental->numrs, fundamental->numzs, fftlen, numzs,offset, norm);
                add_subharm_gpu_list(outpows_gpu, tmpout_gpu, d_rinds_gpu+(int)(((pow(2,stage-1)-1)+harmnum/2)*obs->corr_uselen), d_zinds_gpu+(int)(((pow(2,stage-1)-1)+harmnum/2)*obs->numz), fundamental->numrs, fundamental->numzs, fftlen, numzs, offset, norm, readdatanum, outpows_gpu_xlen);
                // cudaDeviceSynchronize();
                // CUDA_CHECK(cudaGetLastError());
            }
        }
    }

    // vect_free(data);
}



//----------------------initialize cuFFT plans ----------------------
void init_cuFFT_plans(subharminfo **subharminfs, int numharmstages, int inmem)
{
	int harm, harmtosum, stage;		
	int numkern, fftlen;
	
	numkern = subharminfs[0][0].numkern;
	fftlen = subharminfs[0][0].kern[0][0].fftlen;

	cufftPlan1d(&plan_pdata_gpu_array[0][0], fftlen, CUFFT_C2C, 1);
    // CUDA_CHECK(cudaGetLastError());
	cufftPlan1d(&plan_tmpdat_gpu_array[0][0], fftlen, CUFFT_C2C, numkern);
    // CUDA_CHECK(cudaGetLastError());

	if (numharmstages > 1 && !inmem) {
		for(stage=1; stage<numharmstages; stage++){	
			harmtosum = 1 << stage;
			for (harm = 1; harm < harmtosum; harm += 2) {
				numkern = subharminfs[stage][harm-1].numkern;
				fftlen = subharminfs[stage][harm-1].kern[0][0].fftlen;


				cufftPlan1d(&plan_pdata_gpu_array[stage][harm-1], fftlen, CUFFT_C2C, 1);	
                // CUDA_CHECK(cudaGetLastError());	
				cufftPlan1d(&plan_tmpdat_gpu_array[stage][harm-1], fftlen, CUFFT_C2C, numkern);	
                // CUDA_CHECK(cudaGetLastError());	
			}	
		}	
	}
}

void init_cuFFT_plans_list(subharminfo **subharminfs, int numharmstages, int inmem, int readdatanum)
{
	int harm, harmtosum, stage;		
	int numkern, fftlen;
	
	numkern = subharminfs[0][0].numkern;
	fftlen = subharminfs[0][0].kern[0][0].fftlen;

	cufftPlan1d(&plan_pdata_gpu_array[0][0], fftlen, CUFFT_C2C, readdatanum);
    // CUDA_CHECK(cudaGetLastError());
	cufftPlan1d(&plan_tmpdat_gpu_array[0][0], fftlen, CUFFT_C2C, numkern*readdatanum);
    // cufftPlanMany(&plan_tmpdat_gpu_array[0][0], 1, &fftlen, NULL, 1, fftlen, NULL, 1, fftlen, CUFFT_C2C, numkern*readdatanum);
    // CUDA_CHECK(cudaGetLastError());

	if (numharmstages > 1 && !inmem) {
		for(stage=1; stage<numharmstages; stage++){	
			harmtosum = 1 << stage;
			for (harm = 1; harm < harmtosum; harm += 2) {
				numkern = subharminfs[stage][harm-1].numkern;
				fftlen = subharminfs[stage][harm-1].kern[0][0].fftlen;


				cufftPlan1d(&plan_pdata_gpu_array[stage][harm-1], fftlen, CUFFT_C2C, readdatanum);	
                // CUDA_CHECK(cudaGetLastError());	
				cufftPlan1d(&plan_tmpdat_gpu_array[stage][harm-1], fftlen, CUFFT_C2C, numkern*readdatanum);
                // cufftPlanMany(&plan_tmpdat_gpu_array[stage][harm-1], 1, &fftlen, NULL, 1, fftlen, NULL, 1, fftlen, CUFFT_C2C, numkern*readdatanum);
                // CUDA_CHECK(cudaGetLastError());	
			}	
		}	
	}
}

//----------------------destroy cuFFT plans ----------------------
void destroy_cuFFT_plans(subharminfo **subharminfs, int numharmstages, int inmem)
{
	int harm, harmtosum, stage;

	cufftDestroy(plan_pdata_gpu_array[0][0]);
    CUDA_CHECK(cudaGetLastError());
	cufftDestroy(plan_tmpdat_gpu_array[0][0]);
    CUDA_CHECK(cudaGetLastError());

    if (numharmstages > 1  && !inmem) {
        for(stage=1; stage<numharmstages; stage++){	
            harmtosum = 1 << stage;		
            for (harm = 1; harm < harmtosum; harm += 2)
            {
                cufftDestroy(plan_pdata_gpu_array[stage][harm-1]);
                CUDA_CHECK(cudaGetLastError());
                cufftDestroy(plan_tmpdat_gpu_array[stage][harm-1]);
                CUDA_CHECK(cudaGetLastError());
            }
        }
    }
}




/***********************************************************************************************************************************/

float * prep_result_on_gpu(subharminfo **subharminfs, int numharmstages)
//cudaMalloc d_result for the whole search process
{

	int harm, harmtosum, stage;		
	int numkern, fftlen;
	int size_d_result;
	
	float *d_result;	
	size_d_result = 0;

	numkern = subharminfs[0][0].numkern;
	fftlen = subharminfs[0][0].kern[0][0].fftlen;

	size_d_result = numkern * fftlen;

	if (numharmstages > 1) {
		for(stage=1; stage<numharmstages; stage++){	
			harmtosum = 1 << stage;		
			for (harm = 1; harm < harmtosum; harm += 2) {               			
				numkern = subharminfs[stage][harm-1].numkern;
				fftlen = subharminfs[stage][harm-1].kern[0][0].fftlen;
									
				if(size_d_result < numkern * fftlen )
				{
					size_d_result = numkern * fftlen;
				}			
			}	
		}	
	}
		//Alloc mem for result on GPU  
  	cudaMalloc((void **)&d_result, size_d_result * sizeof(float));  
	return d_result ;
}

ffdotpows *copy_ffdotpows(ffdotpows * orig)
{
    int ii;
    ffdotpows *copy;
    
    copy = (ffdotpows *) malloc(sizeof(ffdotpows));
    copy->numrs = orig->numrs;
    copy->numzs = orig->numzs;
    copy->numws = orig->numws;
    copy->rlo = orig->rlo;
    copy->zlo = orig->zlo;
    copy->wlo = orig->wlo;
    copy->powers = gen_f3Darr(orig->numws, orig->numzs, orig->numrs);
    for (ii = 0; ii < (orig->numws * orig->numzs * orig->numrs); ii++)
        copy->powers[0][0][ii] = orig->powers[0][0][ii];
    return copy;
}


void fund_to_ffdotplane(ffdotpows * ffd, accelobs * obs)
{
    // This moves the fundamental's ffdot plane powers
    // into the one for the full array
    int ii;
    long long rlen = (obs->highestbin + obs->corr_uselen) * ACCEL_RDR;
    long long offset;
    float *outpow;

    for (ii = 0; ii < ffd->numzs; ii++) {
        offset = ii * rlen;
        outpow = obs->ffdotplane + offset + ffd->rlo * ACCEL_RDR;
        memcpy(outpow, ffd->powers[0][ii], ffd->numrs * sizeof(float));
    }
}


void fund_to_ffdotplane_trans(ffdotpows * ffd, accelobs * obs)
{
    // This moves the fundamental's ffdot plane powers
    // into the one for the full array, but with a transpose
    // so that points in both r and z directions are more
    // memory local (since numz << numr)
    int ii, jj;
    float *outpow = obs->ffdotplane + ffd->rlo * ACCEL_RDR * ffd->numzs;
    for (ii = 0; ii < ffd->numrs; ii++) {
        float *inpow = ffd->powers[0][0] + ii;
        for (jj = 0; jj < ffd->numzs; jj++, inpow += ffd->numrs) {
            *outpow++ = *inpow;
        }
    }
}


void free_ffdotpows(ffdotpows * ffd)
{
    vect_free(ffd->powers[0][0]);
    vect_free(ffd->powers[0]);
    vect_free(ffd->powers);
    free(ffd);
}

void add_ffdotpows(ffdotpows * fundamental,
                   ffdotpows * subharmonic, int numharm, int harmnum)
{
    int ii, jj, kk, ww, rind, zind, wind, subw;
    const double harm_fract = (double) harmnum / (double) numharm;
    for (ii = 0; ii < fundamental->numws; ii++) {
        ww = fundamental->wlo + ii * ACCEL_DW;
        subw = calc_required_w(harm_fract, ww);
        wind = index_from_w(subw, subharmonic->wlo);
        for (jj = 0; jj < fundamental->numzs; jj++) {
            zind = subharmonic->zinds[jj];
            for (kk = 0; kk < fundamental->numrs; kk++) {
                rind = subharmonic->rinds[kk];
                fundamental->powers[ii][jj][kk] += subharmonic->powers[wind][zind][rind];
            }
        }
    }
}

void add_ffdotpows_ptrs(ffdotpows * fundamental,
                        ffdotpows * subharmonic, int numharm, int harmnum)
{
    int ii, jj, kk, ww, wind, subw;
    const int wlo = fundamental->wlo;
    const int numrs = fundamental->numrs;
    const int numzs = fundamental->numzs;
    const int numws = fundamental->numws;
    const double harm_fract = (double) harmnum / (double) numharm;
    float *outpows, *inpows;
    unsigned short *rindsptr, *zindsptr;

    for (ii = 0; ii < numws; ii++) {
        ww = wlo + ii * ACCEL_DW;
        subw = calc_required_w(harm_fract, ww);
        wind = index_from_w(subw, subharmonic->wlo);
        zindsptr = subharmonic->zinds;
        for (jj = 0; jj < numzs; jj++) {
            inpows = subharmonic->powers[wind][*zindsptr++];
            outpows = fundamental->powers[ii][jj];
            rindsptr = subharmonic->rinds;
            for (kk = 0; kk < numrs; kk++)
                *outpows++ += inpows[*rindsptr++];
        }
    }
}


void inmem_add_ffdotpows(ffdotpows * fundamental, accelobs * obs,
                         int numharm, int harmnum)
{
    const int rlo = fundamental->rlo;
    const int numrs = fundamental->numrs;
    const int numzs = fundamental->numzs;
    const double harm_fract = (double) harmnum / (double) numharm;
    int *rinds;

    // Pre-compute the frequency lookup table
    rinds = gen_ivect(numrs);
    {
        int ii, rrint;
        for (ii = 0, rrint = ACCEL_RDR * rlo; ii < numrs; ii++, rrint++)
            rinds[ii] = (int) (rrint * harm_fract + 0.5);
        // exit(0);
    }

    // Now add all the powers
#ifdef _OPENMP
#pragma omp parallel shared(rinds,fundamental,obs)
#endif
    {
        const int zlo = fundamental->zlo;
        const long long rlen = (obs->highestbin + obs->corr_uselen) * ACCEL_RDR;
        float *powptr = fundamental->powers[0][0]; //output data
        float *fdp = obs->ffdotplane; //input data
        int ii, jj, zz, zind, subz;
        float *inpows, *outpows;
        long long offset;
#ifdef _OPENMP
#pragma omp for
#endif
        for (ii = 0; ii < numzs; ii++) {
            zz = zlo + ii * ACCEL_DZ;
            subz = calc_required_z(harm_fract, zz);
            zind = index_from_z(subz, zlo);
            offset = zind * rlen;
            inpows = fdp + offset;
            outpows = powptr + ii * numrs;
#if (defined(__GNUC__) || defined(__GNUG__)) && \
    !(defined(__clang__) || defined(__INTEL_COMPILER))
#pragma GCC ivdep
#endif
            for (jj = 0; jj < numrs; jj++)
                outpows[jj] += inpows[rinds[jj]];
        }
    }
    vect_free(rinds);
}

void get_rinds_gpu(ffdotpows * fundamental, int *rinds_gpu, int numharmstages)
{
    // printf("Malloc in GPU Done\n"); 

    int *rinds_cpu;
    int *rinds_bk;
    
    int stage;
    const int rlo = fundamental->rlo;
    const int numrs = fundamental->numrs;

    cudaMallocHost((void**)&rinds_cpu, sizeof(int)*numrs*(pow(2,numharmstages)-1));

    int harm;
    int ii, rrint,kk=0;

    for (stage = 1; stage < numharmstages; stage++)
    {
        int harmtosum = 1 << stage;
        for (harm = 1; harm < harmtosum; harm += 2)
        {
            const double harm_fract = (double) harm / (double) harmtosum;
            rinds_bk = rinds_cpu+kk*numrs;
            for (ii = 0, rrint = ACCEL_RDR * rlo; ii < numrs; ii++, rrint++)
                rinds_bk[ii] = (int) (rrint * harm_fract + 0.5);
            kk++;
        }  
    }
    cudaMemcpy(rinds_gpu, rinds_cpu, sizeof(int)*numrs*(pow(2,numharmstages)-1), cudaMemcpyHostToDevice);
    cudaFreeHost(rinds_cpu);
}

void inmem_add_subharm_gpu(ffdotpows * fundamental, accelobs * obs, float *outpows_gpu, float *outpows_gpu_obs, int stage, int *rinds_gpu)
{
    const int numrs = fundamental->numrs;
    const int numzs = fundamental->numzs;
    const int zlo = fundamental->zlo;
    long long rlen = (obs->highestbin + obs->corr_uselen) * ACCEL_RDR;

    inmem_add_ffdotpows_gpu_gpu(outpows_gpu_obs, outpows_gpu, rinds_gpu, zlo, numrs, numzs, stage, rlen);
}

void inmem_add_subharm_gpu_list(ffdotpows * fundamental, accelobs * obs, float *outpows_gpu, float *outpows_gpu_obs, int stage, int *rinds_gpu, int outpows_gpu_xlen, int readdatanum, int outpows_gpu_obs_xlen)
{
    const int numrs = fundamental->numrs;
    const int numzs = fundamental->numzs;
    const int zlo = fundamental->zlo;
    long long rlen = (obs->highestbin + obs->corr_uselen) * ACCEL_RDR;

    inmem_add_ffdotpows_gpu_gpu_list(outpows_gpu_obs, outpows_gpu, rinds_gpu, zlo, numrs, numzs, stage, rlen, outpows_gpu_xlen, readdatanum, outpows_gpu_obs_xlen);
}


void inmem_add_ffdotpows_trans(ffdotpows * fundamental, accelobs * obs,
                               int numharm, int harmnum)
{
    const int rlo = fundamental->rlo;
    const int numrs = fundamental->numrs;
    const int numzs = fundamental->numzs;
    const double harm_fract = (double) harmnum / (double) numharm;
    long *rinds;

    // Pre-compute the frequency lookup table
    rinds = gen_lvect(numrs);
    {
        int ii, rrint;
        for (ii = 0, rrint = ACCEL_RDR * rlo; ii < numrs; ii++, rrint++)
            rinds[ii] = (long) (rrint * harm_fract + 0.5) * numzs;
    }

    // Now add all the powers
#ifdef _OPENMP
#pragma omp parallel shared(rinds,fundamental,obs)
#endif
    {
        const int zlo = fundamental->zlo;
        float *powptr = fundamental->powers[0][0];
        float *fdp = obs->ffdotplane;
        int ii, jj, zz, zind, subz;
        float *inpows, *outpows;
#ifdef _OPENMP
#pragma omp for
#endif
        for (ii = 0; ii < numzs; ii++) {
            zz = zlo + ii * ACCEL_DZ;
            subz = calc_required_z(harm_fract, zz);
            zind = index_from_z(subz, zlo);
            inpows = fdp + zind;
            outpows = powptr + ii * numrs;
#if (defined(__GNUC__) || defined(__GNUG__)) && \
    !(defined(__clang__) || defined(__INTEL_COMPILER))
#pragma GCC ivdep
#endif
            for (jj = 0; jj < numrs; jj++)
                outpows[jj] += inpows[rinds[jj]];
        }
    }
    vect_free(rinds);
}


GSList *search_ffdotpows(ffdotpows * ffdot, int numharm,
                         accelobs * obs, GSList * cands)
{
    int ii;
    float powcut;
    long long numindep;
    
    powcut = obs->powcut[twon_to_index(numharm)];
    numindep = obs->numindep[twon_to_index(numharm)];
    
#ifdef _OPENMP
#pragma omp parallel for shared(ffdot,powcut,obs,numharm,numindep)
#endif
    for (ii = 0; ii < ffdot->numws; ii++) {
        int jj;
        for (jj = 0; jj < ffdot->numzs; jj++) {
            int kk;
            for (kk = 0; kk < ffdot->numrs; kk++) {
                if (ffdot->powers[ii][jj][kk] > powcut) {
                    float pow, sig;
                    double rr, zz, ww;
                    int added = 0;
                    
                    pow = ffdot->powers[ii][jj][kk];
                    sig = candidate_sigma(pow, numharm, numindep);
                    rr = (ffdot->rlo + kk * (double) ACCEL_DR) / (double) numharm;
                    zz = (ffdot->zlo + jj * (double) ACCEL_DZ) / (double) numharm;
                    ww = (ffdot->wlo + ii * (double) ACCEL_DW) / (double) numharm;
#ifdef _OPENMP
#pragma omp critical
#endif
                    {
                        cands = insert_new_accelcand(cands, pow, sig, numharm,
                                                     rr, zz, ww, &added);
                    }
                    if (added && !obs->dat_input)
                        fprintf(obs->workfile,
                                "%-7.2f  %-7.4f  %-2d  %-14.4f  %-14.9f  %-10.4f %-10.4f\n",
                                pow, sig, numharm, rr, rr / obs->T, zz, ww);
                }
            }
        }
    }
    return cands;
}

GSList *search_ffdotpows_sort_gpu_result(ffdotpows * ffdot, int numharm,
                         accelobs * obs, GSList * cands, accel_cand_gpu *cand_gpu_cpu, int nof_cand)
{
   register int ii;
//    register float powcut;
   register long long numindep;

//    powcut = obs->powcut[twon_to_index(numharm)];
   numindep = obs->numindep[twon_to_index(numharm)];	
   
   register float pow, sig;
   register double rr, zz, ww;
   int added = 0;   
   
   register int rind, zind;
   
   if(nof_cand > 0)
   {
   		for(ii=0; ii< nof_cand; ii++)
   		{   	
   			added = 0;      			
   			pow = cand_gpu_cpu[ii].pow; 
   			sig = candidate_sigma(pow, numharm, numindep);   						
   			zind = cand_gpu_cpu[ii].z_ind;
   			rind = cand_gpu_cpu[ii].r_ind;     			
   			rr = (ffdot->rlo + rind * (double) ACCEL_DR) / (double) numharm;   			
   			zz = (ffdot->zlo + zind * (double) ACCEL_DZ) / (double) numharm;
   			ww = 0.0;
   			// cands = insert_new_accelcand(cands, pow, sig, numharm, rr, zz, &added);
            cands = insert_new_accelcand(cands, pow, sig, numharm, rr, zz, ww, &added);
   			if (added && !obs->dat_input)
               fprintf(obs->workfile,
                                "%-7.2f  %-7.4f  %-2d  %-14.4f  %-14.9f  %-10.4f %-10.4f\n",
                                pow, sig, numharm, rr, rr / obs->T, zz, ww);
   		}  		   		
   }
   
   return cands;
}


void deredden(fcomplex * fft, int numamps)
/* Attempt to remove rednoise from a time series by using   */
/* a median-filter of logarithmically increasing width.     */
/* Thanks to Jason Hessels and Maggie Livingstone for the   */
/* initial implementation (in rednoise.c)                   */
{
    int ii, ind, initialbuflen = 6, buflen, lastbuflen, maxbuflen = 200;
    int binnum = 1, numwrote = 1;
    float *powbuf, mean_old, mean_new, dslope = 1.0, norm;
    float powargr, powargi;

    /* Takes care of the DC term */
    fft[0].r = 1.0;
    fft[0].i = 0.0;

    /* Step through the input FFT and create powers */
    powbuf = gen_fvect(numamps);
    for (ii = 0; ii < numamps; ii++)
        powbuf[ii] = POWER(fft[ii].r, fft[ii].i);

    /* Calculate initial values */
    buflen = initialbuflen;
    mean_old = median(powbuf + binnum, buflen) / log(2.0);

    // Write the first half of the normalized block
    // Note that this does *not* include a slope, but since it
    // is only a few bins, that is probably OK.
    norm = invsqrtf(mean_old);
    for (ind = numwrote; ind < binnum + buflen / 2; ind++) {
        fft[ind].r *= norm;
        fft[ind].i *= norm;
    }
    numwrote += buflen / 2;
    binnum += buflen;
    lastbuflen = buflen;
    buflen = initialbuflen * log(binnum);
    if (buflen > maxbuflen)
        buflen = maxbuflen;

    while (binnum + buflen < numamps) {
        // Calculate the next mean
        mean_new = median(powbuf + binnum, buflen) / log(2.0);
        // The slope between the last block median and the current median
        dslope = (mean_new - mean_old) / (0.5 * (lastbuflen + buflen));
        //printf("\n%d %.5g %.5g %.5g\n", buflen, mean_old, mean_new, dslope);

        // Correct the last-half of the old block...
        for (ii = 0, ind = numwrote; ind < binnum + buflen / 2; ii++, ind++) {
            norm = invsqrtf(mean_old + dslope * ii);
            fft[ind].r *= norm;
            fft[ind].i *= norm;
            //printf("  %10ld %4d %.5g\n", ii+numwrote, ii, 1.0/(norm*norm));
        }
        numwrote += ii;

        /* Update our values */
        binnum += buflen;
        lastbuflen = buflen;
        mean_old = mean_new;
        buflen = initialbuflen * log(binnum);
        if (buflen > maxbuflen)
            buflen = maxbuflen;
    }

    // Deal with the last chunk (assume same slope as before)
    for (ii = 0, ind = numwrote; ind < numamps; ii++, ind++) {
        norm = invsqrtf(mean_old + dslope * ii);
        fft[ind].r *= norm;
        fft[ind].i *= norm;
    }

    /* Free the powers */
    vect_free(powbuf);
}


void create_accelobs(accelobs * obs, infodata * idata, Cmdline * cmd, int usemmap)
{
    int ii, rootlen, input_shorts = 0;

    {
        int hassuffix = 0;
        char *suffix;

        hassuffix = split_root_suffix(cmd->argv[0], &(obs->rootfilenm), &suffix);
        if (hassuffix) {
            if (strcmp(suffix, "fft") != 0 &&
                strcmp(suffix, "dat") != 0 && strcmp(suffix, "sdat") != 0) {
                printf
                    ("\nInput file ('%s') must be an '.fft' or '.[s]dat' file!\n\n",
                     cmd->argv[0]);
                free(suffix);
                exit(0);
            }
            /* If the input file is a time series */
            if (strcmp(suffix, "dat") == 0 || strcmp(suffix, "sdat") == 0) {
                obs->dat_input = 1;
                obs->mmap_file = 0;
                if (strcmp(suffix, "sdat") == 0)
                    input_shorts = 1;
            }
            else {
                obs->dat_input = 0;
            }

            // if(strcmp(suffix, "fft") == 0 && obs->cudaP) 
            // {
            //     printf("\nInput file ('%s') must be an '.[s]dat' file  with CUDA !\n\n",cmd->argv[0]);
            //     free(suffix);
            //     exit(0);
            // }

            free(suffix);
        } else {
            printf("\nInput file ('%s') must be an '.fft' or '.[s]dat' file!\n\n",
                   cmd->argv[0]);
            exit(0);
        }
    }



    if (cmd->noharmpolishP)
        obs->use_harmonic_polishing = 0;
    else
        obs->use_harmonic_polishing = 1;        // now default

    /* Read the info file */

    readinf(idata, obs->rootfilenm);
    if (strlen(remove_whitespace(idata->object)) > 0) {
        printf("Analyzing %s data from '%s'.\n\n",
               remove_whitespace(idata->object), cmd->argv[0]);
    } else {
        printf("Analyzing data from '%s'.\n\n", cmd->argv[0]);
    }
    obs->cudaP = cmd->cudaP;
    /* Prepare the input time series if required */

    if (obs->dat_input) {
        FILE *datfile;
        long long filelen;
        float *ftmp;

        printf("Reading and FFTing the time series...");
        fflush(NULL);
        datfile = chkfopen(cmd->argv[0], "rb");

        /* Check the length of the file to see if we can handle it */
        filelen = chkfilelen(datfile, sizeof(float));
        if (input_shorts)
            filelen *= 2;
        // if (filelen > 67108864) {       /* Small since we need memory for the templates */
        //     printf
        //         ("\nThe input time series is too large.  Use 'realfft' first.\n\n");
        //     exit(0);
        // }

        /* Read the time series into a temporary buffer */
        /* Note:  The padding allows us to search very short time series */
        /*        using correlations without having to worry about       */
        /*        accessing data before or after the valid FFT freqs.    */
        if (input_shorts) {
            short *stmp = gen_svect(filelen);
            ftmp = gen_fvect(filelen + 2 * ACCEL_PADDING);
            for (ii = 0; ii < ACCEL_PADDING; ii++) {
                ftmp[ii] = 0.0;
                ftmp[ii + filelen + ACCEL_PADDING] = 0.0;
            }
            chkfread(stmp, sizeof(short), filelen, datfile);
            for (ii = 0; ii < filelen; ii++)
                ftmp[ii + ACCEL_PADDING] = (float) stmp[ii];
            vect_free(stmp);
        } else {
            ftmp =
                read_float_file(datfile, -ACCEL_PADDING,
                                filelen + 2 * ACCEL_PADDING);
        }
        /* Now, offset the pointer so that we are pointing at the first */
        /* bits of valid data.                                          */
        ftmp += ACCEL_PADDING;
        fclose(datfile);

        /* FFT it */
        realfft(ftmp, filelen, -1);
        obs->fftfile = NULL;
        obs->fft = (fcomplex *) ftmp;
        obs->numbins = filelen / 2;
        printf("done.\n");

        /* De-redden it */
        printf("Removing red-noise...");
        deredden(obs->fft, obs->numbins);
        printf("done.\n\n");
    }

    /* Open the FFT file if it exists appropriately */
    if (!obs->dat_input) {
        obs->fftfile = chkfopen(cmd->argv[0], "rb");
        obs->numbins = chkfilelen(obs->fftfile, sizeof(fcomplex));
        if (usemmap) {
            fclose(obs->fftfile);
            obs->fftfile = NULL;
            printf("Memory mapping the input FFT.  This may take a while...\n");
            obs->mmap_file = open(cmd->argv[0], O_RDONLY);
            if (obs->mmap_file == -1) {
                perror("\nError in open() in accel_utils.c");
                printf("\n");
                exit(-1);
            }
            obs->fft =
                (fcomplex *) mmap(0, sizeof(fcomplex) * obs->numbins, PROT_READ,
                                  MAP_SHARED, obs->mmap_file, 0);
            if (obs->fft == MAP_FAILED) {
                perror("\nError in mmap() in accel_utils.c");
                printf("Falling back to a non-mmaped approach\n");
                obs->fftfile = chkfopen(cmd->argv[0], "rb");
                obs->mmap_file = 0;
            }
        } else {
            obs->mmap_file = 0;
        }
    }

    /* Determine the other parameters */

    if (cmd->zmax % ACCEL_DZ)
        cmd->zmax = (cmd->zmax / ACCEL_DZ + 1) * ACCEL_DZ;
    obs->N = (long long) idata->N;
    if (cmd->photonP) {
        if (obs->mmap_file || obs->dat_input) {
            obs->nph = obs->fft[0].r;
        } else {
            obs->nph = get_numphotons(obs->fftfile);
        }
        printf("Normalizing powers using %.0f photons.\n\n", obs->nph);
    } else {
        obs->nph = 0.0;
        /* For short FFTs insure that we don't pick up the DC */
        /* or Nyquist component as part of the interpolation  */
        /* for higher frequencies.                            */
        if (cmd->locpowP) {
            obs->norm_type = 1;
            printf("Normalizing powers using local-power determination.\n\n");
        } else if (cmd->medianP) {
            obs->norm_type = 0;
            printf("Normalizing powers using median-blocks.\n\n");
        } else {
            obs->norm_type = 0;
            printf("Normalizing powers using median-blocks (default).\n\n");
        }
        if (obs->dat_input) {
            obs->fft[0].r = 1.0;
            obs->fft[0].i = 1.0;
        }
    }
    obs->lobin = cmd->lobin;
    if (obs->lobin > 0) {
        obs->nph = 0.0;
        if (cmd->lobin > obs->numbins - 1) {
            printf("\n'lobin' is greater than the total number of\n");
            printf("   frequencies in the data set.  Exiting.\n\n");
            exit(1);
        }
    }
    if (cmd->numharm != 1 &&
        cmd->numharm != 2 &&
        cmd->numharm != 4 &&
        cmd->numharm != 8 &&
        cmd->numharm != 16 &&
        cmd->numharm != 32) {
        printf("\n'numharm' = %d must be a power-of-two!  Exiting\n\n",
               cmd->numharm);
        exit(1);
    }
    obs->numharmstages = twon_to_index(cmd->numharm) + 1;

    // if(cmd->cudaP && (obs->mmap_file || obs->dat_input))
    // {
    //     fcomplex *temp_gpu;
    //     cudaMalloc((void**)&temp_gpu, sizeof(fcomplex)*(obs->numbins+ACCEL_PADDING));
    //     cudaMemcpy(temp_gpu, obs->fft-ACCEL_PADDING/2, sizeof(fcomplex) * (obs->numbins+ACCEL_PADDING), cudaMemcpyHostToDevice);
    //     obs->fft_gpu = temp_gpu+ACCEL_PADDING/2;
    // }
            

    obs->dz = ACCEL_DZ;
    obs->numz = (cmd->zmax / ACCEL_DZ) * 2 + 1;
    
    /* Setting extra parameters for jerk search */
    if (cmd->wmaxP) {
        if (cmd->wmax % ACCEL_DW)
            cmd->wmax = (cmd->wmax / ACCEL_DW + 1) * ACCEL_DW;
        obs->whi = cmd->wmax;
        obs->wlo = -cmd->wmax;
        obs->dw = ACCEL_DW;
        obs->numw = (cmd->wmax / ACCEL_DW) * 2 + 1;
        if (cmd->wmax==0.0)
            obs->numw = 0;
        printf("Jerk search enabled with maximum fdotdot wmax = %d\n", cmd->wmax);
    } else {
        obs->whi = 0.0;
        obs->wlo = 0.0;
        obs->dw = 0.0;
        obs->numw = 0;
    }
    
    /* Determine the output filenames */
    rootlen = strlen(obs->rootfilenm) + 45;
    obs->candnm = (char *) calloc(rootlen, 1);
    obs->accelnm = (char *) calloc(rootlen, 1);
    obs->workfilenm = (char *) calloc(rootlen, 1);
    if (obs->numw) {
        sprintf(obs->candnm, "%s_ACCEL_%d_JERK_%d.cand", obs->rootfilenm, cmd->zmax, cmd->wmax);
        sprintf(obs->accelnm, "%s_ACCEL_%d_JERK_%d", obs->rootfilenm, cmd->zmax, cmd->wmax);
        sprintf(obs->workfilenm, "%s_ACCEL_%d_JERK_%d.txtcand", obs->rootfilenm, cmd->zmax, cmd->wmax);
    } else {
        sprintf(obs->candnm, "%s_ACCEL_%d.cand", obs->rootfilenm, cmd->zmax);
        sprintf(obs->accelnm, "%s_ACCEL_%d", obs->rootfilenm, cmd->zmax);
        sprintf(obs->workfilenm, "%s_ACCEL_%d.txtcand", obs->rootfilenm, cmd->zmax);
    }
    if (!obs->dat_input)
        obs->workfile = chkfopen(obs->workfilenm, "w");

    obs->numbetween = ACCEL_NUMBETWEEN;
    obs->dt = idata->dt;
    obs->T = idata->dt * idata->N;
    if (cmd->floP) {
        obs->rlo = floor(cmd->flo * obs->T);
        if (obs->rlo < obs->lobin)
            obs->rlo = obs->lobin;
        if (obs->rlo > obs->numbins - 1) {
            printf("\nLow frequency to search 'flo' is greater than\n");
            printf("   the highest available frequency.  Exiting.\n\n");
            exit(1);
        }
    } else {
        if (cmd->rloP)
            obs->rlo = cmd->rlo;
        else
            obs->rlo = 1.0;
        if (obs->rlo < obs->lobin)
            obs->rlo = obs->lobin;
        if (obs->rlo > obs->numbins - 1) {
            printf("\nLow frequency to search 'rlo' is greater than\n");
            printf("   the available number of points.  Exiting.\n\n");
            exit(1);
        }
    }
    obs->highestbin = obs->numbins - 1;
    if (cmd->fhiP) {
        obs->highestbin = ceil(cmd->fhi * obs->T);
        if (obs->highestbin > obs->numbins - 1)
            obs->highestbin = obs->numbins - 1;
        obs->rhi = obs->highestbin;
        if (obs->highestbin < obs->rlo) {
            printf("\nHigh frequency to search 'fhi' is less than\n");
            printf("   the lowest frequency to search 'flo'.  Exiting.\n\n");
            exit(1);
        }
    } else if (cmd->rhiP) {
        obs->highestbin = cmd->rhi;
        if (obs->highestbin > obs->numbins - 1)
            obs->highestbin = obs->numbins - 1;
        obs->rhi = obs->highestbin;
        if (obs->highestbin < obs->rlo) {
            printf("\nHigh frequency to search 'rhi' is less than\n");
            printf("   the lowest frequency to search 'rlo'.  Exiting.\n\n");
            exit(1);
        }
    }
    obs->dr = ACCEL_DR;
    obs->zhi = cmd->zmax;
    obs->zlo = -cmd->zmax;
    obs->sigma = cmd->sigma;
    obs->powcut = (float *) malloc(obs->numharmstages * sizeof(float));
    obs->numindep = (long long *) malloc(obs->numharmstages * sizeof(long long));
    for (ii = 0; ii < obs->numharmstages; ii++) {
        if (obs->numz == 1 && obs->numw == 0)
            obs->numindep[ii] = (obs->rhi - obs->rlo) / index_to_twon(ii);
        else if (obs->numz > 1 && obs->numw == 0)
            /* The numz+1 takes care of the small amount of  */
            /* search we get above zmax and below zmin.      */
            obs->numindep[ii] = (obs->rhi - obs->rlo) * (obs->numz + 1) *
                (obs->dz / 6.95) / index_to_twon(ii);
        else
            /* The numw+1 takes care of the small amount of  */
            /* search we get above wmax and below wmin.      */
            obs->numindep[ii] = (obs->rhi - obs->rlo) * \
                (obs->numz + 1) * (obs->dz / 6.95) *        \
                (obs->numw + 1) * (obs->dw / 44.2) / index_to_twon(ii);
        obs->powcut[ii] = power_for_sigma(obs->sigma,
                                          index_to_twon(ii), obs->numindep[ii]);
    }
    obs->numzap = 0;
    /*
       if (zapfile!=NULL)
       obs->numzap = get_birdies(cmd->zapfile, obs->T, obs->baryv, 
       &(obs->lobins), &(obs->hibins));
       else
       obs->numzap = 0;
     */

    /* Determine corr_uselen from zmax and wmax */
    obs->maxkernlen = 2 * ACCEL_NUMBETWEEN * w_resp_halfwidth(obs->zhi, obs->whi, LOWACC);
    obs->fftlen = fftlen_from_kernwidth(obs->maxkernlen);
    if (obs->fftlen < 2048)
        obs->fftlen = 2048;  // This gives slightly better speed empirically
    obs->corr_uselen = obs->fftlen - obs->maxkernlen;
    // Make sure that obs->corr_uselen is an integer number of
    // full (i.e. un-interpolated) Fourier bins
    if (obs->corr_uselen % ACCEL_RDR)
        obs->corr_uselen = obs->corr_uselen / ACCEL_RDR * ACCEL_RDR;

    /* Can we perform the search in-core memory? */
    {
        long long memuse;
        // double gb = (double) (1L << 30);

        if (cmd->wmaxP) {
            memuse = sizeof(float) * (obs->highestbin + obs->corr_uselen)
                * obs->numbetween * obs->numz * obs->numw;
        } else {
            memuse = sizeof(float) * (obs->highestbin + obs->corr_uselen)
                * obs->numbetween * obs->numz;
        }

        // This is the size of powers covering the full f-dot-dot plane to search
        // Need the extra obs->corr_uselen since we generate the plane in blocks
        if (!cmd->wmaxP && cmd->inmemP) 
        // if (!cmd->wmaxP &&  cmd->inmemP) 
        {
            printf("using in-memory accelsearch.\n\n");
            obs->inmem = 1;
            if(!cmd->cudaP)
                obs->ffdotplane = gen_fvect(memuse / sizeof(float));
        } else {
            printf("using standard accelsearch.\n\n");
            obs->inmem = 0;
            obs->ffdotplane = NULL;
        }
    }

    

}

void create_accelobs_list(accelobs *obs, infodata *idata, 
		     Cmdline *cmd, int usemmap, int thread, int kk, int readdatanum)
{
    int ii, jj, input_shorts = 0;
    static int firsttime = 1;
    static float **ftmp_bk;
    static float *ftmp;

    
    {     
        for(ii = 0; ii < readdatanum; ii++)
        {
            int hassuffix = 0;
            char *suffix;
            printf("%s\n", cmd->datalist[ii+kk*cmd->gpu]);
            hassuffix = split_root_suffix(cmd->datalist[ii+kk*cmd->gpu], &(obs->rootfilenmlist[ii]), &suffix);

            if (hassuffix) {
                if (strcmp(suffix, "fft") != 0 &&
                    strcmp(suffix, "dat") != 0 && strcmp(suffix, "sdat") != 0) {
                    printf
                        ("\nInput file ('%s') must be an '.fft' or '.[s]dat' file!\n\n",
                        cmd->datalist[ii+kk*cmd->gpu]);
                    free(suffix);
                    exit(0);
                }
                /* If the input file is a time series */
                if (strcmp(suffix, "dat") == 0 || strcmp(suffix, "sdat") == 0) {
                    obs->dat_input = 1;
                    obs->mmap_file = 0;
                    if (strcmp(suffix, "sdat") == 0)
                        input_shorts = 1;
                }
                else {
                    obs->dat_input = 0;
                }
                free(suffix);
            } else {
                printf("\nInput file ('%s') must be an '.fft' or '.[s]dat' file!\n\n",
                    cmd->datalist[ii+kk*cmd->gpu]);
                exit(0);
            }
        }
    }


    if(firsttime)
    {
        if (cmd->noharmpolishP)
            obs->use_harmonic_polishing = 0;
        else
            obs->use_harmonic_polishing = 1;        // now default
        obs->cudaP = cmd->cudaP;  
    }
    

    
    /* Read the info file */
    for(ii=0; ii<readdatanum; ii++)
    {
        readinf(&idata[ii], obs->rootfilenmlist[ii]);
    }

    
    /* Prepare the input time series if required */
    if(firsttime)
        ftmp_bk = (float **) malloc(cmd->gpu * sizeof(float *));

    if (obs->dat_input) {
        FILE *datfile;
        long long filelen;
        
        
        printf("Reading and FFTing the time series...");
        fflush(NULL);

        for(jj=0; jj < readdatanum; jj++)
        {           
            datfile = chkfopen(cmd->datalist[jj+kk*cmd->gpu], "rb");
            /* Check the length of the file to see if we can handle it */
            filelen = chkfilelen(datfile, sizeof(float));
            if (input_shorts)
                filelen *= 2;
            // if (filelen > 67108864) {       /* Small since we need memory for the templates */
            //     printf
            //         ("\nThe input time series is too large.  Use 'realfft' first.\n\n");
            //     exit(0);
            // }

            /* Read the time series into a temporary buffer */
            /* Note:  The padding allows us to search very short time series */
            /*        using correlations without having to worry about       */
            /*        accessing data before or after the valid FFT freqs.    */
            
            if(firsttime)
            {   // obs->fftlist[jj] = (fcomplex *)malloc((filelen + 2 * ACCEL_PADDING) * sizeof(fcomplex));
                obs->fftlist_f[jj] = (float *)malloc((filelen + 2 * ACCEL_PADDING) * sizeof(float));
                ftmp_bk[jj] = obs->fftlist_f[jj];
            }

            
            if (input_shorts) {
                short *stmp = gen_svect(filelen);
                // ftmp = gen_fvect(filelen + 2 * ACCEL_PADDING);
                // ftmp = obs->fftlist_f[jj];
                ftmp = ftmp_bk[jj];
                for (ii = 0; ii < ACCEL_PADDING; ii++) {
                    ftmp[ii] = 0.0;
                    ftmp[ii + filelen + ACCEL_PADDING] = 0.0;
                }
                chkfread(stmp, sizeof(short), filelen, datfile);
                for (ii = 0; ii < filelen; ii++)
                    ftmp[ii + ACCEL_PADDING] = (float) stmp[ii];
                vect_free(stmp);
            } else {
                // ftmp = read_float_file(datfile, -ACCEL_PADDING,
                //                     filelen + 2 * ACCEL_PADDING);
                // ftmp = obs->fftlist_f[jj];
                ftmp = ftmp_bk[jj];
                read_float_file_list(datfile, -ACCEL_PADDING,
                                    filelen + 2 * ACCEL_PADDING, ftmp);
            }
            /* Now, offset the pointer so that we are pointing at the first */
            /* bits of valid data.                                          */
            ftmp += ACCEL_PADDING;
            fclose(datfile);

            /* FFT it */
            // realfft(ftmp, filelen, -1);
            obs->fftfilelist[jj] = NULL;
            obs->fftlist_f[jj] = ftmp;
            obs->numbins = filelen / 2;

            /* De-redden it */
            // printf("Removing red-noise...");
            // deredden(obs->fftlist[ii], obs->numbins);
            // printf("done.\n\n");
        }
        printf("Removing red-noise...");
        
        if(firsttime)
        {
            realfft(obs->fftlist_f[0], filelen, -1);
            #ifdef _OPENMP
            #pragma omp parallel for num_threads(thread) shared(obs, filelen, cmd)
            #endif
            for(ii=1; ii < readdatanum; ii++)
                realfft(obs->fftlist_f[ii], filelen, -1);
        }
        else
        {
            #ifdef _OPENMP
            #pragma omp parallel for num_threads(thread) shared(obs, filelen, cmd)
            #endif
            for(ii=0; ii < readdatanum; ii++)
                realfft(obs->fftlist_f[ii], filelen, -1);
        }

        obs->fftlist = (fcomplex **) obs->fftlist_f;

        #ifdef _OPENMP
        #pragma omp parallel for num_threads(thread) shared(obs)
        #endif
        for(ii=0; ii < readdatanum; ii++)
            deredden(obs->fftlist[ii], obs->numbins);   
        printf("done.\n\n");
    }


    /* Open the FFT file if it exists appropriately */
    if (!obs->dat_input) {
        for(ii=0; ii < readdatanum; ii++)
        {
            obs->fftfilelist[ii] = chkfopen(cmd->datalist[ii+kk*cmd->gpu], "rb");
            obs->numbins = chkfilelen(obs->fftfilelist[ii], sizeof(fcomplex));
            if (usemmap) {
                fclose(obs->fftfilelist[ii]);
                obs->fftfilelist[ii] = NULL;
                printf("Memory mapping the input FFT.  This may take a while...\n");
                obs->mmap_file = open(cmd->datalist[ii+kk*cmd->gpu], O_RDONLY);
                if (obs->mmap_file == -1) {
                    perror("\nError in open() in accel_utils.c");
                    printf("\n");
                    exit(-1);
                }
                obs->fftlist[ii] =
                    (fcomplex *) mmap(0, sizeof(fcomplex) * obs->numbins, PROT_READ,
                                    MAP_SHARED, obs->mmap_file, 0);
                if (obs->fftlist[ii] == MAP_FAILED) {
                    perror("\nError in mmap() in accel_utils.c");
                    printf("Falling back to a non-mmaped approach\n");
                    obs->fftfilelist[ii] = chkfopen(cmd->datalist[ii+kk*cmd->gpu], "rb");
                    obs->mmap_file = 0;
                }
            } else {
                obs->mmap_file = 0;
            }
        }

    }

    /* Determine the other parameters */
    
    if (cmd->zmax % ACCEL_DZ)
        cmd->zmax = (cmd->zmax / ACCEL_DZ + 1) * ACCEL_DZ;
    obs->N = (long long) idata[0].N;
    if (cmd->photonP) {
        if (obs->mmap_file || obs->dat_input) {
            obs->nph = obs->fftlist[0][0].r;
        } else {
            obs->nph = get_numphotons(obs->fftfilelist[0]);
        }
        printf("Normalizing powers using %.0f photons.\n\n", obs->nph);
    } else {
        obs->nph = 0.0;
        /* For short FFTs insure that we don't pick up the DC */
        /* or Nyquist component as part of the interpolation  */
        /* for higher frequencies.                            */
        if (cmd->locpowP) {
            obs->norm_type = 1;
            printf("Normalizing powers using local-power determination.\n\n");
        } else if (cmd->medianP) {
            obs->norm_type = 0;
            printf("Normalizing powers using median-blocks.\n\n");
        } else {
            obs->norm_type = 0;
            printf("Normalizing powers using median-blocks (default).\n\n");
        }
        if (obs->dat_input) {
            for(ii=0; ii < readdatanum; ii++)
            {
                obs->fftlist[ii][0].r = 1.0;
                obs->fftlist[ii][0].i = 1.0;
            }
        }
    }

    if(firsttime)
    {
        obs->lobin = cmd->lobin;
        if (obs->lobin > 0) {
            obs->nph = 0.0;
            if (cmd->lobin > obs->numbins - 1) {
                printf("\n'lobin' is greater than the total number of\n");
                printf("   frequencies in the data set.  Exiting.\n\n");
                exit(1);
            }
        }
        if (cmd->numharm != 1 &&
            cmd->numharm != 2 &&
            cmd->numharm != 4 &&
            cmd->numharm != 8 &&
            cmd->numharm != 16 &&
            cmd->numharm != 32) {
            printf("\n'numharm' = %d must be a power-of-two!  Exiting\n\n",
                cmd->numharm);
            exit(1);
        }
        obs->numharmstages = twon_to_index(cmd->numharm) + 1;

        obs->dz = ACCEL_DZ;
        obs->numz = (cmd->zmax / ACCEL_DZ) * 2 + 1;
        
        /* Setting extra parameters for jerk search */
        if (cmd->wmaxP) {
            if (cmd->wmax % ACCEL_DW)
                cmd->wmax = (cmd->wmax / ACCEL_DW + 1) * ACCEL_DW;
            obs->whi = cmd->wmax;
            obs->wlo = -cmd->wmax;
            obs->dw = ACCEL_DW;
            obs->numw = (cmd->wmax / ACCEL_DW) * 2 + 1;
            if (cmd->wmax==0.0)
                obs->numw = 0;
            printf("Jerk search enabled with maximum fdotdot wmax = %d\n", cmd->wmax);
        } else {
            obs->whi = 0.0;
            obs->wlo = 0.0;
            obs->dw = 0.0;
            obs->numw = 0;
        }
    }
    
    

    /* Determine the output filenames */
    for(ii=0; ii<readdatanum; ii++)
    {
        if (obs->numw) {
            sprintf(obs->candnmlist[ii], "%s_ACCEL_%d_JERK_%d.cand", obs->rootfilenmlist[ii], cmd->zmax, cmd->wmax);
            sprintf(obs->accelnmlist[ii], "%s_ACCEL_%d_JERK_%d", obs->rootfilenmlist[ii], cmd->zmax, cmd->wmax);
            sprintf(obs->workfilenmlist[ii], "%s_ACCEL_%d_JERK_%d.txtcand", obs->rootfilenmlist[ii], cmd->zmax, cmd->wmax);
        } else {
            sprintf(obs->candnmlist[ii], "%s_ACCEL_%d.cand", obs->rootfilenmlist[ii], cmd->zmax);
            sprintf(obs->accelnmlist[ii], "%s_ACCEL_%d", obs->rootfilenmlist[ii], cmd->zmax);
            sprintf(obs->workfilenmlist[ii], "%s_ACCEL_%d.txtcand", obs->rootfilenmlist[ii], cmd->zmax);
        }
        if (!obs->dat_input)
            obs->workfilelist[ii] = chkfopen(obs->workfilenmlist[ii], "w");
    }
    

    if(firsttime)
    {
        obs->numbetween = ACCEL_NUMBETWEEN;
        obs->dt = idata[0].dt;
        obs->T = idata[0].dt * idata[0].N;
        if (cmd->floP) {
            obs->rlo = floor(cmd->flo * obs->T);
            if (obs->rlo < obs->lobin)
                obs->rlo = obs->lobin;
            if (obs->rlo > obs->numbins - 1) {
                printf("\nLow frequency to search 'flo' is greater than\n");
                printf("   the highest available frequency.  Exiting.\n\n");
                exit(1);
            }
        } else {
            if (cmd->rloP)
                obs->rlo = cmd->rlo;
            else
                obs->rlo = 1.0;
            if (obs->rlo < obs->lobin)
                obs->rlo = obs->lobin;
            if (obs->rlo > obs->numbins - 1) {
                printf("\nLow frequency to search 'rlo' is greater than\n");
                printf("   the available number of points.  Exiting.\n\n");
                exit(1);
            }
        }

        obs->highestbin = obs->numbins - 1;
        if (cmd->fhiP) {
            obs->highestbin = ceil(cmd->fhi * obs->T);
            if (obs->highestbin > obs->numbins - 1)
                obs->highestbin = obs->numbins - 1;
            obs->rhi = obs->highestbin;
            if (obs->highestbin < obs->rlo) {
                printf("\nHigh frequency to search 'fhi' is less than\n");
                printf("   the lowest frequency to search 'flo'.  Exiting.\n\n");
                exit(1);
            }
        } else if (cmd->rhiP) {
            obs->highestbin = cmd->rhi;
            if (obs->highestbin > obs->numbins - 1)
                obs->highestbin = obs->numbins - 1;
            obs->rhi = obs->highestbin;
            if (obs->highestbin < obs->rlo) {
                printf("\nHigh frequency to search 'rhi' is less than\n");
                printf("   the lowest frequency to search 'rlo'.  Exiting.\n\n");
                exit(1);
            }
        }
        obs->dr = ACCEL_DR;
        obs->zhi = cmd->zmax;
        obs->zlo = -cmd->zmax;
        obs->sigma = cmd->sigma;
        obs->powcut = (float *) malloc(obs->numharmstages * sizeof(float));
        obs->numindep = (long long *) malloc(obs->numharmstages * sizeof(long long));
        for (ii = 0; ii < obs->numharmstages; ii++) {
            if (obs->numz == 1 && obs->numw == 0)
                obs->numindep[ii] = (obs->rhi - obs->rlo) / index_to_twon(ii);
            else if (obs->numz > 1 && obs->numw == 0)
                /* The numz+1 takes care of the small amount of  */
                /* search we get above zmax and below zmin.      */
                obs->numindep[ii] = (obs->rhi - obs->rlo) * (obs->numz + 1) *
                    (obs->dz / 6.95) / index_to_twon(ii);
            else
                /* The numw+1 takes care of the small amount of  */
                /* search we get above wmax and below wmin.      */
                obs->numindep[ii] = (obs->rhi - obs->rlo) * \
                    (obs->numz + 1) * (obs->dz / 6.95) *        \
                    (obs->numw + 1) * (obs->dw / 44.2) / index_to_twon(ii);
            obs->powcut[ii] = power_for_sigma(obs->sigma,
                                            index_to_twon(ii), obs->numindep[ii]);
        }
        obs->numzap = 0;
        /*
        if (zapfile!=NULL)
        obs->numzap = get_birdies(cmd->zapfile, obs->T, obs->baryv, 
        &(obs->lobins), &(obs->hibins));
        else
        obs->numzap = 0;
        */
        /* Determine corr_uselen from zmax and wmax */
        obs->maxkernlen = 2 * ACCEL_NUMBETWEEN * w_resp_halfwidth(obs->zhi, obs->whi, LOWACC);
        obs->fftlen = fftlen_from_kernwidth(obs->maxkernlen);
        if (obs->fftlen < 2048)
            obs->fftlen = 2048;  // This gives slightly better speed empirically
        obs->corr_uselen = obs->fftlen - obs->maxkernlen;
        // Make sure that obs->corr_uselen is an integer number of
        // full (i.e. un-interpolated) Fourier bins
        if (obs->corr_uselen % ACCEL_RDR)
            obs->corr_uselen = obs->corr_uselen / ACCEL_RDR * ACCEL_RDR;

        /* Can we perform the search in-core memory? */
        {
            long long memuse;
            // double gb = (double) (1L << 30);

            if (cmd->wmaxP) {
                memuse = sizeof(float) * (obs->highestbin + obs->corr_uselen)
                    * obs->numbetween * obs->numz * obs->numw;
            } else {
                memuse = sizeof(float) * (obs->highestbin + obs->corr_uselen)
                    * obs->numbetween * obs->numz;
            }

            // This is the size of powers covering the full f-dot-dot plane to search
            // Need the extra obs->corr_uselen since we generate the plane in blocks
            if (!cmd->wmaxP && cmd->inmemP) 
            // if (!cmd->wmaxP &&  cmd->inmemP) 
            {
                printf("using in-memory accelsearch.\n\n");
                obs->inmem = 1;
                if(!cmd->cudaP)
                    obs->ffdotplane = gen_fvect(memuse / sizeof(float));
            } else {
                printf("using standard accelsearch.\n\n");
                obs->inmem = 0;
                obs->ffdotplane = NULL;
            }
        }
    }
    else
        firsttime = 0;
    
    
}

void create_accelobs_list_1(accelobs * obs, infodata * idata, Cmdline * cmd, int usemmap, int readid)
{
    int ii, rootlen, input_shorts = 0;
    static int firsttime = 1;
    static float *ftmp_bk;
    static float *ftmp;

    {
        int hassuffix = 0;
        char *suffix;
        
        printf("%s\n", cmd->datalist[readid]);
        hassuffix = split_root_suffix(cmd->datalist[readid], &(obs->rootfilenm), &suffix);
        // hassuffix = split_root_suffix(cmd->argv[0], &(obs->rootfilenm), &suffix);
        if (hassuffix) {
            if (strcmp(suffix, "fft") != 0 &&
                strcmp(suffix, "dat") != 0 && strcmp(suffix, "sdat") != 0) {
                printf
                    ("\nInput file ('%s') must be an '.fft' or '.[s]dat' file!\n\n",
                     cmd->datalist[readid]);
                free(suffix);
                exit(0);
            }
            /* If the input file is a time series */
            if (strcmp(suffix, "dat") == 0 || strcmp(suffix, "sdat") == 0) {
                obs->dat_input = 1;
                obs->mmap_file = 0;
                if (strcmp(suffix, "sdat") == 0)
                    input_shorts = 1;
            }
            else {
                obs->dat_input = 0;
            }

            // if(strcmp(suffix, "fft") == 0 && obs->cudaP) 
            // {
            //     printf("\nInput file ('%s') must be an '.[s]dat' file  with CUDA !\n\n",cmd->argv[0]);
            //     free(suffix);
            //     exit(0);
            // }

            free(suffix);
        } else {
            printf("\nInput file ('%s') must be an '.fft' or '.[s]dat' file!\n\n",
                   cmd->datalist[readid]);
            exit(0);
        }
    }



    if (cmd->noharmpolishP)
        obs->use_harmonic_polishing = 0;
    else
        obs->use_harmonic_polishing = 1;        // now default

    /* Read the info file */

    readinf(idata, obs->rootfilenm);
    if (strlen(remove_whitespace(idata->object)) > 0) {
        printf("Analyzing %s data from '%s'.\n\n",
               remove_whitespace(idata->object), cmd->datalist[readid]);
    } else {
        printf("Analyzing data from '%s'.\n\n", cmd->datalist[readid]);
    }
    obs->cudaP = cmd->cudaP;
    /* Prepare the input time series if required */

    if (obs->dat_input) {
        FILE *datfile;
        long long filelen;
        

        printf("Reading and FFTing the time series...");
        fflush(NULL);
        datfile = chkfopen(cmd->datalist[readid], "rb");

        /* Check the length of the file to see if we can handle it */
        filelen = chkfilelen(datfile, sizeof(float));
        if (input_shorts)
            filelen *= 2;
        // if (filelen > 67108864) {       /* Small since we need memory for the templates */
        //     printf
        //         ("\nThe input time series is too large.  Use 'realfft' first.\n\n");
        //     exit(0);
        // }

        /* Read the time series into a temporary buffer */
        /* Note:  The padding allows us to search very short time series */
        /*        using correlations without having to worry about       */
        /*        accessing data before or after the valid FFT freqs.    */
        if (input_shorts) {
            short *stmp = gen_svect(filelen);
            
            if(firsttime)
            {
                ftmp = gen_fvect(filelen + 2 * ACCEL_PADDING);
                ftmp_bk = ftmp;
            }
            else
                ftmp = ftmp_bk;

            for (ii = 0; ii < ACCEL_PADDING; ii++) {
                ftmp[ii] = 0.0;
                ftmp[ii + filelen + ACCEL_PADDING] = 0.0;
            }
            chkfread(stmp, sizeof(short), filelen, datfile);
            for (ii = 0; ii < filelen; ii++)
                ftmp[ii + ACCEL_PADDING] = (float) stmp[ii];
            vect_free(stmp);
        } else {
            if(firsttime)
            {
                ftmp = gen_fvect(filelen + 2 * ACCEL_PADDING);
                ftmp_bk = ftmp;
            }
            else 
                ftmp = ftmp_bk;
            read_float_file_list(datfile, -ACCEL_PADDING,
                                filelen + 2 * ACCEL_PADDING, ftmp);
            // ftmp = read_float_file(datfile, -ACCEL_PADDING,
            //                     filelen + 2 * ACCEL_PADDING);
        }
        /* Now, offset the pointer so that we are pointing at the first */
        /* bits of valid data.                                          */
        ftmp += ACCEL_PADDING;
        fclose(datfile);

        /* FFT it */
        realfft(ftmp, filelen, -1);
        obs->fftfile = NULL;
        obs->fft = (fcomplex *) ftmp;
        obs->numbins = filelen / 2;
        printf("done.\n");

        /* De-redden it */
        printf("Removing red-noise...");
        deredden(obs->fft, obs->numbins);
        printf("done.\n\n");
    }

    /* Open the FFT file if it exists appropriately */
    if (!obs->dat_input) {
        obs->fftfile = chkfopen(cmd->datalist[readid], "rb");
        obs->numbins = chkfilelen(obs->fftfile, sizeof(fcomplex));
        if (usemmap) {
            fclose(obs->fftfile);
            obs->fftfile = NULL;
            printf("Memory mapping the input FFT.  This may take a while...\n");
            obs->mmap_file = open(cmd->datalist[readid], O_RDONLY);
            if (obs->mmap_file == -1) {
                perror("\nError in open() in accel_utils.c");
                printf("\n");
                exit(-1);
            }
            obs->fft =
                (fcomplex *) mmap(0, sizeof(fcomplex) * obs->numbins, PROT_READ,
                                  MAP_SHARED, obs->mmap_file, 0);
            if (obs->fft == MAP_FAILED) {
                perror("\nError in mmap() in accel_utils.c");
                printf("Falling back to a non-mmaped approach\n");
                obs->fftfile = chkfopen(cmd->datalist[readid], "rb");
                obs->mmap_file = 0;
            }
        } else {
            obs->mmap_file = 0;
        }
    }

    /* Determine the other parameters */

    if (cmd->zmax % ACCEL_DZ)
        cmd->zmax = (cmd->zmax / ACCEL_DZ + 1) * ACCEL_DZ;
    obs->N = (long long) idata->N;
    if (cmd->photonP) {
        if (obs->mmap_file || obs->dat_input) {
            obs->nph = obs->fft[0].r;
        } else {
            obs->nph = get_numphotons(obs->fftfile);
        }
        printf("Normalizing powers using %.0f photons.\n\n", obs->nph);
    } else {
        obs->nph = 0.0;
        /* For short FFTs insure that we don't pick up the DC */
        /* or Nyquist component as part of the interpolation  */
        /* for higher frequencies.                            */
        if (cmd->locpowP) {
            obs->norm_type = 1;
            printf("Normalizing powers using local-power determination.\n\n");
        } else if (cmd->medianP) {
            obs->norm_type = 0;
            printf("Normalizing powers using median-blocks.\n\n");
        } else {
            obs->norm_type = 0;
            printf("Normalizing powers using median-blocks (default).\n\n");
        }
        if (obs->dat_input) {
            obs->fft[0].r = 1.0;
            obs->fft[0].i = 1.0;
        }
    }

    if(firsttime)
    {
        obs->lobin = cmd->lobin;
        if (obs->lobin > 0) {
            obs->nph = 0.0;
            if (cmd->lobin > obs->numbins - 1) {
                printf("\n'lobin' is greater than the total number of\n");
                printf("   frequencies in the data set.  Exiting.\n\n");
                exit(1);
            }
        }
        if (cmd->numharm != 1 &&
            cmd->numharm != 2 &&
            cmd->numharm != 4 &&
            cmd->numharm != 8 &&
            cmd->numharm != 16 &&
            cmd->numharm != 32) {
            printf("\n'numharm' = %d must be a power-of-two!  Exiting\n\n",
                cmd->numharm);
            exit(1);
        }
        obs->numharmstages = twon_to_index(cmd->numharm) + 1;

        // if(cmd->cudaP && (obs->mmap_file || obs->dat_input))
        // {
        //     fcomplex *temp_gpu;
        //     cudaMalloc((void**)&temp_gpu, sizeof(fcomplex)*(obs->numbins+ACCEL_PADDING));
        //     cudaMemcpy(temp_gpu, obs->fft-ACCEL_PADDING/2, sizeof(fcomplex) * (obs->numbins+ACCEL_PADDING), cudaMemcpyHostToDevice);
        //     obs->fft_gpu = temp_gpu+ACCEL_PADDING/2;
        // }

        obs->dz = ACCEL_DZ;
        obs->numz = (cmd->zmax / ACCEL_DZ) * 2 + 1;
        
        /* Setting extra parameters for jerk search */
        if (cmd->wmaxP) {
            if (cmd->wmax % ACCEL_DW)
                cmd->wmax = (cmd->wmax / ACCEL_DW + 1) * ACCEL_DW;
            obs->whi = cmd->wmax;
            obs->wlo = -cmd->wmax;
            obs->dw = ACCEL_DW;
            obs->numw = (cmd->wmax / ACCEL_DW) * 2 + 1;
            if (cmd->wmax==0.0)
                obs->numw = 0;
            printf("Jerk search enabled with maximum fdotdot wmax = %d\n", cmd->wmax);
        } else {
            obs->whi = 0.0;
            obs->wlo = 0.0;
            obs->dw = 0.0;
            obs->numw = 0;
        }
    }
    
    
    /* Determine the output filenames */
    rootlen = strlen(obs->rootfilenm) + 45;
    obs->candnm = (char *) calloc(rootlen, 1);
    obs->accelnm = (char *) calloc(rootlen, 1);
    obs->workfilenm = (char *) calloc(rootlen, 1);
    if (obs->numw) {
        sprintf(obs->candnm, "%s_ACCEL_%d_JERK_%d.cand", obs->rootfilenm, cmd->zmax, cmd->wmax);
        sprintf(obs->accelnm, "%s_ACCEL_%d_JERK_%d", obs->rootfilenm, cmd->zmax, cmd->wmax);
        sprintf(obs->workfilenm, "%s_ACCEL_%d_JERK_%d.txtcand", obs->rootfilenm, cmd->zmax, cmd->wmax);
    } else {
        sprintf(obs->candnm, "%s_ACCEL_%d.cand", obs->rootfilenm, cmd->zmax);
        sprintf(obs->accelnm, "%s_ACCEL_%d", obs->rootfilenm, cmd->zmax);
        sprintf(obs->workfilenm, "%s_ACCEL_%d.txtcand", obs->rootfilenm, cmd->zmax);
    }
    if (!obs->dat_input)
        obs->workfile = chkfopen(obs->workfilenm, "w");

    if(firsttime)
    {
        obs->numbetween = ACCEL_NUMBETWEEN;
        obs->dt = idata->dt;
        obs->T = idata->dt * idata->N;
        if (cmd->floP) {
            obs->rlo = floor(cmd->flo * obs->T);
            if (obs->rlo < obs->lobin)
                obs->rlo = obs->lobin;
            if (obs->rlo > obs->numbins - 1) {
                printf("\nLow frequency to search 'flo' is greater than\n");
                printf("   the highest available frequency.  Exiting.\n\n");
                exit(1);
            }
        } else {
            if (cmd->rloP)
                obs->rlo = cmd->rlo;
            else
                obs->rlo = 1.0;
            if (obs->rlo < obs->lobin)
                obs->rlo = obs->lobin;
            if (obs->rlo > obs->numbins - 1) {
                printf("\nLow frequency to search 'rlo' is greater than\n");
                printf("   the available number of points.  Exiting.\n\n");
                exit(1);
            }
        }
        obs->highestbin = obs->numbins - 1;
        if (cmd->fhiP) {
            obs->highestbin = ceil(cmd->fhi * obs->T);
            if (obs->highestbin > obs->numbins - 1)
                obs->highestbin = obs->numbins - 1;
            obs->rhi = obs->highestbin;
            if (obs->highestbin < obs->rlo) {
                printf("\nHigh frequency to search 'fhi' is less than\n");
                printf("   the lowest frequency to search 'flo'.  Exiting.\n\n");
                exit(1);
            }
        } else if (cmd->rhiP) {
            obs->highestbin = cmd->rhi;
            if (obs->highestbin > obs->numbins - 1)
                obs->highestbin = obs->numbins - 1;
            obs->rhi = obs->highestbin;
            if (obs->highestbin < obs->rlo) {
                printf("\nHigh frequency to search 'rhi' is less than\n");
                printf("   the lowest frequency to search 'rlo'.  Exiting.\n\n");
                exit(1);
            }
        }
        obs->dr = ACCEL_DR;
        obs->zhi = cmd->zmax;
        obs->zlo = -cmd->zmax;
        obs->sigma = cmd->sigma;
        obs->powcut = (float *) malloc(obs->numharmstages * sizeof(float));
        obs->numindep = (long long *) malloc(obs->numharmstages * sizeof(long long));
        for (ii = 0; ii < obs->numharmstages; ii++) {
            if (obs->numz == 1 && obs->numw == 0)
                obs->numindep[ii] = (obs->rhi - obs->rlo) / index_to_twon(ii);
            else if (obs->numz > 1 && obs->numw == 0)
                /* The numz+1 takes care of the small amount of  */
                /* search we get above zmax and below zmin.      */
                obs->numindep[ii] = (obs->rhi - obs->rlo) * (obs->numz + 1) *
                    (obs->dz / 6.95) / index_to_twon(ii);
            else
                /* The numw+1 takes care of the small amount of  */
                /* search we get above wmax and below wmin.      */
                obs->numindep[ii] = (obs->rhi - obs->rlo) * \
                    (obs->numz + 1) * (obs->dz / 6.95) *        \
                    (obs->numw + 1) * (obs->dw / 44.2) / index_to_twon(ii);
            obs->powcut[ii] = power_for_sigma(obs->sigma,
                                            index_to_twon(ii), obs->numindep[ii]);
        }
        obs->numzap = 0;
        /*
        if (zapfile!=NULL)
        obs->numzap = get_birdies(cmd->zapfile, obs->T, obs->baryv, 
        &(obs->lobins), &(obs->hibins));
        else
        obs->numzap = 0;
        */

        /* Determine corr_uselen from zmax and wmax */
        obs->maxkernlen = 2 * ACCEL_NUMBETWEEN * w_resp_halfwidth(obs->zhi, obs->whi, LOWACC);
        obs->fftlen = fftlen_from_kernwidth(obs->maxkernlen);
        if (obs->fftlen < 2048)
            obs->fftlen = 2048;  // This gives slightly better speed empirically
        obs->corr_uselen = obs->fftlen - obs->maxkernlen;
        // Make sure that obs->corr_uselen is an integer number of
        // full (i.e. un-interpolated) Fourier bins
        if (obs->corr_uselen % ACCEL_RDR)
            obs->corr_uselen = obs->corr_uselen / ACCEL_RDR * ACCEL_RDR;

        /* Can we perform the search in-core memory? */
        {
            long long memuse;
            // double gb = (double) (1L << 30);

            if (cmd->wmaxP) {
                memuse = sizeof(float) * (obs->highestbin + obs->corr_uselen)
                    * obs->numbetween * obs->numz * obs->numw;
            } else {
                memuse = sizeof(float) * (obs->highestbin + obs->corr_uselen)
                    * obs->numbetween * obs->numz;
            }

            // This is the size of powers covering the full f-dot-dot plane to search
            // Need the extra obs->corr_uselen since we generate the plane in blocks
            if (!cmd->wmaxP && cmd->inmemP) 
            // if (!cmd->wmaxP &&  cmd->inmemP) 
            {
                printf("using in-memory accelsearch.\n\n");
                obs->inmem = 1;
                if(!cmd->cudaP)
                    obs->ffdotplane = gen_fvect(memuse / sizeof(float));
            } else {
                printf("using standard accelsearch.\n\n");
                obs->inmem = 0;
                obs->ffdotplane = NULL;
            }
        }
    }
    else
        firsttime = 0;
    
}


void free_accelobs(accelobs * obs)
{
    if (obs->mmap_file)
        close(obs->mmap_file);
    else if (obs->dat_input)
        free(obs->fft - ACCEL_PADDING / 2);
    else
        fclose(obs->fftfile);
    free(obs->powcut);
    free(obs->numindep);
    free(obs->rootfilenm);
    free(obs->candnm);
    free(obs->accelnm);
    free(obs->workfilenm);
    if (obs->numzap) {
        free(obs->lobins);
        free(obs->hibins);
    }
    if (obs->inmem && !obs->cudaP) {
        vect_free(obs->ffdotplane);
    }
}

void free_accelobs_list(accelobs *obs)
{
    int ii;
    if (obs->mmap_file)
        close(obs->mmap_file);
    else if (obs->dat_input)
    {
        for(ii=0;ii<obs->gpu;ii++)
            free(obs->fftlist[ii]- ACCEL_PADDING / 2);
        free(obs->fftlist);
    }
    else
    {
        for(ii=0;ii<obs->gpu;ii++)
            free(obs->fftfilelist[ii]);
        free(obs->fftfilelist);
    }
    free(obs->powcut);
    free(obs->numindep);
    
    for(ii=0;ii<obs->gpu;ii++)
    {
        free(obs->rootfilenmlist[ii]);
        free(obs->candnmlist[ii]);
        free(obs->accelnmlist[ii]);
        free(obs->workfilenmlist[ii]);
    }
    free(obs->rootfilenmlist);
    free(obs->candnmlist);
    free(obs->accelnmlist);
    free(obs->workfilenmlist);
    
    if (obs->numzap) {
        free(obs->lobins);
        free(obs->hibins);
    }
    if (obs->inmem && !obs->cudaP) {
        vect_free(obs->ffdotplane);
    }
}
