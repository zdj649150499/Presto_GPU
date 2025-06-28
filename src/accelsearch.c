#include "accel.h"

// #include "cuda.cuh"

/*#undef USEMMAP*/

#ifdef USEMMAP
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif

#include "time.h"

// Use OpenMP
#ifdef _OPENMP
#include <omp.h>
extern void set_openmp_numthreads(int numthreads);
#endif

extern float calc_median_powers(fcomplex * amplitudes, int numamps);
extern void zapbirds(double lobin, double hibin, FILE * fftfile, fcomplex * fft);



static void print_percent_complete(int current, int number, char *what, int reset)
{
    static int newper = 0, oldper = -1;

    if (reset) {
        oldper = -1;
        newper = 0;
    } else {
        newper = (int) (current / (float) (number) * 100.0);
        if (newper < 0)
            newper = 0;
        if (newper > 100)
            newper = 100;
        if (newper > oldper) {
            printf("\rAmount of %s complete = %3d%%", what, newper);
            fflush(stdout);
            oldper = newper;
        }
    }
}

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

int main(int argc, char *argv[])
{
    int ii, rstep;
    double ttim, utim, stim, tott;
    struct tms runtimes;
    subharminfo **subharminfs;
    accelobs obs;
    infodata idata;
    GSList *cands = NULL;
    Cmdline *cmd;

    cufftComplex *pdata_gpu;
    cufftComplex *fkern_gpu;
    cufftComplex *tmpdat_gpu;
    unsigned short *d_rinds_gpu;
    unsigned short *d_zinds_gpu;
    
    unsigned short *d_rinds_cpu;
    unsigned short *d_zinds_cpu;

    float *outpows_gpu;
    float *outpows_gpu_obs;
    int *rinds_gpu;
    int nof_cand = 0 ;

    // cudaStream_t stream_1;// stream_2;  

    accel_cand_gpu *cand_array_search_gpu;
	// accel_cand_gpu *cand_array_sort_gpu;	
    accel_cand_gpu *cand_gpu_cpu;	
    cufftComplex *pdata;
    cufftComplex *data_gpu;
    float *power;
    int **offset_array;
    /* Prep the timer */

    tott = times(&runtimes) / (double) CLK_TCK;

    /* Call usage() if we have no command line arguments */

    if (argc == 1) {
        Program = argv[0];
        printf("\n");
        usage();
        exit(1);
    }

    /* Parse the command line using the excellent program Clig */

    cmd = parseCmdline(argc, argv);

#ifdef DEBUG
    showOptionValues();
#endif

    printf("\n\n");
    printf("    Fourier-Domain Acceleration and Jerk Search Routine\n");
    printf("                    by Scott M. Ransom\n\n");
    printf("            GPU version by Dejiang Zhou, NAOC\n\n");

    if(cmd->cudaP) 
        select_cuda_dev(cmd->cuda);
    /* Create the accelobs structure */
    create_accelobs(&obs, &idata, cmd, 1);

    if(cmd->cudaP) cmd->otheroptP = 0;

    /* The step-size of blocks to walk through the input data */
    rstep = obs.corr_uselen * ACCEL_DR;

    /* Zap birdies if requested and if in memory */
    if (cmd->zaplistP && !obs.mmap_file && obs.fft) {
        int numbirds;
        double *bird_lobins, *bird_hibins, hibin;

        /* Read the Standard bird list */
        numbirds = get_birdies(cmd->zaplist, obs.T, cmd->baryv,
                               &bird_lobins, &bird_hibins);

        /* Zap the birdies */
        printf("Zapping them using a barycentric velocity of %.5gc.\n\n",
               cmd->baryv);
        hibin = obs.N / 2;
        for (ii = 0; ii < numbirds; ii++) {
            if (bird_lobins[ii] >= hibin)
                break;
            if (bird_hibins[ii] >= hibin)
                bird_hibins[ii] = hibin - 1;
            zapbirds(bird_lobins[ii], bird_hibins[ii], NULL, obs.fft);
        }

        vect_free(bird_lobins);
        vect_free(bird_hibins);
    }

    printf("Searching with up to %d harmonics summed:\n",
           1 << (obs.numharmstages - 1));
    printf("  f = %.1f to %.1f Hz\n", obs.rlo / obs.T, obs.rhi / obs.T);
    printf("  r = %.1f to %.1f Fourier bins\n", obs.rlo, obs.rhi);
    printf("  z = %.1f to %.1f Fourier bins drifted\n", obs.zlo, obs.zhi);
    if (obs.numw)
        printf("  w = %.1f to %.1f Fourier-derivative bins drifted\n", obs.wlo, obs.whi);

    /* Generate the correlation kernels */

    printf("\nGenerating correlation kernels:\n");
    subharminfs = create_subharminfos(&obs);
    printf("Done generating kernels.\n\n");
    if (cmd->ncpus > 1) {
#ifdef _OPENMP
        set_openmp_numthreads(cmd->ncpus);
#endif
    } else {
#ifdef _OPENMP
        omp_set_num_threads(1); // Explicitly turn off OpenMP
#endif
        printf("Starting the search.\n\n");
    }
    /* Don't use the *.txtcand files on short in-memory searches */
    if (!obs.dat_input) {
        printf("  Working candidates in a test format are in '%s'.\n\n",
               obs.workfilenm);
    }

    /* Function pointers to make code a bit cleaner */
    void (*fund_to_ffdot)() = NULL;
    void (*add_subharm)() = NULL;
    void (*inmem_add_subharm)() = NULL;
    if (obs.inmem) {
        if (cmd->otheroptP) {
            fund_to_ffdot = &fund_to_ffdotplane_trans;
            inmem_add_subharm = &inmem_add_ffdotpows_trans;
        } else {
            fund_to_ffdot = &fund_to_ffdotplane;
            inmem_add_subharm = &inmem_add_ffdotpows;
        }
    } else {
        if (cmd->otheroptP) {
            add_subharm = &add_ffdotpows_ptrs;
        } else {
            add_subharm = &add_ffdotpows;
        }
    }

    double startr, lastr, nextr;

    ffdotpows *fundamental; 
    ffdotpows *subharmonic;

    int stage, harmtosum, harm;

    int fftlen =  subharminfs[0][0].kern[0][0].fftlen;
    int numzs = subharminfs[0][0].numkern_zdim;

    // __device__ cufftComplex data_gpu[fftlen];


    if(cmd->cudaP)
    {
        // cudaStreamCreate(&stream_1); 
        // cudaStreamCreate(&stream_2); 

        cudaMalloc((void**)&tmpdat_gpu, sizeof(cufftComplex)*fftlen*numzs);
        cudaMalloc((void**)&pdata_gpu, sizeof(cufftComplex)*fftlen);
        // cudaMallocHost((void**)&pdata, sizeof(cufftComplex)*fftlen);
        cudaMallocHost((void**)&power, sizeof(float)*fftlen/ACCEL_NUMBETWEEN);
        cudaMalloc((void**)&data_gpu, sizeof(cufftComplex)*fftlen);

        cudaMalloc((void**)&outpows_gpu, sizeof(float)*subharminfs[0][0].numkern*obs.fftlen);  // only for ffdot->numrs*ffdot->numzs is useful, numrs is changed each time but low than fftlen, numkern = numzs;
        cudaMemset(outpows_gpu, 0.0, sizeof(float)*subharminfs[0][0].numkern*obs.fftlen);
        

        cudaMalloc((void **)&cand_array_search_gpu, subharminfs[0][0].numkern * subharminfs[0][0].kern[0][0].fftlen * sizeof(accel_cand_gpu));
        // cudaMalloc((void **)&cand_array_sort_gpu, subharminfs[0][0].numkern * subharminfs[0][0].kern[0][0].fftlen * sizeof(accel_cand_gpu));
        cudaMallocHost((void**)&cand_gpu_cpu, sizeof(accel_cand_gpu)*subharminfs[0][0].numkern * subharminfs[0][0].kern[0][0].fftlen);


        if(cmd->inmemP)
        {
            cudaMalloc((void**)&fkern_gpu, sizeof(cufftComplex)*fftlen*numzs);
            for(ii=0; ii<numzs; ii++)
                cudaMemcpy(fkern_gpu+ii*fftlen, (cufftComplex *)subharminfs[0][0].kern[0][ii].data, sizeof(cufftComplex)*fftlen, cudaMemcpyHostToDevice);
            
            cudaMalloc((void**)&rinds_gpu, sizeof(int)*fftlen*(pow(2,obs.numharmstages)-1));
            cudaMalloc((void**)&outpows_gpu_obs, sizeof(float)*(obs.highestbin+obs.corr_uselen)*obs.numbetween*obs.numz);
        }
        else
        {
            cudaMalloc((void**)&d_rinds_gpu, sizeof(unsigned short)*obs.corr_uselen*(pow(2,obs.numharmstages)-1));
            cudaMalloc((void**)&d_zinds_gpu, sizeof(unsigned short)*obs.numz*(pow(2,obs.numharmstages)-1));
            cudaMallocHost((void**)&d_rinds_cpu, sizeof(unsigned short)*obs.corr_uselen*(pow(2,obs.numharmstages)-1));
            cudaMallocHost((void**)&d_zinds_cpu, sizeof(unsigned short)*obs.numz*(pow(2,obs.numharmstages)-1));
            // d_rinds_cpu = malloc(sizeof(unsigned short)*obs.corr_uselen*(pow(2,obs.numharmstages)-1));
            // d_zinds_cpu = malloc(sizeof(unsigned short)*obs.numz*(pow(2,obs.numharmstages)-1));

            offset_array = (int **)malloc(obs.numharmstages * sizeof(int *));
            offset_array[0] = (int *)malloc( 1 * sizeof(int) );
            int jj;
            for(ii=1; ii<obs.numharmstages; ii++){	
                    jj = 1 << ii;		
                    offset_array[ii] = (int *)malloc( jj * sizeof(int));
            }
            fkern_gpu =  cp_kernel_array_to_gpu(subharminfs, obs.numharmstages, offset_array);
        }

        if(obs.dat_input&&obs.numbins<100000000)
        {
            cudaMalloc((void**)&obs.fft_gpu, sizeof(cufftComplex)*(obs.numbins+ACCEL_PADDING));
            cudaMemcpy(obs.fft_gpu, obs.fft-ACCEL_PADDING/2, sizeof(cufftComplex)*(obs.numbins+ACCEL_PADDING), cudaMemcpyHostToDevice);
            obs.fft_gpu += ACCEL_PADDING/2;
        }

        init_cuFFT_plans(subharminfs, obs.numharmstages, obs.inmem);
        

        // cudaStreamSynchronize(stream_1);
    }
    
    /* Start the main search loop */
    {
        /* Populate the saved F-Fdot plane at low freqs for in-memory
         * searches of harmonics that are below obs.rlo */
        if (obs.inmem) {
            startr = 8;  // Choose a very low Fourier bin
            lastr = 0;
            nextr = 0;
            while (startr < obs.rlo) {
                nextr = startr + rstep;
                lastr = nextr - ACCEL_DR;
                // Compute the F-Fdot plane
                if(!cmd->cudaP)
                {
                    fundamental = subharm_fderivs_vol(1, 1, startr, lastr,
                                                  &subharminfs[0][0], &obs);
                    fund_to_ffdot(fundamental, &obs);
                    free_ffdotpows(fundamental);
                }
                else 
                {
                    fundamental = ini_subharm_fderivs_vol(1, 1, startr, lastr,
                                                  &subharminfs[0][0], &obs);
                                                  
                    subharm_fderivs_vol_gpu(1, 1, startr, lastr,
                                                  &subharminfs[0][0], &obs, fkern_gpu, pdata_gpu, tmpdat_gpu, tmpdat_gpu, outpows_gpu, outpows_gpu_obs, pdata, 0, d_zinds_gpu, d_rinds_gpu, d_zinds_cpu, d_rinds_cpu, fundamental, offset_array, 0, data_gpu, power);
                    free(fundamental);
                }
                startr = nextr;
            }
        }  

        /* Reset indices if needed and search for real */
        startr = obs.rlo;
        lastr = 0;
        nextr = 0;

        

        // int firsttime=1;
        while (startr + rstep < obs.highestbin) {
            /* Search the fundamental */
            nextr = startr + rstep;
            lastr = nextr - ACCEL_DR;
            if(!cmd->cudaP)
            {
                fundamental = subharm_fderivs_vol(1, 1, startr, lastr,
                                              &subharminfs[0][0], &obs);
                cands = search_ffdotpows(fundamental, 1, &obs, cands);
            }
            else
            {
                fundamental = ini_subharm_fderivs_vol(1, 1, startr, lastr, &subharminfs[0][0], &obs);
                subharm_fderivs_vol_gpu(1, 1, startr, lastr,
                                              &subharminfs[0][0], &obs, fkern_gpu, pdata_gpu, tmpdat_gpu, tmpdat_gpu, outpows_gpu, outpows_gpu_obs, pdata, obs.inmem, d_zinds_gpu, d_rinds_gpu, d_zinds_cpu, d_rinds_cpu, fundamental, offset_array, 0, data_gpu, power);
                
                if(cmd->inmemP)
                {
                    get_rinds_gpu(fundamental, rinds_gpu, obs.numharmstages);
                }
                else
                    get_rind_zind_gpu(d_rinds_gpu, d_zinds_gpu, d_rinds_cpu, d_zinds_cpu, obs.numharmstages, obs, startr);
                nof_cand = search_ffdotpows_gpu(obs.powcut[twon_to_index(1)], outpows_gpu, cand_array_search_gpu, fundamental->numzs, fundamental->numrs, cand_gpu_cpu);
                cands = search_ffdotpows_sort_gpu_result(fundamental, 1, &obs, cands, cand_gpu_cpu, nof_cand);
            }
            if (obs.numharmstages > 1) {        /* Search the subharmonics */

                // Copy the fundamental's ffdot plane to the full in-core one
                if (obs.inmem && !cmd->cudaP)
                    fund_to_ffdot(fundamental, &obs);

                if(!cmd->cudaP)
                {
                    for (stage = 1; stage < obs.numharmstages; stage++) {
                        harmtosum = 1 << stage;
                        for (harm = 1; harm < harmtosum; harm += 2) {
                            if (obs.inmem) 
                                inmem_add_subharm(fundamental, &obs, harmtosum, harm);
                            else 
                            {
                                subharmonic = subharm_fderivs_vol(harmtosum, harm, startr, lastr,
                                                        &subharminfs[stage][harm-1],
                                                        &obs);
                                add_subharm(fundamental, subharmonic, harmtosum, harm);
                                free_ffdotpows(subharmonic);
                            }
                        }
                        cands = search_ffdotpows(fundamental, harmtosum, &obs, cands);
                    }
                }
                else
                {
                    for (stage = 1; stage < obs.numharmstages; stage++) {
                        harmtosum = 1 << stage;
                        if(cmd->inmemP)
                            inmem_add_subharm_gpu(fundamental, &obs, outpows_gpu, outpows_gpu_obs, stage, rinds_gpu);
                        else
                        {
                            for (harm = 1; harm < harmtosum; harm += 2)
                                subharm_fderivs_vol_gpu(harmtosum, harm, startr, lastr,
                                              &subharminfs[stage][harm-1], &obs, fkern_gpu, pdata_gpu, tmpdat_gpu, tmpdat_gpu, outpows_gpu, outpows_gpu_obs, pdata, 0, d_zinds_gpu, d_rinds_gpu, d_zinds_cpu, d_rinds_cpu, fundamental, offset_array, stage, data_gpu, power);
                        }
                        nof_cand = search_ffdotpows_gpu(obs.powcut[twon_to_index(harmtosum)], outpows_gpu, cand_array_search_gpu, fundamental->numzs, fundamental->numrs, cand_gpu_cpu);                        
                        cands = search_ffdotpows_sort_gpu_result(fundamental, harmtosum, &obs, cands, cand_gpu_cpu, nof_cand);
                    }
                }
            }
            if(!cmd->cudaP)
                free_ffdotpows(fundamental);
            else
                free(fundamental);
            startr = nextr;
        }
        print_percent_complete(obs.highestbin - obs.rlo,
                               obs.highestbin - obs.rlo, "search", 0);
    }
    if(cmd->cudaP)
    {
        cudaFree(outpows_gpu);
        CUDA_CHECK(cudaGetLastError());
        if(cmd->inmemP)
        {
            cudaFree(outpows_gpu_obs);
            cudaFree(rinds_gpu);
        }
        // cudaFreeHost(pdata);
        cudaFreeHost(power);
        cudaFree(data_gpu);
        cudaFree(pdata_gpu); 
        cudaFree(tmpdat_gpu);
        cudaFree(fkern_gpu);
        CUDA_CHECK(cudaGetLastError());

        if(obs.dat_input&&obs.numbins<100000000)
            cudaFree(obs.fft_gpu-ACCEL_PADDING/2);


        cudaFree(cand_array_search_gpu);
	    // cudaFree(cand_array_sort_gpu);
        
        cudaFreeHost(cand_gpu_cpu);
        CUDA_CHECK(cudaGetLastError());
        if(!cmd->inmemP)
        {
            cudaFree(d_zinds_gpu);
            CUDA_CHECK(cudaGetLastError());
            cudaFree(d_rinds_gpu);
            CUDA_CHECK(cudaGetLastError());
            cudaFreeHost(d_zinds_cpu);
            CUDA_CHECK(cudaGetLastError());
            cudaFreeHost(d_rinds_cpu);
            CUDA_CHECK(cudaGetLastError());
            free(offset_array);
        }
        CUDA_CHECK(cudaGetLastError());
        destroy_cuFFT_plans(subharminfs, obs.numharmstages, obs.inmem);

        // cudaStreamDestroy(stream_1);
        // cudaStreamDestroy(stream_2);
        Endup_GPU();
    }

    printf("\n\nDone searching.  Now optimizing each candidate.\n\n");
    free_subharminfos(&obs, subharminfs);

    printf("\nTiming summary:\n");
    tott = times(&runtimes) / (double) CLK_TCK - tott;
    utim = runtimes.tms_utime / (double) CLK_TCK;
    stim = runtimes.tms_stime / (double) CLK_TCK;
    ttim = utim + stim;
    printf("    CPU time: %.3f sec (User: %.3f sec, System: %.3f sec)\n",
           ttim, utim, stim);
    printf("  Total time: %.3f sec\n\n", tott);

    {                           /* Candidate list trimming and optimization */
        int numcands = g_slist_length(cands);
        printf("\nFound %d cands\n", numcands);


        GSList *listptr;
        accelcand *cand;
        fourierprops *props;

        if (numcands) {

            /* Sort the candidates according to the optimized sigmas */
            cands = sort_accelcands(cands);

            /* Eliminate (most of) the harmonically related candidates */
            if ((cmd->numharm > 1) && !(cmd->noharmremoveP))
                eliminate_harmonics(cands, &numcands);

            /* Now optimize each candidate and its harmonics */
            print_percent_complete(0, 0, NULL, 1);
            listptr = cands;
            for (ii = 0; ii < numcands; ii++) {
                print_percent_complete(ii, numcands, "optimization", 0);
                cand = (accelcand *) (listptr->data);
                optimize_accelcand(cand, &obs);
                listptr = listptr->next;
            }
            print_percent_complete(ii, numcands, "optimization", 0);

            /* Calculate the properties of the fundamentals */
            props = (fourierprops *) malloc(sizeof(fourierprops) * numcands);
            listptr = cands;
            for (ii = 0; ii < numcands; ii++) {
                cand = (accelcand *) (listptr->data);
                /* In case the fundamental harmonic is not significant,  */
                /* send the originally determined r and z from the       */
                /* harmonic sum in the search.  Note that the derivs are */
                /* not used for the computations with the fundamental.   */
                calc_props(cand->derivs[0], cand->r, cand->z, cand->w, props + ii);
                /* Override the error estimates based on power */
                props[ii].rerr = (float) (ACCEL_DR) / cand->numharm;
                props[ii].zerr = (float) (ACCEL_DZ) / cand->numharm;
                props[ii].werr = (float) (ACCEL_DW) / cand->numharm;
                listptr = listptr->next;
            }

            /* Write the fundamentals to the output text file */
            output_fundamentals(props, cands, &obs, &idata);

            /* Write the harmonics to the output text file */
            output_harmonics(cands, &obs, &idata);
            
            /* Write the fundamental fourierprops to the cand file */
            obs.workfile = chkfopen(obs.candnm, "wb");
            chkfwrite(props, sizeof(fourierprops), numcands, obs.workfile);
            fclose(obs.workfile);
            free(props);
            printf("\n\n");
        } else {
            printf("No candidates above sigma = %.2f were found.\n\n", obs.sigma);
        }
    }

    /* Finish up */

    printf("Searched the following approx numbers of independent points:\n");
    printf("  %d harmonic:   %9lld\n", 1, obs.numindep[0]);
    for (ii = 1; ii < obs.numharmstages; ii++)
        printf("  %d harmonics:  %9lld\n", 1 << ii, obs.numindep[ii]);

    printf("\nTiming summary:\n");
    tott = times(&runtimes) / (double) CLK_TCK - tott;
    utim = runtimes.tms_utime / (double) CLK_TCK;
    stim = runtimes.tms_stime / (double) CLK_TCK;
    ttim = utim + stim;
    printf("    CPU time: %.3f sec (User: %.3f sec, System: %.3f sec)\n",
           ttim, utim, stim);
    printf("  Total time: %.3f sec\n\n", tott);

    printf("Final candidates in binary format are in '%s'.\n", obs.candnm);
    printf("Final Candidates in a text format are in '%s'.\n\n", obs.accelnm);

    free_accelobs(&obs);
    g_slist_foreach(cands, free_accelcand, NULL);
    g_slist_free(cands);


        
    return (0);
}
