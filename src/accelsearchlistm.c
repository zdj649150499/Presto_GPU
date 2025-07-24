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
    int ii, jj, rstep, kk;
    double ttim, utim, stim, tott;
    struct tms runtimes;
    subharminfo **subharminfs;
    accelobs obs;
    infodata *idata;
    // GSList **cands  = NULL ;
    Cmdline *cmd;

    cufftComplex *pdata_gpu;
    cufftComplex *fkern_gpu;
    cufftComplex *tmpdat_gpu;
    fcomplex *data;


    unsigned short *d_rinds_gpu;
    unsigned short *d_zinds_gpu;
    
    unsigned short *d_rinds_cpu;
    unsigned short *d_zinds_cpu;

    double *norm_data_gpu, *norm_data_cpu;

    float *outpows_gpu;
    float *outpows_gpu_obs;
    int *rinds_gpu;
    int *nof_cand, *nof_cand_gpu;

    accel_cand_gpu *cand_array_search_gpu;
	// accel_cand_gpu *cand_array_sort_gpu;	
    accel_cand_gpu *cand_gpu_cpu;	
    cufftComplex *pdata;
    cufftComplex *data_gpu;
    float *power;
    int **offset_array;
    /* Prep the timer */

    static int firsttime = 1;

    

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
    
    cmd->listP = 1;
    if(cmd->listP)
    {
        printf("Read from list: %s\n", cmd->argv[0]);
        cmd->listnum = 0;
        FILE *FP;
        FP = fopen(cmd->argv[0], "r");
        while(!feof(FP))                  //获取list中的NBIN数
        {
            if(fgetc(FP)=='\n')
            {
                cmd->listnum++;
            }
        }
        rewind(FP);

        if(cmd->gpu > cmd->listnum)
            cmd->gpu = cmd->listnum;

        cmd->datalist = (char **) malloc(cmd->listnum * sizeof(char *));
        obs.rootfilenmlist = (char **) malloc(cmd->gpu * sizeof(char *));
        obs.candnmlist = (char **) malloc(cmd->gpu * sizeof(char *));
        obs.accelnmlist = (char **) malloc(cmd->gpu * sizeof(char *));
        obs.workfilenmlist = (char **) malloc(cmd->gpu * sizeof(char *));
        obs.workfilelist = (FILE **) malloc(cmd->gpu * sizeof(FILE *));
        for(ii=0;ii<cmd->listnum;ii++)
        {
            cmd->datalist[ii] = malloc(1024*sizeof(char));
            fscanf(FP, "%s", cmd->datalist[ii]);
        }
        
        for(ii=0; ii<cmd->gpu; ii++)
        {
            obs.rootfilenmlist[ii] = malloc(1024*sizeof(char));
            obs.candnmlist[ii] = malloc(1024*sizeof(char));
            obs.accelnmlist[ii] = malloc(1024*sizeof(char));
            obs.workfilenmlist[ii] = malloc(1024*sizeof(char));
        }

        fclose(FP);
        
        idata = malloc(cmd->gpu * sizeof(infodata));
        obs.fftfilelist = (FILE **) malloc(cmd->gpu * sizeof(FILE *));
        // obs.fftlist = (fcomplex **) malloc(cmd->gpu * sizeof(fcomplex *));
        obs.fftlist_f = (float **) malloc(cmd->gpu * sizeof(float *));
        

        obs.gpu = cmd->gpu;
        
    }

    printf("\n\n");
    printf("    Fourier-Domain Acceleration and Jerk Search Routine\n");
    printf("                    by Scott M. Ransom\n\n");
    printf("            GPU version by Dejiang Zhou, NAOC\n\n");

    if(cmd->cudaP) 
        select_cuda_dev(cmd->cuda);
    /* Create the accelobs structure */
    
    

    
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



    
    int numbirds;
    double *bird_lobins, *bird_hibins, hibin;

    double startr, lastr, nextr;

    ffdotpows *fundamental; 

    int stage, harmtosum, harm;
    int readdatanum;
    int fornum = (cmd->listnum + cmd->gpu - 1)/cmd->gpu;

    int fftlen;
    int numzs;

    
    
    for(kk=0; kk < fornum; kk++)
    {
        if(kk == fornum-1) readdatanum = cmd->listnum - kk*cmd->gpu;
        else readdatanum = cmd->gpu;

        printf("Reading data from %d files:\n", readdatanum);
        create_accelobs_list(&obs, idata, cmd, 1, cmd->ncpus, kk, readdatanum);

        int outpows_gpu_obs_xlen = (obs.highestbin+obs.corr_uselen)*obs.numbetween*obs.numz;
        int cand_cpu_xlen;
        int outpows_xlen;


        GSList **cands  = NULL ;
        cands = (GSList **) malloc(readdatanum * sizeof(GSList *));

        if(firsttime)
        {
            if(cmd->cudaP) cmd->otheroptP = 0;
            
            /* The step-size of blocks to walk through the input data */
            rstep = obs.corr_uselen * ACCEL_DR;

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

            /* Don't use the *.txtcand files on short in-memory searches */
            if (!obs.dat_input) {
                printf("  Working candidates in a test format are in '%s'.\n\n",
                    obs.workfilenmlist[0]);
            }

            fftlen =  subharminfs[0][0].kern[0][0].fftlen;
            numzs = subharminfs[0][0].numkern_zdim;

            cand_cpu_xlen = subharminfs[0][0].numkern * subharminfs[0][0].kern[0][0].fftlen;
            outpows_xlen = subharminfs[0][0].numkern*obs.fftlen;


            data = malloc(sizeof(fcomplex)*fftlen*readdatanum);

            
            // if(cmd->cudaP)
            {
                printf("\nAllocating GPU memory: \n");

                cudaMalloc((void**)&tmpdat_gpu, sizeof(cufftComplex)*fftlen*numzs*readdatanum);
                cudaMalloc((void**)&pdata_gpu, sizeof(cufftComplex)*fftlen*readdatanum);
                cudaMallocHost((void**)&power, sizeof(float)*fftlen/ACCEL_NUMBETWEEN*readdatanum);
                cudaMalloc((void**)&data_gpu, sizeof(cufftComplex)*fftlen*readdatanum);
                // CUDA_CHECK(cudaGetLastError());

                cudaMalloc((void**)&outpows_gpu, sizeof(float)*subharminfs[0][0].numkern*obs.fftlen*readdatanum);  // only for ffdot->numrs*ffdot->numzs is useful, numrs is changed each time but low than fftlen, numkern = numzs;
                // CUDA_CHECK(cudaGetLastError());

                cudaMalloc((void **)&cand_array_search_gpu, cand_cpu_xlen * readdatanum * sizeof(accel_cand_gpu));
                cudaMallocHost((void**)&cand_gpu_cpu, sizeof(accel_cand_gpu) * cand_cpu_xlen * readdatanum);

                cudaMalloc((void**)&norm_data_gpu, sizeof(double)*readdatanum);

                // CUDA_CHECK(cudaGetLastError());


                if(cmd->inmemP)
                {
                    cudaMalloc((void**)&fkern_gpu, sizeof(cufftComplex)*fftlen*numzs);
                    for(ii=0; ii<numzs; ii++)
                        cudaMemcpy(fkern_gpu+ii*fftlen, (cufftComplex *)subharminfs[0][0].kern[0][ii].data, sizeof(cufftComplex)*fftlen, cudaMemcpyHostToDevice);
                    
                    cudaMalloc((void**)&rinds_gpu, sizeof(int)*fftlen*(pow(2,obs.numharmstages)-1));
                    cudaMalloc((void**)&outpows_gpu_obs, sizeof(float)*outpows_gpu_obs_xlen*readdatanum);
                }
                else
                {
                    cudaMalloc((void**)&d_rinds_gpu, sizeof(unsigned short)*obs.corr_uselen*(pow(2,obs.numharmstages)-1));
                    cudaMalloc((void**)&d_zinds_gpu, sizeof(unsigned short)*obs.numz*(pow(2,obs.numharmstages)-1));
                    cudaMallocHost((void**)&d_rinds_cpu, sizeof(unsigned short)*obs.corr_uselen*(pow(2,obs.numharmstages)-1));
                    cudaMallocHost((void**)&d_zinds_cpu, sizeof(unsigned short)*obs.numz*(pow(2,obs.numharmstages)-1));
                    // CUDA_CHECK(cudaGetLastError());

                    offset_array = (int **)malloc(obs.numharmstages * sizeof(int *));
                    offset_array[0] = (int *)malloc( 1 * sizeof(int) );
                    int jj;
                    for(ii=1; ii<obs.numharmstages; ii++){	
                            jj = 1 << ii;		
                            offset_array[ii] = (int *)malloc( jj * sizeof(int));
                    }
                    fkern_gpu =  cp_kernel_array_to_gpu(subharminfs, obs.numharmstages, offset_array);
                }
                init_cuFFT_plans_list(subharminfs, obs.numharmstages, obs.inmem, readdatanum);
                // init_cuFFT_plans(subharminfs, obs.numharmstages, obs.inmem);

                printf("Done\n");
            }

            nof_cand = malloc(sizeof(int)*readdatanum);

            

            firsttime = 0;
        }

        
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
                    
                    #ifdef _OPENMP
                    #pragma omp parallel for num_threads(cmd->ncpus) shared(bird_lobins, bird_hibins, obs, readdatanum, hibin, numbirds)
                    #endif
                    for(jj=0; jj<readdatanum; jj++)
                        zapbirds(bird_lobins[ii], bird_hibins[ii], NULL, obs.fftlist[jj]);
            }

            vect_free(bird_lobins);
            vect_free(bird_hibins);
        }
        
        cudaMemset(outpows_gpu, 0.0, sizeof(float)*subharminfs[0][0].numkern*obs.fftlen*readdatanum);
        // CUDA_CHECK(cudaGetLastError());
        
        for(ii=0; ii<readdatanum; ii++)
            cands[ii]  = NULL;


        /* Start the main search loop */
        // for(ii=0; ii<readdatanum; ii++)
        {
            tott = times(&runtimes) / (double) CLK_TCK;

            // printf("Starting the main search loop data : %s ... \n", cmd->datalist[ii+kk*cmd->gpu]);
            printf("Starting the main search loop data ... \n");
            
            

            if (obs.inmem) {
                startr = 8;  // Choose a very low Fourier bin
                lastr = 0;
                nextr = 0;
                while (startr < obs.rlo) {
                    nextr = startr + rstep;
                    lastr = nextr - ACCEL_DR;
                    // Compute the F-Fdot plane
                    {
                        fundamental = ini_subharm_fderivs_vol(1, 1, startr, lastr,
                                                    &subharminfs[0][0], &obs);
                                                    
                        // subharm_fderivs_vol_gpu(1, 1, startr, lastr, &subharminfs[0][0], &obs, fkern_gpu, pdata_gpu, tmpdat_gpu, tmpdat_gpu, outpows_gpu, outpows_gpu_obs, pdata, 0, d_zinds_gpu, d_rinds_gpu, d_zinds_cpu, d_rinds_cpu, fundamental, offset_array, 0, data_gpu, power);

                        // CUDA_CHECK(cudaGetLastError());
                        subharm_fderivs_vol_gpu_1_list(1, 1, startr, lastr, &subharminfs[0][0], &obs, fkern_gpu, pdata_gpu, tmpdat_gpu, tmpdat_gpu, outpows_gpu, outpows_gpu_obs, pdata, 0, d_zinds_gpu, d_rinds_gpu, d_zinds_cpu, d_rinds_cpu, fundamental, offset_array, 0, data_gpu, power, readdatanum, fftlen, norm_data_gpu, outpows_xlen, outpows_gpu_obs_xlen);
                        // CUDA_CHECK(cudaGetLastError());

                        free(fundamental);
                    }
                    startr = nextr;
                }
            }  


            /* Reset indices if needed and search for real */
            startr = obs.rlo;
            lastr = 0;
            nextr = 0;

            

            while (startr + rstep < obs.highestbin) {
                /* Search the fundamental */
                nextr = startr + rstep;
                lastr = nextr - ACCEL_DR;

                {
                    fundamental = ini_subharm_fderivs_vol(1, 1, startr, lastr, &subharminfs[0][0], &obs);

                    subharm_fderivs_vol_gpu_1_list(1, 1, startr, lastr, &subharminfs[0][0], &obs, fkern_gpu, pdata_gpu, tmpdat_gpu, tmpdat_gpu, outpows_gpu, outpows_gpu_obs, pdata, obs.inmem, d_zinds_gpu, d_rinds_gpu, d_zinds_cpu, d_rinds_cpu, fundamental, offset_array, 0, data_gpu, power, readdatanum, fftlen, norm_data_gpu, outpows_xlen, outpows_gpu_obs_xlen);

                    if(cmd->inmemP)
                        get_rinds_gpu(fundamental, rinds_gpu, obs.numharmstages);
                    else
                        get_rind_zind_gpu(d_rinds_gpu, d_zinds_gpu, d_rinds_cpu, d_zinds_cpu, obs.numharmstages, obs, startr);

                    search_ffdotpows_gpu_list(obs.powcut[twon_to_index(1)], outpows_gpu, cand_array_search_gpu, fundamental->numzs, fundamental->numrs, cand_gpu_cpu, readdatanum, nof_cand, cand_cpu_xlen, outpows_xlen);

                    // #ifdef _OPENMP
                    // #pragma omp parallel for schedule(dynamic) shared(readdatanum, fundamental, cands, obs, ii, cand_gpu_cpu, nof_cand)
                    // #endif
                    for(ii=0; ii<readdatanum; ii++)
                    {
                        // cands[ii] = search_ffdotpows_sort_gpu_result(fundamental, 1, &obs, cands[ii], cand_gpu_cpu+ii*cand_cpu_xlen, nof_cand[ii]);
                        cands[ii] = search_ffdotpows_sort_gpu_result_listm(fundamental, 1, &obs, cands[ii], cand_gpu_cpu+ii*cand_cpu_xlen, nof_cand[ii], ii);
                    }

                }
                if (obs.numharmstages > 1) {        /* Search the subharmonics */

                    // Copy the fundamental's ffdot plane to the full in-core one
                    {
                        for (stage = 1; stage < obs.numharmstages; stage++) {
                            harmtosum = 1 << stage;

                            if(cmd->inmemP)
                                inmem_add_subharm_gpu_list(fundamental, &obs, outpows_gpu, outpows_gpu_obs, stage, rinds_gpu, outpows_xlen, readdatanum, outpows_gpu_obs_xlen);
                            else
                                for (harm = 1; harm < harmtosum; harm += 2)
                                    subharm_fderivs_vol_gpu_1_list(harmtosum, harm, startr, lastr, &subharminfs[stage][harm-1], &obs, fkern_gpu, pdata_gpu, tmpdat_gpu, tmpdat_gpu, outpows_gpu, outpows_gpu_obs, pdata, 0, d_zinds_gpu, d_rinds_gpu, d_zinds_cpu, d_rinds_cpu, fundamental, offset_array, stage, data_gpu, power, readdatanum, fftlen, norm_data_gpu, outpows_xlen, outpows_gpu_obs_xlen);

                            search_ffdotpows_gpu_list(obs.powcut[twon_to_index(harmtosum)], outpows_gpu, cand_array_search_gpu, fundamental->numzs, fundamental->numrs, cand_gpu_cpu, readdatanum, nof_cand, cand_cpu_xlen, outpows_xlen); 

                            // #ifdef _OPENMP
                            // #pragma omp parallel for schedule(dynamic) shared(readdatanum, fundamental, harmtosum, cands, obs, ii, cand_gpu_cpu, nof_cand)
                            // #endif
                            for(ii=0; ii<readdatanum; ii++)
                            {
                                // cands[ii] = search_ffdotpows_sort_gpu_result(fundamental, harmtosum, &obs, cands[ii], cand_gpu_cpu+ii*cand_cpu_xlen, nof_cand[ii]);
                                cands[ii] = search_ffdotpows_sort_gpu_result_listm(fundamental, harmtosum, &obs, cands[ii], cand_gpu_cpu+ii*cand_cpu_xlen, nof_cand[ii], ii);
                            }                            
                        }
                    }
                }
                free(fundamental);
                startr = nextr;
            }
            print_percent_complete(obs.highestbin - obs.rlo,
                                obs.highestbin - obs.rlo, "search", 0);
            
            for(ii=0; ii<readdatanum; ii++)
            {
                int numcands = g_slist_length(cands[ii]);
                printf("\nFound %d cands for %s\n", numcands, cmd->datalist[ii+kk*cmd->gpu]);
            }
            
            printf("\nTiming summary:\n");
            tott = times(&runtimes) / (double) CLK_TCK - tott;
            utim = runtimes.tms_utime / (double) CLK_TCK;
            stim = runtimes.tms_stime / (double) CLK_TCK;
            ttim = utim + stim;
            printf("    CPU time: %.3f sec (User: %.3f sec, System: %.3f sec)\n",
                ttim, utim, stim);
            printf("  Total time: %.3f sec\n\n", tott);
        }
    
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic) shared(readdatanum, cands, cmd, obs, idata, ii)
        // #pragma omp parallel for schedule(dynamic) default(none) shared(cands, idata, obs, cmd, readdatanum)
        #endif
        for(ii=0; ii<readdatanum; ii++)
        {
                int jj;
                int numcands;
                GSList *listptr;
                accelcand *cand;
                /* Candidate list trimming and optimization */
                numcands = g_slist_length(cands[ii]);

                if (numcands) {
                    fourierprops *props;
                    /* Sort the candidates according to the optimized sigmas */
                    cands[ii] = sort_accelcands(cands[ii]);

                    /* Eliminate (most of) the harmonically related candidates */
                    if ((cmd->numharm > 1) && !(cmd->noharmremoveP))
                        eliminate_harmonics(cands[ii], &numcands);

                    /* Now optimize each candidate and its harmonics */
                    listptr = cands[ii];
                    for (jj = 0; jj < numcands; jj++) {
                        cand = (accelcand *) (listptr->data);
                        optimize_accelcand_list(cand, &obs, ii);
                        // optimize_accelcand_list_gpu_noharmpolish(cand, &obs, ii);
                        listptr = listptr->next;
                    }

                    /* Calculate the properties of the fundamentals */
                    props = (fourierprops *) malloc(sizeof(fourierprops) * numcands);
                    listptr = cands[ii];
                    for (jj = 0; jj < numcands; jj++) {
                        cand = (accelcand *) (listptr->data);
                        /* In case the fundamental harmonic is not significant,  */
                        /* send the originally determined r and z from the       */
                        /* harmonic sum in the search.  Note that the derivs are */
                        /* not used for the computations with the fundamental.   */
                        calc_props(cand->derivs[0], cand->r, cand->z, cand->w, props + jj);
                        /* Override the error estimates based on power */
                        props[jj].rerr = (float) (ACCEL_DR) / cand->numharm;
                        props[jj].zerr = (float) (ACCEL_DZ) / cand->numharm;
                        props[jj].werr = (float) (ACCEL_DW) / cand->numharm;
                        listptr = listptr->next;
                    }

                    /* Write the fundamentals to the output text file */
                    output_fundamentals_list(props, cands[ii], &obs, &idata[ii], ii);
                    /* Write the harmonics to the output text file */
                    output_harmonics_list(cands[ii], &obs, &idata[ii], ii);
                    
                    /* Write the fundamental fourierprops to the cand file */
                    obs.workfilelist[ii] = chkfopen(obs.candnmlist[ii], "wb");
                    chkfwrite(props, sizeof(fourierprops), numcands, obs.workfilelist[ii]);
                    fclose(obs.workfilelist[ii]);
                    free(props);
                    
                }

            /* Finish up */
            printf("Final candidates in binary format are in '%s'.\n", obs.candnmlist[ii]);
            printf("Final Candidates in a text format are in '%s'.\n\n", obs.accelnmlist[ii]);
        }
        


        for(ii=0; ii<readdatanum; ii++)
        {
            g_slist_foreach(cands[ii], free_accelcand, NULL);
            g_slist_free(cands[ii]);
        }
        free(cands);
    }
    // free(cands);
    // g_slist_free(cands);


    // if(cmd->cudaP)
    {
        cudaFree(outpows_gpu);
        if(cmd->inmemP)
        {
            cudaFree(outpows_gpu_obs);
            cudaFree(rinds_gpu);
        }
        cudaFreeHost(power);
        cudaFree(data_gpu);
        cudaFree(pdata_gpu); 
        cudaFree(tmpdat_gpu);
        cudaFree(fkern_gpu);
        cudaFree(norm_data_gpu);
        
        cudaFree(cand_array_search_gpu);
        cudaFreeHost(cand_gpu_cpu);

        if(!cmd->inmemP)
        {
            cudaFree(d_zinds_gpu);
            cudaFree(d_rinds_gpu);
            cudaFreeHost(d_zinds_cpu);
            cudaFreeHost(d_rinds_cpu);
            free(offset_array);
        }
        destroy_cuFFT_plans(subharminfs, obs.numharmstages, obs.inmem);
        
        Endup_GPU();

        free(nof_cand);
    }

    if (cmd->zaplistP && !obs.mmap_file && obs.fft)
    {
        vect_free(bird_lobins);
        vect_free(bird_hibins);
    }

 
    free_accelobs_list(&obs);
    
    if(cmd->listP)
    {
        for (ii = 0; ii < cmd->listnum; ii++) {
            free(cmd->datalist[ii]);
        }
        free(cmd->datalist);
        free(idata);
    }

    
        
    return (0);
}
