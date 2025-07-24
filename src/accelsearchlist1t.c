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
#include <pthread.h>
#include <stdbool.h>

// Use OpenMP
#ifdef _OPENMP
#include <omp.h>
extern void set_openmp_numthreads(int numthreads);
#endif

Cmdline *cmd;
accelobs obs;

cufftComplex *pdata_gpu;
cufftComplex *fkern_gpu;
cufftComplex *tmpdat_gpu;
fcomplex *data;

subharminfo **subharminfs_all;


unsigned short *d_rinds_gpu;
unsigned short *d_zinds_gpu;

unsigned short *d_rinds_cpu;
unsigned short *d_zinds_cpu;

float *outpows_gpu;
float *outpows_gpu_obs;
int *rinds_gpu;
int nof_cand;

accel_cand_gpu *cand_array_search_gpu;
// accel_cand_gpu *cand_array_sort_gpu;	
accel_cand_gpu *cand_gpu_cpu;	
cufftComplex *pdata;
cufftComplex *data_gpu;

float *power;
int **offset_array;

static int firsttime = 1;
int fornum;

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



typedef struct {
    GSList *cands;
    accelobs obs;
    infodata idata;
} QueueItem;

typedef struct {
    QueueItem *items;
    int capacity;
    int size;
    int front;
    int rear;
} ThreadSafeQueue;


pthread_mutex_t queue_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t queue_cond = PTHREAD_COND_INITIALIZER;
pthread_cond_t  cv_prod = PTHREAD_COND_INITIALIZER;
pthread_cond_t  cv_cons = PTHREAD_COND_INITIALIZER;
ThreadSafeQueue task_queue;
bool ready1 = false;   // true表示中间结果已准备好交给线程2，单需要等ready2是否将之前的数据处理完毕
bool ready2 = true;    // true表示线程2准备好，可以从线程1接收下一段数据，false表示线程1需要等一会交接数据
bool done = false;
void queue_init(ThreadSafeQueue *q, int capacity) {
    q->items = malloc(capacity * sizeof(QueueItem));
    q->capacity = capacity;
    q->size = 0;
    q->front = 0;
    q->rear = -1;
}

// void queue_push(ThreadSafeQueue *q, QueueItem item) {
//     if (q->size < q->capacity) {
//         q->rear = (q->rear + 1) % q->capacity;
//         q->items[q->rear] = item;
//         q->size++;
//     }
// }

void queue_push(ThreadSafeQueue *q, QueueItem item) {
    // if (q->size >= q->capacity) {
    //     return;  // 或者可以阻塞等待
    // }
    q->rear = (q->rear + 1) % q->capacity;
    q->items[q->rear] = item;
    q->size++;
}

QueueItem queue_pop(ThreadSafeQueue *q) {
    QueueItem item = q->items[q->front];
    q->front = (q->front + 1) % q->capacity;
    q->size--;
    return item;
}

int queue_empty(ThreadSafeQueue *q) {
    return q->size == 0;
}


void *producer_thread(void *arg) 
{
    int ii, jj, rstep, kk;
    subharminfo **subharminfs;

    printf("GPU device: %d\n\n", cmd->cuda);

    if(cmd->cudaP) 
        select_cuda_dev(cmd->cuda);
    accelobs obs_all;

    for (kk = 0; kk < fornum; kk++) {
        accelobs obs;
        double ttim, utim, stim, tott;
        struct tms runtimes;

        accelobs current_obs;// = malloc(sizeof(accelobs));
        infodata current_idata;//  = malloc(sizeof(infodata));
        GSList *cands = NULL;


        printf("Reading data from %d files:\n", 1);

        // 初始化并处理文件，生成候选体
        create_accelobs_list_1(&obs, &current_idata, cmd, 1, kk);

        // obs = current_obs;
        // idata = current_idata;

        // process_file_gpu(current_obs, current_idata, cands); // 提取原处理逻辑至此函数
        {
            rstep = obs.corr_uselen * ACCEL_DR;

            int numbirds;
            double *bird_lobins, *bird_hibins, hibin;

            double startr, lastr, nextr;

            ffdotpows *fundamental; 
            ffdotpows *subharmonic;

            int stage, harmtosum, harm;
            // int readdatanum = 1;
            // fornum = cmd->listnum;

            int fftlen;
            int numzs;

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

            printf("Searching with up to %d harmonics summed:\n",
                    1 << (obs.numharmstages - 1));
            printf("  f = %.1f to %.1f Hz\n", obs.rlo / obs.T, obs.rhi / obs.T);
            printf("  r = %.1f to %.1f Fourier bins\n", obs.rlo, obs.rhi);
            printf("  z = %.1f to %.1f Fourier bins drifted\n", obs.zlo, obs.zhi);
            if (obs.numw)
                printf("  w = %.1f to %.1f Fourier-derivative bins drifted\n", obs.wlo, obs.whi);
            
            /* Generate the correlation kernels */
                      


            if(firsttime)
            {
                if(cmd->cudaP) cmd->otheroptP = 0;
                
                /* The step-size of blocks to walk through the input data */
                printf("\nGenerating correlation kernels:\n");
                subharminfs = create_subharminfos(&obs);
                // subharminfs_all = subharminfs;

                printf("Done generating kernels.\n\n");

                /* Don't use the *.txtcand files on short in-memory searches */
                if (!obs.dat_input) {
                    printf("  Working candidates in a test format are in '%s'.\n\n",
                        obs.workfilenm);
                }
                fftlen =  subharminfs[0][0].kern[0][0].fftlen;
                numzs = subharminfs[0][0].numkern_zdim;
            
                if(cmd->cudaP)
                {
                    printf("\nAllocating GPU memory: \n");

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
                        obs.fft_gpu += ACCEL_PADDING/2;
                    }

                    init_cuFFT_plans(subharminfs, obs.numharmstages, obs.inmem);

                    printf("Done\n");
                }

                // nof_cand = malloc(sizeof(int)*readdatanum);
                // cands = (GSList **) malloc(readdatanum * sizeof(GSList *));

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
                        
                        zapbirds(bird_lobins[ii], bird_hibins[ii], NULL, obs.fft);

                }

                vect_free(bird_lobins);
                vect_free(bird_hibins);
            }

            if(obs.dat_input&&obs.numbins<100000000)
                cudaMemcpy(obs.fft_gpu-ACCEL_PADDING/2, obs.fft-ACCEL_PADDING/2, sizeof(cufftComplex)*(obs.numbins+ACCEL_PADDING), cudaMemcpyHostToDevice);
            
            cudaMemset(outpows_gpu, 0.0, sizeof(float)*subharminfs[0][0].numkern*obs.fftlen);

            {
                tott = times(&runtimes) / (double) CLK_TCK;

                printf("Starting the main search loop data ... \n");
                
                

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
                                                        
                            subharm_fderivs_vol_gpu(1, 1, startr, lastr, &subharminfs[0][0], &obs, fkern_gpu, pdata_gpu, tmpdat_gpu, tmpdat_gpu, outpows_gpu, outpows_gpu_obs, pdata, 0, d_zinds_gpu, d_rinds_gpu, d_zinds_cpu, d_rinds_cpu, fundamental, offset_array, 0, data_gpu, power);

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
                
                // for(ii=0; ii<readdatanum; ii++)
                {
                    int numcands = g_slist_length(cands);
                    printf("\nFound %d cands for %s\n", numcands, cmd->datalist[kk]);
                }
                
                printf("\nTiming summary:\n");
                tott = times(&runtimes) / (double) CLK_TCK - tott;
                utim = runtimes.tms_utime / (double) CLK_TCK;
                stim = runtimes.tms_stime / (double) CLK_TCK;
                ttim = utim + stim;
                printf("    1-CPU time: %.3f sec (User: %.3f sec, System: %.3f sec)\n",
                    ttim, utim, stim);
                printf("  1-Total time: %.3f sec\n\n", tott);
            }
        }

        pthread_mutex_lock(&queue_mutex);
        while (!ready2) {  // 等待消费者准备好
            pthread_cond_wait(&cv_prod, &queue_mutex);
        }
        QueueItem item = { .cands = cands, .obs = obs, .idata = current_idata};
        queue_push(&task_queue, item);
        ready1 = true;    // 数据已准备好
        // ready2 = false;   // 等待消费者处理
        pthread_cond_signal(&cv_cons); // 通知消费者
        pthread_mutex_unlock(&queue_mutex);

        if(kk==fornum-1)
        {
            pthread_mutex_lock(&queue_mutex);
            while (!ready2) {
                pthread_cond_wait(&cv_prod, &queue_mutex);
            }
            ready1 = true;
            done = true;
            // pthread_cond_broadcast(&cv_cons); // 唤醒所有可能阻塞的消费者
            pthread_cond_signal(&cv_cons); // 通知消费者
            pthread_mutex_unlock(&queue_mutex);

            if(cmd->cudaP){
                if(obs.dat_input&&obs.numbins<100000000)
                    cudaFree(obs.fft_gpu-ACCEL_PADDING/2);
                destroy_cuFFT_plans(subharminfs_all, obs.numharmstages, obs.inmem);
            }

            
            // free_accelobs(&obs);
        }
    }

    

    if(cmd->cudaP)
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
        
        Endup_GPU();
        // free(nof_cand);
    }

    

    pthread_exit(NULL);
}


void *consumer_thread(void *arg) 
{
    printf("\n\nDone searching.  Now optimizing each candidate.\n\n");
    while (1) 
    {
        pthread_mutex_lock(&queue_mutex);
        while (!ready1) {
            ready2 = true; // 标记为可接收新数据
            pthread_cond_signal(&cv_prod); // 确保生产者可以继续
            pthread_cond_wait(&cv_cons, &queue_mutex);
        }

        // if (done) {
        //     // pthread_mutex_unlock(&queue_mutex);
        //     break;
        // }
        
        // 取出任务项
        QueueItem item = queue_pop(&task_queue);
        ready1 = false;
        ready2 = false;
        pthread_cond_signal(&cv_prod);
        pthread_mutex_unlock(&queue_mutex);

        {
                int jj;
                int numcands;
                GSList *listptr;
                accelcand *cand;
                GSList *cands = item.cands;
                accelobs obs = item.obs;
                infodata idata = item.idata;
                /* Candidate list trimming and optimization */
                numcands = g_slist_length(cands);

                if (numcands) {
                    fourierprops *props;
                    /* Sort the candidates according to the optimized sigmas */
                    cands = sort_accelcands(cands);

                    /* Eliminate (most of) the harmonically related candidates */
                    if ((cmd->numharm > 1) && !(cmd->noharmremoveP))
                        eliminate_harmonics(cands, &numcands);

                    /* Now optimize each candidate and its harmonics */
                    listptr = cands;
                    for (jj = 0; jj < numcands; jj++) {
                        cand = (accelcand *) (listptr->data);
                        // optimize_accelcand_list(cand, &obs, ii);
                        optimize_accelcand(cand, &obs);
                        // optimize_accelcand_list_gpu_noharmpolish(cand, &obs, ii);
                        listptr = listptr->next;
                    }

                    /* Calculate the properties of the fundamentals */
                    props = (fourierprops *) malloc(sizeof(fourierprops) * numcands);
                    listptr = cands;
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
                    output_fundamentals(props, cands, &obs, &idata);

                    /* Write the harmonics to the output text file */
                    output_harmonics(cands, &obs, &idata);

                    /* Write the fundamentals to the output text file */
                    // output_fundamentals(props, cands[ii], &obs, &idata[ii]);
                    {                       
                        /* Write the fundamental fourierprops to the cand file */
                        obs.workfile = chkfopen(obs.candnm, "wb");
                        chkfwrite(props, sizeof(fourierprops), numcands, obs.workfile);
                        fclose(obs.workfile);
                        free(props);
                    }
                    
                }
                else {
                    printf("No candidates above sigma = %.2f were found.\n\n", obs.sigma);
                }

            /* Finish up */
            {

                printf("Final candidates in binary format are in '%s'.\n", obs.candnm);
                printf("Final Candidates in a text format are in '%s'.\n\n", obs.accelnm);
            }

            // g_slist_foreach(cands, free_accelcand, NULL);
            // g_slist_free(cands);
        }

        if (done) {
            // pthread_mutex_unlock(&queue_mutex);
            break;
        }

        
        // free_accelobs(&obs);
    
    }
    pthread_exit(NULL);
}





int main(int argc, char *argv[])
{
    int ii, jj, rstep, kk;
    double ttim, utim, stim, tott;
    struct tms runtimes;
    
    tott = times(&runtimes) / (double) CLK_TCK;
    
    infodata idata;
    // GSList *cands  = NULL ;
    

    cufftComplex *pdata_gpu;
    cufftComplex *fkern_gpu;
    cufftComplex *tmpdat_gpu;
    fcomplex *data;


    unsigned short *d_rinds_gpu;
    unsigned short *d_zinds_gpu;
    
    unsigned short *d_rinds_cpu;
    unsigned short *d_zinds_cpu;

    float *outpows_gpu;
    float *outpows_gpu_obs;
    int *rinds_gpu;
    int nof_cand;

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

        cmd->datalist = (char **) malloc(cmd->listnum * sizeof(char *));
        for(ii=0;ii<cmd->listnum;ii++)
        {
            cmd->datalist[ii] = malloc(256*sizeof(char));
            fscanf(FP, "%s", cmd->datalist[ii]);
        }

        fclose(FP);
    }

    printf("\n\n");
    printf("    Fourier-Domain Acceleration and Jerk Search Routine\n");
    printf("                    by Scott M. Ransom\n\n");
    printf("            GPU version by Dejiang Zhou, NAOC\n\n");

    
    /* Create the accelobs structure */

    printf("Starting the search.\n\n");

    fornum = cmd->listnum;

    queue_init(&task_queue, 2);
    pthread_t producer, consumer;
    // int ThErr;
    pthread_attr_t ThAttr;		
    pthread_attr_init(&ThAttr);
	pthread_attr_setdetachstate(&ThAttr, PTHREAD_CREATE_JOINABLE);

    pthread_create(&producer, NULL, producer_thread, NULL);
    pthread_create(&consumer, NULL, consumer_thread, NULL);

    pthread_attr_destroy(&ThAttr);

    pthread_join(producer, NULL);
    pthread_join(consumer, NULL);

    // 清理队列
    free(task_queue.items);
    
    
    if(cmd->listP)
    {
        for (ii = 0; ii < cmd->listnum; ii++) {
            free(cmd->datalist[ii]);
        }
        free(cmd->datalist);

        // free(idata);
    }

    // if (cmd->zaplistP && !obs.mmap_file && obs.fft)
    // {
    //     vect_free(bird_lobins);
    //     vect_free(bird_hibins);
    // }


    printf("\nTiming summary:\n");
    tott = times(&runtimes) / (double) CLK_TCK - tott;
    utim = runtimes.tms_utime / (double) CLK_TCK;
    stim = runtimes.tms_stime / (double) CLK_TCK;
    ttim = utim + stim;
    printf("    CPU time: %.3f sec (User: %.3f sec, System: %.3f sec)\n",
        ttim, utim, stim);
    printf("  Total time: %.3f sec\n\n", tott);


        
    return (0);
}
