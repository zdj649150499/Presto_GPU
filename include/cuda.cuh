#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include <cuda.h>
#include <cufft.h>


#ifdef __cplusplus
extern "C" {
#endif


#define CUDA_CHECK(call)                            \
do                                                  \
{                                                   \
    const cudaError_t error_code = call;            \
    if (error_code != cudaSuccess)                  \
    {                                               \
        printf("CUDA Error:\n");                    \
        printf("    File:       %s\n", __FILE__);   \
        printf("    Line:       %d\n", __LINE__);   \
        printf("    Error code: %d\n", error_code); \
        printf("    Error text: %s\n",              \
                cudaGetErrorString(error_code));    \
        exit(1);                                    \
    }                                               \
} while (0)




#ifndef _FCOMPLEX_DECLARED_
typedef struct fcomplex {
    float r, i;
} fcomplex;
#define _FCOMPLEX_DECLARED_
#endif				/* _FCOMPLEX_DECLARED_ */

typedef struct accel_cand_gpu{
	float pow;					/*pow of selected candidate*/
	int		nof_cand;			/*number of candidates in sub_array/plane */
	int		z_ind;				/*z_index of the selected candidate*/
	int		r_ind;				/*r_index of the selected candidate*/
    int		w_ind;				/*w_index of the selected candidate*/
}	accel_cand_gpu;

typedef struct foldstats_gpu {
  double numdata;     /* Number of data bins folded         */
  double data_avg;    /* Average level of the data bins     */
  double data_var;    /* Variance of the data bins          */
  double numprof;     /* Number of bins in the profile      */
  double prof_avg;    /* Average level of the profile bins  */
  double prof_var;    /* Variance of the profile bins       */
  double redchi;      /* Reduced chi-squared of the profile */
} foldstats_gpu;

typedef struct POSITION_gpu {
    float pow;	    /* Power normalized with local power             */
    double p1;	    /* r if rzw search, r_startfft if bin search     */
    double p2;	    /* z if rzw search, r_freqmod if bin search      */
    double p3;	    /* w if rzw search, numfftbins if bin search     */
} position_gpu;

typedef struct orbitparams_gpu {
    double p;	    /* Orbital period (s)                            */
    double e;	    /* Orbital eccentricity                          */
    double x;	    /* Projected semi-major axis (lt-sec)            */
    double w;	    /* Longitude of periapsis (deg)                  */
    double t;	    /* Time since last periastron passage (s)        */
    double pd;	    /* Orbital period derivative (s/yr)              */
    double wd;	    /* Advance of longitude of periapsis (deg/yr)    */
} orbitparams_gpu;

typedef struct PREPFOLDINFO_GPU {
  double *rawfolds;   /* Raw folds (nsub * npart * proflen points) */
  double *dms;        /* DMs used in the trials */
  double *periods;    /* Periods used in the trials */
  double *pdots;      /* P-dots used in the trials */
  foldstats_gpu *stats;   /* Statistics for the raw folds */
  int numdms;         /* Number of 'dms' */
  int numperiods;     /* Number of 'periods' */
  int numpdots;       /* Number of 'pdots' */
  int nsub;           /* Number of frequency subbands folded */
  int npart;          /* Number of folds in time over integration */
  int proflen;        /* Number of bins per profile */
  int numchan;        /* Number of channels for radio data */
  int pstep;          /* Minimum period stepsize in profile phase bins */
  int pdstep;         /* Minimum p-dot stepsize in profile phase bins */
  int dmstep;         /* Minimum DM stepsize in profile phase bins */
  int ndmfact;        /* 2*ndmfact*proflen+1 DMs to search */
  int npfact;         /* 2*npfact*proflen+1 periods and p-dots to search */
  char *filenm;       /* Filename of the folded data */
  char *candnm;       /* String describing the candidate */
  char *telescope;    /* Telescope where observation took place */
  char *pgdev;        /* PGPLOT device to use */
  char rastr[16];     /* J2000 RA  string in format hh:mm:ss.ssss */
  char decstr[16];    /* J2000 DEC string in format dd:mm:ss.ssss */
  double dt;          /* Sampling interval of the data */
  double startT;      /* Fraction of observation file to start folding */
  double endT;        /* Fraction of observation file to stop folding */
  double tepoch;      /* Topocentric eopch of data in MJD */
  double bepoch;      /* Barycentric eopch of data in MJD */
  double avgvoverc;   /* Average topocentric velocity */
  double lofreq;      /* Center of low frequency radio channel */
  double chan_wid;    /* Width of each radio channel in MHz */
  double bestdm;      /* Best DM */
  position_gpu topo;      /* Best topocentric p, pd, and pdd */
  position_gpu bary;      /* Best barycentric p, pd, and pdd */
  position_gpu fold;      /* f, fd, and fdd used to fold the initial data */
  orbitparams_gpu orb;    /* Barycentric orbital parameters used in folds */
} prepfoldinfo_gpu;




void select_cuda_dev(int cuda_inds);
void Endup_GPU();

unsigned int calc_next_32_times(unsigned int x);
// int configure_reduce_sum(dim3 &grid_dim, dim3 &block_dim, unsigned int next_pow_32_times);


float get_mean_gpu(float *indata, int N);
double get_mean_gpu_d(double *indata, int N);
float get_variance_gpu(float *indata, float mean,int N);
void get_maxindex_double_gpu(double *indata, int N, double *maxV, int *maxindex);

void sum_value_gpu(float *indata, float value,int N);
void sum_value_gpu_stream(float *indata, double *mean,int N, int M, cudaStream_t stream_1, cudaStream_t stream_2);
__global__ void Do_sum_value_gpu(float *indata, float value,int N);

float gpu_sum_reduce_float(float *arr, int total_num);
__global__ void block_sum_reduce_float(float *g_idata, float *g_odata, unsigned int n, int blockSize);

float gpu_varience_reduce_float(float *arr, float mean, int total_num);
__global__ void block_varience_reduce_float(float *g_idata, double *g_odata, float mean, unsigned int n, int blockSize) ;

void mask_data_GPU(float *currentdata_gpu, int *maskchans_gpu, float *padvals_gpu, int spectra_per_subint, int num_channels, int nummasked);
__global__ void Do_mask_data_GPU(float *currentdata_gpu, int *maskchans_gpu, float *padvals_gpu, int spectra_per_subint, int num_channels, int nummasked);

void ignorechans_GPU(float *currentdata_gpu, int *ignorechans_gpu, int spectra_per_subint, int num_channels, int num_ignorechans);
__global__ void DO_ignorechans_GPU(float *currentdata_gpu, int *ignorechans_gpu, int spectra_per_subint, int num_channels, int num_ignorechans);

void dedisp_subbands_GPU(float *lastdata_gpu,float *lastdata,
                     int numpts, int numchan, 
                     int *delays, int numsubbands, float *result, int transpose);
void dedisp_subbands_GPU_cache(unsigned char *data_gpu, float *data_gpu_scl, float *data_gpu_offs, unsigned char *lastdata_gpu, float *lastdata_gpu_scl, float *lastdata_gpu_offs,
                     int numpts, int numchan, 
                     int *delays, int numsubbands, float *result, int transpose);
__global__ void Do_dedisp_subbands_GPU(float *data, float *lastdata,
                     int numpts, int numchan, 
                     int *delays, int numsubbands, float *result, int transpose);
__global__ void Do_dedisp_subbands_GPU_cache(unsigned char *data, float *data_scl, float *data_offs, unsigned char *lastdata, float *lastdata_scl, float *lastdata_offs,
                     int numpts, int numchan, 
                     int *delays, int numsubbands, float *result, int transpose);
void downsamp_GPU(float *indata, float *outdata, int numchan, int numpts, int down, int transpose);
__global__ void Do_downsamp_GPU(float *indata, float *outdata, int numchan, int numpts, int down, int transpose);

void float_dedisp_GPU(float *data, float *lastdata,
                  int numpts, int numchan,
                  int *delays, float approx_mean, float *result, int numdms, int transpose);
__global__ void Do_float_dedisp_GPU(float *data, float *lastdata,
                  int numpts, int numchan,
                  int *delays, float approx_mean, float *result, int numdms, int transpose);

void Get_subsdata(float *indata, short *outdata, int nsub, int worklen);
__global__ void Do_Get_subsdata(float *indata, short *outdata, int nsub, int worklen);

void SetSubData4Mask_GPU(float *subbanddata, float *padvals, int *maskchans,int len, int Width);
__global__ void Do_SetSubData4Mask_GPU(float *subbanddata, float *padvals, int *maskchans, int len, int Width);

void ZeroDM_subchan_GPU(float *indata, int len, int Width);
__global__ void Add_data_GPU(float *indata, float value, int len);

int iDivUp(int a, int b);

void transpose_GPU(float *d_src, float *d_dest, int width, int height);
void transpose_GPU_short(short *d_src, float *d_dest, int width, int height);
__global__ void d_transpose(float *odata, float *idata, int width, int height) ;
__global__ void d_transpose_short(float *odata, short *idata, int width, int height);

void FFTonGPU(cufftComplex *GPUTimeDMArray,cufftComplex *GPUTimeDMArrayFFTbk, int x,int y, int index, cufftHandle plan);

void get_power_GPU(cufftComplex * data, int numdata, float *power);
__global__ void Do_get_power_GPU(cufftComplex * data, int numdata, float *power);


float  get_med_gpu(float *data, int N);


void spread_no_pad_gpu(cufftComplex * data, int numdata, cufftComplex * result, int numresult, int numbetween, double norm);
__global__ void Do_spread_with_pad_GPU(cufftComplex * data, int numdata, cufftComplex * result, int numresult, int numbetween, int numpad, double norm);

void spread_no_pad_gpu_list(cufftComplex * data, int numdata, cufftComplex * result, int numresult, int numbetween, int readdatanum, double *norm_data_gpu);
__global__ void Do_spread_with_pad_GPU_list(cufftComplex * data, int numdata, cufftComplex * result, int numresult, int numbetween, int numpad, int readdatanum, double *norm_data_gpu);
                   
void loops_in_GPU_1(cufftComplex *fpdata, cufftComplex *fkern, cufftComplex *outdata, int fftlen, int numzs);
void loops_in_GPU_1_list(cufftComplex *fpdata, cufftComplex *fkern, cufftComplex *outdata, int fftlen, int numzs, int readdatanum);
static __global__ void Do_loops_in_GPU_1(cufftComplex *fpdata, cufftComplex *fkern, cufftComplex *outdata, int fftlen, int numzs);
static __global__ void Do_loops_in_GPU_1_list(cufftComplex *fpdata, cufftComplex *fkern, cufftComplex *outdata, int fftlen, int numzs, int readdatanum);


void loops_in_GPU_2(cufftComplex *fdata,  float *outpows, int numrs, int numzs, int offset, int fftlen, float norm, float *outpows_obs, long long rlen, long long rlo, int tip);
void loops_in_GPU_2_list(cufftComplex *fdata,  float *outpows, int numrs, int numzs, int offset, int fftlen, float norm, float *outpows_obs, long long rlen, long long rlo, int tip, int readdatanum, int outpows_gpu_xlen, int outpows_gpu_obs_xlen);
static __global__ void Do_loops_in_GPU_2(cufftComplex *fdata,  float *outpows, int numrs, int numzs, int offset, int fftlen, float norm, float *outpows_obs, long long rlen, long long rlo, int tip);
static __global__ void Do_loops_in_GPU_2_list(cufftComplex *fdata,  float *outpows, int numrs, int numzs, int offset, int fftlen, float norm, float *outpows_obs, long long rlen, long long rlo, int tip, int readdatanum, int outpows_gpu_xlen, int outpows_gpu_obs_xlen);

void add_subharm_gpu(float *powers_out, cufftComplex *fdata, unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, int fftlen, int numzs,int offset, float norm);
void add_subharm_gpu_list(float *powers_out, cufftComplex *fdata, unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, int fftlen, int numzs,int offset, float norm, int readdatanum, int outpows_gpu_xlen);
static __global__ void Do_add_subharm_gpu(float *powers_out, cufftComplex *fdata,unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, int fftlen, int offset, float norm);
static __global__ void Do_add_subharm_gpu_list(float *powers_out, cufftComplex *fdata,unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, int fftlen, int numzs, int offset, float norm, int readdatanum, int outpows_gpu_xlen);


void inmem_add_ffdotpows_gpu_gpu(float *fdp, float *powptr, int *rinds, int zlo, int numrs, int numzs, int stage, long long rlen);
static __global__ void Do_inmem_add_ffdotpows_gpu_gpu(float *fdp, float *powptr, int *rinds,int zlo, int numrs, int numzs, int stage, long long rlen);

void inmem_add_ffdotpows_gpu_gpu_list(float *fdp, float *powptr, int *rinds, int zlo, int numrs, int numzs, int stage, long long rlen, int outpows_gpu_xlen, int readdatanum, int outpows_gpu_obs_xlen);
static  __global__ void Do_inmem_add_ffdotpows_gpu_gpu_list(float *fdp, float *powptr, int *rinds,int zlo, int numrs, int numzs, int stage, long long rlen, int outpows_gpu_xlen, int readdatanum, int outpows_gpu_obs_xlen);

int  search_ffdotpows_gpu(float powcut, float *d_fundamental, accel_cand_gpu * cand_array_search_gpu, int numzs, int numrs, accel_cand_gpu *cand_gpu_cpu);
void  search_ffdotpows_gpu_list(float powcut, float *d_fundamental, accel_cand_gpu * cand_array_search_gpu, int numzs, int numrs, accel_cand_gpu *cand_gpu_cpu, int readdatanum, int *nof_cand, int output_x_max, int d_fundamental_xlen);
static __global__ void  search_ffdotpows_kernel(float powcut, float *d_fundamental, accel_cand_gpu * cand_array_search_gpu, int numzs, int numrs, int *d_addr);
static __global__ void  search_ffdotpows_kernel_list(float powcut, float *d_fundamental, accel_cand_gpu * cand_array_search_gpu, int numzs, int numrs, int *d_addr, int readdatanum, int put_x_max, int d_fundamental_xlen);

void hunt_CPU(double *xx, int n, double x, int *jlo);
void hunt_GPU(double *xx, int n, double x, int *jlo);
double fold_gpu_cu(float *data, int nsub, int numdata, double dt, double tlo,
            double *prof, int numprof, double startphs,
            double *buffer, double *phaseadded,
            double fo, double fdot, double fdotdot, int flags,
            double *delays, double *delaytimes, int numdelays,
            int *onoffpairs, foldstats_gpu * stats, int standard, int ONOFF, int DELAYS, int worklen);


void fold_2_gpu(float *data, int nsub, int numdata, double dt, double tlo,
            double *prof, int numprof, double startphs,
            double *buffer, double *phaseadded,
            double fo, double fdot, double fdotdot, int flags,
            double *delays, double *delaytimes, int numdelays,
            int *onoffpairs, foldstats_gpu * stats, int standard, int worklen);


void Get_dmdelays_subband_int_gpu(double *search_dms_gpu, int *dmdelays, int numdmtrials, 
 int search_nsub,  double search_fold_p1, int search_proflen, double search_lofreq, 
 int search_numchan, double search_chan_wid, double search_avgvoverc);
__global__ void Get_dmdelays_subband_int_gpu_Do(double *search_dms_gpu, int *dmdelays, int numdmtrials, 
 int search_nsub,  double search_fold_p1, int search_proflen, double search_lofreq, 
 int search_numchan, double search_chan_wid, double search_avgvoverc);

void combine_subbands_1_gpu(double *inprofs,
                      int numparts, int numsubbands, int proflen, int numdmtrials,
                      int *delays, double *outprofs);
__global__ void combine_subbands_1_gpu_Do(double *inprofs,
                      int numparts, int numsubbands, int proflen, int numdmtrials,
                      int *delays, double *outprofs);

void get_delays_gpu(double *delays_gpu, const int numpdds, const int numpds, const int  numps, const int npart, const int numtrials, const int proflen, double *fdotdots, double *fdots, const int good_ipd, const int good_ip, const int searchpddP, const int search_pstep, const long reads_per_part, const double proftime);
__global__ void get_delays_gpu_Do(double *delays_gpu, const int numpdds, const int numpds, const int  numps, const int npart, const int numtrials, const int proflen, double *fdotdots, double *fdots, const int good_ipd, const int good_ip, const int searchpddP, const int search_pstep, const long reads_per_part, const double proftime);

void combine_profs_1_gpu(double *profs, double *delays, int numprofs, int proflen, int numpdd, int numpd, int nump, double outstats_prof_avg, double outstats_prof_var, double *currentstats_redchi);
__global__ void combine_profs_1_gpu_Do(double *profs, double *delays, int numprofs, int proflen, int numpdd, int numpd, int nump, double outstats_prof_avg, double outstats_prof_var, double *currentstats_redchi);



void combine_profs_2_gpu(double *ddprofs_gpu, double *search_pdots_gpu, double *search_periods_gpu, float *parttimes_gpu, int numpdots, int numperiods, int npart, int search_proflen, double outstats_prof_avg, double outstats_prof_var, float *currentstats_redchi, double pfold, double search_fold_p2, double search_fold_p1, double *pdd_delays_gpu, float chifact);
__global__ void combine_profs_2_gpu_Do(double *ddprofs_gpu, double *search_pdots_gpu, double *search_periods_gpu, float *parttimes_gpu, int numpdots, int numperiods, int npart, int search_proflen, double outstats_prof_avg, double outstats_prof_var, float *currentstats_redchi, double pfold, double search_fold_p2, double search_fold_p1, double *pdd_delays_gpu, float chifact);

__device__ double switch_pfdot_device(double pf, double pfdot);
__device__ double fdot2phasedelay_gpu(double fdot, double time);
__device__ float fdot2phasedelay_gpu_float(float fdot, float time);
__device__ double calculate_delays(double pf, double pfdot, double search_fold_p2, double df, double time);

void get_redchi_gpu(double *currentstats_redchi, double *outprof,  double outstats_prof_avg, double outstats_prof_var, int proflen, int parts);
__global__ void get_redchi_gpu_Do(double *currentstats_redchi, double *outprof,  double outstats_prof_avg, double outstats_prof_var, int proflen, int parts);
       

void Set_cufftComplex_date_as_zero_gpu(fcomplex *data, long long num);
__global__ void Set_cufftComplex_date_as_zero_gpu_Do(fcomplex *data, long long num);

void compute_power_gpu(fcomplex *data, float *powers, int numdata);
__global__ void compute_power_kernel(fcomplex *data, float *powers, int numdata);

void sort_and_get_median_gpu(float *data, int numdata, float *median);

#ifdef __cplusplus
}
#endif

