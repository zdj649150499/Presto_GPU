#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cufft.h>


#ifdef __cplusplus
extern "C" {
#endif

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

void select_cuda_dev(int cuda_inds);
void Endup_GPU();

unsigned int calc_next_32_times(unsigned int x);
// int configure_reduce_sum(dim3 &grid_dim, dim3 &block_dim, unsigned int next_pow_32_times);

float get_mean_gpu(float *indata, int N);
double get_mean_gpu_d(float *indata, int N);
float get_variance_gpu(float *indata, float mean,int N);

void sum_value_gpu(float *indata, float value,int N);
__global__ void Do_sum_value_gpu(float *indata, float value,int N);

float gpu_sum_reduce_float(float *arr, int total_num);
__global__ void block_sum_reduce_float(float *g_idata, float *g_odata, unsigned int n, int blockSize);

float gpu_varience_reduce_float(float *arr, float mean, int total_num);
__global__ void block_varience_reduce_float(float *g_idata, double *g_odata, float mean, unsigned int n, int blockSize) ;


void dedisp_subbands_GPU(float *data, float *lastdata,
                     int numpts, int numchan, 
                     int *delays, int numsubbands, float *result, int transpose);

__global__ void Do_dedisp_subbands_GPU(float *data, float *lastdata,
                     int numpts, int numchan, 
                     int *delays, int numsubbands, float *result, int transpose);

void downsamp_GPU(float *indata, float *outdata, int numchan, int numpts, int down);
__global__ void Do_downsamp_GPU(float *indata, float *outdata, int numchan, int numpts, int down);

void float_dedisp_GPU(float *data, float *lastdata,
                  int numpts, int numchan,
                  int *delays, float approx_mean, float *result, int numdms);
__global__ void Do_float_dedisp_GPU(float *data, float *lastdata,
                  int numpts, int numchan,
                  int *delays, float approx_mean, float *result, int numdms);

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

void spread_no_pad_gpu(cufftComplex * data, int numdata,
                   cufftComplex * result, int numresult, int numbetween, double norm);
__global__ void Do_spread_with_pad_GPU(cufftComplex * data, int numdata,
                     cufftComplex * result, int numresult, int numbetween, int numpad, double norm);


void loops_in_GPU_1(cufftComplex *fpdata, cufftComplex *fkern, cufftComplex *outdata, int fftlen, int numzs);
static __global__ void Do_loops_in_GPU_1(cufftComplex *fpdata, cufftComplex *fkern, cufftComplex *outdata, int fftlen, int numzs);



void loops_in_GPU_2(cufftComplex *fdata,  float *outpows, int numrs, int numzs, int offset, int fftlen, float norm, float *outpows_obs, long long rlen, long long rlo, int tip);
static __global__ void Do_loops_in_GPU_2(cufftComplex *fdata,  float *outpows, int numrs, int numzs, int offset, int fftlen, float norm, float *outpows_obs, long long rlen, long long rlo, int tip);

void add_subharm_gpu(float *powers_out, cufftComplex *fdata, unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, int fftlen, int numzs,int offset, float norm);
static __global__ void Do_add_subharm_gpu(float *powers_out, cufftComplex *fdata,unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, int fftlen, int offset, float norm);


void inmem_add_ffdotpows_gpu_gpu(float *fdp, float *powptr, int *rinds, int zlo, int numrs, int numzs, int stage, long long rlen);
static __global__ void Do_inmem_add_ffdotpows_gpu_gpu(float *fdp, float *powptr, int *rinds,int zlo, int numrs, int numzs, int stage, long long rlen);

int  search_ffdotpows_gpu(float powcut, float *d_fundamental, accel_cand_gpu * cand_array_search_gpu, accel_cand_gpu * cand_array_sort_gpu, int numzs, int numrs, accel_cand_gpu *cand_gpu_cpu);
static __global__ void  search_ffdotpows_kernel(float powcut, float *d_fundamental, accel_cand_gpu * cand_array_search_gpu, int numzs, int numrs, int *d_addr);

#ifdef __cplusplus
}
#endif

