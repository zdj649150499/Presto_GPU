#include "cuda.cuh"
// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>


#define MAX_BLOCK_SZ 512
#define BLOCK_DIM 16
#define WARP_SIZE 32
#define MAX_THREAD 1024
#define NEAREST_INT(x) (int) (x<0 ? x-0.5 : x+0.5)
#define ACCEL_RDZ 0.5
#define DBLCORRECT    1e-14

typedef float2 Complex;

//define a texture memory
texture<cufftComplex> tex_d_kernel;
texture<cufftComplex> tex_d_fpdata;
texture<cufftComplex> tex_d_fdata;
texture<unsigned short> tex_d_zinds;
texture<unsigned short> tex_d_rinds;
texture<float> tex_d_fundamental;

typedef struct kernel{
    int z;               /* The fourier f-dot of the kernel */
    int w;               /* The fourier f-dot-dot of the kernel */
    int fftlen;          /* Number of complex points in the kernel */
    int numgoodbins;     /* The number of good points you can get back */
    int numbetween;      /* Fourier freq resolution (2=interbin) */
    int kern_half_width; /* Half width (bins) of the raw kernel. */
    fcomplex *data;      /* The FFTd kernel itself */
} kernel;

typedef struct subharminfo{
    int numharm;       /* The number of sub-harmonics */
    int harmnum;       /* The sub-harmonic number (fundamental = numharm) */
    int zmax;          /* The maximum Fourier f-dot for this harmonic */
    int wmax;          /* The maximum Fourier f-dot-dot for this harmonic */
    int numkern_zdim;  /* Number of kernels calculated in the z dimension */
    int numkern_wdim;  /* Number of kernels calculated in the w dimension */
    int numkern;       /* Total number of kernels in the vector == numzs */
    kernel **kern;     /* A 2D array of the kernels themselves, with dimensions of z and w */
    unsigned short *rinds; /* Table of lookup indices for Fourier Freqs: subharmonic r values corresponding to "fundamental" r values */
    unsigned short *zinds; /* Table of lookup indices for Fourier F-dots */
} subharminfo;

// Round a / b to nearest higher integer value
int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }



void sort_arrat_thrust(float *indata, int N)
{
    thrust::device_ptr<float> indata_prt(indata);
    thrust::sort(indata_prt, indata_prt+N);
}


float get_mean_gpu(float *indata, int N)
{
    thrust::device_ptr<float> indata_bk(indata);
    double sum = thrust::reduce(indata_bk, indata_bk+N);
    return sum/N;
}

double get_mean_gpu_d(float *indata, int N)
{
    thrust::device_ptr<float> indata_bk(indata);
    double sum = thrust::reduce(indata_bk, indata_bk+N);
    return sum/N;
}

struct variance: std::unary_function<float, float>
{
    variance(float m): mean(m){ }
    const float mean;
    __host__ __device__ float operator()(float data) const
    {
        return (data - mean)*(data - mean);
    }
};


float get_variance_gpu(float *indata, float mean,int N)
{
    thrust::device_ptr<float> indata_bk(indata);
    float variance_bk = thrust::transform_reduce(indata_bk, indata_bk+N, variance(mean), 0.0f, thrust::plus<float>())/N;
    return variance_bk;
}


void sum_value_gpu(float *indata, float value,int N)
{
    int BlkPerRow=(N-1+512)/512;
    Do_sum_value_gpu<<< BlkPerRow, 512>>>(indata, value, N);
}

__global__ void Do_sum_value_gpu(float *indata, float value,int N)
{
    const int MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=N) return;

    indata[MYgtid] += value;
}

//----------------------select a cpu to play with --------------------
void select_cuda_dev(int cuda_inds)
{
	cudaSetDevice(cuda_inds);
    // cudaDeviceProp deviceProp;
    // cudaGetDeviceProperties(&deviceProp, cuda_inds);
    // printf("\nGPU Device %d: \"%s\" with Capability: %d.%d\n", cuda_inds, deviceProp.name, deviceProp.major, deviceProp.minor);
	cudaDeviceReset();
}


void Endup_GPU()
{
    cudaDeviceSynchronize();
    cudaDeviceReset();
}


__device__ void warpReduce(volatile float *sdata, unsigned int tid, int blockSize)
{
    if (blockSize >= 64) {sdata[tid] += sdata[tid + 32];}
    if (blockSize >= 32) {sdata[tid] += sdata[tid + 16];}
    if (blockSize >= 16) {sdata[tid] += sdata[tid + 8];}
    if (blockSize >= 8) {sdata[tid] += sdata[tid + 4];}
    if (blockSize >= 4) {sdata[tid] += sdata[tid + 2];}
    if (blockSize >= 2) {sdata[tid] += sdata[tid + 1];}
}

__device__ void warpReduce_double(volatile double *sdata, unsigned int tid, int blockSize)
{
    if (blockSize >= 64) {sdata[tid] += sdata[tid + 32];}
    if (blockSize >= 32) {sdata[tid] += sdata[tid + 16];}
    if (blockSize >= 16) {sdata[tid] += sdata[tid + 8];}
    if (blockSize >= 8) {sdata[tid] += sdata[tid + 4];}
    if (blockSize >= 4) {sdata[tid] += sdata[tid + 2];}
    if (blockSize >= 2) {sdata[tid] += sdata[tid + 1];}
}


unsigned int calc_next_32_times(unsigned int x)
{
    unsigned int next_pow_32_times = WARP_SIZE;
    for(int i=2; next_pow_32_times < x; i++)
        next_pow_32_times = WARP_SIZE*i;
    return next_pow_32_times;
}


int configure_reduce_sum(dim3 &grid_dim, dim3 &block_dim, unsigned int next_pow_32_times)
{
    if(next_pow_32_times < MAX_THREAD)
    {
        block_dim = dim3(next_pow_32_times, 1, 1);
        grid_dim = dim3(1,1,1);
    }
    else
    {
        int grid_x = ceil((double) next_pow_32_times / MAX_THREAD);
        block_dim = dim3(MAX_THREAD, 1, 1);
        grid_dim = dim3(grid_x,1,1);
    }
    return 0;
}



float gpu_sum_reduce_float(float *arr, int total_num)
{
    float  *d_out_arr, *h_out_arr; //*d_arr,
    int next_pow_32_times = calc_next_32_times(total_num);

    // cudaMalloc(&d_arr, next_pow_32_times * sizeof(float));
    // cudaMemset(d_arr, 0, next_pow_32_times * sizeof(float));
    
    dim3 block_dim, grid_dim;
    configure_reduce_sum(grid_dim, block_dim, next_pow_32_times);
    
    int block_num = grid_dim.x * grid_dim.y * grid_dim.z;
   
    cudaMalloc(&d_out_arr, block_num * sizeof(float));
    cudaMemset(d_out_arr, 0, block_num * sizeof(float));

    // cudaMemcpy(d_arr, arr, sizeof(float) * total_num, cudaMemcpyHostToDevice);
    
    // reduce6<<<grid_dim, block_dim, block_dim.x*sizeof(int)>>>(d_arr, d_out_arr, total_num, block_dim.x);
    block_sum_reduce_float<<<grid_dim, block_dim, block_dim.x*sizeof(float)>>>(arr, d_out_arr, total_num, block_dim.x);
    cudaDeviceSynchronize();

    h_out_arr = new float[block_num];
    memset(h_out_arr, 0, sizeof(float)*block_num);
    cudaMemcpy(h_out_arr, d_out_arr, sizeof(float) * block_num, cudaMemcpyDeviceToHost);

    float ret=0;
    for(int i=0; i<block_num; i++)
    {
        ret += h_out_arr[i];
    }

    // cudaFree(d_arr);
    cudaFree(d_out_arr);
    delete []h_out_arr;

    return ret;

}


__global__ void block_sum_reduce_float(float *g_idata, float *g_odata, unsigned int n, int blockSize) 
{
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid; 
    unsigned int gridSize = blockSize*2*gridDim.x; //suit for the grid_dim.y > 1

    sdata[tid] = 0;//set the shared memory to zero is needed
    while (i < n) 
    { 
        sdata[tid] += g_idata[i];
        if ((i+blockSize) < n)
            sdata[tid] += g_idata[i+blockSize];
        
        i += gridSize; 
    }
     __syncthreads();
    if (blockSize >= 1024) 
    { 
        if (tid < 512)
        { 
            sdata[tid] += sdata[tid + 512]; 
        }
        __syncthreads(); 
    } 
    if (blockSize >= 512) 
    { 
        if (tid < 256)
        { 
            sdata[tid] += sdata[tid + 256]; 
        }
        __syncthreads(); 
    } 
    if (blockSize >= 256) 
    { 
        if (tid < 128) 
        { 
            sdata[tid] += sdata[tid + 128]; 
        }
        __syncthreads(); 
    } 
    if (blockSize >= 128)
    { 
        if (tid < 64)
        { 
            sdata[tid] += sdata[tid + 64]; 
        } 
        __syncthreads(); 
    }
    if (tid < 32) warpReduce(sdata, tid, blockSize);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0]; 
}

float gpu_varience_reduce_float(float *arr, float mean, int total_num)
{
    double  *d_out_arr, *h_out_arr; //*d_arr,
    int next_pow_32_times = calc_next_32_times(total_num);

    // cudaMalloc(&d_arr, next_pow_32_times * sizeof(float));
    // cudaMemset(d_arr, 0, next_pow_32_times * sizeof(float));
    
    dim3 block_dim, grid_dim;
    configure_reduce_sum(grid_dim, block_dim, next_pow_32_times);
    
    int block_num = grid_dim.x * grid_dim.y * grid_dim.z;
   
    cudaMalloc(&d_out_arr, block_num * sizeof(double));
    cudaMemset(d_out_arr, 0, block_num * sizeof(double));

    // cudaMemcpy(d_arr, arr, sizeof(float) * total_num, cudaMemcpyHostToDevice);
    
    // reduce6<<<grid_dim, block_dim, block_dim.x*sizeof(int)>>>(d_arr, d_out_arr, total_num, block_dim.x);
    block_varience_reduce_float<<<grid_dim, block_dim, block_dim.x*sizeof(double)>>>(arr, d_out_arr, mean, total_num, block_dim.x);
    cudaDeviceSynchronize();

    h_out_arr = new double[block_num];
    memset(h_out_arr, 0, sizeof(double)*block_num);
    cudaMemcpy(h_out_arr, d_out_arr, sizeof(double) * block_num, cudaMemcpyDeviceToHost);

    double ret=0;
    for(int i=0; i<block_num; i++)
    {
        ret += h_out_arr[i];
    }
    // cudaFree(d_arr);
    cudaFree(d_out_arr);
    delete []h_out_arr;
    return (float)ret;
}


__global__ void block_varience_reduce_float(float *g_idata, double *g_odata, float mean, unsigned int n, int blockSize) 
{
    extern __shared__ double ddata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid; 
    unsigned int gridSize = blockSize*2*gridDim.x; //suit for the grid_dim.y > 1

    double bk;
    ddata[tid] = 0;//set the shared memory to zero is needed
    while (i < n) 
    { 
        bk=g_idata[i]-mean;
        ddata[tid] += bk*bk;
        if ((i+blockSize) < n)
        {
            bk=g_idata[i+blockSize]-mean;
            ddata[tid] += bk*bk;
        }
            
        
        i += gridSize; 
    }
     __syncthreads();

    if (blockSize >= 1024) 
    { 
        if (tid < 512)
        { 
            ddata[tid] += ddata[tid + 512]; 
        }
        __syncthreads(); 
    } 
    if (blockSize >= 512) 
    { 
        if (tid < 256)
        { 
            ddata[tid] += ddata[tid + 256]; 
        }
        __syncthreads(); 
    } 
    if (blockSize >= 256) 
    { 
        if (tid < 128) 
        { 
            ddata[tid] += ddata[tid + 128]; 
        }
        __syncthreads(); 
    } 
    if (blockSize >= 128)
    { 
        if (tid < 64)
        { 
            ddata[tid] += ddata[tid + 64]; 
        } 
        __syncthreads(); 
    }
    if (tid < 32) warpReduce_double(ddata, tid, blockSize);
    if (tid == 0) g_odata[blockIdx.x] = ddata[0]; 
}


void dedisp_subbands_GPU(float *data, float *lastdata,
                     int numpts, int numchan, 
                     int *delays, int numsubbands, float *result, int transpose)
{
    float *data_gpu, *lastdata_gpu;

    cudaMalloc((void**)&data_gpu, sizeof(float)*numpts * numchan);
    cudaMalloc((void**)&lastdata_gpu, sizeof(float)*numpts * numchan);

    cudaMemcpy(data_gpu, data, sizeof(float)*numpts * numchan, cudaMemcpyHostToDevice);
    cudaMemcpy(lastdata_gpu, lastdata, sizeof(float)*numpts * numchan, cudaMemcpyHostToDevice);
    

    int BlkPerRow=(numpts-1+512)/512;
    dim3 dimGrid2D(BlkPerRow,numsubbands);
    Do_dedisp_subbands_GPU<<<dimGrid2D,512>>>(data_gpu, lastdata_gpu, numpts, numchan, delays, numsubbands, result, transpose);
    int cudaStatus = cudaDeviceSynchronize();
    cudaFree(data_gpu);
    cudaFree(lastdata_gpu);
}

__global__ void Do_dedisp_subbands_GPU(float *data, float *lastdata,
                     int numpts, int numchan, 
                     int *delays, int numsubbands, float *result, int transpose)
{
    int i;
    const int MYrow=blockIdx.y;
    const int MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=numpts) return;

    float temp = 0.0f;
    int chan_per_subband = numchan / numsubbands;
    int  in_index, out_index;
    if(transpose)
        out_index = MYrow*numpts + MYgtid;      // transpose=1 for time first
    else
        out_index = MYrow + MYgtid*numsubbands;  // transpose=0 for freq first
    int blk = numpts*numchan;
    for(i=0;i<chan_per_subband; i++)
    {
        int  in_chan = i + MYrow*chan_per_subband;
        int  dind = delays[in_chan];
        in_index = (dind+MYgtid)*numchan + in_chan;
        if(MYgtid < numpts-dind)
        {
            temp += lastdata[in_index];
        }
        else
        {
            in_index -= blk;
            temp += data[in_index];
        }
    }
    result[out_index] = temp;
}

void downsamp_GPU(float *indata, float *outdata, int numchan, int numpts, int down)
{
    int BlkPerRow=(numpts-1+512)/512;
    dim3 dimGrid2D(BlkPerRow,numchan);
    Do_downsamp_GPU<<<dimGrid2D, 512>>>(indata, outdata, numchan, numpts, down);
    int cudaStatus = cudaDeviceSynchronize();
}

__global__ void Do_downsamp_GPU(float *indata, float *outdata, int numchan, int numpts, int down)
{
    int i;
    const int MYrow=blockIdx.y;
    const int MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=numpts) return;

    float ftmp = 0.0f;
    const int out_index = MYrow + MYgtid*numchan;
    int  in_dex;

    /* input freq first */
    
    in_dex = MYgtid*numchan*down + MYrow;
    for(i=0;i<down;i++)
    {
        ftmp+=indata[in_dex];
        in_dex+=numchan;
    }
    outdata[out_index] = ftmp /(1.0f*down);
}

void float_dedisp_GPU(float *data, float *lastdata,
                  int numpts, int numchan,
                  int *delays, float approx_mean, float *result, int numdms)
{

    int BlkPerRow=(numpts-1+512)/512;
    dim3 dimGrid2D(BlkPerRow,numdms);
    Do_float_dedisp_GPU<<< dimGrid2D,512 >>>(data, lastdata, numpts, numchan,
                  delays, approx_mean, result, numdms);
    int cudaStatus = cudaDeviceSynchronize();
}


__global__ void Do_float_dedisp_GPU(float *data, float *lastdata,
                  int numpts, int numchan,
                  int *delays, float approx_mean, float *result, int numdms)
{
    int i;
    const int MYrow=blockIdx.y;
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=numpts) return;

    
    int in_dex;
    const int out_index = MYrow*numpts + MYgtid;
    const int  x=MYrow*numchan;

    float ftmp = -approx_mean;

    for(i=0;i<numchan;i++)
    {

        in_dex = MYgtid+(long long)(delays[x+i]);
        if(in_dex<numpts)
        {
            in_dex=i+in_dex*numchan;
            ftmp += lastdata[in_dex];
        }
        else
        {
            in_dex -=numpts;
            in_dex=i+in_dex*numchan;
            ftmp += data[in_dex];
        }
    }
    result[out_index] = ftmp;
}


void Get_subsdata(float *indata, short *outdata, int nsub, int worklen)
{
    int BlkPerRow=(nsub*worklen-1+512)/512;
    Do_Get_subsdata<<< BlkPerRow,512 >>>(indata, outdata, nsub, worklen);
    int cudaStatus = cudaDeviceSynchronize();
}

__global__ void Do_Get_subsdata(float *indata, short *outdata, int nsub, int worklen)
{
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=worklen) return;

    int x,y;
    x = MYgtid%worklen;
    y = MYgtid/worklen;
    outdata[MYgtid] = (short)(indata[x*nsub+y]+0.5);
}


void SetSubData4Mask_GPU(float *subbanddata, float *padvals, int *maskchans,int len, int Width)
{
    int BlkPerRow=(len-1+512)/512;
    dim3 dimGrid2D(BlkPerRow,Width);
    Do_SetSubData4Mask_GPU<<< dimGrid2D,512 >>>(subbanddata, padvals, maskchans, len, Width);
    int cudaStatus = cudaDeviceSynchronize();
}

__global__ void Do_SetSubData4Mask_GPU(float *subbanddata, float *padvals, int *maskchans, int len, int Width)
{
    const int MYrow=blockIdx.y;
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=len) return;

    int channum = maskchans[MYrow];
    int in_dex = MYgtid*Width+channum;
    subbanddata[in_dex] = padvals[channum];
}


void ZeroDM_subchan_GPU(float *indata, int len, int Width)
{
    int ii;
    float ave;
    int BlkPerRow;
    int cudaStatus;

    for(ii=0; ii< len; ii++)
    {
        ave=gpu_sum_reduce_float(indata+ii*Width,Width);
        ave/=Width;
        BlkPerRow=(Width-1+512)/512;
        Add_data_GPU<<<BlkPerRow,512>>>(indata, -ave, Width);
        cudaStatus = cudaDeviceSynchronize();
    }

}

__global__ void Add_data_GPU(float *indata, float value, int len)
{
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=len) return;

    indata[MYgtid] +=  value;
}


void transpose_GPU(float *d_src, float *d_dest, int width, int height) 
{
  int BlkPerRow = (width*height-1+512)/512;

  BlkPerRow =(height-1+512)/512;
  d_transpose<<<BlkPerRow*width, 512>>>(d_dest, d_src, width, height);
  int cudaStatus =  cudaDeviceSynchronize();
}

__global__ void d_transpose(float *odata, float *idata, int width, int height) 
{
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=(width*height)) return;

    // int yIndex = MYgtid/width;
    // int xIndex = MYgtid-yIndex*width;
    // int index_in = yIndex + xIndex * height;
    // odata[MYgtid] = idata[index_in];

    int yIndex = MYgtid%height;
    int xIndex = MYgtid/height;
    int index_out = yIndex*width + xIndex;

    odata[index_out] = idata[MYgtid];
}




void transpose_GPU_short(short *d_src, float *d_dest, int width, int height) 
{
  int BlkPerRow =(height-1+512)/512;
  d_transpose_short<<<BlkPerRow*width, 512>>>(d_dest, d_src, width, height);

//   dim3 grid(iDivUp(width, BLOCK_DIM), iDivUp(height, BLOCK_DIM), 1);
//   dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
//   d_transpose<<<grid, threads>>>(d_dest, d_src, width, height);

  int cudaStatus =  cudaDeviceSynchronize();
}

__global__ void d_transpose_short(float *odata, short *idata, int width, int height) 
{
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=(width*height)) return;

    int yIndex = MYgtid/width;
    int xIndex = MYgtid-yIndex*width;

    int index_in = yIndex + xIndex * height;
	odata[MYgtid] = idata[index_in];
}

void FFTonGPU(cufftComplex *GPUTimeDMArray,cufftComplex *GPUTimeDMArrayFFTbk, int x,int y, int index, cufftHandle plan)
{
    /*方法一*/
    // cufftPlan1d(&plan, x, CUFFT_C2C, y);
    cufftExecC2C(plan, GPUTimeDMArray, GPUTimeDMArrayFFTbk,index);
    int cudaStatus = cudaDeviceSynchronize();
    // cufftDestroy(plan);
}

void get_power_GPU(cufftComplex * data, int numdata, float *power)
{
    int BlkPerRow = (numdata-1+32)/32;
    Do_get_power_GPU<<<BlkPerRow, 32>>>(data, numdata, power);
}

__global__ void Do_get_power_GPU(cufftComplex * data, int numdata, float *power)
{
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=numdata) return;

    cufftComplex data_bk = data[MYgtid];
    power[MYgtid] = data_bk.x* data_bk.x + data_bk.y*data_bk.y;
}

float  get_med_gpu(float *data, int N)
{
    sort_arrat_thrust(data, N);
    float med;
    cudaMemcpy(&med, data+N/2, sizeof(float), cudaMemcpyDeviceToHost);
    return med;
}

void spread_no_pad_gpu(cufftComplex * data, int numdata,
                   cufftComplex * result, int numresult, int numbetween, double norm)
{
    int BlkPerRow = (numresult-1+32)/32;
    Do_spread_with_pad_GPU<<<BlkPerRow, 32>>>(data, numdata, result, numresult, numbetween, 0,norm);
}

__global__ void Do_spread_with_pad_GPU(cufftComplex * data, int numdata,
                     cufftComplex * result, int numresult, int numbetween, int numpad, double norm)
{
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=numresult) return;


    int numtoplace;
    cufftComplex zeros = { 0.0, 0.0 };
    

    result[MYgtid] = zeros;
    numtoplace = (numresult - numpad) / numbetween;
    if (numtoplace > numdata)
        numtoplace = numdata;
    if(MYgtid>=numtoplace) return;
    
    cufftComplex data_bk = data[MYgtid];
    data_bk.x *= norm;
    data_bk.y *= norm;

    result[MYgtid*numbetween] = data_bk;
}



void loops_in_GPU_1(cufftComplex *fpdata, cufftComplex *fkern, cufftComplex *outdata, int fftlen, int numzs)
{
    // cudaBindTexture(NULL, tex_d_fpdata, fpdata, sizeof(cufftComplex) * fftlen);
    // cudaBindTexture(NULL, tex_d_kernel, fkern, sizeof(cufftComplex) * fftlen*numzs);

    int BlkPerRow = (fftlen*numzs-1+512)/512;
    Do_loops_in_GPU_1<<<BlkPerRow,512>>>(fpdata, fkern, outdata, fftlen, numzs);

    // cudaUnbindTexture(tex_d_fpdata);
    // cudaUnbindTexture(tex_d_kernel);
}

__global__ void Do_loops_in_GPU_1(cufftComplex *fpdata, cufftComplex *fkern, cufftComplex *outdata, int fftlen, int numzs)
{
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=fftlen*numzs) return;
    const int xnum = MYgtid%fftlen;

    // Complex dfp, dfk;
    // dfp = tex1Dfetch(tex_d_fpdata, xnum);
    // dfk = tex1Dfetch(tex_d_kernel, MYgtid);
    // outdata[MYgtid].x = dfp.x * dfk.x + dfp.y * dfk.y;
    // outdata[MYgtid].y = dfp.y * dfk.x - dfp.x * dfk.y;

    const float dr = fpdata[xnum].x, di = fpdata[xnum].y;
    const float kr = fkern[MYgtid].x, ki = fkern[MYgtid].y;
    outdata[MYgtid].x = dr * kr + di * ki;
    outdata[MYgtid].y = di * kr - dr * ki;
}

void loops_in_GPU_2(cufftComplex *fdata,  float *outpows, int numrs, int numzs, int offset, int fftlen, float norm, float *outpows_obs, long long rlen, long long rlo, int tip)
{
    // cudaBindTexture(NULL, tex_d_fdata, fdata, sizeof(cufftComplex) * fftlen*numzs);
    
    int BlkPerRow = (numrs*numzs-1+512)/512;
    Do_loops_in_GPU_2<<<BlkPerRow, 512>>>(fdata, outpows, numrs, numzs, offset, fftlen, norm, outpows_obs, rlen, rlo, tip);

    // cudaUnbindTexture(tex_d_fdata);
}

static __global__ void Do_loops_in_GPU_2(cufftComplex *fdata,  float *outpows, int numrs, int numzs, int offset, int fftlen, float norm, float *outpows_obs, long long rlen, long long rlo, int tip)
{
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=numrs*numzs) return;

    const int ynum = MYgtid/numrs;
    const int xnum = MYgtid - ynum*numrs;
    const int ind = xnum+offset + ynum*fftlen;

    // Complex fdata_b;
    // fdata_b = tex1Dfetch(tex_d_fdata, ind);
    // if(tip)
    //     outpows[MYgtid] = outpows_obs[ynum*rlen+xnum+rlo] = (fdata_b.x*fdata_b.x +  fdata_b.y*fdata_b.y) * norm;
    // else
    //     outpows[MYgtid] = (fdata_b.x*fdata_b.x + fdata_b.y*fdata_b.y) * norm;

    // float fdata1 = fdata[ind].x;
    // float fdata2 = fdata[ind].y;
    // if(tip)
    //     outpows[MYgtid] = outpows_obs[ynum*rlen+xnum+rlo] = (fdata1*fdata1 +  fdata2*fdata2) * norm;
    // else
    //     outpows[MYgtid] = (fdata1*fdata1 +  fdata2*fdata2) * norm;

    Complex fdata_b = fdata[ind];
    if(tip)
        outpows[MYgtid] = outpows_obs[ynum*rlen+xnum+rlo] = (fdata_b.x*fdata_b.x +  fdata_b.y*fdata_b.y) * norm;
    else
        outpows[MYgtid] = (fdata_b.x*fdata_b.x + fdata_b.y*fdata_b.y) * norm;
}

void add_subharm_gpu(float *powers_out, cufftComplex *fdata, unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, int fftlen, int numzs, int offset, float norm)
{
    // cudaBindTexture(NULL, tex_d_fdata, fdata, sizeof(cufftComplex) * fftlen*numzs);
    // cudaBindTexture(NULL, tex_d_zinds, zinds, sizeof(unsigned short) * numzs_0 );
    // cudaBindTexture(NULL, tex_d_rinds, rinds, sizeof(unsigned short) * numrs_0 );		


    int BlkPerRow = (numrs_0*numzs_0-1+512)/512;
    Do_add_subharm_gpu<<< BlkPerRow, 512 >>>(powers_out, fdata, rinds, zinds, numrs_0, numzs_0, fftlen, offset, norm);

    // cudaUnbindTexture(tex_d_fdata);
    // cudaUnbindTexture(tex_d_zinds);
    // cudaUnbindTexture(tex_d_rinds);
}

static __global__ void Do_add_subharm_gpu(float *powers_out, cufftComplex *fdata,unsigned short *rinds, unsigned short *zinds, int numrs_0, int numzs_0, int fftlen, int offset, float norm)
{
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=numrs_0*numzs_0) return;
    
    // int yy = MYgtid/numrs_0 ;
    // int addr_z = tex1Dfetch(tex_d_zinds, yy);
    // int xx = MYgtid -  yy * numrs_0 ;
    // int addr_r = tex1Dfetch(tex_d_rinds, xx);
    // int addr_result = addr_z * fftlen + offset + addr_r ;
    // Complex fdata_b;
    // fdata_b = tex1Dfetch(tex_d_fdata, addr_result);
    // powers_out[MYgtid] += ((fdata_b.x*fdata_b.x +  fdata_b.y*fdata_b.y)*norm);

    int yy = MYgtid/numrs_0 ;
    int addr_z = zinds[yy];
    int xx = MYgtid -  yy * numrs_0 ;
    int addr_r = rinds[xx];
    int addr_result = addr_z * fftlen + offset + addr_r ;

    // float fdata1 = fdata[addr_result].x;
    // float fdata2 = fdata[addr_result].y;
    // powers_out[MYgtid] += ((fdata1*fdata1 +  fdata2*fdata2)*norm);

    Complex fdata_b = fdata[addr_result];
    powers_out[MYgtid] += ((fdata_b.x*fdata_b.x +  fdata_b.y*fdata_b.y)*norm);
}


static __device__ __host__ inline int calc_required_z_gpu(double harm_fract, double zfull)
{
    return NEAREST_INT(0.5 * zfull * harm_fract) * 2;
}


static __device__ __host__ inline int index_from_z_gpu(double z, double loz)
{
    return (int) ((z - loz) * ACCEL_RDZ + DBLCORRECT);
}

void inmem_add_ffdotpows_gpu_gpu(float *fdp, float *powptr, int *rinds, int zlo, int numrs, int numzs, int stage, long long rlen)
{
    int BlkPerRow = (numrs*numzs-1+512)/512;
    Do_inmem_add_ffdotpows_gpu_gpu<<< BlkPerRow, 512>>>(fdp, powptr, rinds, zlo, numrs, numzs, (int)(pow(2,stage-1)), rlen);
}

static  __global__ void Do_inmem_add_ffdotpows_gpu_gpu(float *fdp, float *powptr, int *rinds,int zlo, int numrs, int numzs, int stage, long long rlen)
{
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=numrs*numzs) return;

    const int ii = MYgtid/numrs;
    const int jj = MYgtid - ii*numrs;
    int zz = zlo + ii * 2;
    int zind, subz;
    int kk;
    float tmp = 0.0f;
    
    for(kk=0; kk<stage; kk++)
    {
        subz = calc_required_z_gpu((2.0*kk+1.0)/(stage*2), zz);
        zind = index_from_z_gpu(subz, zlo);
        int offset = zind * rlen;
        tmp += fdp[offset+rinds[jj+ (kk+stage-1)*numrs]];
    }
    powptr[MYgtid] += tmp;
}


int  search_ffdotpows_gpu(float powcut, float *d_fundamental, accel_cand_gpu * cand_array_search_gpu, accel_cand_gpu * cand_array_sort_gpu, int numzs, int numrs, accel_cand_gpu *cand_gpu_cpu)
{
	int *d_addr;
	int h_addr;
	
	cudaMalloc((void **)&d_addr, sizeof(int) * 1);	
	cudaMemset(d_addr, 0, sizeof(int)); // set d_addr to 0

	int BlkPerRow=(numrs*numzs-1+512)/512;
	search_ffdotpows_kernel<<<BlkPerRow, 512>>>(powcut, d_fundamental, cand_array_search_gpu, numzs, numrs, d_addr);
	cudaMemcpy(&h_addr, d_addr, sizeof(int) * 1, cudaMemcpyDeviceToHost);	
	cudaMemcpy(cand_gpu_cpu, cand_array_search_gpu, sizeof(accel_cand_gpu) * h_addr, cudaMemcpyDeviceToHost);
	cudaFree(d_addr);
	return h_addr ;
}

static __global__ void  search_ffdotpows_kernel(float powcut, float *d_fundamental, accel_cand_gpu * cand_array_search_gpu, int numzs, int numrs, int *d_addr)
{
    const int MYgtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_num = numrs*numzs;
    if(MYgtid >= total_num) return;
    
    int nof_cand;
    float pow ;
    int addr_search;
    accel_cand_gpu cand_tmp ;
    int z_ind, r_ind;
    nof_cand = 0;
    pow = d_fundamental[MYgtid];
    if(pow > powcut)
    {
        cand_tmp.pow = pow ;
        nof_cand += 1 ;
        cand_tmp.nof_cand = nof_cand ;
        z_ind = (int)(MYgtid/numrs);
        cand_tmp.z_ind = z_ind;
        r_ind = MYgtid - z_ind * numrs ;
        cand_tmp.r_ind = r_ind;
        cand_tmp.w_ind = 0;
        addr_search = atomicAdd(&d_addr[0], 1);
        cand_array_search_gpu[ addr_search ] = cand_tmp ;
    }
}


