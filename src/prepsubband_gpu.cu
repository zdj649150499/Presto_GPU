#include <cuda_runtime.h>
#include <cufft.h>

#include <stdio.h>

extern "C" void select_cuda_dev(int cuda_inds);
extern "C" void Endup_GPU();


extern "C" void dedisp_subbands_GPU(float *data, float *lastdata,
                     int numpts, int numchan, 
                     int *delays, int numsubbands, float *result, int transpose);

__global__ void Do_dedisp_subbands_GPU(float *data, float *lastdata,
                     int numpts, int numchan, 
                     int *delays, int numsubbands, float *result, int transpose);
extern "C" void downsamp_GPU(float *indata, float *outdata, int numchan, int numpts, int down);
__global__ void Do_downsamp_GPU(float *indata, float *outdata, int numchan, int numpts, int down);
extern "C" void float_dedisp_GPU(float *data, float *lastdata,
                  int numpts, int numchan,
                  int *delays, float approx_mean, float *result, int numdms);
__global__ void Do_float_dedisp_GPU(float *data, float *lastdata,
                  int numpts, int numchan,
                  int *delays, float approx_mean, float *result, int numdms);

extern "C" void Get_subsdata(float *indata, short *outdata, int nsub, int worklen);
__global__ void Do_Get_subsdata(float *indata, short *outdata, int nsub, int worklen);

//----------------------select a cpu to play with --------------------
void select_cuda_dev(int cuda_inds)
{
	
	cudaSetDevice(cuda_inds);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, cuda_inds);

    printf("\nGPU Device %d: \"%s\" with Capability: %d.%d\n", cuda_inds, deviceProp.name, deviceProp.major, deviceProp.minor);
	
	cudaDeviceReset();
}


void Endup_GPU()
{
    cudaDeviceSynchronize();
    cudaDeviceReset();
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

        // if(MYgtid < numpts-dind)
        // {
        //     in_index = (MYgtid+dind)*numchan+in_chan;
        //     temp += lastdata[in_index];
        // }
        // else
        // {
        //     in_index = (MYgtid+dind -numpts)*numchan+in_chan;
        //     temp += data[in_index];
        // }
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

    /* input time first */
    // out_index = MYrow*numpts + MYgtid;
    // in_dex = out_index*down;
    // for(i=0;i<down;i++)
    // {
    //     ftmp+=indata[in_dex];
    //     in_dex+=1;
    // }
    // outdata[out_index] = ftmp/(1.0f*down);
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

    // extern __shared__ int delays_share[];
    // int *delays_share_bk = (int *)delays_share;



    for(i=0;i<numchan;i++)
    {
        // delays_share_bk[i] = delays[x+i];
        // __syncthreads();
        /****/
        /* input freq first */
        // in_dex = MYgtid+(long long)(delays_share_bk[i]);
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

        /* input time first */
        // in_dex = delays[x+i];
        // if(MYgtid<(numpts-in_dex))
        // {
        //     in_dex=i*numpts+in_dex+MYgtid;
        //     ftmp += lastdata[in_dex];
        // }
        // else
        // {
        //     in_dex=i*numpts+(MYgtid-numpts+in_dex);
        //     ftmp += data[in_dex];
        // }
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
