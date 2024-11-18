#include <cuda_runtime.h>
#include <cufft.h>

#include <stdio.h>

extern "C" void select_cuda_dev(int cuda_inds);
extern "C" void Endup_GPU();


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


__global__ void Do_Get_subsdata(float *indata, short *outdata, int nsub, int worklen)
{
    const int  MYgtid = blockDim.x * blockIdx.x + threadIdx.x;
    if(MYgtid>=worklen) return;

    // int x,y;
    // x = MYgtid%worklen;
    // y = MYgtid/worklen;
    // outdata[MYgtid] = (short)(indata[x*nsub+y]+0.5);

    outdata[MYgtid] = (short)(indata[MYgtid]+0.5);
}


