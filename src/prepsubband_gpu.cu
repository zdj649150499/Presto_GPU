#include <cuda_runtime.h>
#include <cufft.h>
//#include <helper_functions.h>
// #include <helper_cuda.h>
#include <stdio.h>
#include "device_functions.h" 

extern "C" void select_cuda_dev(int cuda_inds);
extern "C" void Endup_GPU();

extern "C" void DedispersionOnGPU(float *oridata_gpu, float *outdata_gpu,  int worklen, int bs, int nsub, int numdms, int *offsets);
__global__ void getTimeDMArrayGPUonceGPU(float *oridata_gpu, float *outdata_gpu,  int worklen, int bs, int nsub, int numdms, int *offsets);
__global__ void DownsampDataArrayGPU(float *oridata, float *outdata, int nsub, int worklen, int bs);

extern "C" void dedisp_subbandsGPU(float *data, float *lastdata, int numpts, int numchan, int *delays, int numsubbands, float *result);

__global__ void dedisp_subbandsGPU_GPU(float *data, float *lastdata, int numpts, int numchan, int *delays, int numsubbands, int chan_per_subband, float *result);



texture<float> texRef1;
texture<float> texRef2;

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


// void DedispersionOnGPU(float *oridata_gpu, float *outdata_gpu,  int worklen, int bs, int nsub, int numdms, int *offsets)
// {
//     int BlkPerRow=(worklen-1+512)/512;
//     dim3 dimGrid2D(BlkPerRow,numdms);
//     getTimeDMArrayGPUonceGPU<<<dimGrid2D,512>>>(oridata_gpu, outdata_gpu, worklen, bs, nsub, numdms, offsets);
// }

// __global__ void getTimeDMArrayGPUonceGPU(float *oridata_gpu, float *outdata_gpu,  int worklen, int bs, int nsub, int numdms, int *offsets)
// {
//     int i,j,c,b;
//     int MYrow=blockIdx.y;
//     int ThrPerBlk = blockDim.x;
// 	int MYbid = blockIdx.x;
// 	int MYtid = threadIdx.x;
//     int MYgtid = ThrPerBlk * MYbid + MYtid;
//     if(MYgtid>=worklen) return;

//     // c=MYgtid*numdms+MYrow;   // col first
//     c=MYrow*worklen+MYgtid;     // row first
//     float val=0;
//     for(i=0;i<nsub;i++)
//     {
//         b=MYgtid+offsets[MYrow*nsub+i];
//         for(j=0;j<bs;j++)
//         {
//             b=i+(b*bs+j)*nsub;
//         }
//         val+=(oridata_gpu[b]);
//     }
//     outdata_gpu[c]=val;
// }

/* same as above*/

void DedispersionOnGPU(float *oridata_gpu, float *outdata_gpu,  int worklen, int bs, int nsub, int numdms, int *offsets)
{
    int BlkPerRow;

    if(bs>1)
    {
        float *arraybk;
        int custatus = cudaMalloc((void**)&arraybk, sizeof(float)*nsub*worklen*2);

        BlkPerRow=(worklen*nsub*2-1+1024)/1024;
        DownsampDataArrayGPU<<<BlkPerRow,1024>>>(oridata_gpu,arraybk, nsub,worklen*2,bs);

        BlkPerRow=(worklen*numdms-1+1024)/1024;
        getTimeDMArrayGPUonceGPU<<<BlkPerRow,1024>>>(arraybk, outdata_gpu, worklen, 1, nsub, numdms, offsets);

        cudaFree(arraybk);
    }
    else
    {
        BlkPerRow=(worklen*numdms-1+1024)/1024;
        getTimeDMArrayGPUonceGPU<<<BlkPerRow,1024>>>(oridata_gpu, outdata_gpu, worklen, 1, nsub, numdms, offsets);
    }
}

__global__ void DownsampDataArrayGPU(float *oridata, float *outdata, int nsub, int worklen, int bs)
{
    int ThrPerBlk = blockDim.x;
	int MYbid = blockIdx.x;
	int MYtid = threadIdx.x;
    int MYgtid = ThrPerBlk * MYbid + MYtid;
    if(MYgtid>=worklen*nsub) return;

    int j;
    float val=0;
    int mychan = (int )(MYgtid/worklen);
    int mybin = MYgtid - mychan*worklen;

    int index = mychan + mybin*bs*nsub;
    for(j=0;j<bs;j++)
    {
        val += (oridata[index]);
        index += nsub;
    }
    outdata[mychan+mybin*nsub]=val/bs;
}


__global__ void getTimeDMArrayGPUonceGPU(float *oridata_gpu, float *outdata_gpu,  int worklen, int bs, int nsub, int numdms, int *offsets)
{
    int i,j,b;
    int ThrPerBlk = blockDim.x;
	int MYbid = blockIdx.x;
	int MYtid = threadIdx.x;
    int MYgtid = ThrPerBlk * MYbid + MYtid;
    if(MYgtid>=worklen*numdms) return;

    float val=0;
    int dmi= (int)(MYgtid/worklen);
    int xi= MYgtid - dmi*worklen;
    for(i=0;i<nsub;i++)
    {
        b=xi*bs+offsets[dmi*nsub+i]*bs;
        b=i+b*nsub;
        for(j=0;j<bs;j++)
        {
            val+=(oridata_gpu[b]);
            b+=nsub;
        }
    }
    outdata_gpu[MYgtid]=val/bs;
}


void dedisp_subbandsGPU(float *data, float *lastdata, int numpts, int numchan, int *delays, int numsubbands, float *result)

{
    float *data_gpu;
    float *lastdata_gpu;
    float *result_gpu;
    int *delays_gpu;
    int chan_per_subband = numchan / numsubbands;

    cudaMalloc((void**)&data_gpu, sizeof(float)*numpts*numchan);
    cudaMalloc((void**)&lastdata_gpu, sizeof(float)*numpts*numchan);
    cudaMalloc((void**)&result_gpu, sizeof(float)*numpts*numsubbands);
    cudaMalloc((void**)&delays_gpu, sizeof(int)*numchan);

    cudaMemcpy(data_gpu, lastdata, sizeof(float)*numpts*numchan, cudaMemcpyHostToDevice);
    cudaMemcpy(data_gpu+numpts*numchan, data, sizeof(float)*numpts*numchan, cudaMemcpyHostToDevice);
    cudaMemcpy(delays_gpu, delays, sizeof(int)*numchan, cudaMemcpyHostToDevice);

    int BlkPerRow=(numpts*numsubbands-1+1024)/1024;
    dedisp_subbandsGPU_GPU<<<BlkPerRow,512>>>(data_gpu, lastdata_gpu, numpts,  numchan,  delays_gpu,  numsubbands, chan_per_subband, result_gpu);

    cudaMemcpy(result, result_gpu, sizeof(float)*numpts*numsubbands, cudaMemcpyDeviceToHost);

    cudaFree(data_gpu);
    cudaFree(lastdata_gpu);
    cudaFree(result_gpu);
    cudaFree(delays_gpu);
}


__global__ void dedisp_subbandsGPU_GPU(float *data, float *lastdata, int numpts, int numchan, int *delays, int numsubbands, int chan_per_subband, float *result)
{
    int i;
    int ThrPerBlk = blockDim.x;
	int MYbid = blockIdx.x;
	int MYtid = threadIdx.x;
    int MYgtid = ThrPerBlk * MYbid + MYtid;
    if(MYgtid>=numpts*numsubbands) return;

    int yi= (int)(MYgtid/numpts);
    int xi= MYgtid - yi*numpts;
    int oir_yi;
    int ori_xi;
    float val=0.0;
    for(i=0; i<chan_per_subband; i++)
    {
        oir_yi = i + yi*chan_per_subband;
        ori_xi = xi + delays[oir_yi];
        if(ori_xi < numpts)
            val += lastdata[ori_xi + oir_yi*numpts];
        else val += data[ori_xi - numpts + oir_yi*numpts];
    }
    result[MYgtid] = val;
}
// {
//     int i;
//     int ThrPerBlk = blockDim.x;
//     int MYbid = blockIdx.x;
//     int MYtid = threadIdx.x;
//     int MYgtid = ThrPerBlk * MYbid + MYtid;
//     if(MYgtid>=numpts*numsubbands) return;

//     int yi= (int)(MYgtid/numpts);
//     int xi= MYgtid - yi*numpts;
//     int oir_yi;
//     int ori_xi;
//     float val=0.0;
//     for(i=0; i<chan_per_subband; i++)
//     {
//         oir_yi = i + yi*chan_per_subband;
//         ori_xi = xi + delays[oir_yi];
//         val += data[ori_xi*numchan +oir_yi];
//     }
//     result[MYgtid] = val;
// }
