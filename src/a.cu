
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <ctype.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>


#define WARP_SIZE 32
#define MAX_THREAD 512

__device__ void warpReduce(volatile float *sdata, unsigned int tid, int blockSize)
{
    if (blockSize >= 64) {sdata[tid] += sdata[tid + 32];}
    if (blockSize >= 32) {sdata[tid] += sdata[tid + 16];}
    if (blockSize >= 16) {sdata[tid] += sdata[tid + 8];}
    if (blockSize >= 8) {sdata[tid] += sdata[tid + 4];}
    if (blockSize >= 4) {sdata[tid] += sdata[tid + 2];}
    if (blockSize >= 2) {sdata[tid] += sdata[tid + 1];}
}

__global__ void reduce6(float *g_idata, float *g_odata, unsigned int n, int blockSize) 
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

extern "C" 
float API_cuda_reduce_sum(float *arr, int total_num)
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
    reduce6<<<grid_dim, block_dim, block_dim.x*sizeof(int)>>>(arr, d_out_arr, total_num, block_dim.x);
    cudaDeviceSynchronize();

    h_out_arr = new float[block_num];
    memset(h_out_arr, 0, sizeof(int)*block_num);
    cudaMemcpy(h_out_arr, d_out_arr, sizeof(int) * block_num, cudaMemcpyDeviceToHost);

    long ret=0;
    for(int i=0; i<block_num; i++)
    {
        ret += h_out_arr[i];
    }

    // cudaFree(d_arr);
    cudaFree(d_out_arr);
    delete []h_out_arr;

    return ret;

}

