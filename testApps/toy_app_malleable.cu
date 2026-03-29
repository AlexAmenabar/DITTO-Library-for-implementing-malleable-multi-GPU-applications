#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "toy_app_malleable.hpp"


__global__ void simulateKernel(float* arr, int N, int K){

    size_t threadId = 
        ((size_t)blockIdx.x  + (size_t)gridDim.x  * (size_t)blockIdx.y + (size_t)gridDim.x * (size_t)gridDim.y * (size_t)blockIdx.z) *
        ((size_t)blockDim.x * (size_t)blockDim.y * (size_t)blockDim.z) +
        ((size_t)threadIdx.x + (size_t)blockDim.x * (size_t)threadIdx.y + (size_t)blockDim.x * (size_t)blockDim.y * (size_t)threadIdx.z);

    if(threadId < N){
    
        float val = arr[threadId];
        for (int k = 0; k < K; k++) {
            val = val * 1.000001f + 0.000001f; //(val * 1.01) * 0.999;// * 0.99;// / 1.5;//1.000001f + 0.000001f;
        }
        val = val * 0.5;

        arr[threadId] = val;
    }
}


void runKernel(float* arr, int N, int K){

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    simulateKernel<<<blocks, threads>>>(arr, N, K);
    cudaDeviceSynchronize();
}