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
        float v0 = val, v1 = val, v2 = val, v3 = val;

        for (int k = 0; k < K; k++) {
            v0 = v0 * 1.000001f + 0.000001f;
            v1 = v1 * 1.000001f + 0.000001f;
            v2 = v2 * 1.000001f + 0.000001f;
            v3 = v3 * 1.000001f + 0.000001f;
        }

        val = (v0 + v1 + v2 + v3) * 0.125f;

        arr[threadId] = val;
    }
}


void runKernel(float* arr, int N, int K){

    int threads = 512;
    int blocks = (N + threads - 1) / threads;

    simulateKernel<<<blocks, threads>>>(arr, N, K);
    cudaDeviceSynchronize();
}