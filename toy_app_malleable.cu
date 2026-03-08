#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "toy_app_malleable.hpp"


__global__ void simulateKernel(int* arr, int N){

    size_t threadId = 
        ((size_t)blockIdx.x  + (size_t)gridDim.x  * (size_t)blockIdx.y + (size_t)gridDim.x * (size_t)gridDim.y * (size_t)blockIdx.z) *
        ((size_t)blockDim.x * (size_t)blockDim.y * (size_t)blockDim.z) +
        ((size_t)threadIdx.x + (size_t)blockDim.x * (size_t)threadIdx.y + (size_t)blockDim.x * (size_t)blockDim.y * (size_t)threadIdx.z);

    if(threadId < N){

        arr[threadId] *= 2;
    }
}


void runKernel(int* arr, int N){

    simulateKernel<<<1, N>>>(arr, N);
}