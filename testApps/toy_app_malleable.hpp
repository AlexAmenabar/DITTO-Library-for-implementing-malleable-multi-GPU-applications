#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

typedef struct appStruct_t {

    int *arr;
    int N;

} appStruct_t;


void runKernel(int* arr, int N);