#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

typedef struct appStruct_t {

    int *arr;
    int N;
    int T;

} appStruct_t;


void runKernel(int* arr, int N);

void launch(int argc, void* argv[]);