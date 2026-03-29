#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

typedef struct appStruct_t {

    float *arr;
    int N;
    int T;
    int K;
    int cpuK;

    int P;
    int *phases;
} appStruct_t;


void runCPU(float *arr, int N, int K);
void runKernel(float* arr, int N, int K);

void launch_iterative_app(int argc, void* argv[]);
void launch_phases_app(int argc, void* argv[]);