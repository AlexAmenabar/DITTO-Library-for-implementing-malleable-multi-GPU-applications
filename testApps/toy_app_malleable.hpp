#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

typedef struct appStruct_t {

    // iterative application
    float *arr;
    char *charArr;
    size_t N;
    size_t T;
    size_t K;
    size_t cpuK;

    // iterative application with phases
    size_t P;
    size_t *phases;

    // iterative application with phases and communication
    size_t nIterationsForCommunications; // number of iterations to be executed before each communication
    double communicationTimeSrcDst = 0.0;
    double communicationTimeDstSrc = 0.0;

    // whether the application is malleable or not (this decides if there are reconfigurations or not)
    size_t malleable; // 0 / 1
} appStruct_t;


void runCPU(float *arr, size_t N, size_t K);
void runKernel(float* arr, size_t N, size_t K);
void launch_iterative_app(int argc, void* argv[]);
void launch_phases_app(int argc, void* argv[]);
void launch_communications_app(int argc, void* argv[]);

// test application 
void launch_reconf_test_app(int argc, void* argv[]);

// test communication application
