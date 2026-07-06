#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

typedef struct appStruct_t {

    // iterative application
    float *arr;
    size_t *indexArr;
    char *charArr;
    size_t N;
    size_t T;
    size_t K;
    size_t cpuK;
    size_t s;


    float *gValues; // graph values
    float *tmpAcc;
    size_t *indices;
    size_t *offPerNode;
    size_t *nPerNode;

    // iterative application with phases
    size_t P;
    size_t *phases;


    // iterative application with phases and communication
    size_t nIterationsForCommunications; // number of iterations to be executed before each communication
    double communicationTimeSrcDst = 0.0;
    double communicationTimeDstSrc = 0.0;

    // whether the application is malleable or not (this decides if there are reconfigurations or not)
    size_t malleable; // 0 / 1

    size_t async; // whether communications are performed asynchronously

} appStruct_t;


void runCPU(float *arr, size_t N, size_t K);
void runKernel(float* arr, size_t N, size_t K);
void runGraphKernel(float* arr, size_t N, size_t *n, size_t *off, float *tmpAcc, size_t *indices, size_t K);
void runUpdateNodesKernel(float* arr, size_t N, size_t *n, float *tmpAcc);


void launch_iterative_app(int argc, void* argv[]);
void launch_phases_app(int argc, void* argv[]);
void launch_communications_app(int argc, void* argv[]);

// test application 
void launch_reconf_test_app(int argc, void* argv[]);
void launch_malloc_test_app(int argc, void* argv[]);

void launch_reconfs_test_app_new(int argc, void* argv[]);
void launch_NCCL_communications_app(int argc, void* argv[]);
void launch_unified_memory_app(int argc, void* argv[]);

// test communication application
