#include <cuda_runtime.h>
#include <nccl.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>


#include <omp.h>
#include <iostream>
#include <unistd.h>
#include <cstddef>
#include <cstring>
#include <time.h>

#include <nccl.h>

#include "toy_app_malleable.hpp"
#include "DITO_API.hpp"
#include "DTM.hpp"


#include "RMS.hpp"

void runCommunicationWorkload(size_t N, size_t nGPUs, size_t *idGPUs, ncclComm_t *comms, cudaStream_t *streams, size_t iterations) {
    
    // Size of the data block each GPU sends to every other GPU
    // Total matrix per GPU is N x N
    size_t total_elements = N * N;
    size_t chunk_elements = total_elements / nGPUs; // each GPU stores a chunk
    size_t chunk_bytes = chunk_elements * sizeof(float); // number of bytes per chunk

    // buffers to send and receive data
    float **d_send_buffer = (float**)malloc(nGPUs * sizeof(float*));
    float **d_recv_buffer = (float**)malloc(nGPUs * sizeof(float*));

    // allocate memory for execution times
    float *execution_times = (float*)calloc(iterations, sizeof(float));

    // Allocate memory on each designated GPU
    for (size_t i = 0; i < nGPUs; i++) {
        
        cudaSetDevice(idGPUs[i]);
        cudaMalloc(&(d_send_buffer[i]), total_elements * sizeof(float));
        cudaMalloc(&(d_recv_buffer[i]), total_elements * sizeof(float));
    }

    // start measuring time
    struct timespec startComm, endComm;

    // Benchmark: loop
    for (size_t iter = 0; iter < iterations; iter++) {
        
        clock_gettime(CLOCK_MONOTONIC, &startComm);

        #pragma omp parallel num_threads(nGPUs)
        {
            // get the rank of the thread
            size_t rank = omp_get_thread_num();
            cudaSetDevice(idGPUs[rank]);

            // Begin grouped communication to flood the network simultaneously
            ncclGroupStart();

            // send data to all GPUs and receive from all GPUs 
            for (size_t peer = 0; peer < nGPUs; peer++) {
                
                // address offset for the unique chunk destined for 'peer'
                // send chunk to the GPU
                float* send_ptr = d_send_buffer[rank] + (peer * chunk_elements);
                
                // receive chunk from other GPU and store in its position
                float* recv_ptr = d_recv_buffer[rank] + (peer * chunk_elements);

                // Send my chunk to peer
                ncclSend(send_ptr, chunk_elements, ncclFloat, peer, comms[rank], streams[idGPUs[rank]]);
                
                // Receive peer's chunk into my buffer
                ncclRecv(recv_ptr, chunk_elements, ncclFloat, peer, comms[rank], streams[idGPUs[rank]]);
            }

            ncclGroupEnd();
        }

        // Synchronize all streams to enforce exact completion on the hardware
        #pragma omp parallel num_threads(nGPUs)
        {
            size_t rank = omp_get_thread_num();
            cudaSetDevice(idGPUs[rank]);
            cudaStreamSynchronize(streams[idGPUs[rank]]);
        }
        clock_gettime(CLOCK_MONOTONIC, &endComm);
        double elapsed = (endComm.tv_sec - startComm.tv_sec) + (endComm.tv_nsec - startComm.tv_nsec) / 1e9;

        execution_times[iter] = elapsed;
    }

    // Cleanup
    for (size_t i = 0; i < nGPUs; i++) {
        cudaSetDevice(idGPUs[i]);
        cudaFree(d_send_buffer[i]);
        cudaFree(d_recv_buffer[i]);
    }
    free(d_send_buffer);
    free(d_recv_buffer);

    // compute mean and standard deviation
    float meanExecTime = 0.0;
    float stdv = 0.0;
    for(size_t i = 0; i<iterations; i++){

        meanExecTime += execution_times[i];
    }
    meanExecTime /= (float)iterations;

    for(size_t i = 0; i<iterations; i++){

        float var = (execution_times[i] - meanExecTime);
        var *= var;
        stdv += var;
    }
    stdv /= (float)(iterations - 1);

    printf("%lf %lf\n", meanExecTime, stdv);

    free(execution_times);
}

void launch_communication_bench(int argc, void *argv[]){


    size_t i;

    // initialize DITTO environment
    // temporal: simulate information received from the scheduler: number of GPUs and identifiers of the GPUs (pass argv to the initDITTO function?)
    initDITTO(argv[argc+1]);
    initNCCLComm(getState()->jobResources);

    appStruct_t *appData = (appStruct_t*)calloc(1, sizeof(appStruct_t));
    appData->N = *(size_t*)argv[0]; // number of elements in the matrix
    appData->T = *(size_t*)argv[1]; // number of iterations
    appData->s = *(size_t*)argv[2]; 

    // communication info
    size_t pinned = *(size_t*)argv[3];
    size_t async = *(size_t*)argv[4];
    size_t steps = *(size_t*)argv[5];
    size_t cores = *(size_t*)argv[6];
    

    appData->malleable = *(size_t*)argv[argc];

    
    // allocate memory for App Data
    if(pinned) // pinned memory
        cudaMallocHost(&appData->arr, appData->N * appData->N * sizeof(float));
    else
        appData->arr = (float*)calloc(appData->N * appData->N, sizeof(float));
    
    // initialize data
    for(i = 0; i<appData->N * appData->N; i++)
        appData->arr[i] = (float)i;//(float)rand()/(float)(RAND_MAX);


    // set communication type
    communicationType_t commType;
    if(pinned)
        commType.cudaMemoryType = pinnedComm;
    else 
        commType.cudaMemoryType = nonPinnedComm;

    if(async)
        commType.transmissionType = asyncComm;
    else
        commType.transmissionType = syncComm;
    
    if(steps == 0)
        commType.transferSteps = oneStepComm;
    else if(steps == 1)
        commType.transferSteps = twoStepsComm;
    else
        commType.transferSteps = stridedComm;

    if(cores)
        commType.transferCores = multiCoreComm;
    else
        commType.transferCores = singleCoreComm;


    // Initialize DTIs for automatically managing CPU-GPU communications
    DTI_t *appDataDTI;
    if(appData->s > 0)
        appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->N * appData->N, sizeof(float), "appData", initializeDTIDescription(simple, ordered, commType));
    else
        appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->N * appData->N, sizeof(float), "appData", initializeDTIDescription(simple, ordered, commType));



    // configure all DTIs for automatic data transference // TODO: configure in initialization
    configureDTIs(getState()->jobResources, NULL, CPU2GPU); // number of GPUs, old number of GPUs

    // send data to the GPU
    transferDataCPU2GPU();

    // wait for signal before starting
    size_t nGPUs = getState()->jobResources->nGPUs;
    size_t *idGPUs = getState()->jobResources->idGPUs;
    runCommunicationWorkload(appData->N, nGPUs, idGPUs, getNCCLComms(), getCudaStreams(), appData->T);

    // transfer data fron the GPUs to the CPU
    transferDataGPU2CPU();


    if(pinned)
        cudaFreeHost(appDataDTI->cpuData);
    else
        free(appDataDTI->cpuData);

    // destroy streams
    freeNCCLComm(getState()->jobResources);
    freeDITTO();


    // signal that job finished
    jobFinished(getJobControl());

    pthread_exit(NULL);
}