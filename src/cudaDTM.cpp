#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstring>

#include "DTM.hpp"
#include "DDM.hpp"
#include "DITO_API.hpp"

// TODO: transference only valid when there is only 1 partition per GPU

// map using the idGPUs established by the scheduler
void setGPUDevice(size_t i){

    state_t *state = getState();
    cudaSetDevice(state->idGPUs[i]);
}

void cpyDataCPU2GPU(DTI_t *DTI){

    size_t i, j, nPartitionsGPU, nPerPartitionGPU, fPerPartitionGPU, nextIndex;
    size_t nGPUs, *idGPUs;
    cudaError_t err;

    size_t size = DTI->size;

    // get application state
    state_t *state = getState();
    nGPUs = state->nGPUs;
    idGPUs = state->idGPUs;

    // copy data from the CPU to the GPUs following the information in the DTI structure
    DTIDesctiption_t *description = DTI->description;

    // accumulate partitions corresponding to the GPU contiguously
    void *cData = (void*)calloc(DTI->N, size);
    nextIndex = 0;
    for(i = 0; i<nGPUs; i++){

        // get the total number of elements to be transferred to the GPU
        //nElementsInGPU = infoDTI->nElementsPerGPU[i];
        nPartitionsGPU = description->nPartitionsPerGPU[i];

        // loop over partitions corresponding to the GPU and build the contiguous array
        for(j = 0; j<nPartitionsGPU; j++){

            // get partition data
            nPerPartitionGPU = description->nElementsPerPartition[i][j];
            fPerPartitionGPU = description->firstElementPerPartition[i][j];

            // copy the partition from the source to the destination
            void *src = (char*)(DTI->cpuData) + fPerPartitionGPU * size;
            void *dst = (char*)cData + nextIndex * size;
            size_t nBytes = nPerPartitionGPU * size;
            memcpy(dst, src, nBytes);

            nextIndex += nPerPartitionGPU;
        }
    }

    printf(" Printing cData = ");
    int* intCData = (int*)cData;
    for(i = 0; i<DTI->N; i++){

        printf("%d ", intCData[i]);
    }
    printf("\n");

    // allocate memory and move data to the GPU
    size_t firstElement = 0;
    for(i = 0; i<nGPUs; i++){

        // set device
        setGPUDevice(i);

        // allocate memory
        err = cudaMalloc(&(DTI->gpuData[i]), description->nElementsPerGPU[i] * size); 
        if (err != cudaSuccess) 
            printf("Allocation failed in %zu:  %s\n", i, cudaGetErrorString(err));

        // move data
        err = cudaMemcpy(DTI->gpuData[i], (char*)cData + firstElement * size, description->nElementsPerGPU[i] * size, cudaMemcpyHostToDevice); 	
        if (err != cudaSuccess) 
            printf("Memcpy failed in %zu:  %s\n", i, cudaGetErrorString(err));

        firstElement += description->nElementsPerGPU[i];
    }

    // deallocate termporal intermediate structure
    free(cData);
}

void cpyDataGPU2CPU(DTI_t *DTI){

    size_t i, j, nPartitionsGPU, nPerPartitionGPU, fPerPartitionGPU, nextIndex;
    cudaError_t err;
    size_t size = DTI->size;
    state_t *state = getState();

    size_t nGPUs, *idGPUs;
    nGPUs = state->nGPUs;
    idGPUs = state->idGPUs;

    // copy data from the CPU to the GPUs following the information in the DTI structure
    DTIDesctiption_t *description = DTI->description;

    // intermediate array to recive data from the GPUs contiguously and then redistribute
    void *cData = (void*)calloc(DTI->N, size);

    // allocate memory and move data to the GPU
    size_t firstElement = 0;
    for(i = 0; i<nGPUs; i++){

        // set device
        setGPUDevice(i);
        
        // move data from the GPU to the CPU
        err = cudaMemcpy((char*)cData + firstElement * size, DTI->gpuData[i], description->nElementsPerGPU[i] * size, cudaMemcpyDeviceToHost); 	
        if (err != cudaSuccess) 
            printf("Memcpy GPU2CPU failed in %zu:  %s\n", i, cudaGetErrorString(err));

        firstElement += description->nElementsPerGPU[i];

        // deallocate memory
        err = cudaFree(DTI->gpuData[i]); 
        if (err != cudaSuccess) 
            printf("Deallocation failed in %zu:  %s\n", i, cudaGetErrorString(err));

    }

    printf(" Printing cData = ");
    int* intCData = (int*)cData;
    for(i = 0; i<DTI->N; i++){

        printf("%d ", intCData[i]);
    }
    

    nextIndex = 0;
    for(i = 0; i<nGPUs; i++){
        
        // get the total number of elements to be transferred to the GPU
        nPartitionsGPU = description->nPartitionsPerGPU[i];

        // loop over partitions corresponding to the GPU and build the contiguous array
        for(j = 0; j<nPartitionsGPU; j++){

            // get partition data
            nPerPartitionGPU = description->nElementsPerPartition[i][j];
            fPerPartitionGPU = description->firstElementPerPartition[i][j];

            // copy the partition from the source to the destination
            void *dst = (char*)(DTI->cpuData) + fPerPartitionGPU * size;
            void *src = (char*)cData + nextIndex * size;
            size_t nBytes = nPerPartitionGPU * size;

            memcpy(dst, src, nBytes);

            nextIndex += nPerPartitionGPU;
        }
    }
    printf("\n");
}