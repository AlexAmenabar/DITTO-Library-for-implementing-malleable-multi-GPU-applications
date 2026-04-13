#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstring>

#include "DTM.hpp"
#include "DDM.hpp"
#include "DITO_API.hpp"
#include "mockSch.hpp"

// TODO: this module should be revised since it calls functions that it shouldn't

// TODO: transference only valid when there is only 1 partition per GPU

// map using the idGPUs established by the RMS
void setGPUDevice(size_t i){

    state_t *state = getState();
    //printf(" Setting dev %zu (%zu)\n", i, state->idGPUs[i]);
    //fflush(stdout);
    cudaSetDevice(getGPUIds()[i]);
}

void cpyDataCPU2GPU(DTI_t *DTI){

    size_t i, j, nPartitionsGPU, nPartitionGPU, offPartitionGPU, nextIndex;
    size_t nGPUs, *idGPUs;
    cudaError_t err;

    size_t size = DTI->size;

    // get application state
    state_t *state = getState();
    nGPUs = getNumberOfGPUs();
    idGPUs = getGPUIds();

    // copy data from the CPU to the GPUs following the information in the DTI structure
    DTIDesctiption_t *description = DTI->description;
    nPartitionsGPU = description->nPartitions;

    // accumulate partitions corresponding to the GPU contiguously
    void *cData = (void*)calloc(DTI->N, size);
    nextIndex = 0;
    for(i = 0; i<nGPUs; i++){

        // loop over partitions corresponding to the GPU and build the contiguous array
        for(j = 0; j<nPartitionsGPU; j++){

            // get partition data
            nPartitionGPU = DTI->nPerPartition[i][j];
            offPartitionGPU = DTI->offsetPerPartition[i][j];

            if(nPartitionGPU > 0){
                
                // copy the partition from the source to the destination
                void *src = (char*)(DTI->cpuData) + offPartitionGPU * size;
                void *dst = (char*)cData + nextIndex * size;
                size_t nBytes = nPartitionGPU * size;
                memcpy(dst, src, nBytes);

                nextIndex += nPartitionGPU;
            }
        }
    }

    // allocate memory and move data to the GPU
    size_t firstElement = 0;
    for(i = 0; i<nGPUs; i++){

        // set device
        setGPUDevice(i);
	cudaDeviceSynchronize();

        // allocate memory
        err = cudaMalloc(&(DTI->gpuData[i]), DTI->nPerGPU[i] * size); 
        if (err != cudaSuccess) 
            printf("Job%zu: Allocation in GPU failed in %zu (%zu):  %s\n", getJobControl()->jobId, i, getGPUIds()[i], cudaGetErrorString(err));

        // move data
        err = cudaMemcpy(DTI->gpuData[i], (char*)cData + firstElement * size, DTI->nPerGPU[i] * size, cudaMemcpyHostToDevice); 	
        if (err != cudaSuccess) 
            printf("Job%zu: Memcpy CPU2GPU failed in %zu (%zu):  %s\n", getJobControl()->jobId, i, getGPUIds()[i], cudaGetErrorString(err));

        firstElement += DTI->nPerGPU[i];
    }

    // deallocate termporal intermediate structure
    free(cData);
}

void cpyDataGPU2CPU(DTI_t *DTI){

    size_t i, j, nPartitionsGPU, nPartitionGPU, offPartitionGPU, nextIndex;
    cudaError_t err;
    size_t size = DTI->size;
    state_t *state = getState();

    size_t nGPUs, *idGPUs;
    nGPUs = getNumberOfGPUs();
    idGPUs = getGPUIds();

    // copy data from the CPU to the GPUs following the information in the DTI structure
    DTIDesctiption_t *description = DTI->description;

    // intermediate array to recive data from the GPUs contiguously and then redistribute
    void *cData = (void*)calloc(DTI->N, size);

    // allocate memory and move data to the GPU
    size_t firstElement = 0;
    for(i = 0; i<nGPUs; i++){

        // set device
        setGPUDevice(i);
//        cudaDeviceSynchronize();

        // move data from the GPU to the CPU
        err = cudaMemcpy((char*)cData + firstElement * size, DTI->gpuData[i], DTI->nPerGPU[i] * size, cudaMemcpyDeviceToHost); 	
        if (err != cudaSuccess) 
            printf("Job%zu: Memcpy GPU2CPU failed in %zu (%zu):  %s\n", getJobControl()->jobId, i, getGPUIds()[i], cudaGetErrorString(err));

        firstElement += DTI->nPerGPU[i];

        // deallocate memory
        err = cudaFree(DTI->gpuData[i]); 
        if (err != cudaSuccess) 
            printf("Job%zu: Deallocation failed in %zu (%zu):  %s\n", getJobControl()->jobId, i, getGPUIds()[i], cudaGetErrorString(err));

    }


    nPartitionsGPU = description->nPartitions;
    nextIndex = 0;
    for(i = 0; i<nGPUs; i++){
        
        // get the total number of elements to be transferred to the GPU

        // loop over partitions corresponding to the GPU and build the contiguous array
        for(j = 0; j<nPartitionsGPU; j++){

            // get partition data
            nPartitionGPU = DTI->nPerPartition[i][j];
            offPartitionGPU = DTI->offsetPerPartition[i][j];

            if(nPartitionGPU>0){
            
                // copy the partition from the source to the destination
                void *dst = (char*)(DTI->cpuData) + offPartitionGPU * size;
                void *src = (char*)cData + nextIndex * size;
                size_t nBytes = nPartitionGPU * size;

                memcpy(dst, src, nBytes);

                nextIndex += nPartitionGPU;
            }
        }
    }
    //printf("\n");
}

void resetGPUs(){
    
    // reinit all GPUs
    for(size_t i = 0; i<getNumberOfGPUs(); i++){

        cudaSetDevice(getGPUIds()[i]);
        cudaDeviceReset();
    }
}
