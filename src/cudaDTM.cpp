#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstring>

#include "DTM.hpp"
#include "DDM.hpp"
#include "DITO_API.hpp"
#include "RMS.hpp"

// TODO: this module should be revised since it calls functions that it shouldn't

// TODO: transference only valid when there is only 1 partition per GPU

// map using the idGPUs established by the RMS
void setGPUDevice(size_t i){

    state_t *state = getState();
    //printf(" -- [APP] Setting dev %zu (real dev %zu)\n", i, getGPUIds()[i]);
    //fflush(stdout);
    cudaSetDevice(getGPUIds()[i]);
}

void initializeStreams(){

    for(size_t i = 0; i<getNumberOfGPUs(); i++){

        setGPUDevice(i);
        cudaStreamCreate(&(getCudaStreams()[i]));
    }
}

void destroyStreams(){
    
    for(size_t i = 0; i<getNumberOfGPUs(); i++){

        setGPUDevice(i);
        cudaStreamDestroy(getCudaStreams()[i]);
    } 
}


void resetGPUs(){
    
    // reinit all GPUs
    for(size_t i = 0; i<getNumberOfGPUs(); i++){

        cudaSetDevice(getGPUIds()[i]);
        cudaDeviceReset();
    }
}


void cpyDataCPU2GPUtwoSteps(DTI_t *DTI){

    size_t i, j, nPartitionsGPU, nPartitionGPU, offPartitionGPU, nextIndex;
    cudaError_t err;

    size_t size = DTI->size;

    // get application state
    size_t nGPUs = getNumberOfGPUs();

    DTIDesctiption_t *description = DTI->description;
    communicationType_t commType = description->commType;
    
    size_t *gpuIds = getGPUIds();
    cudaStream_t *cudaStreams = getCudaStreams();
    jobControl_t *jobControl = getJobControl();

    // set number of threads
    size_t n_threads = 1;
    if(commType.transferCores == multiCoreComm)
        n_threads = nGPUs;


    // allocate memory for arrays to accumulate partitions for the same GPU
    void **cData = (void**)malloc(nGPUs * sizeof(void*));

    // loop over GPUs
    #pragma omp parallel for num_threads(n_threads) private(nPartitionsGPU, nextIndex, nPartitionGPU, offPartitionGPU, err, i, j)
    for(i = 0; i<nGPUs; i++){

        // allocate memory for the number of elements necessary in the GPU i
        if(commType.cudaMemoryType == nonPinnedComm) // non-pinned memory and sync (async requires pinned memory)
            cData[i] = (void*)calloc(DTI->nPerGPU[i], size);
        else // pinned memory or async 
            cudaMallocHost(&cData[i], DTI->nPerGPU[i] * size);
        

        nPartitionsGPU = DTI->nPartitionsPerGPU[i];
        nextIndex = 0;

        // loop over partitions corresponding to the GPU and build the contiguous array
        for(j = 0; j<nPartitionsGPU; j++){

            // get partition data
            nPartitionGPU = DTI->nPerPartition[i][j];
            offPartitionGPU = DTI->offsetPerPartition[i][j];

            if(nPartitionGPU > 0){
                
                // copy the partition from the source to the destination
                void *src = (char*)(DTI->cpuData) + offPartitionGPU * size;
                void *dst = (char*)(cData[i]) + nextIndex * size;
                size_t nBytes = nPartitionGPU * size;

                memcpy(dst, src, nBytes);

                nextIndex += nPartitionGPU;
            }
        }
        
        // move data to the GPU

        // set device
        //setGPUDevice(i);
        cudaSetDevice(gpuIds[i]);


        // allocate memory in the GPU for the contiguous array
        err = cudaMalloc(&(DTI->gpuData[i]), DTI->nPerGPU[i] * size); 
        if (err != cudaSuccess) 
            printf("Job%zu: Allocation in GPU failed in %zu (%zu):  %s\n", jobControl->jobId, i, gpuIds[i], cudaGetErrorString(err));
        
        // copy data
        if(commType.transmissionType == asyncComm) // async
            err = cudaMemcpyAsync(DTI->gpuData[i], (char*)cData[i], DTI->nPerGPU[i] * size, cudaMemcpyHostToDevice, cudaStreams[i]);
        else // sync
            err = cudaMemcpy(DTI->gpuData[i], (char*)cData[i], DTI->nPerGPU[i] * size, cudaMemcpyHostToDevice); 	
 	
        if (err != cudaSuccess) 
            printf("Job%zu: Memcpy CPU2GPU failed in %zu (%zu):  %s\n", jobControl->jobId, i, gpuIds[i], cudaGetErrorString(err));


        // if sync, deallocate cData[i]
        if(commType.transmissionType == syncComm && commType.cudaMemoryType == nonPinnedComm) // non-pinned
            free(cData[i]);
        else if(commType.transmissionType == syncComm && commType.cudaMemoryType == pinnedComm) // pinned
            cudaFreeHost(cData[i]);
    }
    
    // if async transmission, sync streams and deallocate cData
    if(commType.transmissionType == asyncComm){

        #pragma omp parallel for num_threads(n_threads)
        for(i = 0; i<nGPUs; i++){

            //setGPUDevice(i);
            cudaSetDevice(gpuIds[i]);
            
            // if async communication, wait until the stream finishes moving data
            cudaStreamSynchronize(cudaStreams[i]);

            // if sync, deallocate cData[i]
            if(commType.transmissionType == asyncComm && commType.cudaMemoryType == nonPinnedComm) // non-pinned
                free(cData[i]);
            else if(commType.transmissionType == asyncComm && commType.cudaMemoryType == pinnedComm) // pinned
                cudaFreeHost(cData[i]);
        }
    }

    // deallocate termporal intermediate structure
    free(cData);
}

void cpyDataCPU2GPUstep(DTI_t *DTI){

    size_t i, j, nPartitionsGPU, nPartitionGPU, offPartitionGPU, nextIndex;
    cudaError_t err;

    size_t size = DTI->size;

    // get application state
    size_t nGPUs = getNumberOfGPUs();

    DTIDesctiption_t *description = DTI->description;
    communicationType_t commType = description->commType;

    size_t *gpuIds = getGPUIds();
    cudaStream_t *cudaStreams = getCudaStreams();
    jobControl_t *jobControl = getJobControl();
    
    // set number of threads
    size_t n_threads = 1;
    if(commType.transferCores == multiCoreComm)
        n_threads = nGPUs;


    // loop over GPUs for sending data to the GPUs
    #pragma omp parallel for num_threads(n_threads) private(nPartitionsGPU, nextIndex, nPartitionGPU, offPartitionGPU, err, i, j)
    for(i = 0; i<nGPUs; i++){

        // set device
        //setGPUDevice(i);
        cudaSetDevice(gpuIds[i]);

        // get information about partitions
        nPartitionsGPU = DTI->nPartitionsPerGPU[i];
        nextIndex = 0;

        // allocate memory in the GPU
        err = cudaMalloc(&(DTI->gpuData[i]), DTI->nPerGPU[i] * size); 
        if (err != cudaSuccess) 
            printf("Job%zu: Allocation in GPU failed in %zu (%zu):  %s\n", jobControl->jobId, i, gpuIds[i], cudaGetErrorString(err));


        // loop over partitions corresponding to the GPU and send each partition
        if(commType.transferSteps == oneStepComm){

            for(j = 0; j<nPartitionsGPU; j++){

                // get partition data
                nPartitionGPU = DTI->nPerPartition[i][j];
                offPartitionGPU = DTI->offsetPerPartition[i][j];

                if(nPartitionGPU > 0){
    
                    // copy the partition from the source to the destination
                    void *src = (char*)(DTI->cpuData) + offPartitionGPU * size;
                    void *dst = (char*)(DTI->gpuData[i]) + nextIndex * size;
                    size_t nBytes = nPartitionGPU * size;

                    // async communication
                    if(commType.transmissionType == asyncComm)
                        err = cudaMemcpyAsync(dst, src, nBytes, cudaMemcpyHostToDevice, cudaStreams[i]);
                    else // one step (pinned and not pinned)
                        err = cudaMemcpy(dst, src, nBytes, cudaMemcpyHostToDevice);
    
                    if (err != cudaSuccess) 
                        printf("Job%zu: Memcpy CPU2GPU failed in %zu (%zu):  %s\n", jobControl->jobId, i, gpuIds[i], cudaGetErrorString(err));

                    nextIndex += nPartitionGPU;
                }
            }
        }
        else { // stridedComm

            size_t w, h, dpitch, spitch;

            // number of elements per partition (supposing the same number of elements on all partitions)
            nPartitionGPU = DTI->nPerPartition[i][0]; // TODO: all partitions have the same size

            w = nPartitionGPU * size; // contiguous bytes to send on each partition
            h  = nPartitionsGPU; // number of partitions to send to GPU
            spitch = w * nGPUs; // offset between partitions on the CPU
            dpitch = w; // offset between partition on the GPU (store contiguously, so w is the offset)

            // src and dst pointers
            void *src = (char*)(DTI->cpuData) + DTI->offsetPerPartition[i][0] * size; // first element to send to gpu i
            void *dst = (char*)(DTI->gpuData[i]); // dst first position in the GPU

            if(commType.transmissionType == asyncComm)
                err = cudaMemcpy2DAsync(dst, dpitch, src, spitch, w, h, cudaMemcpyHostToDevice, cudaStreams[i]);

            else // commType.transmissionType == asyncComm && commType.transferSteps == oneStepComm
                err = cudaMemcpy2D(dst, dpitch, src, spitch, w, h, cudaMemcpyHostToDevice);
        }
    }

    // if async, sync streams
    if(commType.transmissionType == asyncComm){
        
        #pragma omp parallel for num_threads(n_threads)
        for(i = 0; i<nGPUs; i++){

            //setGPUDevice(i);
            cudaSetDevice(gpuIds[i]);
        
            // synchronize streams
            cudaStreamSynchronize(cudaStreams[i]);
        }   
    }
}

void cpyDataCPU2GPU(DTI_t *DTI){

    // copy data from the CPU to the GPUs following the information in the DTI structure
    DTIDesctiption_t *description = DTI->description;
    communicationType_t commType = description->commType;

    // two steps
    if(commType.transferSteps == twoStepsComm){ 
        cpyDataCPU2GPUtwoSteps(DTI);
    }
    // one step of 2D cuda memcpy
    else{ 
        cpyDataCPU2GPUstep(DTI);
    }
}

void cpyDataGPU2CPUtwoSteps(DTI_t *DTI){

    size_t i, j, nPartitionsGPU, nPartitionGPU, offPartitionGPU, nextIndex;
    cudaError_t err;
    size_t size = DTI->size;
    size_t nGPUs = getNumberOfGPUs();

    DTIDesctiption_t *description = DTI->description;
    communicationType_t commType = description->commType;

    size_t *gpuIds = getGPUIds();
    cudaStream_t *cudaStreams = getCudaStreams();
    jobControl_t *jobControl = getJobControl();


    // set number of threads
    size_t n_threads = 1;
    if(commType.transferCores == multiCoreComm)
        n_threads = nGPUs;


    // intermediate array to recive data from the GPUs contiguously and then redistribute
    void **cData = (void**)calloc(nGPUs, sizeof(void*));

    // allocate memory and move data to the GPU
    #pragma omp parallel for num_threads(n_threads) private(nPartitionsGPU, nextIndex, nPartitionGPU, offPartitionGPU, err, i, j)
    for(i = 0; i<nGPUs; i++){

        // set device
        //setGPUDevice(i);
        cudaSetDevice(gpuIds[i]);


        // allocate memory for cData        
        if(commType.cudaMemoryType == nonPinnedComm) // non-pinned memory
            cData[i] = (void*)calloc(DTI->nPerGPU[i], size);
        else // pinned memory
            cudaMallocHost(&cData[i], DTI->nPerGPU[i] * size);


        // move data from the GPU to the CPU in a contiguous array
        if(commType.transmissionType == asyncComm)
            err = cudaMemcpyAsync((char*)cData[i], DTI->gpuData[i], DTI->nPerGPU[i] * size, cudaMemcpyDeviceToHost, cudaStreams[i]);
        else if (commType.transmissionType == syncComm)
            err = cudaMemcpy((char*)cData[i], DTI->gpuData[i], DTI->nPerGPU[i] * size, cudaMemcpyDeviceToHost); 	
        
        if (err != cudaSuccess) 
            printf("Job%zu: Memcpy GPU2CPU failed in %zu (%zu):  %s\n", jobControl->jobId, i, gpuIds[i], cudaGetErrorString(err));


        // if sync, copy to the corresponding position
        if(commType.transmissionType == syncComm){

            // deallocate GPU memory
            err = cudaFree(DTI->gpuData[i]); 
            if (err != cudaSuccess) 
                printf("Job%zu: Deallocation failed in %zu (%zu):  %s\n", jobControl->jobId, i, gpuIds[i], cudaGetErrorString(err));


            // get the total number of elements to be transferred to the GPU
            nPartitionsGPU = DTI->nPartitionsPerGPU[i];
            nextIndex = 0;

            // loop over partitions and redistribute the contigious array
            for(j = 0; j<nPartitionsGPU; j++){

                // get partition data
                nPartitionGPU = DTI->nPerPartition[i][j];
                offPartitionGPU = DTI->offsetPerPartition[i][j];

                if(nPartitionGPU>0){
                
                    // copy the partition from the source to the destination
                    void *dst = (char*)(DTI->cpuData) + offPartitionGPU * size;
                    void *src = (char*)cData[i] + nextIndex * size;
                    size_t nBytes = nPartitionGPU * size;

                    memcpy(dst, src, nBytes);

                    nextIndex += nPartitionGPU;
                }
            }

            // deallocate cData memory
            if(commType.cudaMemoryType == nonPinnedComm)
                free(cData[i]);
            else  // pinned
                cudaFreeHost(cData[i]); 
        }
    }

    // if async, sync streams and copy data to the corresponding positions
    if(commType.transmissionType == asyncComm){
        
        #pragma omp parallel for num_threads(n_threads) private(nPartitionsGPU, nextIndex, nPartitionGPU, offPartitionGPU, err, i, j)
        for(i = 0; i<nGPUs; i++){
            
            //setGPUDevice(i);
            cudaSetDevice(gpuIds[i]);


            // get the total number of elements to be transferred to the GPU
            nPartitionsGPU = DTI->nPartitionsPerGPU[i];
            nextIndex = 0;
            
            // if async, wait until data transmission from the stream finishes
            cudaStreamSynchronize(cudaStreams[i]);

            // deallocate GPU memory
            err = cudaFree(DTI->gpuData[i]); 
            if (err != cudaSuccess) 
                printf("Job%zu: Deallocation failed in %zu (%zu):  %s\n", jobControl->jobId, i, gpuIds[i], cudaGetErrorString(err));


            // loop over partitions corresponding to the GPU and build the contiguous array
            for(j = 0; j<nPartitionsGPU; j++){

                // get partition data
                nPartitionGPU = DTI->nPerPartition[i][j];
                offPartitionGPU = DTI->offsetPerPartition[i][j];

                if(nPartitionGPU>0){
                
                    // copy the partition from the source to the destination
                    void *dst = (char*)(DTI->cpuData) + offPartitionGPU * size;
                    void *src = (char*)(cData[i]) + nextIndex * size;
                    size_t nBytes = nPartitionGPU * size;

                    memcpy(dst, src, nBytes);

                    nextIndex += nPartitionGPU;
                }
            }

            // deallocate memory depending on if the memory is pinned or not
            if(commType.cudaMemoryType == nonPinnedComm) // non-pinned
                free(cData[i]);
            else if(commType.cudaMemoryType == pinnedComm) // pinned
                cudaFreeHost(cData[i]); 
        }   
    }
    free(cData);
}

void cpyDataGPU2CPUstep(DTI_t *DTI){

    size_t i, j, nPartitionsGPU, nPartitionGPU, offPartitionGPU, nextIndex;
    cudaError_t err;
    size_t size = DTI->size;
    size_t nGPUs = getNumberOfGPUs();

    DTIDesctiption_t *description = DTI->description;
    communicationType_t commType = description->commType;
    
    size_t *gpuIds = getGPUIds();
    cudaStream_t *cudaStreams = getCudaStreams();
    jobControl_t *jobControl = getJobControl();

    // set number of threads
    size_t n_threads = 1;
    if(commType.transferCores == multiCoreComm)
        n_threads = nGPUs;

    // loop over devices to copy data from the GPUs to the CPU
    #pragma omp parallel for num_threads(n_threads) private(nPartitionsGPU, nextIndex, nPartitionGPU, offPartitionGPU, err, i, j)
    for(i = 0; i<nGPUs; i++){
        
        // set device
        //setGPUDevice(i);
        cudaSetDevice(gpuIds[i]);


        // get the total number of elements to be transferred to the GPU
        nPartitionsGPU = DTI->nPartitionsPerGPU[i];
        nextIndex = 0;
        
        // loop over partitions corresponding to the device and build the contiguous array
        if(commType.transferSteps == oneStepComm){
        
            for(j = 0; j<nPartitionsGPU; j++){

                // get partition data
                nPartitionGPU = DTI->nPerPartition[i][j];
                offPartitionGPU = DTI->offsetPerPartition[i][j];

                // if partition has elements, copy to the CPU
                if(nPartitionGPU>0){
                
                    // copy the partition from the source to the destination
                    void *dst = (char*)(DTI->cpuData) + offPartitionGPU * size;
                    void *src = (char*)(DTI->gpuData[i]) + nextIndex * size;
                    size_t nBytes = nPartitionGPU * size;

                    // send data async or sync
                    if(commType.transmissionType == asyncComm)
                        err = cudaMemcpyAsync(dst, src, nBytes, cudaMemcpyDeviceToHost, cudaStreams[i]);
                    else if(commType.transmissionType == syncComm && commType.transferSteps == oneStepComm)
                        err = cudaMemcpy(dst, src, nBytes, cudaMemcpyDeviceToHost);
                    
                    if (err != cudaSuccess) 
                        printf("Job%zu: Memcpy GPU2CPU failed in %zu (%zu):  %s\n", jobControl->jobId, i, gpuIds[i], cudaGetErrorString(err));

                    // update next partition index
                    nextIndex += nPartitionGPU;
                }
            }
        }
        else { // stridedComm

            size_t w, h, dpitch, spitch;

            // number of elements per partition (supposing the same number of elements on all partitions)
            nPartitionGPU = DTI->nPerPartition[i][0]; // TODO: all partitions have the same size

            w = nPartitionGPU * size; // contiguous bytes to send on each partition
            h  = nPartitionsGPU; // number of partitions to send to GPU
            spitch = w; // offset between partitions on the CPU
            dpitch = w * nGPUs; // offset between partition on the GPU (store in the original partitions, so let space for the data sent to other GPUs)

            // src and dst pointers
            void *dst = (char*)(DTI->cpuData) + DTI->offsetPerPartition[i][0] * size;
            void *src = (char*)(DTI->gpuData[i]);

            if(commType.transmissionType == asyncComm)
                err = cudaMemcpy2DAsync(dst, dpitch, src, spitch, w, h, cudaMemcpyDeviceToHost, cudaStreams[i]);

            else // commType.transmissionType == asyncComm && commType.transferSteps == oneStepComm
                err = cudaMemcpy2D(dst, dpitch, src, spitch, w, h, cudaMemcpyDeviceToHost);
        }


        // if sync method is used, deallocate GPU memory
        if(commType.transmissionType == syncComm)
            err = cudaFree(DTI->gpuData[i]); 
    }

    // if async, sync stream and deallocate GPU memory
    if(commType.transmissionType == asyncComm){

        #pragma omp parallel for num_threads(n_threads) private(err)
        for(i = 0; i<nGPUs; i++){
            //setGPUDevice(i);
            cudaSetDevice(gpuIds[i]);
            
            cudaStreamSynchronize(cudaStreams[i]);
            err = cudaFree(DTI->gpuData[i]); 
        }
    }
}

void cpyDataGPU2CPU(DTI_t *DTI){

    // copy data from the CPU to the GPUs following the information in the DTI structure
    DTIDesctiption_t *description = DTI->description;
    communicationType_t commType = description->commType;
   
    // two steps
    if(commType.transferSteps == twoStepsComm){ 
        cpyDataGPU2CPUtwoSteps(DTI);
    }
    // one step of 2D cuda memcpy
    else{ 
        cpyDataGPU2CPUstep(DTI);
    }
}


void cpyDataP2P(DTI_t *DTI, jobResources_t *oldResources, jobResources_t *newResources){

    DTIDesctiption_t *description = DTI->description;
    communicationType_t commType = description->commType;
    cudaError_t err;

    cudaStream_t *cudaStreams = getCudaStreams();
    jobControl_t *jobControl = getJobControl();


    size_t i,j;

    size_t nOldGPUs = oldResources->nGPUs;
    size_t nNewGPUs = newResources->nGPUs;

    // check whether any GPU repeats (copy the minimum amount of data)
    size_t *oldGPUs = oldResources->idGPUs;
    size_t *newGPUs = newResources->idGPUs;


    // find GPUs that are not repeated between old and new GPU sets
    size_t nDiff = nOldGPUs; // nOldGPUs == nNewGPUs
    size_t *src = (size_t*)calloc(nNewGPUs, sizeof(size_t));
    size_t *dst = (size_t*)calloc(nNewGPUs, sizeof(size_t));

    for(i = 0; i<nNewGPUs; i++){

        for(j = 0; j<nOldGPUs; j++){

            if(newGPUs[i] == oldGPUs[j]){
                
                nDiff--;

                // indicate that it appears in both
                src[j] = 1;
                dst[i] = 1; 
            }
        }
    }

    
    // allocate memory for nDiff temporal buffers
    void** tmpGPUBuffers = (void**)calloc(nDiff, sizeof(void*));

    // if dst has a 1 value, then, not move, if is has a 0 value, them, find where to move
    j = 0;
    size_t next = 0;
    for(i = 0; i<nNewGPUs; i++){

        if(!dst[i]){

            // find the first 
            while(src[j])
                j++;

            // move data from src[j] gpu to dst[i] gpu
            size_t srcDev = oldGPUs[j];
            size_t dstDev = newGPUs[i];

            // allocate memory for data in the dst device
            cudaSetDevice(dstDev); 
            err = cudaMalloc(&(tmpGPUBuffers[next]), DTI->nPerGPU[j] * sizeof(float)); 


            // copy data from source to destination by P2P communication
            cudaSetDevice(srcDev);

            // try to enable peer access from src to dst
            int canAccess = 0;
            cudaDeviceCanAccessPeer(&canAccess, dstDev, srcDev);

            // if can access, move data directly, else, use the CPU
            if (canAccess) {
                
                // enable peer access
                cudaDeviceEnablePeerAccess(dstDev, 0);

                // cpy data directly to a temporal buffer
                if(commType.transmissionType == asyncComm){
                    cudaMemcpyPeerAsync(tmpGPUBuffers[next], dstDev, DTI->gpuData[j], srcDev, DTI->nPerGPU[j] * (size_t)sizeof(float), cudaStreams[j]);
                }
                else{
                    cudaMemcpyPeer(tmpGPUBuffers[next], dstDev, DTI->gpuData[j], srcDev, DTI->nPerGPU[j] * (size_t)sizeof(float)); // copy P2P
                }
                cudaDeviceSynchronize();
            }
            else{

                printf(" -- [APP]: Not implemented yet!\n");
                exit(0);
            }

            j++;
        }
    }

    // if async, sync stream and deallocate GPU memory
    /*if(commType.transmissionType == asyncComm){

        for(i = 0; i<nDiff; i++){
            //setGPUDevice(i);
            cudaSetDevice(gpuIds[i]);
            
            cudaStreamSynchronize(cudaStreams[i]);
            err = cudaFree(DTI->gpuData[i]); 

            // store new pointer
            //DTI->gpuData[i] = 
        }
    }   */ 
}
