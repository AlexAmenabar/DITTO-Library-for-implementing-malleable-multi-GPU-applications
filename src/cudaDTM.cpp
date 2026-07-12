#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstring>
#include <vector>

#include "DTM.hpp"
#include "DDM.hpp"
#include "DITO_API.hpp"
#include "RMS.hpp"

// TODO: this module should be revised since it calls functions that it shouldn't

// TODO: transference only valid when there is only 1 partition per GPU

// map using the idGPUs established by the RMS
void setGPUDevice(size_t i){

    state_t *state = getState();
    cudaSetDevice(state->jobResources->idGPUs[i]);
}

void initializeStreams(jobResources_t *jobResources){

    size_t *idGPUs = jobResources->idGPUs;
    size_t nGPUs = jobResources->nGPUs;

    for(size_t i = 0; i<nGPUs; i++){

        // get GPU id
        size_t gpu = idGPUs[i];

        // set device
        cudaSetDevice(gpu);

        // create stream for the GPU
        cudaStreamCreate(&(getCudaStreams()[i]));
    }
}

void destroyStreams(jobResources_t *jobResources){
    
    size_t *idGPUs = jobResources->idGPUs;
    size_t nGPUs = jobResources->nGPUs;

    for(size_t i = 0; i<nGPUs; i++){

        // get GPU
        size_t gpu = idGPUs[i];

        // set device
        cudaSetDevice(gpu);

        // destroy stream to the GPU
        cudaStreamDestroy(getCudaStreams()[i]);
    } 
}


void initializeNCCLComm(jobResources_t *jobResources){

    size_t *idGPUs = jobResources->idGPUs;
    size_t nGPUs = jobResources->nGPUs;

    std::vector<int> devs(nGPUs);
    for (size_t i = 0; i < nGPUs; i++)
        devs[i] = static_cast<int>(idGPUs[i]);

    ncclComm_t *ncclComms = getNCCLComms();

    ncclComms = (ncclComm_t*)malloc(nGPUs * sizeof(ncclComm_t));
    appData->ncclComms = ncclComms;

    ncclCommInitAll(getNCCLComms(), nGPUs, devs.data());
}

void destroyNCCLComm(jobResources_t *jobResources){

    size_t nGPUs = jobResources->nGPUs;
    ncclComm_t* comms = getNCCLComms();

    for (size_t i = 0; i < nGPUs; i++) {
        ncclCommDestroy(comms[i]);
    }

    free(getNCCLComms());
}



// TODO: revise, not used actually
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

        // get the GPU id of the i.th GPU
        size_t gpuId = gpuIds[i];

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
        cudaSetDevice(gpuId);


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



// [P2P data redistributions] //

void reconfExpand(DTI_t *DTI){

    size_t i,j;

    // helper variables
    reconfData_t *localReconfData = reconfData; // get reconfiguration data
    size_t **gpusToSplit = localReconfData->gpusToSplit;

    DTIDesctiption_t *description = DTI->description;
    communicationType_t commType = description->commType;
    cudaError_t err;

    cudaStream_t *cudaStreams = getCudaStreams();
    state_t *state = getState();

    // get resource data
    jobResources_t *resources = state->jobResources;
    jobResources_t *reconfResources = state->reconfJobResources;
    
    size_t *idGPUs = resources->idGPUs;

    size_t nGPUs = resources->nGPUs; 
    size_t nReconfGPUs = reconfResources->nGPUs;

    // compute the number of GPUs to distribute information from each GPU
    size_t M = nReconfGPUs / nGPUs;


    // set number of threads
    size_t n_threads = 1;
    if(commType.transferCores == multiCoreComm)
        n_threads = nGPUs;

    // loop over GPUs for sending data to the GPUs
    #pragma omp parallel for num_threads(n_threads) private(err, i, j)
    for(i = 0; i<nGPUs; i++){

        // source GPU
        size_t srcDev = idGPUs[i];

        // number of partitions on each partition
        size_t nPartitions = DTI->prev_nPartitionsPerGPU[i];

        // loop over the destination GPUs and redistribute data
        for(j = 0; j<M; j++){

            // get destination GPU identifier
            size_t dstDev = gpusToSplit[i][j];
            size_t dstDevIndex = 0;

            // find the index of the GPU in the current resource array
            while(reconfResources->idGPUs[dstDevIndex] != dstDev){
                
                dstDevIndex ++;
            }

            // allocate memory in the GPU
            cudaSetDevice(dstDev);
            err = cudaMalloc(&(DTI->gpuData[dstDevIndex]), DTI->nPerGPU[dstDevIndex] * sizeof(float)); 

            if (err != cudaSuccess) 
                printf("cuda malloc failed in expand: %s\n", cudaGetErrorString(err));

            // check and enable peer access
            int canAccess = 0;
            cudaSetDevice(srcDev);
            cudaDeviceCanAccessPeer(&canAccess, srcDev, dstDev);

            if(canAccess)
                cudaDeviceEnablePeerAccess(dstDev, 0);

            // not strided reconfiguration
            if(commType.transferSteps != stridedComm){
                
                // loop over partitions and copy each one
                for(size_t nP = 0; nP < nPartitions; nP++){

                    // get the number of elements in the partition
                    size_t prevN = DTI->prev_nPerPartition[i][nP];
                    size_t n = prevN / M;

                    // compute offsets in source and destination arrays
                    size_t offDst = n * nP; // each partition has n elements, stored contiguously
                    size_t offSrc = prevN * nP + n * j; // each partition has prevN elements [prevN * nP], and the partition is divided into M GPUs, so compute offset using [n * j]
                    // j because is the j.th GPU to redistribute data from this, and n because each new subpartition has n elements

                    //printf(" canAccess = %d from %zu to %zu: %zu elements (prev N = %zu, M = %zu)\n", canAccess, srcDev, dstDev, prevN, n, M);
                    //printf(" -- prev[%zu:%zu], post[%zu]\n", srcDev, offSrc, dstDev);
                    //fflush(stdout);


                    // copy the partition from the source to the destination
                    void *dst = (char*)(DTI->gpuData[dstDevIndex]) + offDst * DTI->size;
                    void *src = (char*)(DTI->prevGpuData[i]) + offSrc * DTI->size;
                    size_t nBytes = n * DTI->size;

                    // if can access, move data directly, else, use the CPU
                    if (canAccess){

                        // cpy data directly to a temporal buffer
                        if(commType.transmissionType == asyncComm){
                            
                            err = cudaMemcpyPeerAsync(dst, dstDev, src, srcDev, nBytes, cudaStreams[i]);
                        }
                        else{

                            err = cudaMemcpyPeer(dst, dstDev, src, srcDev, nBytes); // copy P2P
                            
                            // free old GPU data
                            cudaFree(DTI->prevGpuData[i]);
                        }
                    }
                    else if(srcDev == dstDev){

                        // cpy data directly to a temporal buffer
                        if(commType.transmissionType == asyncComm){
                            
                            err = cudaMemcpyAsync(dst, src, nBytes, cudaMemcpyDeviceToDevice, cudaStreams[i]);
                            //cudaMemcpyPeerAsync(dst, dstDev, src, srcDev, DTI->nPerGPU[i] * (size_t)sizeof(float), cudaStreams[i]);
                        }
                        else{

                            err = cudaMemcpy(dst, src, nBytes, cudaMemcpyDeviceToDevice);
                            //cudaMemcpyPeer(dst, dstDev, src, srcDev, DTI->nPerGPU[i] * (size_t)sizeof(float)); // copy P2P
                            
                            // free old GPU data
                            cudaFree(DTI->prevGpuData[i]);
                        }

                        if (err != cudaSuccess) 
                            printf("Cpy failed in expand: %s\n", cudaGetErrorString(err));
                    }
                    else{

                        printf(" [APP]: P2P not supported between GPUs %zu and %zu. Exit\n", srcDev, dstDev);
                        exit(1);
                    }
                }
            }
            // strided communication function
            else{

                size_t w, h, dpitch, spitch, size = DTI->size;

                // number of elements per partition (supposing the same number of elements on all partitions)
                size_t prev_nPartitionGPU = DTI->prev_nPerPartition[i][0];
                size_t nPartitionGPU = DTI->nPerPartition[i][0]; // TODO: all partitions have the same size

                w = nPartitionGPU * size; // contiguous bytes to send on each partition (nPartitionGPU is the new partition size)
                h  = nPartitions; // number of partitions to send to GPU
                spitch = prev_nPartitionGPU * size; // offset between partitions on the source GPU (the partition size in the previous configuration was prev_nPartitionGPU)
                dpitch = nPartitionGPU * size; // offset between partition on the GPU (store in the original partitions, so let space for the data sent to other GPUs)

                /* NOTE
                Although from the perspective of the global data array, each GPU has an offset that can be higher than 0, the offsets of the firts local element for each
                GPU is 0, therefore, the source structure offset do not take into account the global offsets of the data element. src is [j * nPartitionGPU] because we
                start always from the first element, and then the offset depends on the new subpartition. If the previous size is prevN, and the new one n, then, the offset
                is [n * j], since we are dividing the partition in subpartitions
                */

                // src and dst pointers
                void *dst = (char*)(DTI->gpuData[dstDevIndex]);
                void *src = (char*)(DTI->prevGpuData[i])  + (j * nPartitionGPU) * size;

                if(commType.transmissionType == asyncComm){
                    err = cudaMemcpy2DAsync(dst, dpitch, src, spitch, w, h, cudaMemcpyDeviceToDevice, cudaStreams[i]);
                }

                else { // commType.transmissionType == asyncComm && commType.transferSteps == oneStepComm
                    err = cudaMemcpy2D(dst, dpitch, src, spitch, w, h, cudaMemcpyDeviceToDevice);
                    cudaFree(DTI->prevGpuData[i]);
                }

                if (err != cudaSuccess) 
                    printf("Cpy failed in expand: %s\n", cudaGetErrorString(err));
            }
        }        
    }

    //printf(" Data moved!\n");
    //fflush(stdout);

    // deallocate old memory
    if(commType.transmissionType == asyncComm){

        #pragma omp parallel for num_threads(n_threads) private(err, i)
        for(i = 0; i<nGPUs; i++){

            cudaSetDevice(idGPUs[i]);
            
            cudaStreamSynchronize(cudaStreams[i]);
            err = cudaFree(DTI->prevGpuData[i]); 

            if (err != cudaSuccess) 
                printf("Cuda Free failed: %s\n", cudaGetErrorString(err));
        }
    }
}

void reconfShrink(DTI_t *DTI){

    size_t i,j;

    // helper variables
    reconfData_t *localReconfData = reconfData; // get reconfiguration data
    size_t **gpusToSplit = localReconfData->gpusToSplit;

    DTIDesctiption_t *description = DTI->description;
    communicationType_t commType = description->commType;
    cudaError_t err;

    cudaStream_t *cudaStreams = getCudaStreams();
    state_t *state = getState();

    // get resource data
    jobResources_t *resources = state->jobResources;
    jobResources_t *reconfResources = state->reconfJobResources;
    
    size_t *idGPUs = resources->idGPUs;
    size_t *idReconfGPUs = reconfResources->idGPUs;

    size_t nGPUs = resources->nGPUs; 
    size_t nReconfGPUs = reconfResources->nGPUs;

    // compute the number of GPUs to distribute information from each GPU
    size_t M =  nGPUs / nReconfGPUs;

    // allocate memory in the destination GPUs
    for(i = 0; i<nReconfGPUs; i++){

        //printf("Allocating on GPU %zu (%zu): %zu\n", idReconfGPUs[i], i, DTI->nPerGPU[i] * sizeof(float));

        // allocate memory in the destination GPU
        cudaSetDevice(idReconfGPUs[i]);
        err = cudaMalloc(&(DTI->gpuData[i]), DTI->nPerGPU[i] * sizeof(float)); 

        if (err != cudaSuccess) 
            printf("Cuda malloc failed: %s\n", cudaGetErrorString(err));
    }


    // set number of threads
    size_t n_threads = 1;
    if(commType.transferCores == multiCoreComm)
        n_threads = nGPUs;

    // loop over GPUs for sending data to the GPUs
    #pragma omp parallel for num_threads(n_threads) private(err, i, j)
    for(i = 0; i<nGPUs; i++){

        err = cudaSuccess;


        // source GPU
        size_t srcDev = idGPUs[i];

        // number of partitions in the data
        size_t nPartitions = DTI->prev_nPartitionsPerGPU[i];

        // get destination GPU identifier
        size_t dstDev = gpusToSplit[i][0]; // there is only one device
        size_t dstDevIndex = 0;

        // find the index of the GPU in the current resource array
        while(reconfResources->idGPUs[dstDevIndex] != dstDev){
            
            dstDevIndex ++;
        }


        // check and enable peer access
        int canAccess = 0;
        cudaSetDevice(srcDev);
        cudaDeviceCanAccessPeer(&canAccess, srcDev, dstDev);

        if(canAccess)
            cudaDeviceEnablePeerAccess(dstDev, 0);


        // loop over the partitions and copy each one
        if(commType.transferSteps != stridedComm){

            for(size_t nP = 0; nP < nPartitions; nP++){

                // get the number of elements in the previous and new partitions
                size_t prevN = DTI->prev_nPerPartition[i][nP];
                size_t n = prevN * M;

                // get offset of the partition
                size_t offset = (DTI->prev_offsetPerPartition[i][nP]) % (DTI->N / nPartitions);
                offset = (offset / prevN);

                if(nReconfGPUs > 1)
                    offset %= M;

                //printf(" Prev offset of GPU %zu (%zu) = %zu, normalized offset = %zu  (prevN = %zu, n = %zu, prev_offset = %zu, N = %zu, nPartitions = %zu)\n", 
                //        srcDev, i, DTI->prev_offsetPerPartition[i][nP], offset, prevN, n, DTI->prev_offsetPerPartition[i][nP], DTI->N, nPartitions);
                //fflush(stdout);


                // compute offsets
                //size_t offDst = nElements * M * nP + dstDevBaseOffset + DTI->prev_offsetPerPartition[i][nP];
                size_t offDst = n * nP + offset * prevN;
                size_t offSrc = prevN * nP;

                if(err != cudaSuccess){
                    printf(" canAccess = %d from %zu to %zu: %zu elements (prev N = %zu, M = %zu)\n", canAccess, srcDev, dstDev, prevN, n, M);
                    printf(" -- prev[%zu:%zu], post[%zu:%zu]\n", srcDev, offSrc, dstDev, offDst);
                    fflush(stdout);
                }


                // copy the partition from the source to the destination
                void *dst = (char*)(DTI->gpuData[dstDevIndex]) + offDst * DTI->size;
                void *src = (char*)(DTI->prevGpuData[i]) + offSrc * DTI->size;
                size_t nBytes = prevN * DTI->size;

                // if can access, move data directly, else, use the CPU
                if (canAccess){
                    
                    // cpy data directly to a temporal buffer
                    if(commType.transmissionType == asyncComm){
                        
                        err = cudaMemcpyPeerAsync(dst, dstDev, src, srcDev, nBytes, cudaStreams[i]);

                        if (err != cudaSuccess) 
                            printf("Mem cpy failed: %s\n", cudaGetErrorString(err));
                    }
                    else{

                        cudaMemcpyPeer(dst, dstDev, src, srcDev, nBytes); // copy P2P
                        
                        // free old GPU data
                        cudaFree(DTI->prevGpuData[i]);
                    }
                }
                else if(srcDev == dstDev){

                    // cpy data directly to a temporal buffer
                    if(commType.transmissionType == asyncComm){
                        
                        err = cudaMemcpyAsync(dst, src, nBytes, cudaMemcpyDeviceToDevice, cudaStreams[i]);
                        //cudaMemcpyPeerAsync(dst, dstDev, src, srcDev, DTI->nPerGPU[i] * (size_t)sizeof(float), cudaStreams[i]);

                        if (err != cudaSuccess) 
                            printf("Mem cpy (self) failed: %s\n", cudaGetErrorString(err));
                    }
                    else{

                        err = cudaMemcpy(dst, src, nBytes, cudaMemcpyDeviceToDevice);
                        //cudaMemcpyPeer(dst, dstDev, src, srcDev, DTI->nPerGPU[i] * (size_t)sizeof(float)); // copy P2P
                        
                        // free old GPU data
                        cudaFree(DTI->prevGpuData[i]);
                    }
                }
            }
        }
        else{

            size_t w, h, dpitch, spitch, size = DTI->size;

            // number of elements per partition (supposing the same number of elements on all partitions)
            size_t prev_nPartitionGPU = DTI->prev_nPerPartition[i][0];
            size_t nPartitionGPU = DTI->nPerPartition[dstDevIndex][0]; // TODO: all partitions have the same size

            w = prev_nPartitionGPU * size; // contiguous bytes to send on each partition
            h  = nPartitions; // number of partitions to send to GPU
            spitch = prev_nPartitionGPU * size; // offset between partitions on the source GPU
            dpitch = nPartitionGPU * size; // offset between partition on the GPU (store in the original partitions, so let space for the data sent to other GPUs)


            // get offset of the partition
            // this offset is [0.D-1], where, it means the logical position of the segment the GPU receives. For example,
            // if the offset is 0, it means that segment is the first subsegment in the new segment. 1 is the following...
            // this set of computations get the offset, which is later used for computing the index in which data must be copied
            size_t offset = (DTI->prev_offsetPerPartition[i][0]) % (DTI->N / nPartitions);
            offset = (offset / prev_nPartitionGPU);

            if(nReconfGPUs > 1)
                offset %= M;

            if(err != cudaSuccess){
            
                printf(" Src dev = %zu, dst dev = %zu, offset %zu (prevN %zu, N %zu, w %zu, h %zu, spitch %zu dpitch %zu)\n", 
                    srcDev, dstDev, offset, prev_nPartitionGPU, nPartitionGPU, w / size, h, spitch / size, dpitch / size);
                fflush(stdout);
            }

            // src and dst pointers
            void *dst = (char*)(DTI->gpuData[dstDevIndex]) + offset * prev_nPartitionGPU * size;
            void *src = (char*)(DTI->prevGpuData[i]);

            if(commType.transmissionType == asyncComm){
                err = cudaMemcpy2DAsync(dst, dpitch, src, spitch, w, h, cudaMemcpyDeviceToDevice, cudaStreams[i]);
            
                if (err != cudaSuccess) 
                    printf("Mem cpy (2D) failed: %s\n", cudaGetErrorString(err));
            }

            else{
                err = cudaMemcpy2D(dst, dpitch, src, spitch, w, h, cudaMemcpyDeviceToDevice);
                err = cudaFree(DTI->prevGpuData[i]); 
            }
        }
    }        

    //printf(" Data moved!\n");
    //fflush(stdout);

    // deallocate old memory
    if(commType.transmissionType == asyncComm){

        #pragma omp parallel for num_threads(n_threads) private(err, i, j)
        for(i = 0; i<nGPUs; i++){

            cudaSetDevice(idGPUs[i]);
            
            cudaStreamSynchronize(cudaStreams[i]);
            err = cudaFree(DTI->prevGpuData[i]); 

            if (err != cudaSuccess) 
                printf("Free failed: %s\n", cudaGetErrorString(err));
        }
    }
}

void reconfN2N(DTI_t *DTI){

    size_t i,j;

    // helper variables
    reconfData_t *localReconfData = reconfData; // get reconfiguration data
    size_t **gpusToSplit = localReconfData->gpusToSplit;

    DTIDesctiption_t *description = DTI->description;
    communicationType_t commType = description->commType;
    cudaError_t err;

    cudaStream_t *cudaStreams = getCudaStreams();
    state_t *state = getState();

    // get resource data
    jobResources_t *resources = state->jobResources;
    jobResources_t *reconfResources = state->reconfJobResources;
    
    size_t *idGPUs = resources->idGPUs;
    size_t nGPUs = resources->nGPUs; 


    // set number of threads
    size_t n_threads = 1;
    if(commType.transferCores == multiCoreComm)
        n_threads = nGPUs;

    // loop over GPUs for sending data to the GPUs
    #pragma omp parallel for num_threads(n_threads) private(err, i, j)    
    for(i = 0; i<nGPUs; i++){

        // source GPU
        size_t srcDev = idGPUs[i];

        // number of partitions in the data
        size_t nPartitions = DTI->prev_nPartitionsPerGPU[i];


        // get destination GPU identifier
        size_t dstDev = gpusToSplit[i][0];
        size_t dstDevIndex = 0;

        // find the index of the GPU in the current resource array
        while(reconfResources->idGPUs[dstDevIndex] != dstDev){
            
            dstDevIndex ++;
        }


        // allocate memory in the GPU
        cudaSetDevice(dstDev);
        err = cudaMalloc(&(DTI->gpuData[dstDevIndex]), DTI->nPerGPU[dstDevIndex] * sizeof(float)); 

        if (err != cudaSuccess) 
            printf("Cpy failed in expand: %s\n", cudaGetErrorString(err));


        // get the number of elements in the partition
        size_t n = DTI->prev_nPerPartition[i][0];
        
        // copy the partition from the source to the destination
        void *dst = (char*)(DTI->gpuData[dstDevIndex]);
        void *src = (char*)(DTI->prevGpuData[i]);
        size_t nBytes = n * nPartitions * DTI->size;


        // check and enable peer access
        int canAccess = 0;
        cudaSetDevice(srcDev);
        cudaDeviceCanAccessPeer(&canAccess, srcDev, dstDev);

        if(canAccess)
            cudaDeviceEnablePeerAccess(dstDev, 0);

        // if can access, move data directly, else, use the CPU
        if (canAccess){
            
            // cpy data directly to a temporal buffer
            if(commType.transmissionType == asyncComm){
                
                err = cudaMemcpyPeerAsync(dst, dstDev, src, srcDev, nBytes, cudaStreams[i]);
            }
            else{

                err = cudaMemcpyPeer(dst, dstDev, src, srcDev, nBytes); // copy P2P
                
                // free old GPU data
                cudaFree(DTI->prevGpuData[i]);
            }

            if (err != cudaSuccess) 
                printf("Cpy failed in expand: %s\n", cudaGetErrorString(err));
        }
        else if(srcDev == dstDev){

            // cpy data directly to a temporal buffer
            if(commType.transmissionType == asyncComm){
                
                cudaMemcpyAsync(dst, src, nBytes, cudaMemcpyDeviceToDevice, cudaStreams[i]);
                //cudaMemcpyPeerAsync(dst, dstDev, src, srcDev, DTI->nPerGPU[i] * (size_t)sizeof(float), cudaStreams[i]);
            }
            else{

                cudaMemcpy(dst, src, nBytes, cudaMemcpyDeviceToDevice);
                //cudaMemcpyPeer(dst, dstDev, src, srcDev, DTI->nPerGPU[i] * (size_t)sizeof(float)); // copy P2P
                
                // free old GPU data
                cudaFree(DTI->prevGpuData[i]);
            }
        }
    }

    // deallocate old memory
    if(commType.transmissionType == asyncComm){

        #pragma omp parallel for num_threads(n_threads) private(err, i, j)
        for(i = 0; i<nGPUs; i++){

            cudaSetDevice(idGPUs[i]);
            
            cudaStreamSynchronize(cudaStreams[i]);
            err = cudaFree(DTI->prevGpuData[i]); 
        }
    }
}