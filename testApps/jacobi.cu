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





// JACOBI STENCIL 

// jacobi_multi_gpu_naive.cu
// Naive multi-GPU Jacobi example using NCCL halo exchange.
// NOTE: This is a standalone template illustrating the algorithm.
// It is not yet wired into DITTO.

#include <cstdio>


__global__ void jacobiKernel(const float* A, const float* upperHalo, const float* lowerHalo, float* Anew, size_t width, size_t localRows, size_t gpu, size_t nGPUs) {
    
    size_t x = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t y = (size_t)blockIdx.y * (size_t)blockDim.y + (size_t)threadIdx.y;

    if (x < width - 1 && y < localRows && x > 0) {
        
        size_t id = y * width + x;

        float up = 0.0, down = 0.0, right = 0.0, left = 0.0;
        
        // up
        if(gpu > 0){

            up = y == 0 ? upperHalo[x] : A[id - width];
        }
        else{

            if(y > 0){

                up = A[id - width];
            }
        }

        // down
        if(gpu < nGPUs - 1){

            down = y == (localRows-1) ? lowerHalo[x] : A[id + width];
        }
        // gpu == nGPUs - 1
        else{

            if(y < localRows-1){

                down = A[id + width];
            }
        }

        left = A[id-1];
        right = A[id+1];

        if(gpu == 0 && y == 0 || gpu == nGPUs-1 && y == localRows-1){
            Anew[id] = A[id];
        }
        else{
            Anew[id] = 0.25f * (up + down + left + right);
        }
    }
}



__global__ void copyKernel(const float *src, float *dst, size_t width, size_t localRows){
    
    size_t x = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t y = (size_t)blockIdx.y * (size_t)blockDim.y + (size_t)threadIdx.y;

    if (x < width && y < localRows) {

        size_t id = y * width + x;
        dst[id] = src[id];
    }
}


void exchangeHalos(float* dA, float* upperHalo, float* lowerHalo, size_t nCols, size_t nRows, size_t rank, size_t nranks, ncclComm_t comm, cudaStream_t stream){
    
    ncclGroupStart();

    if (rank > 0) {

        ncclSend(dA, nCols, ncclFloat, rank - 1, comm, stream);
        ncclRecv(upperHalo, nCols, ncclFloat, rank - 1, comm, stream);
    }

    if (rank < nranks - 1) {
        
        ncclSend(dA + (nRows-1) * nCols, nCols, ncclFloat, rank + 1, comm, stream);
        ncclRecv(lowerHalo, nCols, ncclFloat, rank + 1, comm, stream);
    }

    ncclGroupEnd();
}


void simulateIterativeJacobi(appStruct_t *data){
    
    // get number of time steps and state
    size_t i;
    size_t T = data->T;

    double etComputation = 0.0, etCommunication = 0.0;
    struct timespec startComputation, endComputation;
    struct timespec startCommunication, endCommunication;


    // allocate memory for the halos in the GPUs
    float **dANew = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));
    float **upperHalo = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));
    float **lowerHalo = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));

    // loop over GPUs and allocate memory for helper variables
    for(size_t i = 0; i<getState()->jobResources->nGPUs; i++){

        // each GPU has N / nGPUs elements
        allocateMemoryGPU(&(dANew[i]), getState()->jobResources->idGPUs[i], data->nRows / getState()->jobResources->nGPUs * data->nCols);
        
        // allocate memory for halos
        allocateMemoryGPU(&(upperHalo[i]), getState()->jobResources->idGPUs[i], data->nCols);
        allocateMemoryGPU(&(lowerHalo[i]), getState()->jobResources->idGPUs[i], data->nCols);
    }
    

    // simulate iterations
    for(size_t t = 0; t<T; t++){

        // reconfiguration point, check if there is any pending reconfiguration
        // if there is a reconfiguration, reallocate memory for halos
        if(checkIfReconfiguration(getJobControl())){

            // deallocate memory of old halos
            for(size_t i = 0; i<getState()->jobResources->nGPUs; i++){

                deallocateMemoryGPU(dANew[i], getState()->jobResources->idGPUs[i]);
                deallocateMemoryGPU(upperHalo[i], getState()->jobResources->idGPUs[i]);
                deallocateMemoryGPU(lowerHalo[i], getState()->jobResources->idGPUs[i]);
            }

            free(dANew);
            free(upperHalo);
            free(lowerHalo);

            // reconfigure and manage NCCL communicators
            freeNCCLComm(getState()->jobResources);
            reconfigure(CPU);
            initNCCLComm(getState()->jobResources);

            // allocate memory for the halos in the GPUs

            dANew = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));
            upperHalo = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));
            lowerHalo = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));

            for(size_t i = 0; i<getState()->jobResources->nGPUs; i++){
                
                allocateMemoryGPU(&(dANew[i]), getState()->jobResources->idGPUs[i], data->nRows / getState()->jobResources->nGPUs * data->nCols);

                allocateMemoryGPU(&(upperHalo[i]), getState()->jobResources->idGPUs[i], data->nCols);
                allocateMemoryGPU(&(lowerHalo[i]), getState()->jobResources->idGPUs[i], data->nCols);
            }
        }

        // copy data from global variables to local variables for enabling visibility to openMP threads (they are thread private)
        public_APP_Data_t *localData = appData;
        DTI_t **localArrDTI = arrDTI;
        size_t localnDTI = nDTI;
        size_t localmaxDTI = maxDTI;
        ncclComm_t *localComms = getNCCLComms();
        cudaStream_t *localStreams = getCudaStreams();

        // get the information of the GPUs
        size_t nGPUs = getNumberOfGPUs();
        size_t *idGPUs = getGPUIds();

        size_t nRowsPerGPU = data->nRows / getState()->jobResources->nGPUs;
        size_t nCols = data->nCols;

        dim3 block(32,32);
        dim3 grid((data->nCols)/32 + 1, (data->nRows)/32 + 1);


        if(nGPUs > 1){
            
            clock_gettime(CLOCK_MONOTONIC, &startCommunication);

            #pragma omp parallel for num_threads (nGPUs) private(i)
            for (i = 0; i < nGPUs; i++) {
                
                // get global data copies
                appData = localData;
                arrDTI = localArrDTI;
                nDTI = localnDTI;
                maxDTI = localmaxDTI;
                ncclComm_t *comms = localComms;
                cudaStream_t *streams = localStreams;
                // set device
                cudaSetDevice(idGPUs[i]);

                // get the pointer to the GPU data
                float *dA = (float*)(getDTIByIndex(0)->gpuData[i]);

                // exchange halos  
                exchangeHalos(dA, upperHalo[i], lowerHalo[i], nCols, nRowsPerGPU, i, nGPUs, comms[i], streams[idGPUs[i]]);
                cudaError_t err = cudaStreamSynchronize(localStreams[idGPUs[i]]); 

                if(err != cudaSuccess){
                    printf("Failed in exchangeHalos: %s\n", cudaGetErrorString(err));
                    fflush(stdout);
                }          
            }
            clock_gettime(CLOCK_MONOTONIC, &endCommunication);
            etCommunication += (endCommunication.tv_sec - startCommunication.tv_sec) + (endCommunication.tv_nsec - startCommunication.tv_nsec) / 1e9;
        }

        clock_gettime(CLOCK_MONOTONIC, &startComputation);
        #pragma omp parallel for num_threads (nGPUs)
        for (i = 0; i < nGPUs; i++) {
            
            // get global data copies
            appData = localData;
            arrDTI = localArrDTI;
            nDTI = localnDTI;
            maxDTI = localmaxDTI;
            ncclComm_t *comms = localComms;
            cudaStream_t *streams = localStreams;
            
            // get the pointer to the GPU data
            float *dA = (float*)(getDTIByIndex(0)->gpuData[i]);

            // set device
            cudaSetDevice(idGPUs[i]);
            jacobiKernel<<<grid, block, 0, streams[idGPUs[i]]>>>(dA, upperHalo[i], lowerHalo[i], dANew[i], nCols, nRowsPerGPU, i, nGPUs);
            cudaError_t err = cudaStreamSynchronize(localStreams[idGPUs[i]]); 

            err = cudaGetLastError();
            if (err != cudaSuccess)
                printf("Kernel launch failed: %s\n", cudaGetErrorString(err));


            if(err != cudaSuccess){
                printf("Failed in exchangeHalos: %s\n", cudaGetErrorString(err));
                fflush(stdout);
            }          
        }  
        clock_gettime(CLOCK_MONOTONIC, &endComputation);
        etComputation += (endComputation.tv_sec - startComputation.tv_sec) + (endComputation.tv_nsec - startComputation.tv_nsec) / 1e9;

        #pragma omp parallel for num_threads(nGPUs) private(i)
        for (i = 0; i < nGPUs; i++) {
         
            cudaSetDevice(idGPUs[i]);
            cudaStreamSynchronize(localStreams[idGPUs[i]]); 

            // get the pointer to the GPU data
            float *dA = (float*)(getDTIByIndex(0)->gpuData[i]);

            // Pointer swap logic fixed per GPU
            localArrDTI[0]->gpuData[i] = (void*)dANew[i];
            dANew[i] = dA; 
        }
    }

    pthread_mutex_lock(&(printLock));
    printf("%lf %lf %zu\n", etComputation, etCommunication, T);
    fflush(stdout);
    pthread_mutex_unlock(&(printLock));

    for (i = 0; i < getState()->jobResources->nGPUs; i++) {

        deallocateMemoryGPU(dANew[i], getState()->jobResources->idGPUs[i]);
        deallocateMemoryGPU(upperHalo[i], getState()->jobResources->idGPUs[i]);
        deallocateMemoryGPU(lowerHalo[i], getState()->jobResources->idGPUs[i]);
    }
}


// launch jacobi
void launch_jacobi_stencil(int argc, void* argv[]){

    size_t i;

    // initialize DITTO environment
    // temporal: simulate information received from the scheduler: number of GPUs and identifiers of the GPUs (pass argv to the initDITTO function?)
    initDITTO(argv[argc+1]);
    initNCCLComm(getState()->jobResources);

    appStruct_t *appData = (appStruct_t*)calloc(1, sizeof(appStruct_t));
    appData->nRows = *(size_t*)argv[0]; // number of elements in the matrix
    appData->nCols = *(size_t*)argv[1]; // number of elements in the matrix
    appData->T = *(size_t*)argv[2]; // number of iterations
    appData->s = *(size_t*)argv[3]; 

    // communication info
    size_t pinned = *(size_t*)argv[4];
    size_t async = *(size_t*)argv[5];
    size_t steps = *(size_t*)argv[6];
    size_t cores = *(size_t*)argv[7];
    
    size_t testing = *(size_t*)argv[8];

    appData->malleable = *(size_t*)argv[argc];

    
    // allocate memory for App Data
    if(pinned) // pinned memory
        cudaMallocHost(&appData->arr, appData->nRows * appData->nCols * sizeof(float));
    else
        appData->arr = (float*)calloc(appData->nRows * appData->nCols, sizeof(float));
    
    // initialize data
    for(i = 0; i<appData->nRows * appData->nCols; i++){
        appData->arr[i] = (float)i;//(float)rand()/(float)(RAND_MAX);
    }
    
    // if testing
    if(testing){

        for(i = 0; i<appData->nRows * appData->nCols; i++){
            appData->arr[i] = 0.0;//(float)rand()/(float)(RAND_MAX);
        }

        appData->arr[0] = 100.0;
        appData->arr[1] = 100.0;
        appData->arr[2] = 100.0;
        appData->arr[3] = 100.0;
        appData->arr[4] = 100.0;
        appData->arr[5] = 100.0;
    }


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
        appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->nRows * appData->nCols, sizeof(float), "appData", initializeDTIDescription(simple, ordered, commType));
    else
        appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->nRows * appData->nCols, sizeof(float), "appData", initializeDTIDescription(simple, ordered, commType));



    // configure all DTIs for automatic data transference // TODO: configure in initialization
    configureDTIs(getState()->jobResources, NULL, CPU2GPU); // number of GPUs, old number of GPUs

    // send data to the GPU
    transferDataCPU2GPU();

    // wait for signal before starting
    simulateIterativeJacobi(appData);

    // transfer data fron the GPUs to the CPU
    transferDataGPU2CPU();

    //printf(" [RES]: %zu bytes; from %zu to %zu; %lf %lf\n", 
    //        appData->N * sizeof(float), getGPUIds()[0], getGPUIds()[1], appData->communicationTimeSrcDst, appData->communicationTimeDstSrc);

    if(testing){

        for(i = 0; i<appData->nRows; i++){
            
            for(size_t j = 0; j<appData->nCols; j++){

                printf("%f ", appData->arr[i * appData->nCols + j]);//(float)rand()/(float)(RAND_MAX);
            }
            printf("\n");
        }
    }


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


