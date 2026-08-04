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
#include <cuComplex.h>

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

#include <cuComplex.h>
#include <math.h>

// reverse bits
__device__ __forceinline__ size_t reverseBits(size_t x, size_t nBits){
    
    size_t y = 0;

    // copy x in y in reverse order
    for (size_t i = 0; i < nBits; i++) {
        y <<= 1; // move one bit to the left
        y |= (x & 1); // store the first bit of x
        x >>= 1; // move one bit to the right
    }

    return y;
}

// reverse bits and swap positions
__global__ void bitReverseKernel(float *real, float *imag, size_t localRows, size_t nCols, size_t logN){
    
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    size_t total = localRows * nCols;

    if (tid < total){

        // row and column
        size_t row = tid / nCols;
        size_t col = tid % nCols;

        // get the bit reversed column
        size_t rev = reverseBits(col, logN);

        // swap (two elements are swapped at once, so only if col < rev)
        if (rev > col){

            // get element indexes
            size_t idx1 = row * nCols + col;
            size_t idx2 = row * nCols + rev;

            // temporal value to storage
            float tmp;

            // swap real value
            tmp = real[idx1];
            real[idx1] = real[idx2];
            real[idx2] = tmp;

            // swap imaginary value
            tmp = imag[idx1];
            imag[idx1] = imag[idx2];
            imag[idx2] = tmp;
        }
    }
}

// perform the DFTs following the butterfly strategy to decompose the work
__global__ void butterflyKernel(float *real, float *imag, size_t localRows, size_t nCols, size_t stage){
    
    // One thread = one butterfly
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    size_t butterfliesPerRow = nCols / 2;
    size_t totalButterflies  = localRows * butterfliesPerRow; // number of butterflies per stage

    if (tid < totalButterflies){

        // compute row and butterfly indices
        size_t row = tid / butterfliesPerRow; // row identifier
        size_t butterfly = tid % butterfliesPerRow; // butterfly identifier

        // stride and distance
        size_t stride   = 1 << stage; // distance between the two butterfly partners
        size_t distance = stride * 2;//stride << 1; // size of the of one FFT block at the current stage 

        // group and offset
        size_t group  = butterfly / stride; // the FFT block id
        size_t offset = butterfly % stride; // offset in the FFT block 

        // compute indices of the two elements for the DFT
        size_t i = group * distance + offset; // block begins in group * distance, first element is in the offset
        size_t j = i + stride; // second elements is + stride
        size_t idx1 = row * nCols + i;
        size_t idx2 = row * nCols + j;

        // compute W
        float angle = -2.0f * 3.1416 * offset / distance;

        // compute W
        float wr = cosf(angle);
        float wi = sinf(angle);

        // 
        float ar = real[idx1];
        float ai = imag[idx1];
        
        float br = real[idx2];
        float bi = imag[idx2];

        // multiply W * b
        float tr = wr * br - wi * bi;
        float ti = wr * bi + wi * br;

        // a + t; b - t
        real[idx1] = ar + tr;
        imag[idx1] = ai + ti;

        real[idx2] = ar - tr;
        imag[idx2] = ai - ti;
    }
}


// perform the DFTs following the butterfly strategy to decompose the work
__global__ void pack(float *real, float *imag, size_t nRows, size_t nCols, size_t nGPUs, float *sendBufferReal, float *sendBufferImag, size_t N){
    
    // One thread = one butterfly
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N){
        
        size_t nRowsPerGPU = nRows / nGPUs;
        size_t nColsPerGPU = nCols / nGPUs;
        size_t sliceSize = nRowsPerGPU * nColsPerGPU;

        // get row and column
        size_t row = tid / nCols;
        size_t col = tid % nCols;

        // get destination gpu and local column
        size_t dstGPU   = col / nColsPerGPU; // gpu offset (slice)
        size_t localCol = col % nColsPerGPU; // column offset (inside the row/slice)

        // compute the destiantion index and store in the send buffer
        size_t dstIndex = dstGPU * sliceSize + row * nColsPerGPU + localCol;
        sendBufferReal[dstIndex] = real[row*nCols + col];
        sendBufferImag[dstIndex] = imag[row*nCols + col];
    }
}

// perform the DFTs following the butterfly strategy to decompose the work
__global__ void unpack(float *real, float *imag, size_t nRows, size_t nCols, size_t nGPUs, float *recvBufferReal, float *recvBufferImag, size_t N){
    
    // One thread = one butterfly
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N){
        
        size_t nRowsPerGPU = nRows / nGPUs;
        size_t nColsPerGPU = nCols / nGPUs;
        size_t sliceSize = nRowsPerGPU * nColsPerGPU;

        // get row and column
        size_t row = tid % nColsPerGPU;
        size_t col = tid / nColsPerGPU;

        // get destination gpu and local column
        size_t srcGPU   = col / nColsPerGPU; // gpu offset (slice)
        size_t localCol = col % nColsPerGPU; // column offset (inside the row/slice)

        // compute the destiantion index and store in the send buffer
        size_t srcIndex = tid;
        size_t dstIndex = row * nCols + col;
        real[dstIndex] = recvBufferReal[srcIndex];
        imag[dstIndex] = recvBufferImag[srcIndex];
    }
}

// perform the DFTs following the butterfly strategy to decompose the work
__global__ void printMatrix(float *real, float *imag, size_t nRows, size_t nCols, size_t nGPUs, float *recvBufferReal, float *recvBufferImag, size_t N){
    
    // One thread = one butterfly
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N){
        
        for(size_t row = 0; row < nRows / nGPUs; row++){

            for(size_t col = 0; col < nCols ; col++){
            
                printf("%f ", real[row * nCols + col]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

__global__ void printBuffer(size_t nRows, size_t nCols, size_t nGPUs, float *recvBufferReal, float *recvBufferImag, size_t N){
    
    // One thread = one butterfly
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N){
                
        size_t rowsPerGPU = nRows / nGPUs;
        size_t colsPerGPU = nCols / nGPUs;
        size_t sliceSize  = rowsPerGPU * colsPerGPU;

        for(size_t gpu = 0; gpu < nGPUs; gpu++){
        
            for(size_t row = 0; row < nRows / nGPUs; row++){

                for(size_t col = 0; col < nCols / nGPUs ; col++){
                
                    printf("%f ", recvBufferReal[gpu * sliceSize + row * colsPerGPU + col]);
                }
            }
            printf("\n");
        }
        printf("\n");
    }
}


__global__ void transposeMatrix(float *real, float *imag, size_t nRows){

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < nRows * nRows){

        size_t row = tid / nRows;
        size_t col = tid % nRows;

        if(row < col){
        
            float tmp = real[row * nRows + col]; 
            real[row * nRows + col] = real[col * nRows + row];
            real[col * nRows + row] = tmp;
            
            tmp = imag[row * nRows + col]; 
            imag[row * nRows + col] = imag[col * nRows + row];
            imag[col * nRows + row] = tmp;
        }
    }
}

/*
void distributedTranspose(float *real, float *imag, size_t nRowsPerGPU, size_t nCols, size_t rank, float **sendBufferReal, float **sendBufferImag, float **recvBufferReal, float **recvBufferImag, ncclComm_t comm, cudaStream_t stream){
    
    // first, accumulate the column
    size_t nColsPerGPU = nCols / nGPUs;
    size_t sliceSize = nColsPerGPU * nRowsPerGPU;

    // loop over the rows of each GPU
    for(size_t row = 0; row < nRowsPerGPU; row++){

        // loop over destination GPUs to store data in slices
        for(size_t dstgpu = 0; dstgpu < nGPUs; dstgpu ++){

            // loop over columns for each destination GPU
            for(size_t col = 0; col < nColsPerGPU; col++){

                // [row * nColsPerGPU + col] element position inside slice; [dstgpu * sliceSize] = [row * nCols], row offset; [dstgpu * nColsPerGPU + col] slice offset + position
                sendBufferReal[rank][row * nColPerGPU + col + dstgpu * sliceSize] = real[row * nCols + dstgpu * nColsPerGPU + col]; 
                sendBufferImag[rank][row * nColPerGPU + col + dstgpu * sliceSize] = imag[row * nCols + dstgpu * nColsPerGPU + col];
            }
        }
    }



    // start NCCL communication
    ncclGroupStart();
    
    for(size_t gpu = 0; gpu < nGPUs; gpu++){
    
        ncclSend(sendBufferReal[rank] + gpu * sliceSize, sliceSize, ncclFloat, gpu, comm, stream);
        ncclSend(sendBufferImag[rank] + gpu * sliceSize, sliceSize, ncclFloat, gpu, comm, stream);
    }

    for(size_t gpu = 0; gpu < nGPUs; gpu++){
    
        ncclRecv(recvBufferReal[rank] + gpu * sliceSize, sliceSize, ncclFloat, gpu, comm, stream);
        ncclRecv(recvBufferImag[rank] + gpu * sliceSize, sliceSize, ncclFloat, gpu, comm, stream);
    }

    // perform NCCL communication
    ncclGroupEnd();

    // synchronize stream
    cudaStreamSynchronize(stream); 


    // tranpose while storing
    for(size_t row = 0; row < nRowsPerGPU; row++){

        for(size_t srcgpu = 0; srcgpu < nGPUs; srcgpu ++){

            for(size_t col = 0; col < nColsPerGPU; col++){

                real[col * nColsPerGPU + row + srcgpu * sliceSize] = recvBufferReal[rank][row * nColPerGPU + col + srcgpu * sliceSize];
                imag[col * nColsPerGPU + row + srcgpu * sliceSize] = recvBufferImag[rank][row * nColPerGPU + col + srcgpu * sliceSize];
            }
        }
    }
}
*/

void simulateFFT(appStruct_t *data){
    
    // get number of time steps and state
    size_t i;
    size_t T = data->T;

    double etComputation = 0.0, etCommunication = 0.0;
    struct timespec startComputation, endComputation;
    struct timespec startCommunication, endCommunication;

    // allocate memory for matrix transpose buffers
    float **sendBufferReal = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));
    float **sendBufferImag = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));
    float **recvBufferReal = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));
    float **recvBufferImag = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));

    for(i = 0; i<getState()->jobResources->nGPUs; i++){

        allocateMemoryGPU(&(sendBufferReal[i]), getState()->jobResources->idGPUs[i], data->N / getState()->jobResources->nGPUs);
        allocateMemoryGPU(&(sendBufferImag[i]), getState()->jobResources->idGPUs[i], data->N / getState()->jobResources->nGPUs);
        allocateMemoryGPU(&(recvBufferReal[i]), getState()->jobResources->idGPUs[i], data->N / getState()->jobResources->nGPUs);
        allocateMemoryGPU(&(recvBufferImag[i]), getState()->jobResources->idGPUs[i], data->N / getState()->jobResources->nGPUs);
    }

    // simulate iterations
    for(size_t t = 0; t<T; t++){

        // reconfiguration point, check if there is any pending reconfiguration
        // if there is a reconfiguration, reallocate memory for halos
        // TODO
        if(checkIfReconfiguration(getJobControl())){

            // deallocate memory of old buffers
            for(i = 0; i<getState()->jobResources->nGPUs; i++){

                deallocateMemoryGPU(sendBufferReal[i], getState()->jobResources->idGPUs[i]);
                deallocateMemoryGPU(sendBufferImag[i], getState()->jobResources->idGPUs[i]);
                deallocateMemoryGPU(recvBufferReal[i], getState()->jobResources->idGPUs[i]);
                deallocateMemoryGPU(recvBufferImag[i], getState()->jobResources->idGPUs[i]);
            }

            free(sendBufferReal);
            free(sendBufferImag);
            free(recvBufferReal);
            free(recvBufferImag);


            // reconfigure and manage NCCL communicators
            freeNCCLComm(getState()->jobResources);
            reconfigure(CPU);
            initNCCLComm(getState()->jobResources);


            // allocate memory for matrix transpose buffers
            sendBufferReal = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));
            sendBufferImag = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));
            recvBufferReal = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));
            recvBufferImag = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));

            for(i = 0; i<getState()->jobResources->nGPUs; i++){

                allocateMemoryGPU(&(sendBufferReal[i]), getState()->jobResources->idGPUs[i], data->N / getState()->jobResources->nGPUs);
                allocateMemoryGPU(&(sendBufferImag[i]), getState()->jobResources->idGPUs[i], data->N / getState()->jobResources->nGPUs);
                allocateMemoryGPU(&(recvBufferReal[i]), getState()->jobResources->idGPUs[i], data->N / getState()->jobResources->nGPUs);
                allocateMemoryGPU(&(recvBufferImag[i]), getState()->jobResources->idGPUs[i], data->N / getState()->jobResources->nGPUs);
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

        // full matrix data
        size_t N = data->N;
        size_t nCols = data->nRows;
        size_t nRows = data->nCols;

        // local matrix data
        size_t nPerGPU = N / nGPUs;
        size_t localRows = nRows / nGPUs;
        size_t sliceSize = localRows * localRows;

        size_t block = 512;
        size_t grid = (N / nGPUs) / block + 1;

        //printf(" N = %zu, nCols = %zu, nRows = %zu, nGPUS = %zu, nPerGPU = %zu, localRows = %zu, sliceSize = %zu\n", 
        //        N, nCols, nRows, nGPUs, nPerGPU, localRows, sliceSize);
        //fflush(stdout);

        #pragma omp parallel num_threads (nGPUs) private(i)
        {
            i = omp_get_thread_num();
            cudaError_t err;

            // get global data copies
            appData = localData;
            arrDTI = localArrDTI;
            nDTI = localnDTI;
            maxDTI = localmaxDTI;
            ncclComm_t *comms = localComms;
            cudaStream_t *streams = localStreams;

            // set cuda device
            cudaSetDevice(idGPUs[i]);

            // get GPU data pointers
            float *dReal = (float*)(getDTIByIndex(0)->gpuData[i]);
            float *dImag = (float*)(getDTIByIndex(1)->gpuData[i]);

            // execute two times
            for(size_t rep = 0; rep<2; rep++){
                
                /* Computation */
                if(i == 0){
                    clock_gettime(CLOCK_MONOTONIC, &startComputation);
                }

                size_t logN = log2(nCols);

                // reverse positions using bits
                bitReverseKernel<<<localRows * nCols / 512 + 1, 512, 0, streams[idGPUs[i]]>>>(dReal, dImag, localRows, nCols, logN);
                err = cudaStreamSynchronize(streams[idGPUs[i]]); 

                if(err != cudaSuccess){
                    printf("Failed in bitReverse: %s\n", cudaGetErrorString(err));
                    fflush(stdout);
                }   

                // compute local fft stages
                for (size_t stage = 0; stage < logN; stage++){
                    
                    butterflyKernel<<<(localRows * nCols / 2) / 512 + 1, 512, 0, streams[idGPUs[i]]>>>(dReal, dImag, localRows, nCols, stage);
                    err = cudaStreamSynchronize(streams[idGPUs[i]]); 
                }
                //err = cudaStreamSynchronize(streams[idGPUs[i]]); 

                if(err != cudaSuccess){
                    printf("Failed in butterfly: %s\n", cudaGetErrorString(err));
                    fflush(stdout);
                }   

                #pragma omp barrier

                if(i == 0){
                    
                    clock_gettime(CLOCK_MONOTONIC, &endComputation);
                    etComputation += (endComputation.tv_sec - startComputation.tv_sec) + (endComputation.tv_nsec - startComputation.tv_nsec) / 1e9;
                }

                /* Transpose */

                if(nGPUs > 1){

                    // pack data into buffers
                    if(i == 0) {
                        clock_gettime(CLOCK_MONOTONIC, &startComputation);
                    }

                    pack<<<grid, block, 0, streams[idGPUs[i]]>>>(dReal, dImag, nRows, nCols, nGPUs, sendBufferReal[i], sendBufferImag[i], nPerGPU);
                    err = cudaStreamSynchronize(streams[idGPUs[i]]);

                    #pragma omp barrier

                    
                    if(err != cudaSuccess){
                        printf("Failed in pack: %s\n", cudaGetErrorString(err));
                        fflush(stdout);
                    } 

                    #pragma omp barrier

                    if(i == 0){
                        clock_gettime(CLOCK_MONOTONIC, &endComputation);
                        etComputation += (endComputation.tv_sec - startComputation.tv_sec) + (endComputation.tv_nsec - startComputation.tv_nsec) / 1e9;
                        clock_gettime(CLOCK_MONOTONIC, &startCommunication);
                    }

                    // start NCCL communication
                    ncclGroupStart();
                    for(size_t gpu = 0; gpu < nGPUs; gpu++){
                    
                        ncclSend(sendBufferReal[i] + gpu * sliceSize, sliceSize, ncclFloat, gpu, comms[i], streams[idGPUs[i]]);
                        ncclSend(sendBufferImag[i] + gpu * sliceSize, sliceSize, ncclFloat, gpu, comms[i], streams[idGPUs[i]]);
                    }
                    for(size_t gpu = 0; gpu < nGPUs; gpu++){
                    
                        ncclRecv(recvBufferReal[i] + gpu * sliceSize, sliceSize, ncclFloat, gpu, comms[i], streams[idGPUs[i]]);
                        ncclRecv(recvBufferImag[i] + gpu * sliceSize, sliceSize, ncclFloat, gpu, comms[i], streams[idGPUs[i]]);
                    }
                    // perform NCCL communication
                    ncclGroupEnd();
                    err = cudaStreamSynchronize(streams[idGPUs[i]]);


                    #pragma omp barrier


                    if(i == 0){

                        clock_gettime(CLOCK_MONOTONIC, &endCommunication);
                        etCommunication += (endCommunication.tv_sec - startCommunication.tv_sec) + (endCommunication.tv_nsec - startCommunication.tv_nsec) / 1e9;
                        clock_gettime(CLOCK_MONOTONIC, &startComputation);
                    }

                    unpack<<<grid, block, 0, streams[idGPUs[i]]>>>(dReal, dImag, nRows, nCols, nGPUs, recvBufferReal[i], recvBufferImag[i], nPerGPU);
                    err = cudaStreamSynchronize(streams[idGPUs[i]]);

                    if(err != cudaSuccess){
                        printf("Failed in unpack: %s\n", cudaGetErrorString(err));
                        fflush(stdout);
                    } 

                    #pragma omp barrier

                    if(i == 0){
                    
                        clock_gettime(CLOCK_MONOTONIC, &endComputation);
                        etComputation += (endComputation.tv_sec - startComputation.tv_sec) + (endComputation.tv_nsec - startComputation.tv_nsec) / 1e9;
                    }
                        
                    if(err != cudaSuccess){
                        printf("Failed in resetForces: %s\n", cudaGetErrorString(err));
                        fflush(stdout);
                    } 
                }
                else{

                    clock_gettime(CLOCK_MONOTONIC, &startComputation);
                    transposeMatrix<<<grid, block, 0, streams[idGPUs[i]]>>>(dReal, dImag, nRows);
                    clock_gettime(CLOCK_MONOTONIC, &endComputation);
                    etComputation += (endComputation.tv_sec - startComputation.tv_sec) + (endComputation.tv_nsec - startComputation.tv_nsec) / 1e9;

                    err = cudaStreamSynchronize(streams[idGPUs[i]]);

                    if(err != cudaSuccess){
                        printf("Failed in transpose matrix: %s\n", cudaGetErrorString(err));
                        fflush(stdout);
                    } 
                }
            }
        }  
    }

    pthread_mutex_lock(&(printLock));
    printf("%lf %lf %zu\n", etComputation, etCommunication, T);
    fflush(stdout);
    pthread_mutex_unlock(&(printLock));

    // deallocate memory of buffers
    for(i = 0; i<getState()->jobResources->nGPUs; i++){

        deallocateMemoryGPU(sendBufferReal[i], getState()->jobResources->idGPUs[i]);
        deallocateMemoryGPU(sendBufferImag[i], getState()->jobResources->idGPUs[i]);
        deallocateMemoryGPU(recvBufferReal[i], getState()->jobResources->idGPUs[i]);
        deallocateMemoryGPU(recvBufferImag[i], getState()->jobResources->idGPUs[i]);
    }

    free(sendBufferReal);
    free(sendBufferImag);
    free(recvBufferReal);
    free(recvBufferImag);
}


// launch jacobi
void launch_FFT(int argc, void* argv[]){

    size_t i;

    // initialize DITTO environment
    // temporal: simulate information received from the scheduler: number of GPUs and identifiers of the GPUs (pass argv to the initDITTO function?)
    initDITTO(argv[argc+1]);
    initNCCLComm(getState()->jobResources);

    appStruct_t *appData = (appStruct_t*)calloc(1, sizeof(appStruct_t));
    appData->nRows = *(size_t*)argv[0]; // number of elements in the matrix
    appData->nCols = *(size_t*)argv[1]; // number of rows / columns
    appData->T = *(size_t*)argv[2]; // number of iterations
    appData->s = *(size_t*)argv[3]; 

    // communication info
    size_t pinned = *(size_t*)argv[4];
    size_t async = *(size_t*)argv[5];
    size_t steps = *(size_t*)argv[6];
    size_t cores = *(size_t*)argv[7];
    
    size_t testing = *(size_t*)argv[8];

    appData->malleable = *(size_t*)argv[argc];

    // compute total number of elements
    appData->N = appData->nRows * appData->nCols;
    
    // allocate memory for App Data
    if(pinned){ // pinned memory
        cudaMallocHost(&appData->real, appData->N * sizeof(float));
        cudaMallocHost(&appData->imag, appData->N * sizeof(float));
    }
    else{
        appData->real = (float*)calloc(appData->N, sizeof(float));
        appData->imag = (float*)calloc(appData->N, sizeof(float));
    }
    
    // initialize data
    for(i = 0; i<appData->N; i++){
        appData->real[i] = (float)rand()/(float)(RAND_MAX);
        appData->imag[i] = (float)rand()/(float)(RAND_MAX);
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


    // if testing
    if(testing == 1){

        for(i = 0; i<appData->nRows * appData->nCols; i++){
            appData->real[i] = 0.0;//(float)rand()/(float)(RAND_MAX);
            appData->imag[i] = 0.0;//(float)rand()/(float)(RAND_MAX);
        }

        appData->real[0] = 1.0;
    }
    else if(testing == 2){

        for(i = 0; i<appData->nRows * appData->nCols; i++){
            appData->real[i] = 1.0;//(float)rand()/(float)(RAND_MAX);
            appData->imag[i] = 0.0;//(float)rand()/(float)(RAND_MAX);
        }
    }


    // Initialize DTIs for automatically managing CPU-GPU communications    
    DTI_t *realDTI = createAutomaticDTI((void*)(appData->real), appData->N, sizeof(float), "real", initializeDTIDescription(simple, ordered, commType));
    DTI_t *imagDTI = createAutomaticDTI((void*)(appData->imag), appData->N, sizeof(float), "imag", initializeDTIDescription(simple, ordered, commType));


    // configure all DTIs for automatic data transference // TODO: configure in initialization
    configureDTIs(getState()->jobResources, NULL, CPU2GPU); // number of GPUs, old number of GPUs

    // send data to the GPU
    transferDataCPU2GPU();

    // wait for signal before starting
    simulateFFT(appData);

    // transfer data fron the GPUs to the CPU
    transferDataGPU2CPU();


    if(testing){

        for(i = 0; i<appData->nRows; i++){
            
            for(size_t j = 0; j<appData->nCols; j++){

                printf("%f ", appData->real[i * appData->nCols + j]);//(float)rand()/(float)(RAND_MAX);
            }
            printf("\n");
        }
        printf("\n");

        for(i = 0; i<appData->nRows; i++){
            
            for(size_t j = 0; j<appData->nCols; j++){

                printf("%f ", appData->imag[i * appData->nCols + j]);//(float)rand()/(float)(RAND_MAX);
            }
            printf("\n");
        }
    }

    free(realDTI->cpuData);
    free(imagDTI->cpuData);

    // destroy streams
    freeNCCLComm(getState()->jobResources);
    freeDITTO();


    // signal that job finished
    jobFinished(getJobControl());

    pthread_exit(NULL);
}


