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

__global__ void resetForces(size_t N, float *fx, float *fy, float *fz){

    size_t i = blockIdx.x * blockDim.x + threadIdx.x; 

    if (i<N) {

        fx[i] = 0.0;
        fy[i] = 0.0;
        fz[i] = 0.0;
    }
}


__global__ void computeForces(size_t N, float *x, float *y, float *z, float *fx, float *fy, float *fz, float *m,
                                        float *rx, float *ry, float *rz, float *rm) {
    
    size_t i = blockIdx.x * blockDim.x + threadIdx.x; // TODO: compute

    if (i<N) {
        
        float G = 1.0; 

        float dstx, dsty, dstz;
        float EPS = 0.01;

        for(size_t p = 0; p<N; p++){
            
            dstx = rx[p] - x[i];
            dsty = ry[p] - y[i];
            dstz = rz[p] - z[i];

            float r2 = dstx * dstx + dsty * dsty + dstz * dstz + EPS;
            float invR = rsqrtf(r2);
            float invR3 = invR * invR * invR;
            float F = G * m[i] * rm[p] * invR3;

            fx[i] += F * dstx;
            fy[i] += F * dsty;
            fz[i] += F * dstz;
        }
    }
}

__global__ void integrate(size_t N, float *ax, float *ay, float *az, float *fx, float *fy, float *fz, float *x, float *y, float *z, float *vx, float *vy, float *vz, float* m){
    
    size_t i = blockIdx.x * blockDim.x + threadIdx.x; // TODO: compute
    
    float dt = 0.001f;

    if (i < N) {

        ax[i] = fx[i] / m[i];
        ay[i] = fy[i] / m[i];
        az[i] = fz[i] / m[i];

        vx[i] += ax[i] * dt;
        vy[i] += ay[i] * dt;
        vz[i] += az[i] * dt;

        x[i] += vx[i] * dt;
        y[i] += vy[i] * dt;
        z[i] += vz[i] * dt;
    }
}


void simulateNBody(appStruct_t *data){
    
    // get number of time steps and state
    size_t T = data->T;

    double etComputation = 0.0, etCommunication = 0.0;
    struct timespec startComputation, endComputation;
    struct timespec startCommunication, endCommunication;


    // allocate memory for non-- GPU data
    float **fx = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));
    float **fy = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));
    float **fz = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));
    float **ax = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));
    float **ay = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));
    float **az = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));

    // allocate memory remote particles data    
    float **rx = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));
    float **ry = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));
    float **rz = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));
    float **rm = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));

    for(size_t i = 0; i<getState()->jobResources->nGPUs; i++){

        // allocate memory for GPU particles
        allocateMemoryGPU(&(fx[i]), getState()->jobResources->idGPUs[i], data->N / getState()->jobResources->nGPUs);
        allocateMemoryGPU(&(fy[i]), getState()->jobResources->idGPUs[i], data->N / getState()->jobResources->nGPUs);
        allocateMemoryGPU(&(fz[i]), getState()->jobResources->idGPUs[i], data->N / getState()->jobResources->nGPUs);
        allocateMemoryGPU(&(ax[i]), getState()->jobResources->idGPUs[i], data->N / getState()->jobResources->nGPUs);
        allocateMemoryGPU(&(ay[i]), getState()->jobResources->idGPUs[i], data->N / getState()->jobResources->nGPUs);
        allocateMemoryGPU(&(az[i]), getState()->jobResources->idGPUs[i], data->N / getState()->jobResources->nGPUs);

        // allocate memory for necessary remote data
        allocateMemoryGPU(&(rx[i]), getState()->jobResources->idGPUs[i], data->N / getState()->jobResources->nGPUs);
        allocateMemoryGPU(&(ry[i]), getState()->jobResources->idGPUs[i], data->N / getState()->jobResources->nGPUs);
        allocateMemoryGPU(&(rz[i]), getState()->jobResources->idGPUs[i], data->N / getState()->jobResources->nGPUs);
        allocateMemoryGPU(&(rm[i]), getState()->jobResources->idGPUs[i], data->N / getState()->jobResources->nGPUs);
    }
    

    // simulate iterations
    for(size_t t = 0; t<T; t++){

        // reconfiguration point, check if there is any pending reconfiguration
        // if there is a reconfiguration, reallocate memory for halos
        if(checkIfReconfiguration(getJobControl())){

            // deallocate memory of old halos
            for(size_t i = 0; i<getState()->jobResources->nGPUs; i++){

                deallocateMemoryGPU(fx[i], getState()->jobResources->idGPUs[i]);
                deallocateMemoryGPU(fy[i], getState()->jobResources->idGPUs[i]);
                deallocateMemoryGPU(fz[i], getState()->jobResources->idGPUs[i]);
                deallocateMemoryGPU(ax[i], getState()->jobResources->idGPUs[i]);
                deallocateMemoryGPU(ay[i], getState()->jobResources->idGPUs[i]);
                deallocateMemoryGPU(az[i], getState()->jobResources->idGPUs[i]);

                // each GPU has N / nGPUs elements
                deallocateMemoryGPU(rx[i], getState()->jobResources->idGPUs[i]);
                deallocateMemoryGPU(ry[i], getState()->jobResources->idGPUs[i]);
                deallocateMemoryGPU(rz[i], getState()->jobResources->idGPUs[i]);
                deallocateMemoryGPU(rm[i], getState()->jobResources->idGPUs[i]);
            }

            free(fx);
            free(fy);
            free(fz);
            free(ax);
            free(ay);
            free(az);

            free(rx);
            free(ry);
            free(rz);
            free(rm);


            // reconfigure and manage NCCL communicators
            freeNCCLComm(getState()->jobResources);
            reconfigure(CPU);
            initNCCLComm(getState()->jobResources);
            
            // allocate memory for GPUs own 
            fx = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));
            fy = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));
            fz = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));
            ax = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));
            ay = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));
            az = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));

            // allocate memory remote particles data
            rx = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));
            ry = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));
            rz = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));
            rm = (float**)calloc(getState()->jobResources->nGPUs, sizeof(float*));

            for(size_t i = 0; i<getState()->jobResources->nGPUs; i++){

                allocateMemoryGPU(&(fx[i]), getState()->jobResources->idGPUs[i], data->N / getState()->jobResources->nGPUs);
                allocateMemoryGPU(&(fy[i]), getState()->jobResources->idGPUs[i], data->N / getState()->jobResources->nGPUs);
                allocateMemoryGPU(&(fz[i]), getState()->jobResources->idGPUs[i], data->N / getState()->jobResources->nGPUs);
                allocateMemoryGPU(&(ax[i]), getState()->jobResources->idGPUs[i], data->N / getState()->jobResources->nGPUs);
                allocateMemoryGPU(&(ay[i]), getState()->jobResources->idGPUs[i], data->N / getState()->jobResources->nGPUs);
                allocateMemoryGPU(&(az[i]), getState()->jobResources->idGPUs[i], data->N / getState()->jobResources->nGPUs);

                // each GPU has N / nGPUs elements
                allocateMemoryGPU(&(rx[i]), getState()->jobResources->idGPUs[i], data->N / getState()->jobResources->nGPUs);
                allocateMemoryGPU(&(ry[i]), getState()->jobResources->idGPUs[i], data->N / getState()->jobResources->nGPUs);
                allocateMemoryGPU(&(rz[i]), getState()->jobResources->idGPUs[i], data->N / getState()->jobResources->nGPUs);
                allocateMemoryGPU(&(rm[i]), getState()->jobResources->idGPUs[i], data->N / getState()->jobResources->nGPUs);
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

        size_t gpuN = data->N / nGPUs;

        dim3 block(128);
        dim3 grid(data->N / 128 + 1);
        
        #pragma omp parallel num_threads(nGPUs)
        {
            size_t i = omp_get_thread_num();
            cudaError_t err;

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
            float *x = (float*)(getDTIByIndex(0)->gpuData[i]);
            float *y = (float*)(getDTIByIndex(1)->gpuData[i]);
            float *z = (float*)(getDTIByIndex(2)->gpuData[i]);
            float *vx = (float*)(getDTIByIndex(3)->gpuData[i]);
            float *vy = (float*)(getDTIByIndex(4)->gpuData[i]);
            float *vz = (float*)(getDTIByIndex(5)->gpuData[i]);
            float *m = (float*)(getDTIByIndex(6)->gpuData[i]);

            resetForces<<<grid, block, 0, streams[idGPUs[i]]>>>(gpuN, fx[i], fy[i], fz[i]);
            err = cudaStreamSynchronize(streams[idGPUs[i]]);

            if(err != cudaSuccess){
                printf("Failed in resetForces: %s\n", cudaGetErrorString(err));
                fflush(stdout);
            }          

            if(nGPUs > 1){
                
                for(size_t g = 0; g<nGPUs; g++){

                    if (i == 0)
                        clock_gettime(CLOCK_MONOTONIC, &startCommunication);

                    // send particles to next GPU and receive from prev
                    size_t nextGPU = (i + g) % nGPUs;
                    size_t prevGPU = i >= g ? i - g : nGPUs + i - g;

                    ncclGroupStart();

                    ncclSend(x, gpuN, ncclFloat, nextGPU, comms[i], streams[idGPUs[i]]);
                    ncclSend(y, gpuN, ncclFloat, nextGPU, comms[i], streams[idGPUs[i]]);
                    ncclSend(z, gpuN, ncclFloat, nextGPU, comms[i], streams[idGPUs[i]]);
                    ncclSend(m, gpuN, ncclFloat, nextGPU, comms[i], streams[idGPUs[i]]);

                    ncclRecv(rx[i], gpuN, ncclFloat, prevGPU, comms[i], streams[idGPUs[i]]);
                    ncclRecv(ry[i], gpuN, ncclFloat, prevGPU, comms[i], streams[idGPUs[i]]);
                    ncclRecv(rz[i], gpuN, ncclFloat, prevGPU, comms[i], streams[idGPUs[i]]);
                    ncclRecv(rm[i], gpuN, ncclFloat, prevGPU, comms[i], streams[idGPUs[i]]);

                    ncclGroupEnd();

                    cudaStreamSynchronize(streams[idGPUs[i]]);
                    #pragma omp barrier

                    if(i == 0){

                        clock_gettime(CLOCK_MONOTONIC, &endCommunication);
                        etCommunication += (endCommunication.tv_sec - startCommunication.tv_sec) + (endCommunication.tv_nsec - startCommunication.tv_nsec) / 1e9;
                        clock_gettime(CLOCK_MONOTONIC, &startComputation);
                    }

                    // compute forces
                    computeForces<<<grid, block, 0, streams[idGPUs[i]]>>>(gpuN, x, y, z, fx[i], fy[i], fz[i], m, rx[i], ry[i], rz[i], rm[i]);
                    err = cudaStreamSynchronize(streams[idGPUs[i]]);

                    if(err != cudaSuccess){
                        printf("Failed in computeForces: %s\n", cudaGetErrorString(err));
                        fflush(stdout);
                    } 

                    #pragma omp barrier
                    
                    if(i == 0){
                    
                        clock_gettime(CLOCK_MONOTONIC, &endComputation);
                        etComputation += (endComputation.tv_sec - startComputation.tv_sec) + (endComputation.tv_nsec - startComputation.tv_nsec) / 1e9;
                    }
                }
            }
            else{

                clock_gettime(CLOCK_MONOTONIC, &startComputation);

                // compute forces
                computeForces<<<grid, block, 0, streams[idGPUs[i]]>>>(gpuN, x, y, z, fx[i], fy[i], fz[i], m, x, y, z, m);
                err = cudaStreamSynchronize(streams[idGPUs[i]]);

                if(err != cudaSuccess){
                    printf("Failed in computeForces: %s\n", cudaGetErrorString(err));
                    fflush(stdout);
                } 
                
                clock_gettime(CLOCK_MONOTONIC, &endComputation);
                etComputation += (endComputation.tv_sec - startComputation.tv_sec) + (endComputation.tv_nsec - startComputation.tv_nsec) / 1e9;
            }

            // update velocities and positions
            if(i == 0)
                clock_gettime(CLOCK_MONOTONIC, &startComputation);


            integrate<<<grid, block, 0, streams[idGPUs[i]]>>>(gpuN, ax[i], ay[i], az[i], fx[i], fy[i], fz[i], x, y, z, vx, vy, vz, m);
            err = cudaStreamSynchronize(streams[idGPUs[i]]);

            if(err != cudaSuccess){
                printf("Failed in integrate: %s\n", cudaGetErrorString(err));
                fflush(stdout);
            } 

            #pragma omp barrier

            if(i == 0){
            
                clock_gettime(CLOCK_MONOTONIC, &endComputation);
                etComputation += (endComputation.tv_sec - startComputation.tv_sec) + (endComputation.tv_nsec - startComputation.tv_nsec) / 1e9;
            }
        }
    }

    pthread_mutex_lock(&(printLock));
    printf("%lf %lf %zu\n", etComputation, etCommunication, T);
    fflush(stdout);
    pthread_mutex_unlock(&(printLock));


    // deallocate memory of old halos
    for(size_t i = 0; i<getState()->jobResources->nGPUs; i++){

        // each GPU has N / nGPUs elements
        deallocateMemoryGPU(fx[i], getState()->jobResources->idGPUs[i]);
        deallocateMemoryGPU(fy[i], getState()->jobResources->idGPUs[i]);
        deallocateMemoryGPU(fz[i], getState()->jobResources->idGPUs[i]);
        deallocateMemoryGPU(ax[i], getState()->jobResources->idGPUs[i]);
        deallocateMemoryGPU(ay[i], getState()->jobResources->idGPUs[i]);
        deallocateMemoryGPU(az[i], getState()->jobResources->idGPUs[i]);

        // each GPU has N / nGPUs elements
        deallocateMemoryGPU(rx[i], getState()->jobResources->idGPUs[i]);
        deallocateMemoryGPU(ry[i], getState()->jobResources->idGPUs[i]);
        deallocateMemoryGPU(rz[i], getState()->jobResources->idGPUs[i]);
        deallocateMemoryGPU(rm[i], getState()->jobResources->idGPUs[i]);
    }

    free(fx);
    free(fy);
    free(fz);
    free(ax);
    free(ay);
    free(az);

    free(rx);
    free(ry);
    free(rz);
    free(rm);
}


// launch jacobi
void launch_n_body(int argc, void* argv[]){

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
    
    size_t testing = *(size_t*)argv[7];

    appData->malleable = *(size_t*)argv[argc];
 
    // allocate memory for App Data
    if(pinned){ // pinned memory

        cudaMallocHost(&appData->x, appData->N * sizeof(float));
        cudaMallocHost(&appData->y, appData->N * sizeof(float));
        cudaMallocHost(&appData->z, appData->N * sizeof(float));
        cudaMallocHost(&appData->vx, appData->N * sizeof(float));
        cudaMallocHost(&appData->vy, appData->N * sizeof(float));
        cudaMallocHost(&appData->vz, appData->N * sizeof(float));
        cudaMallocHost(&appData->m, appData->N * sizeof(float));
    }
    else {

        appData->x = (float*)calloc(appData->N, sizeof(float));
        appData->y = (float*)calloc(appData->N, sizeof(float));
        appData->z = (float*)calloc(appData->N, sizeof(float));
        appData->vx = (float*)calloc(appData->N, sizeof(float));
        appData->vy = (float*)calloc(appData->N, sizeof(float));
        appData->vz = (float*)calloc(appData->N, sizeof(float));
        appData->m = (float*)calloc(appData->N, sizeof(float));
    }
    
    // initialize data
    for(i = 0; i<appData->N; i++){

        appData->x[i] = (float)rand()/(float)(RAND_MAX);
        appData->y[i] = (float)rand()/(float)(RAND_MAX);
        appData->z[i] = (float)rand()/(float)(RAND_MAX);
        appData->vx[i] = (float)rand()/(float)(RAND_MAX);
        appData->vy[i] = (float)rand()/(float)(RAND_MAX);
        appData->vz[i] = (float)rand()/(float)(RAND_MAX);
        appData->m[i] = 1.0f + (float)rand()/(float)(RAND_MAX);
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
    if(testing){

        for(i = 0; i<appData->nRows * appData->nCols; i++){
    
            appData->x[i] = (float)rand()/(float)(RAND_MAX);
            appData->y[i] = (float)rand()/(float)(RAND_MAX);
            appData->z[i] = (float)rand()/(float)(RAND_MAX);
            appData->vx[i] = (float)rand()/(float)(RAND_MAX);
            appData->vy[i] = (float)rand()/(float)(RAND_MAX);
            appData->vz[i] = (float)rand()/(float)(RAND_MAX);
            appData->m[i] = 1.0f + (float)rand()/(float)(RAND_MAX);
        }

        appData->x[0] = -10.0;
        appData->y[0] = 10.0;
        appData->z[0] = -10.0;
        
        appData->x[1] = 10.0;
        appData->y[1] = 10.0;
        appData->z[1] = 10.0;
        
        appData->x[2] = 10.0;
        appData->y[2] = 10.0;
        appData->z[2] = -10.0;
        
        appData->x[3] = -10.0;
        appData->y[3] = 10.0;
        appData->z[3] = 10.0;

        appData->x[4] = -10.0;
        appData->y[4] = -10.0;
        appData->z[4] = -10.0;
        
        appData->x[5] = 10.0;
        appData->y[5] = -10.0;
        appData->z[5] = 10.0;
        
        appData->x[6] = 10.0;
        appData->y[6] = -10.0;
        appData->z[6] = -10.0;
        
        appData->x[7] = -10.0;
        appData->y[7] = -10.0;
        appData->z[7] = 10.0;


        appData->vx[0] = 1.0;
        appData->vy[0] = 0.0;
        appData->vz[0] = 1.0;
        
        appData->vx[1] = 1.0;
        appData->vy[1] = 0.0;
        appData->vz[1] = 1.0;
        
        appData->vx[2] = 1.0;
        appData->vy[2] = 0.0;
        appData->vz[2] = 1.0;
        
        appData->vx[3] = 1.0;
        appData->vy[3] = 0.0;
        appData->vz[3] = 1.0;

        appData->vx[4] = 1.0;
        appData->vy[4] = 0.0;
        appData->vz[4] = 1.0;
        
        appData->vx[5] = 1.0;
        appData->vy[5] = 0.0;
        appData->vz[5] = 1.0;
        
        appData->vx[6] = 1.0;
        appData->vy[6] = 0.0;
        appData->vz[6] = 1.0;
        
        appData->vx[7] = 1.0;
        appData->vy[7] = 0.0;
        appData->vz[7] = 1.0;


        appData->m[0] = 1.0;
        appData->m[1] = 1.0;
        appData->m[2] = 1.0;
        appData->m[3] = 1.0;

        appData->m[4] = 2.0;
        appData->m[5] = 2.0;
        appData->m[6] = 2.0;
        appData->m[7] = 2.0;
    }


    // create DTIs
    createAutomaticDTI((void*)(appData->x), appData->N, sizeof(float), "x", initializeDTIDescription(simple, ordered, commType));
    createAutomaticDTI((void*)(appData->y), appData->N, sizeof(float), "y", initializeDTIDescription(simple, ordered, commType));
    createAutomaticDTI((void*)(appData->z), appData->N, sizeof(float), "z", initializeDTIDescription(simple, ordered, commType));
    createAutomaticDTI((void*)(appData->vx), appData->N, sizeof(float), "vx", initializeDTIDescription(simple, ordered, commType));
    createAutomaticDTI((void*)(appData->vy), appData->N, sizeof(float), "vy", initializeDTIDescription(simple, ordered, commType));
    createAutomaticDTI((void*)(appData->vz), appData->N, sizeof(float), "vz", initializeDTIDescription(simple, ordered, commType));
    createAutomaticDTI((void*)(appData->m), appData->N, sizeof(float), "m", initializeDTIDescription(simple, ordered, commType));

    // configure all DTIs for automatic data transference // TODO: configure in initialization
    configureDTIs(getState()->jobResources, NULL, CPU2GPU); // number of GPUs, old number of GPUs

    // send data to the GPU
    transferDataCPU2GPU();

    
    // wait for signal before starting
    simulateNBody(appData);

    // transfer data fron the GPUs to the CPU
    transferDataGPU2CPU();


    if(testing){

        for(i = 0; i<appData->N; i++){
            
            printf("Particle %zu: %f %f %f %f %f %f %f\n", 
                i, appData->x[i], appData->y[i], appData->z[i], appData->vx[i], appData->vy[i], appData->vz[i], appData->m[i]);//(float)rand()/(float)(RAND_MAX);
        }
    }


    // deallocate DTIs
    deallocateDTIs();

    // destroy streams
    freeNCCLComm(getState()->jobResources);
    freeDITTO();


    // signal that job finished
    jobFinished(getJobControl());

    pthread_exit(NULL);
}


