#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <unistd.h>
#include <cstddef>
#include <stdio.h>
#include <cstring>

#include <cuda.h>
#include <cuda_runtime.h>

#include "toy_app_malleable.hpp"
#include "DITO_API.hpp"


void printArr(appStruct_t *appData){

    size_t i;

    for(i = 0; i<appData->N; i++){

        printf("%f ", appData->arr[i]);
    }
}

void runCPU(float *arr, int N, int K){

    for(int i = 0; i<N; i++){

        float val = arr[i];
        
        for (int k = 0; k < K; k++) {
        
            val = val * 1.000001f + 0.000001f; //(val * 1.01) * 0.999;// * 0.99;// / 1.5;//1.000001f + 0.000001f;
        }

        val = val * 0.5;
        arr[i] = val;
    }
}

void simulatePhases(appStruct_t *appData){

    int T = appData->T;
    int P = appData->P;
    state_t *state = getState();

    for(int t = 0; t<T; t++){

        if(t%100 == 0){
            
            printf(" In t = %d\n", t);
            fflush(stdout);
        }

        // reconfiguration point
        if(checIfkReconfiguration()){
            printf(" > Reconfiguring application\n"); 
            reconfigure();
        }

        // loop over app phases
        for(int p = 0; p<P; p++){

            // CPU
            if(appData->phases[p] == 0){

                // no GPUs needed in this pahse, so move data to the CPU and send signal to the RMS
                notifySigGPUs(); 

                // move data from the GPU to the CPU
                runCPU((float*)(getDTIByIndex(0)->cpuData), getDTIByIndex(0)->N, appData->cpuK);
            }
            // GPU
            else{

                // since this phase requires GPUs, check if it has, and, if not, request
                if(getState()->nGPUs == 0){
                    
                    notifyReqGPUs();
                }

                // app
                for(int j = 0; j<(int)(state->nGPUs); j++){

                    // set device
                    cudaSetDevice(state->idGPUs[j]);

                    // run kernel
                    runKernel((float*)(getDTIByIndex(0)->gpuData[j]), getDTIByIndex(0)->nPerGPU[j], appData->K);     

                    // sync devices
                    cudaDeviceSynchronize();
                }
            }
        }
    }

}

void simulateIterative(appStruct_t *appData){

    int T = appData->T;
    state_t *state = getState();

    for(int t = 0; t<T; t++){

        if(t%100 == 0){
            
            printf(" In t = %d\n", t);
            fflush(stdout);
        }

        // reconfiguration point
        if(checIfkReconfiguration()){
            printf(" > Reconfiguring application\n"); 
            reconfigure();
        }

        // app
        for(int j = 0; j<(int)(state->nGPUs); j++){

            // set device
            cudaSetDevice(state->idGPUs[j]);

            // run kernel
            runKernel((float*)(getDTIByIndex(0)->gpuData[j]), getDTIByIndex(0)->nPerGPU[j], appData->K);     

            // sync devices
            cudaDeviceSynchronize();
        }
    }
}


void launch_iterative_app(int argc, void* argv[]){

    size_t i;

    // initialize DITTO environment
    // temporal: simulate information received from the scheduler: number of GPUs and identifiers of the GPUs (pass argv to the initDITTO function?)
    initDITTO(argv[argc-1]);

    // APP: initialize data structure used by the application
    appStruct_t *appData = (appStruct_t*)calloc(1, sizeof(appStruct_t));
    appData->N = *(int*)argv[0];
    appData->T = *(int*)argv[1];
    appData->K = *(int*)argv[2];

    appData->arr = (float*)calloc(appData->N, sizeof(float));
    for(i = 0; i<appData->N; i++)
        appData->arr[i] = (float)rand()/(float)(RAND_MAX);

    // Initialize DTIs for automatically managing CPU-GPU communications
    DTI_t *appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->N, sizeof(int), "appData", initializeDTIDescription(simple, ordered));


    // configure all DTIs for automatic data transference // TODO: configure in initialization
    configureDTIs(getState()->nGPUs, 0); // number of GPUs, old number of GPUs

    // send data to the GPU
    transferDataCPU2GPU();

    // program code (simulations)
    //printArr(appData);
    simulateIterative(appData);

    // transfer data fron the GPUs to the CPU
    transferDataGPU2CPU();
    
    pthread_exit(NULL);
}

void launch_phases_app(int argc, void* argv[]){

    size_t i;

    // initialize DITTO environment
    // temporal: simulate information received from the scheduler: number of GPUs and identifiers of the GPUs (pass argv to the initDITTO function?)
    initDITTO(argv[argc-1]);

    // APP: initialize data structure used by the application
    appStruct_t *appData = (appStruct_t*)calloc(1, sizeof(appStruct_t));
    appData->N = *(int*)argv[0];
    appData->T = *(int*)argv[1];
    appData->K = *(int*)argv[2];

    appData->arr = (float*)calloc(appData->N, sizeof(float));
    for(i = 0; i<appData->N; i++)
        appData->arr[i] = (float)rand()/(float)(RAND_MAX);

    // Initialize DTIs for automatically managing CPU-GPU communications
    DTI_t *appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->N, sizeof(int), "appData", initializeDTIDescription(simple, ordered));


    // configure all DTIs for automatic data transference // TODO: configure in initialization
    configureDTIs(getState()->nGPUs, 0); // number of GPUs, old number of GPUs

    // send data to the GPU
    transferDataCPU2GPU();

    // program code (simulations)
    //printArr(appData);
    simulatePhases(appData);

    // transfer data fron the GPUs to the CPU
    transferDataGPU2CPU();
    
    pthread_exit(NULL);
}