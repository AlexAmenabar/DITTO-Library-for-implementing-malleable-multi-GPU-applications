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

        printf("%d ", appData->arr[i]);
    }
}


/*void moveDataToGPU(){


}

void moveDataToCPU(){

}

void simulate(appStruct_t *appData, int T, size_t nGPUs){

    for(int i = 0; i<T; i++){

        // app
        for(int j = 0; j<nGPUs; j++){

            // set device
            cudaSetDevice(j);

            // run kernel
            runKernel(appData->gpuData[i], arrDTI[0]->infoCustom->nElementsPerGPU[j]);

            // print
            printf(" -- %d iteration done in GPU %d\n", i, j);        
        }

        sleep(4);
    }
}*/

void simulate(appStruct_t *appData){

    int T = appData->T;

    state_t *state = getState();

    for(int i = 0; i<T; i++){

        // reconfiguration point
        int pdgRcf = checIfkReconfiguration();
        printf(" -- Pending reconfiguration = %d, number of GPUs = %zu\n", pdgRcf, state->nGPUs);
        fflush(stdout);

        if(pdgRcf) reconfigure();
        fflush(stdout);

        // app
        //sleep(2);
        printf(" nGPUs = %d\n", (int)(state->nGPUs));
        for(int j = 0; j<(int)(state->nGPUs); j++){

            // set device
            cudaSetDevice(state->idGPUs[j]);

            // run kernel
            runKernel((int*)(getDTIByIndex(0)->gpuData[j]), getDTIByIndex(0)->description->nElementsPerGPU[j]);

            // print
            printf(" -- %d iteration done in GPU %d\n", i, j);        
        }

        sleep(4);
    }
}


// original application main
/*int main(int argc, char* argv[]){
    
    // initialize data structure used in the application
    appStruct_t *appData = (appStruct_t*)calloc(1, sizeof(appStruct_t));
    appData->N = 5;
    appData->arr = (int*)calloc(appData->N, sizeof(int));

    for(size_t i = 0; i<appData->N; i++){
        
        appData->arr[i] = i;
    }


    // program code (simulations)
    printArr(appData);
    simulate(appData, 2);

    // transfer data fron the GPUs to the CPU
    transferDataGPU2CPU();

    printArr(appData);
}*/

appStruct_t* moveAppStructToGPU(appStruct_t *appStruct, int N, int M){

    return appStruct;
} 

//int main(int argc, char* argv[]){

void launch(int argc, void* argv[]){

    printf(" In toy application\n");
    fflush(stdout);

    // initialize DITTO environment
    // temporal: simulate information received from the scheduler: number of GPUs and identifiers of the GPUs (pass argv to the initDITTO function?)
    initDITTO(argv[argc-1]);

    // initialize data structure used in the application
    appStruct_t *appData = (appStruct_t*)calloc(1, sizeof(appStruct_t));
    appData->N = *(int*)argv[0];
    appData->T = *(int*)argv[1];
    appData->arr = (int*)calloc(appData->N, sizeof(int));
    for(size_t i = 0; i<appData->N; i++){
        
        appData->arr[i] = i;
    }

    // Initialize DTIs for automatically managing CPU-GPU communications
    DTI_t *appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->N, sizeof(int), "appData", initializeDTIDescription(simple, ordered));

    // configure all DTIs for automatic data transference
    reconfigureDTIs(getState()->nGPUs, 0); // number of GPUs, old number of GPUs

    // send data to the GPU
    transferDataCPU2GPU();

    // program code (simulations)
    printArr(appData);
    simulate(appData);


    // transfer data fron the GPUs to the CPU
    transferDataGPU2CPU();

    printArr(appData);
}