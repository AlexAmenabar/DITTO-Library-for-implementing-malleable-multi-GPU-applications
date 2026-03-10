#include <cstddef>
#include <omp.h>
#include <unistd.h>
#include <stdio.h>      
#include <stdlib.h>     
#include <pthread.h>

#include "DITO_API.hpp"
#include "DDM.hpp"
#include "DTM.hpp"

// scheduler coomunication
reconfData_t *reconfData = NULL;
int pendingReconf = 0;
pthread_mutex_t lockPendingReconf;
// sch thread
pthread_t thrSch;


// app state and reconfiguration data
state_t *state;
reconfData_t *appReconfData;


// DTIs
DTI_t **arrDTI;
size_t nDTI;
size_t maxDTI;


/* DITTO initialization */

/**
* Initialize environment for communicating the scheduler and the application 
* 
* The function initializes the necessary global variables for communicating both the scheduler and the application,
* which uses a lock variable to avoid race conditions when writing global variables. 
*
*/
void initDITTO(size_t nGPUs, size_t *idGPUs){

    /* [Scheduler]: in the future should be executed in another place and communicated using MPI */

    // initialize variables for communicating the scheduler and the application
    lockPendingReconf = PTHREAD_MUTEX_INITIALIZER;
    reconfData = (reconf_data_t*)calloc(1, sizeof(reconf_data_t));
    pendingReconf = 0; 
    // create the thread that runs the scheduler
    pthread_create(&thrSch, NULL, runMockScheduler, NULL);


    // initialize array of DTIs
    nDTI = 0;
    maxDTI = 10;
    arrDTI = (DTI_t**)calloc(maxDTI, sizeof(DTI_t*));


    // initialize application state and reconfiguration data
    initState(nGPUs, idGPUs);
    initReconfigurationData();
}

void initState(size_t nGPUs, size_t *idGPUs){

    state = (state_t*)calloc(1,sizeof(state_t));
    state->nGPUs = nGPUs;
    state->idGPUs = idGPUs;
}

state_t* getState(){

    return state;
}

void initReconfigurationData(){

    appReconfData = (reconf_data_t*)calloc(1,sizeof(reconf_data_t));
}

reconfData_t* getReconfigurationData(){

    return appReconfData;
}

void* runMockScheduler(void *arg){

    // launch reconfigurations
    sleep(2);
    printf(" Notifying reconfiguration\n");
    fflush(stdout);
    notifyReconfiguration(4, NULL, NULL, NULL);
    printf(" Reconfiguration notified\n");
    fflush(stdout);
}

/*DTI_t* createDTI(void* cpuData, size_t N, size_t size, transmissionPatternsEnum tpttEnum, remainingElementsEnum rmEnum){

    // call DDM function to initialize the DTI
    DTI_t *DTI = initializeDTI(cpuData, N, size, tpttEnum, rmEnum);

    // add DTI to the global array of DTIs
    addDTI(DTI);

    // return DTI structure
    return DTI;
}*/

DTI_t* createAutomaticDTI(void* cpuData, size_t N, size_t size, const char* name, transmissionPatternsEnum tpttEnum, remainingElementsEnum rmEnum){

    // call DDM function to initialize the DTI
    DTI_t *DTI = initializeDTI(0, cpuData, N, size, name, NULL, NULL, tpttEnum, rmEnum);

    // add DTI to the global array of DTIs
    addDTI(DTI);

    // return DTI structure
    return DTI;
}

DTI_t* createManualDTI(void* cpuData, size_t N, size_t size, const char *name, GenericFunction cpu2gpu, GenericFunction gpu2cpu){

    // call DDM function to initialize the DTI
    DTI_t *DTI = initializeDTI(1, cpuData, N, size, name, cpu2gpu, gpu2cpu, nonettp, nonerme);

    // add DTI to the global array of DTIs
    addDTI(DTI);

    // return DTI structure
    return DTI;
}

void addDTI(DTI_t *DTI){

    // array full, expand
    if(nDTI == maxDTI){

        // double the size of the array
        maxDTI *= 2;
        DTI_t **arrDTIExp = (DTI_t**)calloc(maxDTI, sizeof(DTI_t*));

        for(size_t i = 0; i<nDTI; i++)
            arrDTIExp[i] = arrDTI[i];

        // free old array pointer memory
        free(arrDTI);

        // set new pointer
        arrDTI = arrDTIExp;
    }

    // add new DTI
    arrDTI[nDTI] = DTI;
    nDTI += 1;
}

DTI_t* getDTI(const char *dtiName){

    size_t i = 0;
    int found = 0;

    while(!found && i < nDTI){

        if(arrDTI[i]->name == dtiName){
            return arrDTI[i];
        }
    }
    
    // not found
    return NULL;
}



/* [Reconfigurations] */

void notifyReconfiguration(size_t nGPUs, size_t *idGPUs, size_t *src, size_t *dst){

    // lock
    pthread_mutex_lock(&lockPendingReconf);

    // program the reconfiguration if there are no pending reconfigurations
    if(pendingReconf == 0){

        // store data for the new reconfiguration in the global variable: number of GPUs and identifiers
        reconfData->nGPUs = nGPUs;

        if(reconfData->idGPUs) free(reconfData->idGPUs);
        reconfData->idGPUs = (size_t*)calloc(nGPUs, sizeof(size_t));

        for(size_t i = 0; i<nGPUs; i++){
            
            reconfData->idGPUs[i] = i;
        }
        
        // new pending reconfiguration
        pendingReconf = 1;
    }

    // unlock
    pthread_mutex_unlock(&lockPendingReconf);
}

int checIfkReconfiguration(){

    int localPendingReconf;

    pthread_mutex_lock(&lockPendingReconf);

    // check if there is any pending reconfiguration
    if(pendingReconf == 1){

        // copy data from the global reconfData, used by the scheduler, to the local reconfiguration data
        appReconfData->nGPUs = reconfData->nGPUs;

        // free old GPU ids of the appReconfData
        if(appReconfData && appReconfData->idGPUs) free(appReconfData->idGPUs);

        // allocate and copy
        appReconfData->idGPUs = (size_t*)malloc(appReconfData->nGPUs * sizeof(size_t));
        for(size_t i = 0; i<appReconfData->nGPUs; i++){
            
            appReconfData->idGPUs[i] = reconfData->idGPUs[i]; 
        }
    }

    localPendingReconf = pendingReconf;

    pthread_mutex_unlock(&lockPendingReconf);

    return localPendingReconf;
}

void notifyReconfigurationDone(){

    pthread_mutex_lock(&lockPendingReconf);

    // reconfiguration done
    pendingReconf = 0;

    pthread_mutex_unlock(&lockPendingReconf);
}


/**
* Function to perform the application reconfiguration
*
* The function perform the complete reconfiguration process by first reconfiguring the data by moving from the GPU to the CPU,
* deciding how to redistribute it and moving again to the GPU. Then, kernels are reconfigured and finally the scheduler is informed
* that the reconfiguration is done.
* 
* 
*/
void reconfigure(){

    // move data from the GPUs to the CPU
    transferDataGPU2CPU(NULL, NULL, NULL); // move data from the GPUs to the CPU
    
    // configure DTIs
    reconfigureDTIs(appReconfData->nGPUs, state->nGPUs);
    
    // update state
    state->nGPUs = appReconfData->nGPUs;

    if(state->idGPUs) free(state->idGPUs);

    state->idGPUs = (size_t*)malloc(appReconfData->nGPUs * sizeof(size_t));
    for(size_t i = 0; i<appReconfData->nGPUs; i++){

        state->idGPUs[i] = appReconfData->idGPUs[i];
    }

    // move data again from the CPU to the GPU
    transferDataCPU2GPU(NULL, NULL, NULL);

    // reconfigure the kernels
    reconfigureKernels(NULL, NULL);


    // notify that the reconfiguration has finished
    notifyReconfigurationDone();

    printf(" -- Reconfiguration done!\n");
}

void reconfigureDTIs(size_t nGPUs, size_t nOldGPUs){

    size_t i;
    for(i = 0; i<nDTI; i++){

        configureDTI(arrDTI[i], nGPUs, nOldGPUs, NULL, NULL);
    }
}

void* reconfigureKernels(GenericFunction f, void* params){

    if(f != NULL)
        return f(params);
    return NULL;
}

state_t* storeState(state_t *state){
    return NULL;
}

/* [Scheduler communication: notification signals] */



/* [Data transmission] */

void* transferDataCPU2GPU(GenericFunction f, void* ret, void* args){

    size_t i;

    printf(" CPU2GPU ==> nDTI = %zu\n", nDTI);

    for(i = 0; i<nDTI; i++){

        // call the DTM module for copying data to CUDA
        if(arrDTI[i]->type > 0)
            arrDTI[i]->moveCPU2GPU(arrDTI[i]);
        else
            cpyDataCPU2GPU(arrDTI[i]);
    }

    //ret = f(args);
    return NULL;
}

void* transferDataGPU2CPU(GenericFunction f, void* ret, void* args){
    
    size_t i;

    printf(" GPU2CPU ==> nDTI = %zu\n", nDTI);

    for(i = 0; i<nDTI; i++){

        // call the DTM module for copying data from CUDA to the CPu
        if(arrDTI[i]->type > 0)
            arrDTI[i]->moveGPU2CPU(arrDTI[i]);
        else
            cpyDataGPU2CPU(arrDTI[i]);
    }
    
    //ret = f(args);
    return NULL;
}
