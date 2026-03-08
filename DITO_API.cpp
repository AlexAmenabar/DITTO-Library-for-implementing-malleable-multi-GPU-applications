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

// app state and reconfiguration data
state_t *state;
reconfData_t *appReconfData;

// sch thread
pthread_t thrSch;

// DTIs
DTI_t **arrDTI;
size_t nDTI;
size_t maxDTI;

// data transference functions
UserFunction *userCPU2GPUFunction;
UserFunction *userGPU2CPUFunction;
size_t nFuncs;
size_t maxFuncs;


/* DITTO initialization */

/**
* Initialize environment for communicating the scheduler and the application 
* 
* The function initializes the necessary global variables for communicating both the scheduler and the application,
* which uses a lock variable to avoid race conditions when writing global variables. 
*
*/
void initDITTO(size_t nGPUs, size_t *idGPUs){

    lockPendingReconf = PTHREAD_MUTEX_INITIALIZER;
    //omp_init_lock(&lockPendingReconf); // initialize lock to avoid race conditions
    reconfData = (reconf_data_t*)calloc(1, sizeof(reconf_data_t)); // initialize a global variable to store the new reconfigurations data
    pendingReconf = 0; // no pending jobs yet

    // initialize DTI
    nDTI = 0;
    maxDTI = 10;
    arrDTI = (DTI_t**)calloc(maxDTI, sizeof(DTI_t*));

    nFuncs = 0;
    maxFuncs = 10;
    userCPU2GPUFunction = (UserFunction*)calloc(nFuncs, sizeof(UserFunction));
    userGPU2CPUFunction = (UserFunction*)calloc(nFuncs, sizeof(UserFunction));

    // APP initialization
    initState(nGPUs, idGPUs);
    initReconfigurationData();

    // create the thread that runs the scheduler
    pthread_create(&thrSch, NULL, runMockScheduler, NULL);
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

void *runMockScheduler(void *arg){

    // launch reconfigurations
    sleep(2);
    printf(" Notifying reconfiguration\n");
    fflush(stdout);
    notifyReconfiguration(4, NULL, NULL, NULL);
    printf(" Reconfiguration notified\n");
    fflush(stdout);
}

void setCommunicationFunctions(UserFunction funcCPU2GPU, UserFunction funcGPU2CPU){

    if(nFuncs == maxFuncs){
        
        maxFuncs *= 2;
        UserFunction *funcCPU2GPUExp = (UserFunction*)calloc(maxFuncs, sizeof(UserFunction));
        UserFunction *funcGPU2CPUExp = (UserFunction*)calloc(maxFuncs, sizeof(UserFunction));

        for(size_t i = 0; i<nFuncs; i++){
            funcCPU2GPUExp[i] = userCPU2GPUFunction[i];
            funcGPU2CPUExp[i] = userGPU2CPUFunction[i];
        }

        free(userCPU2GPUFunction);
        free(userGPU2CPUFunction);

        userCPU2GPUFunction = funcCPU2GPUExp;
        userGPU2CPUFunction = funcGPU2CPUExp;
    }

    // add new functions
    userCPU2GPUFunction[nFuncs] = funcCPU2GPU;
    userGPU2CPUFunction[nFuncs] = funcGPU2CPU;
    nFuncs += 1;
}


DTI_t* createDTI(void* cpuData, size_t N, size_t size, transmissionPatternsEnum tpttEnum, remainingElementsEnum rmEnum){

    // call DDM function to initialize the DTI
    DTI_t *DTI = initializeDTI(cpuData, N, size, tpttEnum, rmEnum);

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
    
    // configure the DTIs
    configureDTI(arrDTI[0], appReconfData->nGPUs, state->nGPUs, NULL, NULL);
    
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


    // notify that the reconfiguration has finished
    notifyReconfigurationDone();

    printf(" -- Reconfiguration done!\n");
}

void reconfigureKernels(){

    printf("");
}

state_t* storeState(state_t *state){
    return NULL;
}

/* [Scheduler communication: notification signals] */



/* [Data transmission] */

void* transferDataCPU2GPU(UserFunction userFunction, void* ret, void* args){

    size_t i;

    printf(" CPU2GPU ==> nDTI = %zu\n", nDTI);

    for(i = 0; i<nDTI; i++){

        // call the DTM module for copying data to CUDA
        cpyDataCPU2GPU(arrDTI[i]);
    }

    ret = userFunction(args);
    return ret;
}

void* transferDataGPU2CPU(UserFunction userFunction, void* ret, void* args){
    
    size_t i;

    printf(" GPU2CPU ==> nDTI = %zu\n", nDTI);

    for(i = 0; i<nDTI; i++){

        // call the DTM module for copying data to CUDA
        cpyDataGPU2CPU(arrDTI[i]);
    }
    
    ret = userFunction(args);
    return ret;
}

// tmemp?
/*template<typename Func, typename... Args>
void* transferDataCPU2GPU_impl(Func f, Args... args)
{
    for(size_t i = 0; i < nDTI; i++)
        cpyDataCPU2GPU(arrDTI[i]);

    return f(args...);
}*/