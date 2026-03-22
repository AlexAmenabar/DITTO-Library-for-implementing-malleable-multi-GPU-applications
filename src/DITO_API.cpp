#include <cstddef>
#include <omp.h>
#include <unistd.h>
#include <stdio.h>      
#include <stdlib.h>     
#include <pthread.h>

#include "DITO_API.hpp"
#include "priv_DITTO_API.hpp"
#include "DDM.hpp"
#include "DTM.hpp"
#include "mockSch.hpp"


/* [Global variables] */

// App Data
public_APP_data_t *appData = NULL;

// DTIs
size_t nDTI, maxDTI;
DTI_t** arrDTI = NULL;


/**
=================================================
|| PUBLIC DITTO API: APP - DITTO COMMUNICATION ||
=================================================
*/



/*
==========
| PUBLIC |
==========
*/


/* DITTO initialization */

/**
* Initialize environment for communicating the scheduler and the application 
* 
* The function initializes the necessary global variables for communicating both the scheduler and the application,
* which uses a lock variable to avoid race conditions when writing global variables. 
*
*/
void initDITTO(void *jobControl){

    // allocate memory for appData structure
    appData = (public_APP_Data_t*)calloc(1, sizeof(public_APP_Data_t));
    
    // initialize reconfiguration related variables
    appData->pendingReconf = 0; // indicates whether there is any pending reconfiguration
    appData->lockPendingReconf = PTHREAD_MUTEX_INITIALIZER; // initialize lock
    //appData->reconfData = initReconfigurationData(); // initialize structure for storing storeing reconfiguration data
    appData->jobControl = (jobControl_t*)jobControl; // communication with the scheduler

    // initialize application state // TODO: probably it should be a function
    appData->state = initState(appData->jobControl->nGPUs, appData->jobControl->idGPUs);


    // initialize array of DTIs
    nDTI = 0;
    maxDTI = 10;
    arrDTI = (DTI_t**)calloc(maxDTI, sizeof(DTI_t*));
}

void freeDITTO(){

    printf(" -- Not implemented yet!\n");

    // free appData, 
}

state_t* getState(){

    return appData->state;
}

/*reconfData_t* getReconfigurationData(){

    return appData->reconfData;
}*/

size_t getNumberOfGPUs(){

    return getState()->nGPUs;
}

size_t* getGPUIds(){

    return getState()->idGPUs;
}

state_t* updateState(state_t *state, jobControl_t *jobControl){
    
    size_t i, nGPUs, *idGPUs;

    nGPUs = jobControl->nGPUs;
    idGPUs = jobControl->idGPUs;

    // update state (copy number of GPUs and GPU ids)
    state->nGPUs = nGPUs;

    if(state->idGPUs) 
        free(state->idGPUs);

    state->idGPUs = (size_t*)malloc(nGPUs * sizeof(size_t));
    for(i = 0; i<nGPUs; i++)
        state->idGPUs[i] = idGPUs[i];

    return state;
}

state_t* storeState(state_t *state){

    printf(" -- Store state not implemented yet\n");
    return state;
}



/* [Reconfigurations] */

// scheduler notifies the application a new reconfiguration
void notifyReconfiguration(size_t nGPUs, size_t *idGPUs){

    size_t i;
    state_t *state = getState();
    public_APP_Data_t *appData = appData;

    // lock
    pthread_mutex_lock(&(appData->lockPendingReconf));

    // program the reconfiguration if there are no pending reconfigurations
    if(appData->pendingReconf == 0){

        state->nGPUs = nGPUs;

        if(state->idGPUs) 
            free(state->idGPUs);
 
        state->idGPUs = (size_t*)calloc(nGPUs, sizeof(size_t));

        for(i = 0; i<nGPUs; i++)
            state->idGPUs[i] = i;
        

        // new pending reconfiguration
        appData->pendingReconf = 1;
    }

    // unlock
    pthread_mutex_unlock(&(appData->lockPendingReconf));
}

// application checks whether there is a pending reconfiguration
int checIfkReconfiguration(){

    public_APP_Data_t *appData = appData;
    int localPendingReconf;

    pthread_mutex_lock(&(appData->lockPendingReconf));
    localPendingReconf = appData->pendingReconf;
    pthread_mutex_unlock(&appData->lockPendingReconf);

    return appData->pendingReconf;
}

// application notifies reconfiguration finished
void notifyReconfigurationDone(){

    public_APP_Data_t *appData = appData;

    pthread_mutex_lock(&(appData->lockPendingReconf));

    // reconfiguration done
    appData->pendingReconf = 0;

    pthread_mutex_unlock(&(appData->lockPendingReconf));
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

    state_t *state = getState();
    jobControl_t *jobControl = appData->jobControl;

    // move data from the GPUs to the CPU
    transferDataGPU2CPU(); // move data from the GPUs to the CPU
    
    size_t nOldGPUs, nGPUs;

    // number of GPUs before reconfiguration
    nOldGPUs = state->nGPUs;

    // update state
    updateState(state, jobControl);

    // number of GPUs after reconfiguration
    nGPUs = state->nGPUs;

    // [option to update descriptions depending on the number of GPUs???]

    // configure DTIs
    reconfigureDTIs(nGPUs, nOldGPUs);
    
    // move data again from the CPU to the GPU
    transferDataCPU2GPU();

    // reconfigure the kernels
    reconfigureKernels(NULL, NULL);

    // notify that the reconfiguration has finished
    notifyReconfigurationDone();

    printf(" -- Reconfiguration done!\n");
}

void reconfigureDTIs(size_t nGPUs, size_t nOldGPUs){

    size_t i;
    for(i = 0; i<nDTI; i++){

        redistributeDTI(arrDTI[i], nGPUs, nOldGPUs);
    }
}

void* reconfigureKernels(GenericFunction f, void* params){

    // if there is a function for reconfigurating the kernels,
    // run, else, return NULL
    if(f != NULL)
        return f(params);

    return NULL;
}


/* [DTIs] */

DTI_t* createAutomaticDTI(void* cpuData, size_t N, size_t size, const char* name, DTIDesctiption_t *description){

    // call DDM function to initialize the DTI
    DTI_t *DTI = initializeDTI(0, 0, cpuData, NULL, N, size, name, NULL, NULL, description);

    // add DTI to the global array of DTIs
    addDTI(DTI);

    // return DTI structure
    return DTI;
}

DTI_t* createManualDTI(void* cpuData, void** gpuData, size_t N, size_t size, const char *name, GenericFunction cpu2gpu, GenericFunction gpu2cpu){

    // call DDM function to initialize the DTI
    DTI_t *DTI = initializeDTI(1, 0, cpuData, gpuData, N, size, name, cpu2gpu, gpu2cpu, NULL);

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

DTI_t* getDTIByIndex(int i){

    if(i<maxDTI)
        return arrDTI[i];
    return NULL;
}

void setfDTICPU2GPU(DTI_t *DTI, GenericFunction f){

    DTI->moveCPU2GPU = f;
}

void setfDTIGPU2CPU(DTI_t *DTI, GenericFunction f){

    DTI->moveGPU2CPU = f;
}

void setDTIDescription(DTI_t *DTI, DTIDesctiption_t *description){

    DTI->description = description;
}


DTIDesctiption_t* getDTIDescription(DTI_t *DTI){

    return DTI->description;
}

void setCPUData(DTI_t *DTI, void* cpuData){

    DTI->cpuData = cpuData;
}

void* getCPUData(DTI_t *DTI){

    return DTI->cpuData;
}

void setMultiGPUData(DTI_t *DTI, void** gpuData){

    DTI->gpuData = gpuData;
}

void** getMultiGPUData(DTI_t *DTI){

    return DTI->gpuData;
}

void setGPUData(DTI_t *DTI, void *gpuData, int i){

    DTI->gpuData[i] = gpuData;
}

void* getGPUData(DTI_t *DTI, int i){

    return DTI->gpuData[i];
}

DTIDesctiption_t* initializeDTIDescription(transmissionPatternsEnum tpttEnum, remainingElementsEnum rmEnum){

    DTIDesctiption_t *description = (DTIDesctiption_t*)calloc(1, sizeof(DTIDesctiption_t));

    if(tpttEnum != all && tpttEnum != simple){

        printf(" -- 'initializeDTIDescription' only supports 'all' and 'simple' options. Use 'initializeComplexDTIDescription' or 'initializeCustomDTIDescription' instead.\n");
        return NULL;
    }

    description->tpttEnum = tpttEnum;
    description->rmEnum = rmEnum;

    return description;
}

void printDTI(DTI_t *DTI){

    size_t i, j;
    state_t *state = getState();
    size_t nGPUs = state->nGPUs;
    DTIDesctiption_t *description = DTI->description;

    for(i = 0; i<nGPUs; i++){

        printf(" -- Printing GPU %zu data --\n", i);

        printf("     Number of partitions = %zu\n", description->nPartitionsPerGPU[i]);
        printf("     Number of elements   = %zu\n", description->nElementsPerGPU[i]);

        for(j = 0; j<description->nPartitionsPerGPU[i]; j++){
            printf("     Number of elements in partition %zu = %zu\n", j, description->nElementsPerPartition[i][j]);
            printf("     First element of partition %zu = %zu\n", j, description->firstElementPerPartition[i][j]);
        }
        printf("\n");
    }
    fflush(stdout);
}


/* [Scheduler communication: notification signals] */



/* [Data transmission] */

void transferDataCPU2GPU(){

    size_t i;

    printf(" CPU2GPU ==> nDTI = %zu\n", nDTI);

    for(i = 0; i<nDTI; i++){

        // if automatic, call DTM for moving data. Else, call user-defined function
        if(arrDTI[i]->type == 0)
            cpyDataCPU2GPU(arrDTI[i]);
        else
            arrDTI[i]->moveCPU2GPU(arrDTI[i]);
    }
}

void transferDataGPU2CPU(){
    
    size_t i;

    printf(" GPU2CPU ==> nDTI = %zu\n", nDTI);

    for(i = 0; i<nDTI; i++){

        // if automatic, call DTM for moving data. Else, call user-defined function
        if(arrDTI[i]->type == 0)
            cpyDataGPU2CPU(arrDTI[i]);
        else
            arrDTI[i]->moveGPU2CPU(arrDTI[i]);
    }
}

/*
===========
| PRIVATE |
===========
*/

state_t* initState(size_t nGPUs, size_t *idGPUs){

    state_t *state = (state_t*)calloc(1,sizeof(state_t));
    state->nGPUs = nGPUs;
    state->idGPUs = idGPUs;

    appData->state = state;
    return state;
}

void freeState(state_t *state){

    if(state){

        if(state->idGPUs) 
            free(state->idGPUs);
        
        //if(state->stateDeallocator) state->stateDeallocator();

        free(state);
    }
}

/*reconfData_t* initReconfigurationData(){

    appReconfData = (reconfData_t*)calloc(1,sizeof(reconfData_t));
    return appReconfData;
}

void freeReconfigurationData(reconfData_t *reconfData){

    if(reconfData){

        if(reconfData->idGPUs) free(reconfData->idGPUs);

        free(reconfData);
    }
}*/

DTI_t* initializeDTI(size_t type, size_t dataLocation, void* cpuData, void** gpuData, size_t N, size_t size, const char* name, GenericFunction cpu2gpu, GenericFunction gpu2cpu, DTIDesctiption_t *description){

    // allocate memory
    DTI_t *DTI = (DTI_t*)calloc(1, sizeof(DTI_t));

    // intiialize data
    DTI->type = type;
    DTI->dataLocation = dataLocation; // CPU or GPU (unified memory in the future)

    // store DTI data
    DTI->cpuData = cpuData; // point to CPU data array
    DTI->gpuData = gpuData;
    DTI->N = N;
    DTI->size = size;
    DTI->name = name;

    // if type is > 0, then manually defined functions are used for the data transmission
    if(type > 0){
        
        DTI->moveCPU2GPU = cpu2gpu;
        DTI->moveGPU2CPU = gpu2cpu;
    }

    // store description if type is 0, since the description is used for the data redistribution
    if(type == 0){

        DTI->description = description; // used in automatic 
    }
    
    return DTI;
}

void freeDTI(DTI_t *DTI, size_t nGPUs){
    
    // TODO
    printf(" -- Not implemented yet!\n");
}

void freeDescription(DTIDesctiption_t *description, size_t nGPUs){

    if(description){

        if(description->n) 
            free(description->n);
        if(description->j) 
            free(description->j);
        if(description->off) 
            free(description->off);
        if(description->tn) 
            free(description->tn);


        if(description->nPartitionsPerGPU) 
            free(description->nPartitionsPerGPU);
        if(description->nElementsPerGPU) 
            free(description->nElementsPerGPU);

        for(size_t i = 0; i<nGPUs; i++){

            if(description->nElementsPerPartition[i]) 
                free(description->nElementsPerPartition[i]);
            if(description->firstElementPerPartition[i]) 
                free(description->firstElementPerPartition[i]);
        }

        if(description->nElementsPerPartition)
            free(description->nElementsPerPartition);

        if(description->firstElementPerPartition)
            free(description->firstElementPerPartition);
    }
}

/*void* runMockScheduler(void *arg){

    // launch reconfigurations
    sleep(2);
    printf(" Notifying reconfiguration\n");
    fflush(stdout);
    notifyReconfiguration(4, NULL, NULL, NULL);
    printf(" Reconfiguration notified\n");
    fflush(stdout);
}*/