#include <cstddef>
#include <omp.h>
#include <unistd.h>
#include <stdio.h>      
#include <stdlib.h>     
#include <pthread.h>
#include <time.h>

#include "DITO_API.hpp"
#include "priv_DITTO_API.hpp"
#include "DDM.hpp"
#include "DTM.hpp"
#include "mockSch.hpp"


/* [Global variables] */

// App Data
thread_local public_APP_data_t *appData = NULL;

// DTIs
thread_local DTI_t** arrDTI = NULL;
thread_local size_t nDTI; 
thread_local size_t maxDTI;


/* [INITIALIZATION] */

void initDITTO(void *jobControl){

    jobControl_t *jobCtrl = (jobControl_t*)jobControl;

    // allocate memory for appData structure
    appData = (public_APP_Data_t*)calloc(1, sizeof(public_APP_Data_t));

    //appData->reconfData = initReconfigurationData(); // initialize structure for storing storeing reconfiguration data
    appData->jobControl = jobCtrl; // communication with the scheduler

    // initialize application state // TODO: probably it should be a function
    appData->state = initState(jobCtrl->jobResources);


    // initialize array of DTIs
    printf("nDTI = %zu\n", nDTI);
    fflush(stdout);
    nDTI = 0;
    maxDTI = 10;
    arrDTI = (DTI_t**)calloc(maxDTI, sizeof(DTI_t*));
}

void freeDITTO(){

    printf(" -- freeDITTO() not implemented yet!\n");

    // free appData, 
}

jobControl_t* getJobControl(){

    return appData->jobControl;
}


/* [STATE] */


void freeJobResources(jobResources_t *jobResources){
        
    if(jobResources){

        if(jobResources->idGPUs){

            free(jobResources->idGPUs);
        }

        free(jobResources);
    }
}

void cpyJobResourcesToState(state_t *state, jobResources_t *rmsJobResources){

    jobResources_t *appJobResources = state->jobResources;

    // free old resources
    freeJobResources(state->jobResources);

    printf(" -- RMS GPUS = %zu\n", rmsJobResources->nGPUs);
    fflush(stdout);

    // allocate and copy job resources
    jobResources_t *jobResources = (jobResources_t*)calloc(1,sizeof(jobResources_t));
    state->jobResources = jobResources;

    // store the number of GPUs
    jobResources->nGPUs = rmsJobResources->nGPUs;
    
    // allocate array for GPU identifiers
    jobResources->idGPUs = (size_t*)malloc(jobResources->nGPUs * sizeof(size_t));
    for(size_t i = 0; i<jobResources->nGPUs; i++)
        jobResources->idGPUs[i] = rmsJobResources->idGPUs[i];
}

state_t* initState(jobResources_t *rmsJobResources){

    // allocat memory for state variable
    state_t *state = (state_t*)calloc(1,sizeof(state_t));

    // copy job resources from the RMS structure to the application structure
    cpyJobResourcesToState(state, rmsJobResources);

    // store state in the global variable and return
    appData->state = state; 
    return state;
}

void freeState(state_t *state){

    if(state){

        freeJobResources(state->jobResources);
        free(state);
    }
}

state_t* getState(){

    return appData->state;
}

state_t* updateState(state_t *state, jobControl_t *jobControl){
    
    // update job resources
    cpyJobResourcesToState(state, jobControl->jobResources);

    // return the new state
    return state;
}

state_t* storeState(state_t *state){

    printf(" -- Store state not implemented yet\n");
    return state;
}


size_t getNumberOfGPUs(){

    return getState()->jobResources->nGPUs;
}

size_t* getGPUIds(){

    return getState()->jobResources->idGPUs;
}

void reconfigure(){

    double tGPU2CPU = 0.0, tCPU2GPU = 0.0, tRecfg = 0.0;
    struct timespec startGPU2CPU, startCPU2GPU, startRecfg, endGPU2CPU, endCPU2GPU, endRecfg;

    state_t *state = getState();
    jobControl_t *jobControl = getJobControl();

    size_t nGPUs, nOldGPUs;

    // number of GPUs before reconfiguration
    nOldGPUs = getNumberOfGPUs();

    clock_gettime(CLOCK_MONOTONIC, &startGPU2CPU);
    if(nOldGPUs>0){

        // move data from the GPUs to the CPU using the DTIs
        transferDataGPU2CPU(); // move data from the GPUs to the CPU
    }
    clock_gettime(CLOCK_MONOTONIC, &endGPU2CPU);
    tGPU2CPU = (endGPU2CPU.tv_sec - startGPU2CPU.tv_sec) + (endGPU2CPU.tv_nsec - startGPU2CPU.tv_nsec) / 1e9;

    // update state using the information provided by the jobController
    clock_gettime(CLOCK_MONOTONIC, &startRecfg);
    updateState(state, jobControl);
    clock_gettime(CLOCK_MONOTONIC, &endRecfg);
    tRecfg += (endRecfg.tv_sec - startRecfg.tv_sec) + (endRecfg.tv_nsec - startRecfg.tv_nsec) / 1e9;


    // number of GPUs after reconfiguration
    nGPUs = getNumberOfGPUs();
    
    // reconfigure DTIs and resend data to the GPUs, if necessary
    if(nGPUs>0){

        // configure DTIs
        clock_gettime(CLOCK_MONOTONIC, &startRecfg);
        configureDTIs(nGPUs, nOldGPUs);
        clock_gettime(CLOCK_MONOTONIC, &endRecfg);
        tRecfg += (endRecfg.tv_sec - startRecfg.tv_sec) + (endRecfg.tv_nsec - startRecfg.tv_nsec) / 1e9;

        // move data again from the CPU to the GPUs
        clock_gettime(CLOCK_MONOTONIC, &startCPU2GPU);
        transferDataCPU2GPU();
        clock_gettime(CLOCK_MONOTONIC, &endCPU2GPU);
        tCPU2GPU += (endCPU2GPU.tv_sec - startCPU2GPU.tv_sec) + (endCPU2GPU.tv_nsec - startCPU2GPU.tv_nsec) / 1e9;

        // reconfigure the kernels
        reconfigureKernels(NULL, NULL);
    }

    printf("%zu %zu %lf %lf %lf", nOldGPUs, nGPUs, tGPU2CPU, tRecfg, tCPU2GPU);

    // notify that the reconfiguration has been done
    notifyReconfigurationDone(getJobControl());
}

// TODO
void* reconfigureKernels(GenericFunction f, void* params){

    // if there is a function for reconfigurating the kernels,
    // run, else, return NULL
    if(f != NULL)
        return f(params);

    return NULL;
}

void transferDataCPU2GPU(){

    size_t i;

    for(i = 0; i<nDTI; i++){

        // if automatic, call DTM for moving data. Else, call user-defined function
        if(arrDTI[i]->type == 0){
            cpyDataCPU2GPU(arrDTI[i]);
        }
        else{
            arrDTI[i]->moveCPU2GPU(arrDTI[i]);
        }
    }
}

void transferDataGPU2CPU(){
    
    size_t i;

    for(i = 0; i<nDTI; i++){

        // if automatic, call DTM for moving data. Else, call user-defined function
        if(arrDTI[i]->type == 0){
            cpyDataGPU2CPU(arrDTI[i]);
        }
        else{
            arrDTI[i]->moveGPU2CPU(arrDTI[i]);
        }
    }
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

void addDTI(DTI_t *DTI){

    // array full, expand // TODO: std::vector
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

void configureDTIs(size_t nGPUs, size_t nOldGPUs){

    size_t i;

    // loop over all DTIs and configure following the description
    for(i = 0; i<nDTI; i++){

        configureDTI(arrDTI[i], nGPUs, nOldGPUs);
    }
}

DTI_t* getDTI(const char *dtiName){

    size_t i = 0;
    int found = 0;

    // found DTI by name in the array of DTIs
    while(!found && i < nDTI){

        if(arrDTI[i]->name == dtiName){
            return arrDTI[i];
        }
    }
    
    // not found
    return NULL;
}

// get DTI by index
DTI_t* getDTIByIndex(int i){

    if(i<maxDTI)
        return arrDTI[i];
    return NULL;
}

DTIDesctiption_t* initializeDTIDescription(transmissionPatternsEnum tpttEnum, remainingElementsEnum rmEnum){

    DTIDesctiption_t *description = (DTIDesctiption_t*)calloc(1, sizeof(DTIDesctiption_t));

    if(tpttEnum != entire && tpttEnum != simple){

        printf(" -- 'initializeDTIDescription' only supports 'all' and 'simple' options. Use 'initializeComplexDTIDescription' instead.\n");
        exit(1);
    }

    description->tpttEnum = tpttEnum;
    description->rmEnum = rmEnum;
    description->nPartitions = 1;

    // return created DTI description
    return description;
}

DTIDesctiption_t* initializeComplexDTIDescription(transmissionPatternsEnum tpttEnum, remainingElementsEnum rmEnum, size_t nPartitions){

    DTIDesctiption_t *description = (DTIDesctiption_t*)calloc(1, sizeof(DTIDesctiption_t));

    if(tpttEnum != complex){

        printf(" -- 'initializeComplexDTIDescription' only supports 'complex'option. Use 'initializeDTIDescription' instead.\n");
        exit(1);
    }

    if(nPartitions <= 0){

        printf(" -- nPartitions must be a positive integer value.\n");
        exit(1);
    }

    description->tpttEnum = tpttEnum;
    description->rmEnum = rmEnum;
    description->nPartitions = nPartitions;

    // return created DTI description
    return description;
}

void printDTI(DTI_t *DTI){

    size_t i, j;
    state_t *state = getState();
    size_t nGPUs = getNumberOfGPUs();
    DTIDesctiption_t *description = DTI->description;

    for(i = 0; i<nGPUs; i++){

        printf(" -- Printing GPU %zu data --\n", i);

        printf("     Number of elements   = %zu\n", DTI->nPerGPU[i]);

        for(j = 0; j<description->nPartitions; j++){
            printf("     Number of elements in partition %zu = %zu\n", j, DTI->nPerPartition[i][j]);
            printf("     First element of partition %zu = %zu\n", j, DTI->offsetPerPartition[i][j]);
        }
        printf("\n");
    }
    fflush(stdout);
}



void freeDTI(DTI_t *DTI, size_t nGPUs){
    
    // TODO
    printf(" -- Not implemented yet!\n");
}

void freeDescription(DTIDesctiption_t *description, size_t nGPUs){

    if(description){
        
        free(description);
    }
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
