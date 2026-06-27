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
#include "RMS.hpp"


/* [Global variables] */

// App Data
thread_local public_APP_data_t *appData = NULL;
thread_local reconfData_t *reconfData = NULL;
thread_local size_t *virtualTopology = NULL;

// DTIs
thread_local DTI_t** arrDTI = NULL;
thread_local size_t nDTI; 
thread_local size_t maxDTI;

/* [INITIALIZATION] */

void initDITTO(void *jobControl){

    // allocate memory for general information
    appData = (public_APP_Data_t*)calloc(1, sizeof(public_APP_Data_t));

    // initialize job control
    appData->jobControl = (jobControl_t*)jobControl; 

    // initialize application state (job resources)
    appData->state = initState(appData->jobControl->jobResources);

    // allocate memory for reconfData structure
    reconfData = (reconfData_t*)calloc(1, sizeof(reconfData_t));

    // initialize virtual topology
    virtualTopology = (size_t*)calloc(appData->state->jobResources->nGPUs, sizeof(size_t));
    for(size_t i = 0; i<appData->state->jobResources->nGPUs; i++){

        virtualTopology[i] = i;
    }

    // TODO: This should me moved to a cuda dependant place, and initialized only when communications are asynchronous
    appData->cudaStreams = (cudaStream_t*)malloc(8 * sizeof(cudaStream_t)); // 8 because we consider that it is the maximum number of GPUs we will use for now, in intra-node configurations
    initializeStreams(appData->jobControl->jobResources);

    // initialize DTI data
    nDTI = 0;
    maxDTI = 10;
    arrDTI = (DTI_t**)calloc(maxDTI, sizeof(DTI_t*));
}

// TODO
void freeDITTO(){

    // free all the memory used by DITTO

    // destroy streams
    destroyStreams(getState()->jobResources);
}

jobControl_t* getJobControl(){

    return appData->jobControl;
}



state_t* initState(jobResources_t *rmsJobResources){

    // allocat memory for state variable
    state_t *state = (state_t*)calloc(1, sizeof(state_t));

    // copy resources from jobControl
    cpyJobResourcesToState(state, rmsJobResources);
    
    // initialize reconfiguration resources
    state->reconfJobResources = NULL;

    // store state in the global variable and return
    appData->state = state; 

    return state;
}

state_t* getState(){

    return appData->state;
}

state_t* updateState(state_t *state, jobControl_t *jobControl){
    
    // update job resources
    updateApplicationResources(state);

    // return the new state
    return state;
}

state_t* storeState(state_t *state){

    printf(" -- Store state not implemented yet\n");
    return state;
}

// TODO
void freeState(state_t *state){

    if(state){

        freeJobResources(state->jobResources);
        free(state);
    }
}

void updateApplicationResources(state_t *state){

    // move reconfJobResources to jobResources
    freeJobResources(state->jobResources);
    state->jobResources = state->reconfJobResources;

    // initialize reconfJobResources pointer
    state->reconfJobResources = NULL;
}

// copy job resources to state (never used)
void cpyJobResourcesToState(state_t *state, jobResources_t *rmsJobResources){

    // deallocate job resources in case it is allocated
    if(state->jobResources)
        freeJobResources(state->jobResources);

    // copy rmsJobResources to state
    state->jobResources = cpyJobResources(rmsJobResources);
}


void cpyReconfResourcesToState(state_t *state, jobResources_t *reconfJobResources){

    // if it exist, deallocate old data
    freeJobResources(state->reconfJobResources);

    // copy job resources from the jobControl to the job state 
    state->reconfJobResources = cpyJobResources(reconfJobResources);
}

jobResources_t* cpyJobResources(jobResources_t *jobResources){

    // allocate memory for the new jobResources structure
    jobResources_t *cpJobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    
    // copy data
    cpJobResources->nGPUs = jobResources->nGPUs;
    cpJobResources->idGPUs = (size_t*)calloc(jobResources->nGPUs, sizeof(size_t));
    for(size_t i = 0; i<jobResources->nGPUs; i++){

        cpJobResources->idGPUs[i] = jobResources->idGPUs[i];
    }

    // return copy
    return cpJobResources;
}

void freeJobResources(jobResources_t *jobResources){
        
    if(jobResources){

        if(jobResources->idGPUs){

            free(jobResources->idGPUs);
        }

        free(jobResources);
    }
}

size_t getNumberOfGPUs(){

    return getState()->jobResources->nGPUs;
}

size_t* getGPUIds(){

    return getState()->jobResources->idGPUs;
}

cudaStream_t* getCudaStreams(){
    
    return appData->cudaStreams;
}


void reconfigure(reconfDirEnum reconfDir){

    // TODO: revise DDM and DTM functions
    // TODO: deallocations
    // TODO: GPU-CPU-GPU correction

    double tRecfg = 0.0, tConf = 0.0, tComm = 0.0, tGPU2CPU = 0.0, tCPU2GPU = 0.0;
    struct timespec startRecfg, endRecfg, startConf, endConf, startComm, endComm, startGPU2CPU, startCPU2GPU, endGPU2CPU, endCPU2GPU;

    jobResources_t *reconfJobResources, *jobResources;

    state_t *state = getState();
    jobControl_t *jobControl = getJobControl();

    // copy reconfiguration job resources to job
    cpyReconfResourcesToState(state, jobControl->reconfJobResources);

    // get job resources
    jobResources = state->jobResources;
    reconfJobResources = state->reconfJobResources;
    reconfTypeEnum reconfEnum;

    // set the reconfiguration type in case we perform a GPU2GPU reconfiguraiton 
    if(reconfDir == GPU2GPU)
    {
        if(jobResources->nGPUs < reconfJobResources->nGPUs) 
            reconfEnum = expand;
        else if(jobResources->nGPUs > reconfJobResources->nGPUs)
            reconfEnum = shrink;
        else 
            reconfEnum = keep;
    }

    // configure reconfiguration - GPUs to move data from each GPU (DDM)
    // if goes from GPU to GPU, it is necessary to decide which GPUs will use
    // each GPU
    if(reconfDir == GPU2GPU){
       
        clock_gettime(CLOCK_MONOTONIC, &startConf);
        if(reconfEnum == expand){

            configureExpansion(reconfJobResources, jobResources);
        }
        else if(reconfEnum == shrink){
            
            configureShrink(reconfJobResources, jobResources);
        }
        else{

            configureN2N(reconfJobResources, jobResources);
        }
        clock_gettime(CLOCK_MONOTONIC, &endConf);
        tConf += (endConf.tv_sec - startConf.tv_sec) + (endConf.tv_nsec - startConf.tv_nsec) / 1e9;
    }
    

    // if it goes through the CPU, move data to the CPU first
    if(reconfDir != GPU2GPU){

        clock_gettime(CLOCK_MONOTONIC, &startGPU2CPU);
        // move data from the GPUs to the CPU using the DTIs
        transferDataGPU2CPU(); // move data from the GPUs to the CPU
        clock_gettime(CLOCK_MONOTONIC, &endGPU2CPU);
        tGPU2CPU += (endGPU2CPU.tv_sec - startGPU2CPU.tv_sec) + (endGPU2CPU.tv_nsec - startGPU2CPU.tv_nsec) / 1e9;
        tComm += tGPU2CPU;
    }

    
    // generate metadata for moving to each GPUs (DDM)
    clock_gettime(CLOCK_MONOTONIC, &startConf);
    configureDTIs(jobResources, reconfJobResources, reconfDir);
    clock_gettime(CLOCK_MONOTONIC, &endConf);
    tConf += (endConf.tv_sec - startConf.tv_sec) + (endConf.tv_nsec - startConf.tv_nsec) / 1e9;


    // perform data redistributions (DTM)
    if(reconfDir == GPU2GPU){
        clock_gettime(CLOCK_MONOTONIC, &startComm);
        transferDataGPU2GPU(reconfEnum);
        clock_gettime(CLOCK_MONOTONIC, &endComm);
        tComm += (endComm.tv_sec - startComm.tv_sec) + (endComm.tv_nsec - startComm.tv_nsec) / 1e9;
    }

    // destroy old streams and initialize new ones
    destroyStreams(jobResources);
    initializeStreams(reconfJobResources);

    // update state
    updateState(state, jobControl);

    if(reconfDir != GPU2GPU){
        
        // move data again from the CPU to the GPUs
        clock_gettime(CLOCK_MONOTONIC, &startCPU2GPU);
        transferDataCPU2GPU();
        clock_gettime(CLOCK_MONOTONIC, &endCPU2GPU);
        tCPU2GPU += (endCPU2GPU.tv_sec - startCPU2GPU.tv_sec) + (endCPU2GPU.tv_nsec - startCPU2GPU.tv_nsec) / 1e9;
        tComm += tCPU2GPU;
    }

    // destroy reconfiguration context TODO
    //destroyReconfContext(); // deallocate variables that were only necessary for performing the reconfiguration
    // - gpusToSplit

    // notify that the reconfiguration has been done
    notifyReconfigurationDone(getJobControl());

    printf("%lf %lf %lf %lf ", tConf, tComm, tGPU2CPU, tCPU2GPU);
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

/*void transferDataGPU2GPU(){

    size_t i;

    for(i = 0; i<nDTI; i++){

        cpyDataGPU2GPU(arrDTI[i]);
    }
}*/

void transferDataGPU2GPU(reconfTypeEnum reconfEnum){

    size_t i;

    for(i = 0; i<nDTI; i++){
        
        if(reconfEnum == expand)
            reconfExpand(arrDTI[i]);
        else if(reconfEnum == shrink)
            reconfShrink(arrDTI[i]);
        else if(reconfEnum == keep)
            reconfN2N(arrDTI[i]);
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

void configureDTIs(jobResources_t *jobResources, jobResources_t *reconfJobResources, reconfDirEnum reconfDir){

    size_t i;

    // loop over all DTIs and configure following the description
    for(i = 0; i<nDTI; i++){

        configureDTI(arrDTI[i], jobResources, reconfJobResources, reconfDir);
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


DTIDesctiption_t* initializeDTIDescription(transmissionPatternsEnum tpttEnum, remainingElementsEnum rmEnum, communicationType_t commType){

    DTIDesctiption_t *description = (DTIDesctiption_t*)calloc(1, sizeof(DTIDesctiption_t));

    description->tpttEnum = tpttEnum;
    description->rmEnum = rmEnum;
    description->commType = commType;

    // return created DTI description
    return description;
}


DTIDesctiption_t* initializeComplexDTIDescription(transmissionPatternsEnum tpttEnum, remainingElementsEnum rmEnum, communicationType_t commType, size_t s){

    if(tpttEnum != complex){

        printf(" -- 'initializeComplexDTIDescription' only supports 'complex'option. Use 'initializeDTIDescription' instead.\n");
        exit(1);
    }

    if(s <= 0){

        printf(" -- s must be a positive integer value.\n");
        exit(1);
    }

    DTIDesctiption_t *description = initializeDTIDescription(tpttEnum,rmEnum, commType);
    description->s = s;


    // return created DTI description
    return description;
}

void printDTI(DTI_t *DTI){

    size_t i, j;
    state_t *state = getState();
    size_t nGPUs = getNumberOfGPUs();

    for(i = 0; i<nGPUs; i++){

        printf(" -- Printing GPU %zu data --\n", i);

        printf("     Number of elements = %zu\n", DTI->nPerGPU[i]);

        for(j = 0; j<DTI->nPartitionsPerGPU[i]; j++){
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
