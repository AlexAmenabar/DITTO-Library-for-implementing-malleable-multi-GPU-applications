#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <unistd.h>
#include <cstddef>
#include <cstring>
#include <stdlib.h>     
#include <pthread.h>
#include <time.h>

#ifndef TESTRMS

// include CUDA related libraries
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvml.h>
#include <nccl.h>

// include apps that use CUDA
#include "../testApps/toy_app_malleable.hpp"

#endif

#include "DITO_API.hpp"
#include "RMS.hpp"
#include "SCH.hpp"
#include "RCF.hpp"
#include "jobQueue.hpp"
#include "eventQueue.hpp"


// initialize global variables
schInfo_t *schInfo;
int timer = 0;
FILE *fEventRecord = NULL, *fUsage = NULL, *fOutput = NULL;
size_t nextJobId = 1;

int invoqueScheduler = 0;

size_t usageGPUs = 0;
double usagePower = 0.0;
size_t registeredUsages = 0;

// temp
int *gpuTopology;
int *gpuTopologyRank;
size_t gNGPUs;

// app functions
void expandTest1(int argc, void* argv[]){

    size_t i;

    // initialize DITTO environment
    initDITTO(argv[argc+1]);

    // APP: initialize data structure used by the application
    appStruct_t *appData = (appStruct_t*)calloc(1, sizeof(appStruct_t));
    appData->N = *(size_t*)argv[0];
    appData->s = *(size_t*)argv[1];
    //appData->async = *(size_t*)argv[2];

    // communication info
    size_t pinned = *(size_t*)argv[2];
    size_t async = *(size_t*)argv[3];
    size_t steps = *(size_t*)argv[4];
    size_t cores = *(size_t*)argv[5];

    appData->malleable = *(size_t*)argv[argc];

    // allocate memory for App Data
    if(pinned) // pinned memory
        cudaMallocHost(&appData->arr, appData->N * sizeof(float));
    else
        appData->arr = (float*)calloc(appData->N, sizeof(float));
    
    // initialize data
    for(i = 0; i<appData->N; i++)
        appData->arr[i] = (float)i;//(float)rand()/(float)(RAND_MAX);

    
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
        appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->N, sizeof(float), "appData", initializeComplexDTIDescription(complex, ordered, commType, appData->s));
    else
        appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->N, sizeof(float), "appData", initializeDTIDescription(simple, ordered, commType));

    for(size_t i = 0; i<appData->N; i++){

        printf("%f ", appData->arr[i]);
    }
    printf("\n");
    fflush(stdout);


    // configure all DTIs for automatic data transference // TODO: configure in initialization
    configureDTIs(getState()->jobResources, NULL, CPU2GPU); // number of GPUs, old number of GPUs

    // send data to the GPU
    transferDataCPU2GPU();

    struct timespec start, end;
    double etime = 0.0;

    // wait reconfiguration
    while(checkIfReconfiguration(getJobControl()) == 0){
        
        /*printf(" -- Checking reconf\n");
        fflush(stdout);*/ 
        sleep(1);
    }

    // print GPUs
    printf(" -- Printing current GPU configuration (%zu): ", getJobControl()->jobResources->nGPUs);
    for(size_t i = 0; i<getJobControl()->jobResources->nGPUs; i++){
        printf("%zu ", getJobControl()->jobResources->idGPUs[i]);
    }
    printf("\n");
    printf(" -- Printing GPU configuration for reconfiguration (%zu): ", getJobControl()->reconfJobResources->nGPUs);
    for(size_t i = 0; i<getJobControl()->reconfJobResources->nGPUs; i++){
        printf("%zu ", getJobControl()->reconfJobResources->idGPUs[i]);
    }
    printf("\n");
    fflush(stdout);


    clock_gettime(CLOCK_MONOTONIC, &start);
    reconfigure(GPU2GPU);
    clock_gettime(CLOCK_MONOTONIC, &end);
    etime += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("Reconfiguration finished in: %lf", etime);
    fflush(stdout);

    // transfer data fron the GPUs to the CPU
    for(i = 0; i<appData->N; i++){
        ((float*)(appDataDTI->cpuData))[i] = 0;
    }

    printf(" APP: moving data from GPU to CPU\n");
    fflush(stdout);
    transferDataGPU2CPU();
    printf(" APP: data from GPU to CPU moved\n");
    fflush(stdout);

    for(size_t i = 0; i<appData->N; i++){

        printf("%f ", appData->arr[i]);
    }
    printf("\n");
    fflush(stdout);

    int correct = 1;
    for(i = 0; i<appData->N; i++){
        if( ((float*)(appDataDTI->cpuData))[i] != (float)i){
            correct = 0;
            break;        
        }
    }
    
    if(!correct)
        printf(" -- [APP]: Incorrect result!!!");
    else
        printf(" -- [APP]: Correct results!!!!");

    if(pinned)
        cudaFreeHost(appDataDTI->cpuData);
    else
        free(appDataDTI->cpuData);


    // destroy streams
    freeDITTO();

    // signal that job finished
    jobFinished(getJobControl());

    pthread_exit(NULL);
}



// app functions
void expandTest2(int argc, void* argv[]){

    size_t i;

    // initialize DITTO environment
    initDITTO(argv[argc+1]);

    // APP: initialize data structure used by the application
    appStruct_t *appData = (appStruct_t*)calloc(1, sizeof(appStruct_t));
    appData->N = *(size_t*)argv[0];
    appData->s = *(size_t*)argv[1];
    //appData->async = *(size_t*)argv[2];

    // communication info
    size_t pinned = *(size_t*)argv[2];
    size_t async = *(size_t*)argv[3];
    size_t steps = *(size_t*)argv[4];
    size_t cores = *(size_t*)argv[5];

    appData->malleable = *(size_t*)argv[argc];

    // allocate memory for App Data
    if(pinned) // pinned memory
        cudaMallocHost(&appData->arr, appData->N * sizeof(float));
    else
        appData->arr = (float*)calloc(appData->N, sizeof(float));
    
    // initialize data
    for(i = 0; i<appData->N; i++)
        appData->arr[i] = (float)i;//(float)rand()/(float)(RAND_MAX);

    
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
        appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->N, sizeof(float), "appData", initializeComplexDTIDescription(complex, ordered, commType, appData->s));
    else
        appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->N, sizeof(float), "appData", initializeDTIDescription(simple, ordered, commType));

    for(size_t i = 0; i<appData->N; i++){

        printf("%f ", appData->arr[i]);
    }
    printf("\n");
    fflush(stdout);


    // configure all DTIs for automatic data transference // TODO: configure in initialization
    configureDTIs(getState()->jobResources, NULL, CPU2GPU); // number of GPUs, old number of GPUs

    // send data to the GPU
    transferDataCPU2GPU();

    struct timespec start, end;
    double etime = 0.0;

    // first reconfiguration
    while(checkIfReconfiguration(getJobControl()) == 0){
        
        /*printf(" -- Checking reconf\n");
        fflush(stdout);*/ 
        sleep(1);
    }

    // print GPUs
    printf(" -- Printing current GPU configuration (%zu): ", getJobControl()->jobResources->nGPUs);
    for(size_t i = 0; i<getJobControl()->jobResources->nGPUs; i++){
        printf("%zu ", getJobControl()->jobResources->idGPUs[i]);
    }
    printf("\n");
    printf(" -- Printing GPU configuration for reconfiguration (%zu): ", getJobControl()->reconfJobResources->nGPUs);
    for(size_t i = 0; i<getJobControl()->reconfJobResources->nGPUs; i++){
        printf("%zu ", getJobControl()->reconfJobResources->idGPUs[i]);
    }
    printf("\n");
    fflush(stdout);


    clock_gettime(CLOCK_MONOTONIC, &start);
    reconfigure(GPU2GPU);
    clock_gettime(CLOCK_MONOTONIC, &end);
    etime += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("Reconfiguration finished in: %lf", etime);
    fflush(stdout);


    // second reconfiguration
    etime = 0.0;

    // wait reconfiguration
    while(checkIfReconfiguration(getJobControl()) == 0){
        
        /*printf(" -- Checking reconf\n");
        fflush(stdout);*/ 
        sleep(1);
    }

    // print GPUs
    printf(" -- Printing current GPU configuration (%zu): ", getJobControl()->jobResources->nGPUs);
    for(size_t i = 0; i<getJobControl()->jobResources->nGPUs; i++){
        printf("%zu ", getJobControl()->jobResources->idGPUs[i]);
    }
    printf("\n");
    printf(" -- Printing GPU configuration for reconfiguration (%zu): ", getJobControl()->reconfJobResources->nGPUs);
    for(size_t i = 0; i<getJobControl()->reconfJobResources->nGPUs; i++){
        printf("%zu ", getJobControl()->reconfJobResources->idGPUs[i]);
    }
    printf("\n");
    fflush(stdout);


    clock_gettime(CLOCK_MONOTONIC, &start);
    reconfigure(GPU2GPU);
    clock_gettime(CLOCK_MONOTONIC, &end);
    etime += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("Reconfiguration finished in: %lf", etime);
    fflush(stdout);


    // transfer data fron the GPUs to the CPU
    for(i = 0; i<appData->N; i++){
        ((float*)(appDataDTI->cpuData))[i] = 0;
    }

    printf(" APP: moving data from GPU to CPU\n");
    fflush(stdout);
    transferDataGPU2CPU();
    printf(" APP: data from GPU to CPU moved\n");
    fflush(stdout);

    for(size_t i = 0; i<appData->N; i++){

        printf("%f ", appData->arr[i]);
    }
    printf("\n");
    fflush(stdout);

    int correct = 1;
    for(i = 0; i<appData->N; i++){
        if( ((float*)(appDataDTI->cpuData))[i] != (float)i){
            correct = 0;
            break;        
        }
    }
    
    if(!correct)
        printf(" -- [APP]: Incorrect result!!!");
    else
        printf(" -- [APP]: Correct results!!!!");

    if(pinned)
        cudaFreeHost(appDataDTI->cpuData);
    else
        free(appDataDTI->cpuData);


    // destroy streams
    freeDITTO();

    // signal that job finished
    jobFinished(getJobControl());

    pthread_exit(NULL);
}


// RMS function


int reconfExpandTest1(schInfo_t *schInfo){

    size_t nGPUs = schInfo->nGPUs;

    size_t nCommunicationModes = 8;
    const char *communicationModes[8] = {"OASYNC", "OCASYNC", "SASYNC", "SCASYNC","OPASYNC", "OPCASYNC", "SPASYNC", "SPCASYNC"};
    const size_t async[8] =  {1, 1, 1, 1, 1, 1, 1, 1};
    const size_t pinned[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    const size_t steps[8] =  {0, 0, 2, 2, 0, 0, 2, 2};
    const size_t cores[8] =  {0, 1, 0, 1, 0, 1, 0, 1};

    // weak scalability
    size_t N = (size_t)1024 / (size_t)4;

    // n partitions
    size_t nPartitions = 3;
    size_t arrPartitions[3] = {1, 8, 64}; // number of partitions per GPU


    // argv for the input of the application
    void* jargs[8];

    for(size_t gpus = 1; gpus<= nGPUs / 2; gpus *= 2){

        for(size_t dstgpus = gpus * 2; dstgpus <= nGPUs; dstgpus*=2){

            for(size_t nP = 0; nP<nPartitions; nP++){

                for(size_t commMode = 0; commMode < nCommunicationModes; commMode++){
                                                
                    // set job resources for experiment
                    jobResources_t *jobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
                    jobResources->nGPUs = gpus;
                    jobResources->idGPUs = (size_t*)malloc(jobResources->nGPUs * sizeof(size_t));
                    for(size_t g = 0; g<gpus; g++){
                        jobResources->idGPUs[g] = g;
                    }
                    

                    // weak scalability
                    //ja = N; // the amount of data is fixed, each partition changes depending on the number of GPUs and partitions
                    // each partition has a length of N, and each partition is divided into for all the GPUs
                    //jS = 0;//arrPartitions[nP]; // number of partitions per GPU
                    
                    // weak scalability
                    size_t ja = N; // the amount of data is fixed, each partition changes depending on the number of GPUs and partitions
                    size_t jS = arrPartitions[nP]; // number of partitions per GPU

                    // communication method
                    size_t jPinned = pinned[commMode];
                    size_t jAsync = async[commMode];
                    size_t jSteps = steps[commMode];
                    size_t jCores = cores[commMode];
                    size_t jmall = 1;
                    jargs[0] = &ja;
                    jargs[1] = &jS;
                    jargs[2] = &jPinned;
                    jargs[3] = &jAsync; // async
                    jargs[4] = &jSteps;
                    jargs[5] = &jCores;
                    jargs[6] = &jmall;

                    // launch job
                    jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));
                    jobLauncher->jobType = MALLEABLE;
                    jobLauncher->jobPriority = LOW;
                    jobLauncher->nReqGPUs = 8; // no matter
                    jobLauncher->nReqMinGPUs = 1;
                    jobLauncher->launchTimeStep = 1; // no matter
                    jobLauncher->appType = 2;
                    jobLauncher->argc = 6;
                    jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
                    jobLauncher->launchFunc = &expandTest1;

                    printf(" -- [RMS]: [%s, %zu, %zu, %zu, %zu]: ", communicationModes[commMode], ja, jS, arrPartitions[nP], gpus);
                    fflush(stdout);
                    addPendingJob(jobLauncher); // add to pending list
                    launchJob(getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources); // launch from pending list
                    printf(" -- [RMS]: Job launched!\n");
                    fflush(stdout);

                    // get job control
                    jobControl_t *jobControl = getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl;
                    
                    // schedule reconfiguration
                    jobResources_t *reconfJobResources = (jobResources_t*)calloc(1,sizeof(jobResources_t));
                    reconfJobResources->nGPUs = dstgpus;
                    reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));
                    for(size_t j = 0; j<reconfJobResources->nGPUs; j++)
                        reconfJobResources->idGPUs[j] = j;

                    // schedule
                    scheduleReconfiguration(getJobFromQueue(&(schInfo->runningJobs), 0), 0, reconfJobResources);

                    // wait until the reconfiguration finishes and finish it

                    printf(" [RMS]: checking reconfiguration done\n");
                    fflush(stdout);
                    while(checkReconfigurationDone(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl) == 1){
                        sleep(0.1);
                    }
                    printf(" [RMS]: finishing reconfiguration\n");
                    fflush(stdout);
                    jobFinishedReconfiguration(getJobFromQueue(&(schInfo->reconfiguringJobs), 0), 0);


                    // wait until job finishes and finish job
                    printf(" [RMS]: checking if job finished\n");
                    fflush(stdout);
                    while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                        sleep(1);
                    }

                    printf(" [RMS]: finishing job\n");
                    fflush(stdout);
                    finishJob(getJobFromQueue(&(schInfo->runningJobs), 0), 0);
                    removeJobFromQueueByIndex(&schInfo->finishedJobs, 0);


                    if(jobControl->jobResources){
                        free(jobControl->jobResources->idGPUs);
                        free(jobControl->jobResources);
                    }

                    
                    printf(" -- [RMS]: Moving to next job!\n");
                    fflush(stdout);

                    printf("\n");
                    fflush(stdout);  
                }
            }
        }
    }

    return 1;
}


int reconfExpandTest2(schInfo_t *schInfo){

    size_t nGPUs = schInfo->nGPUs;

    size_t nCommunicationModes = 8;
    const char *communicationModes[8] = {"OASYNC", "OCASYNC", "SASYNC", "SCASYNC","OPASYNC", "OPCASYNC", "SPASYNC", "SPCASYNC"};
    const size_t async[8] =  {1, 1, 1, 1, 1, 1, 1, 1};
    const size_t pinned[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    const size_t steps[8] =  {0, 0, 2, 2, 0, 0, 2, 2};
    const size_t cores[8] =  {0, 1, 0, 1, 0, 1, 0, 1};

    // weak scalability
    size_t N = (size_t)1024 / (size_t)4;

    // n partitions
    size_t nPartitions = 3;
    size_t arrPartitions[3] = {1, 8, 64}; // number of partitions per GPU


    // argv for the input of the application
    void* jargs[8];

    for(size_t gpus = 1; gpus<= nGPUs / 2; gpus *= 2){

        for(size_t dstgpus = gpus * 2; dstgpus <= nGPUs; dstgpus*=2){

            for(size_t nP = 0; nP<nPartitions; nP++){

                for(size_t commMode = 0; commMode < nCommunicationModes; commMode++){
                                                
                    // set job resources for experiment
                    jobResources_t *jobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
                    jobResources->nGPUs = gpus;
                    jobResources->idGPUs = (size_t*)malloc(jobResources->nGPUs * sizeof(size_t));
                    for(size_t g = 0; g<gpus; g++){
                        jobResources->idGPUs[g] = g;
                    }
                    

                    // weak scalability
                    //ja = N; // the amount of data is fixed, each partition changes depending on the number of GPUs and partitions
                    // each partition has a length of N, and each partition is divided into for all the GPUs
                    //jS = 0;//arrPartitions[nP]; // number of partitions per GPU
                    
                    // weak scalability
                    size_t ja = N; // the amount of data is fixed, each partition changes depending on the number of GPUs and partitions
                    size_t jS = arrPartitions[nP]; // number of partitions per GPU

                    // communication method
                    size_t jPinned = pinned[commMode];
                    size_t jAsync = async[commMode];
                    size_t jSteps = steps[commMode];
                    size_t jCores = cores[commMode];
                    size_t jmall = 1;
                    jargs[0] = &ja;
                    jargs[1] = &jS;
                    jargs[2] = &jPinned;
                    jargs[3] = &jAsync; // async
                    jargs[4] = &jSteps;
                    jargs[5] = &jCores;
                    jargs[6] = &jmall;

                    // launch job
                    jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));
                    jobLauncher->jobType = MALLEABLE;
                    jobLauncher->jobPriority = LOW;
                    jobLauncher->nReqGPUs = 8; // no matter
                    jobLauncher->nReqMinGPUs = 1;
                    jobLauncher->launchTimeStep = 1; // no matter
                    jobLauncher->appType = 2;
                    jobLauncher->argc = 6;
                    jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
                    jobLauncher->launchFunc = &expandTest1;

                    printf(" -- [RMS]: [%s, %zu, %zu, %zu, %zu]: ", communicationModes[commMode], ja, jS, arrPartitions[nP], gpus);
                    fflush(stdout);
                    addPendingJob(jobLauncher); // add to pending list
                    launchJob(getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources); // launch from pending list
                    printf(" -- [RMS]: Job launched!\n");
                    fflush(stdout);

                    // get job control
                    jobControl_t *jobControl = getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl;
                    
                    // schedule reconfiguration
                    jobResources_t *reconfJobResources = (jobResources_t*)calloc(1,sizeof(jobResources_t));
                    reconfJobResources->nGPUs = dstgpus;
                    reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));
                    for(size_t j = 0; j<reconfJobResources->nGPUs; j++)
                        reconfJobResources->idGPUs[j] = nGPUs - 1 - j;

                    // schedule
                    scheduleReconfiguration(getJobFromQueue(&(schInfo->runningJobs), 0), 0, reconfJobResources);

                    // wait until the reconfiguration finishes and finish it

                    printf(" [RMS]: checking reconfiguration done\n");
                    fflush(stdout);
                    while(checkReconfigurationDone(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl) == 1){
                        sleep(0.1);
                    }
                    printf(" [RMS]: finishing reconfiguration\n");
                    fflush(stdout);
                    jobFinishedReconfiguration(getJobFromQueue(&(schInfo->reconfiguringJobs), 0), 0);


                    // wait until job finishes and finish job
                    printf(" [RMS]: checking if job finished\n");
                    fflush(stdout);
                    while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                        sleep(1);
                    }

                    printf(" [RMS]: finishing job\n");
                    fflush(stdout);
                    finishJob(getJobFromQueue(&(schInfo->runningJobs), 0), 0);
                    removeJobFromQueueByIndex(&schInfo->finishedJobs, 0);


                    if(jobControl->jobResources){
                        free(jobControl->jobResources->idGPUs);
                        free(jobControl->jobResources);
                    }

                    
                    printf(" -- [RMS]: Moving to next job!\n");
                    fflush(stdout);

                    printf("\n");
                    fflush(stdout);  
                }
            }
        }
    }

    return 1;
}

// manual reconfigurations
int reconfExpandTest3(schInfo_t *schInfo){

    size_t nGPUs = schInfo->nGPUs;

    size_t nCommunicationModes = 8;
    const char *communicationModes[8] = {"OASYNC", "OCASYNC", "SASYNC", "SCASYNC","OPASYNC", "OPCASYNC", "SPASYNC", "SPCASYNC"};
    const size_t async[8] =  {1, 1, 1, 1, 1, 1, 1, 1};
    const size_t pinned[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    const size_t steps[8] =  {0, 0, 2, 2, 0, 0, 2, 2};
    const size_t cores[8] =  {0, 1, 0, 1, 0, 1, 0, 1};

    // weak scalability
    size_t N = (size_t)1024 / (size_t)4;

    // n partitions
    size_t nPartitions = 3;
    size_t arrPartitions[3] = {1, 8, 64}; // number of partitions per GPU


    // argv for the input of the application
    void* jargs[8];

    for(size_t nP = 0; nP<nPartitions; nP++){

        for(size_t commMode = 0; commMode < nCommunicationModes; commMode++){
                       
            size_t gpus = 1;

            // set job resources for experiment
            jobResources_t *jobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
            jobResources->nGPUs = gpus;
            jobResources->idGPUs = (size_t*)malloc(jobResources->nGPUs * sizeof(size_t));
            jobResources->idGPUs[0] = 0;

            // weak scalability
            //ja = N; // the amount of data is fixed, each partition changes depending on the number of GPUs and partitions
            // each partition has a length of N, and each partition is divided into for all the GPUs
            //jS = 0;//arrPartitions[nP]; // number of partitions per GPU
            
            // weak scalability
            size_t ja = N; // the amount of data is fixed, each partition changes depending on the number of GPUs and partitions
            size_t jS = arrPartitions[nP]; // number of partitions per GPU

            // communication method
            size_t jPinned = pinned[commMode];
            size_t jAsync = async[commMode];
            size_t jSteps = steps[commMode];
            size_t jCores = cores[commMode];
            size_t jmall = 1;
            jargs[0] = &ja;
            jargs[1] = &jS;
            jargs[2] = &jPinned;
            jargs[3] = &jAsync; // async
            jargs[4] = &jSteps;
            jargs[5] = &jCores;
            jargs[6] = &jmall;

            // launch job
            jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));
            jobLauncher->jobType = MALLEABLE;
            jobLauncher->jobPriority = LOW;
            jobLauncher->nReqGPUs = 8; // no matter
            jobLauncher->nReqMinGPUs = 1;
            jobLauncher->launchTimeStep = 1; // no matter
            jobLauncher->appType = 2;
            jobLauncher->argc = 6;
            jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
            jobLauncher->launchFunc = &expandTest2;

            printf(" -- [RMS]: [%s, %zu, %zu, %zu, %zu]: ", communicationModes[commMode], ja, jS, arrPartitions[nP], gpus);
            fflush(stdout);
            addPendingJob(jobLauncher); // add to pending list
            launchJob(getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources); // launch from pending list
            printf(" -- [RMS]: Job launched!\n");
            fflush(stdout);

            // get job control
            jobControl_t *jobControl = getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl;
            
            // schedule reconfiguration
            jobResources_t *reconfJobResources = (jobResources_t*)calloc(1,sizeof(jobResources_t));
            reconfJobResources->nGPUs = 2;
            reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));
            reconfJobResources->idGPUs[0] = 0;
            reconfJobResources->idGPUs[1] = 1;

            // schedule
            scheduleReconfiguration(getJobFromQueue(&(schInfo->runningJobs), 0), 0, reconfJobResources);

            // wait until the reconfiguration finishes and finish it

            printf(" [RMS]: checking reconfiguration done\n");
            fflush(stdout);
            while(checkReconfigurationDone(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl) == 1){
                sleep(0.1);
            }
            printf(" [RMS]: finishing reconfiguration\n");
            fflush(stdout);
            jobFinishedReconfiguration(getJobFromQueue(&(schInfo->reconfiguringJobs), 0), 0);


            // second reconfiguration
            reconfJobResources = (jobResources_t*)calloc(1,sizeof(jobResources_t));
            reconfJobResources->nGPUs = 4;
            reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));
            reconfJobResources->idGPUs[0] = 3;
            reconfJobResources->idGPUs[1] = 1;
            reconfJobResources->idGPUs[2] = 0;
            reconfJobResources->idGPUs[3] = 2;

            // schedule
            scheduleReconfiguration(getJobFromQueue(&(schInfo->runningJobs), 0), 0, reconfJobResources);

            // wait until the reconfiguration finishes and finish it

            printf(" [RMS]: checking reconfiguration done\n");
            fflush(stdout);
            while(checkReconfigurationDone(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl) == 1){
                sleep(0.1);
            }
            printf(" [RMS]: finishing reconfiguration\n");
            fflush(stdout);
            jobFinishedReconfiguration(getJobFromQueue(&(schInfo->reconfiguringJobs), 0), 0);



            // wait until job finishes and finish job
            printf(" [RMS]: checking if job finished\n");
            fflush(stdout);
            while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                sleep(1);
            }

            printf(" [RMS]: finishing job\n");
            fflush(stdout);
            finishJob(getJobFromQueue(&(schInfo->runningJobs), 0), 0);
            removeJobFromQueueByIndex(&schInfo->finishedJobs, 0);


            if(jobControl->jobResources){
                free(jobControl->jobResources->idGPUs);
                free(jobControl->jobResources);
            }

            
            printf(" -- [RMS]: Moving to next job!\n");
            fflush(stdout);

            printf("\n");
            fflush(stdout);
        }
    }

    return 1;
}


// topology-aware
int reconfExpandTest4(schInfo_t *schInfo){

    size_t nGPUs = schInfo->nGPUs;

    size_t nCommunicationModes = 8;
    const char *communicationModes[8] = {"OASYNC", "OCASYNC", "SASYNC", "SCASYNC","OPASYNC", "OPCASYNC", "SPASYNC", "SPCASYNC"};
    const size_t async[8] =  {1, 1, 1, 1, 1, 1, 1, 1};
    const size_t pinned[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    const size_t steps[8] =  {0, 0, 2, 2, 0, 0, 2, 2};
    const size_t cores[8] =  {0, 1, 0, 1, 0, 1, 0, 1};

    // weak scalability
    size_t N = (size_t)1024 / (size_t)4;

    // n partitions
    size_t nPartitions = 3;
    size_t arrPartitions[3] = {1, 8, 64}; // number of partitions per GPU

    gpuTopology[0] = 0;
    gpuTopology[1] = 2;
    gpuTopology[2] = 2;
    gpuTopology[3] = 1;

    gpuTopology[4] = 2;
    gpuTopology[5] = 0;
    gpuTopology[6] = 1;
    gpuTopology[7] = 2;

    gpuTopology[8] = 2;
    gpuTopology[9] = 1;
    gpuTopology[10] = 0;
    gpuTopology[11] = 2;

    gpuTopology[12] = 1;
    gpuTopology[13] = 2;
    gpuTopology[14] = 2;
    gpuTopology[15] = 0;

    // argv for the input of the application
    void* jargs[8];

    for(size_t nP = 0; nP<nPartitions; nP++){

        for(size_t commMode = 0; commMode < nCommunicationModes; commMode++){
                       
            size_t gpus = 2;

            // set job resources for experiment
            jobResources_t *jobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
            jobResources->nGPUs = gpus;
            jobResources->idGPUs = (size_t*)malloc(jobResources->nGPUs * sizeof(size_t));
            jobResources->idGPUs[0] = 0;
            jobResources->idGPUs[1] = 2;

            // weak scalability
            //ja = N; // the amount of data is fixed, each partition changes depending on the number of GPUs and partitions
            // each partition has a length of N, and each partition is divided into for all the GPUs
            //jS = 0;//arrPartitions[nP]; // number of partitions per GPU
            
            // weak scalability
            size_t ja = N; // the amount of data is fixed, each partition changes depending on the number of GPUs and partitions
            size_t jS = arrPartitions[nP]; // number of partitions per GPU

            // communication method
            size_t jPinned = pinned[commMode];
            size_t jAsync = async[commMode];
            size_t jSteps = steps[commMode];
            size_t jCores = cores[commMode];
            size_t jmall = 1;
            jargs[0] = &ja;
            jargs[1] = &jS;
            jargs[2] = &jPinned;
            jargs[3] = &jAsync; // async
            jargs[4] = &jSteps;
            jargs[5] = &jCores;
            jargs[6] = &jmall;

            // launch job
            jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));
            jobLauncher->jobType = MALLEABLE;
            jobLauncher->jobPriority = LOW;
            jobLauncher->nReqGPUs = 8; // no matter
            jobLauncher->nReqMinGPUs = 1;
            jobLauncher->launchTimeStep = 1; // no matter
            jobLauncher->appType = 2;
            jobLauncher->argc = 6;
            jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
            jobLauncher->launchFunc = &expandTest1;

            printf(" -- [RMS]: [%s, %zu, %zu, %zu, %zu]: ", communicationModes[commMode], ja, jS, arrPartitions[nP], gpus);
            fflush(stdout);
            addPendingJob(jobLauncher); // add to pending list
            launchJob(getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources); // launch from pending list
            printf(" -- [RMS]: Job launched!\n");
            fflush(stdout);

            // get job control
            jobControl_t *jobControl = getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl;
            
            // schedule reconfiguration
            jobResources_t *reconfJobResources = (jobResources_t*)calloc(1,sizeof(jobResources_t));
            reconfJobResources->nGPUs = 4;
            reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));
            reconfJobResources->idGPUs[0] = 0;
            reconfJobResources->idGPUs[1] = 1;
            reconfJobResources->idGPUs[2] = 2;
            reconfJobResources->idGPUs[3] = 3;

            // schedule
            scheduleReconfiguration(getJobFromQueue(&(schInfo->runningJobs), 0), 0, reconfJobResources);

            // wait until the reconfiguration finishes and finish it

            printf(" [RMS]: checking reconfiguration done\n");
            fflush(stdout);
            while(checkReconfigurationDone(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl) == 1){
                sleep(0.1);
            }
            printf(" [RMS]: finishing reconfiguration\n");
            fflush(stdout);
            jobFinishedReconfiguration(getJobFromQueue(&(schInfo->reconfiguringJobs), 0), 0);


            // wait until job finishes and finish job
            printf(" [RMS]: checking if job finished\n");
            fflush(stdout);
            while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                sleep(1);
            }

            printf(" [RMS]: finishing job\n");
            fflush(stdout);
            finishJob(getJobFromQueue(&(schInfo->runningJobs), 0), 0);
            removeJobFromQueueByIndex(&schInfo->finishedJobs, 0);


            if(jobControl->jobResources){
                free(jobControl->jobResources->idGPUs);
                free(jobControl->jobResources);
            }

            
            printf(" -- [RMS]: Moving to next job!\n");
            fflush(stdout);

            printf("\n");
            fflush(stdout);
        }
    }

    return 1;
}


// main function that launches tests
int main(int argc, char* argv[]){

    pthread_t thrTime, thrJobs;

    int finished = 0; // whether simulation finished or not 
    int nGPUs;
    srand(time(NULL));
    cudaGetDeviceCount(&nGPUs); // get number of available devices
    nGPUs = strtoul(argv[1], NULL, 10); // TMP: load number of GPUs from terminal input
    gNGPUs = (size_t)nGPUs;

    // file directories to store results
    char *recordFileName = argv[3];
    char *gpuUsageFileName = argv[4];
    char *resultsFileName = argv[5];

    // initialize RMS with resource availability information
    schInfo = (schInfo_t *)calloc(1, sizeof(schInfo_t));

    // initialize system resources
    schInfo->nGPUs = (size_t)nGPUs; // number of GPUs
    schInfo->nAvGPUs = schInfo->nGPUs; // available gpus = number of gpus (all GPUs are available in the beginning)
    schInfo->avGPUs = (char*)malloc(schInfo->nGPUs * sizeof(char)); // GPU is available (1) or no (0)
    for(size_t gpu = 0; gpu < (size_t)nGPUs; gpu++){
        schInfo->avGPUs[gpu] = 1; // available
    }
    schInfo->gpuJob = (unsigned int*)calloc(schInfo->nGPUs, sizeof(unsigned int)); 

    schInfo->sched = &greedy;
    schInfo->rconf = &utilization;

    // initialize internal GPU topology information
    initializeTopology(schInfo);

    // tmp: store in global variable for being used by threads
    gpuTopology = schInfo->gpuTopology;
    gpuTopologyRank = schInfo->gpuTopologyRank;


    // init queues
    initQueue(&(schInfo->pendingJobs));
    initQueue(&(schInfo->runningJobs));
    initQueue(&(schInfo->finishedJobs));
    initQueue(&(schInfo->reconfiguringJobs));


    // initialize scheduler jobs control information
    printf(" -- [RMS] Initialized!\n");
    fflush(stdout);


    // loads jobs timeline (jobs information and when they are launched)
    jobsTimeline_t *jobsTimeline = loadJobsFromFile(argv[2]);
    printf(" -- [RMS] Jobs loaded!\n");
    fflush(stdout);


    // [TIMER AND RESOURCE MONITOR]
    // File for recording events: jobs pending, jobs running, reconfigurations...
    fEventRecord = fopen(recordFileName, "w");
    if (fEventRecord == NULL) {
        perror("fopen\n");
        printf(" -- Error opening the file\n");
        return -1;  // or handle error appropriately
    }

    // file for storing resource utilization info
    fUsage = fopen(gpuUsageFileName, "w");
    if (fUsage == NULL) {
        perror("fopen\n");
        printf(" -- Error opening the file\n");
        return -1;  // or handle error appropriately
    }

    // file to store outputs? (TODO: I don't remember what stores this file)
    fOutput = fopen(resultsFileName, "w");
    if (fUsage == NULL) {
        perror("fopen\n");
        printf(" -- Error opening the file\n");
        return -1;  // or handle error appropriately
    }


    // write first line in event record file
    fprintf(fEventRecord, "GPU,Job,time,event\n");
    fflush(fEventRecord);

    // inform
    printf(" -- [RMS] Files opened!\n");
    fflush(stdout);

    // thread for monitorization
    pthread_create(&thrTime, NULL, resourceMonitoring, (void*)(&finished));

    // test function
    printf("\n\n\n[TEST 1]\n");
    fflush(stdout);
    reconfExpandTest1(schInfo);

    printf("\n\n\n[TEST 2]\n");
    fflush(stdout);
    reconfExpandTest2(schInfo);

    printf("\n\n\n[TEST 3]\n");
    fflush(stdout);
    reconfExpandTest3(schInfo);

    printf("\n\n\n[TEST 4]\n");
    fflush(stdout);
    reconfExpandTest4(schInfo);
}