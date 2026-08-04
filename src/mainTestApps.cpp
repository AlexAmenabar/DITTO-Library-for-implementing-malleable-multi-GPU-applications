#include <cstddef>
#include <omp.h>
#include <unistd.h>
#include <stdio.h>      
#include <stdlib.h>     
#include <pthread.h>
#include <time.h>

#include "RMS.hpp"
#include "SCH.hpp"
#include "RCF.hpp"
#include "jobQueue.hpp"
#include "eventQueue.hpp"


#ifndef TESTRMS

// include CUDA related libraries
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvml.h>

// include apps that use CUDA
#include "../testApps/toy_app_malleable.hpp"

#endif



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

pthread_mutex_t printLock;


// Experiments

int testJacobi(schInfo_t *schInfo, size_t nGPUs){

    size_t nCommunicationModes = 1;
    const char *communicationModes[1] = {"OASYNC"};
    const size_t async[1] =  {1};
    const size_t pinned[1] = {0};
    const size_t steps[1] =  {0};
    const size_t cores[1] =  {0};

    size_t nT = 1;
    size_t rows = 1;
    size_t cols = 1;

    size_t nRows[rows] = {(size_t)8};
    size_t nCols[cols] = {(size_t)6};
    size_t T[nT] = {2};
    
    size_t nRepetitions = 1;

    size_t nConfigurations = 3;
    jobResources_t **jobResources = (jobResources_t**)calloc(nConfigurations, sizeof(jobResources_t*));

    jobResources[0] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    jobResources[0]->nGPUs = 1;
    jobResources[0]->idGPUs = (size_t*)calloc(1, sizeof(size_t));
    jobResources[0]->idGPUs[0] = 0;

    jobResources[1] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    jobResources[1]->nGPUs = 2;
    jobResources[1]->idGPUs = (size_t*)calloc(2, sizeof(size_t));
    jobResources[1]->idGPUs[0] = 0;
    jobResources[1]->idGPUs[1] = 1;

    jobResources[2] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    jobResources[2]->nGPUs = 4;
    jobResources[2]->idGPUs = (size_t*)calloc(4, sizeof(size_t));
    jobResources[2]->idGPUs[0] = 0;
    jobResources[2]->idGPUs[1] = 1;
    jobResources[2]->idGPUs[2] = 2;
    jobResources[2]->idGPUs[3] = 3;

    // argv for the input of the application
    void* jargs[12];

    for(size_t i = 0; i<rows; i++){

        for(size_t j = 0; j<cols; j++){ 

            for(size_t t = 0; t<nT; t++){
                for(size_t rec = 0; rec < nConfigurations; rec++){

                    for(size_t rep = 0; rep<nRepetitions; rep++){
                                                        
                        // weak scalability
                        size_t jRows = nRows[i]; // the amount of data is fixed, each partition changes depending on the number of GPUs and partitions
                        size_t jCols = nCols[j];
                        size_t jT = T[t];
                        size_t jS = 1; //arrPartitions[nP]; // number of partitions per GPU

                        // communication method
                        size_t jPinned = pinned[0];
                        size_t jAsync = async[0];
                        size_t jSteps = steps[0];
                        size_t jCores = cores[0];
                        int jReconfDir = 1;
                        size_t jmall = 1;

                        jargs[0] = &jRows;
                        jargs[1] = &jCols;
                        jargs[2] = &jT;
                        jargs[3] = &jS;
                        jargs[4] = &jPinned;
                        jargs[5] = &jAsync; // async
                        jargs[6] = &jSteps;
                        jargs[7] = &jCores;
                        jargs[8] = &jReconfDir;
                        jargs[9] = &jmall;

                        // launch job
                        jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));
                        jobLauncher->jobType = MALLEABLE;
                        jobLauncher->jobPriority = LOW;
                        jobLauncher->nReqGPUs = 8; // no matter
                        jobLauncher->nReqMinGPUs = 1;
                        jobLauncher->launchTimeStep = 1; // no matter
                        jobLauncher->appType = 2; // no matter
                        jobLauncher->argc = 9;
                        jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
                        jobLauncher->launchFunc = &launch_jacobi_stencil;


                        printf(" [%zu, %zu, %zu, %zu]: ", jRows, jCols, jT, rec);
                        fflush(stdout);
                        addPendingJob(jobLauncher); // add to pending list
                        launchJob(schInfo, getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources[rec]); // launch from pending list
                        
                        // notify the job to start running
                        notifyStartRunning(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl);

                        // wait until job finishes and finish job
                        while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                            sleep(1);
                        }

                        finishJob(schInfo, getJobFromQueue(&(schInfo->runningJobs), 0), 0);
                        removeJobFromQueueByIndex(&schInfo->finishedJobs, 0);
                        
                        fflush(stdout);
                    }
                }
            }
        }
    }
    return 1;
}



int testnbody(schInfo_t *schInfo, size_t nGPUs){

    size_t nCommunicationModes = 1;
    const char *communicationModes[1] = {"OASYNC"};
    const size_t async[1] =  {1};
    const size_t pinned[1] = {0};
    const size_t steps[1] =  {0};
    const size_t cores[1] =  {0};

    size_t nT = 1;
    size_t nN = 1;

    size_t N[nN] = {(size_t)8};
    //size_t T[nT] = {1000, 2500, 3000};
    size_t T[nT] = {10};


    size_t nRepetitions = 1;

    size_t nConfigurations = 3;
    jobResources_t **jobResources = (jobResources_t**)calloc(nConfigurations, sizeof(jobResources_t*));

    jobResources[0] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    jobResources[0]->nGPUs = 1;
    jobResources[0]->idGPUs = (size_t*)calloc(1, sizeof(size_t));
    jobResources[0]->idGPUs[0] = 0;

    jobResources[1] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    jobResources[1]->nGPUs = 2;
    jobResources[1]->idGPUs = (size_t*)calloc(2, sizeof(size_t));
    jobResources[1]->idGPUs[0] = 0;
    jobResources[1]->idGPUs[1] = 1;

    jobResources[2] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    jobResources[2]->nGPUs = 4;
    jobResources[2]->idGPUs = (size_t*)calloc(4, sizeof(size_t));
    jobResources[2]->idGPUs[0] = 0;
    jobResources[2]->idGPUs[1] = 1;
    jobResources[2]->idGPUs[2] = 2;
    jobResources[2]->idGPUs[3] = 3;


    // argv for the input of the application
    void* jargs[12];
    for(size_t i = 0; i<nN; i++){

        for(size_t t = 0; t<nT; t++){

            for(size_t rec = 0; rec < nConfigurations; rec++){

                for(size_t rep = 0; rep<nRepetitions; rep++){
                                                    
                    // weak scalability
                    size_t jN = N[i]; // the amount of data is fixed, each partition changes depending on the number of GPUs and partitions
                    size_t jT = T[t];
                    size_t jS = 1; //arrPartitions[nP]; // number of partitions per GPU

                    // communication method
                    size_t jPinned = pinned[0];
                    size_t jAsync = async[0];
                    size_t jSteps = steps[0];
                    size_t jCores = cores[0];
                    int jReconfDir = 1;
                    size_t jmall = 1;

                    jargs[0] = &jN;
                    jargs[1] = &jT;
                    jargs[2] = &jS;
                    jargs[3] = &jPinned;
                    jargs[4] = &jAsync; // async
                    jargs[5] = &jSteps;
                    jargs[6] = &jCores;
                    jargs[7] = &jReconfDir;
                    jargs[8] = &jmall;

                    // launch job
                    jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));
                    jobLauncher->jobType = MALLEABLE;
                    jobLauncher->jobPriority = LOW;
                    jobLauncher->nReqGPUs = 8; // no matter
                    jobLauncher->nReqMinGPUs = 1;
                    jobLauncher->launchTimeStep = 1; // no matter
                    jobLauncher->appType = 2; // no matter
                    jobLauncher->argc = 8;
                    jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
                    jobLauncher->launchFunc = &launch_n_body;


                    printf(" [%zu, %zu, %zu]: ", jN, jT, rec);
                    fflush(stdout);
                    addPendingJob(jobLauncher); // add to pending list
                    launchJob(schInfo, getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources[rec]); // launch from pending list
                    
                    // notify the job to start running
                    notifyStartRunning(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl);

                    // wait until job finishes and finish job
                    while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                        sleep(1);
                    }

                    finishJob(schInfo, getJobFromQueue(&(schInfo->runningJobs), 0), 0);
                    removeJobFromQueueByIndex(&schInfo->finishedJobs, 0);
                    
                    fflush(stdout);
                }
            }
        }
    }
    return 1;
}



int testfft(schInfo_t *schInfo, size_t nGPUs){

    size_t nCommunicationModes = 1;
    const char *communicationModes[1] = {"OASYNC"};
    const size_t async[1] =  {1};
    const size_t pinned[1] = {0};
    const size_t steps[1] =  {0};
    const size_t cores[1] =  {0};

    size_t nT = 1;
    size_t nN = 1;

    size_t nRows[nN] = {(size_t)8};
    size_t nCols[nN] = {(size_t)8};

    //size_t T[nT] = {10000, 20000};
    size_t T[nT] = {1};

    size_t nTest = 2;

    size_t nRepetitions = 1;

    size_t nConfigurations = 3;
    jobResources_t **jobResources = (jobResources_t**)calloc(nConfigurations, sizeof(jobResources_t*));

    jobResources[0] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    jobResources[0]->nGPUs = 1;
    jobResources[0]->idGPUs = (size_t*)calloc(1, sizeof(size_t));
    jobResources[0]->idGPUs[0] = 0;

    jobResources[1] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    jobResources[1]->nGPUs = 2;
    jobResources[1]->idGPUs = (size_t*)calloc(2, sizeof(size_t));
    jobResources[1]->idGPUs[0] = 0;
    jobResources[1]->idGPUs[1] = 1;

    jobResources[2] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    jobResources[2]->nGPUs = 4;
    jobResources[2]->idGPUs = (size_t*)calloc(4, sizeof(size_t));
    jobResources[2]->idGPUs[0] = 0;
    jobResources[2]->idGPUs[1] = 1;
    jobResources[2]->idGPUs[2] = 2;
    jobResources[2]->idGPUs[3] = 3;


    // argv for the input of the application
    void* jargs[12];
    for(size_t i = 0; i<nN; i++){

        for(size_t t = 0; t<nT; t++){

            for(size_t rec = 0; rec < nConfigurations; rec++){

                for(size_t test = 0; test<nTest; test++){

                    for(size_t rep = 0; rep<nRepetitions; rep++){
                                                        
                        // weak scalability
                        size_t jNROWS = nRows[i]; // the amount of data is fixed, each partition changes depending on the number of GPUs and partitions
                        size_t jNCOLS = nCols[i]; // the amount of data is fixed, each partition changes depending on the number of GPUs and partitions
                        size_t jT = T[t];
                        size_t jS = 1; //arrPartitions[nP]; // number of partitions per GPU

                        // communication method
                        size_t jPinned = pinned[0];
                        size_t jAsync = async[0];
                        size_t jSteps = steps[0];
                        size_t jCores = cores[0];
                        int jReconfDir = 1;
                        size_t jmall = 1;

                        size_t jTest = test + 1;

                        jargs[0] = &jNROWS;
                        jargs[1] = &jNCOLS;
                        jargs[2] = &jT;
                        jargs[3] = &jS;
                        jargs[4] = &jPinned;
                        jargs[5] = &jAsync; // async
                        jargs[6] = &jSteps;
                        jargs[7] = &jCores;
                        jargs[8] = &jTest;
                        jargs[9] = &jmall;

                        // launch job
                        jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));
                        jobLauncher->jobType = MALLEABLE;
                        jobLauncher->jobPriority = LOW;
                        jobLauncher->nReqGPUs = 8; // no matter
                        jobLauncher->nReqMinGPUs = 1;
                        jobLauncher->launchTimeStep = 1; // no matter
                        jobLauncher->appType = 2; // no matter
                        jobLauncher->argc = 9;
                        jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
                        jobLauncher->launchFunc = &launch_FFT;


                        printf(" [%zu, %zu, %zu, %zu]: ", jNROWS, jT, rec, test);
                        fflush(stdout);
                        addPendingJob(jobLauncher); // add to pending list
                        launchJob(schInfo, getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources[rec]); // launch from pending list
                        
                        // notify the job to start running
                        notifyStartRunning(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl);

                        // wait until job finishes and finish job
                        while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                            sleep(1);
                        }

                        finishJob(schInfo, getJobFromQueue(&(schInfo->runningJobs), 0), 0);
                        removeJobFromQueueByIndex(&schInfo->finishedJobs, 0);
                        
                        fflush(stdout);
                    }
                }
            }
        }
    }
    return 1;
}

int commBench(schInfo_t *schInfo, size_t nGPUs){

    
    size_t nCommunicationModes = 1;
    const char *communicationModes[1] = {"OASYNC"};
    const size_t async[1] =  {1};
    const size_t pinned[1] = {0};
    const size_t steps[1] =  {0};
    const size_t cores[1] =  {0};

    size_t nT = 1;
    size_t nN = 4;

    size_t N[nN] = {(size_t)64, (size_t)1024, (size_t)16384, (size_t)65536};
    size_t T[nT] = {100};


    size_t nRepetitions = 100;

    size_t nConfigurations = 6;
    jobResources_t **jobResources = (jobResources_t**)calloc(nConfigurations, sizeof(jobResources_t*));

    jobResources[0] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    jobResources[0]->nGPUs = 2;
    jobResources[0]->idGPUs = (size_t*)calloc(2, sizeof(size_t));
    jobResources[0]->idGPUs[0] = 0;
    jobResources[0]->idGPUs[1] = 1;
  
    jobResources[1] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    jobResources[1]->nGPUs = 2;
    jobResources[1]->idGPUs = (size_t*)calloc(2, sizeof(size_t));
    jobResources[1]->idGPUs[0] = 0;
    jobResources[1]->idGPUs[1] = 2;

    jobResources[2] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    jobResources[2]->nGPUs = 2;
    jobResources[2]->idGPUs = (size_t*)calloc(2, sizeof(size_t));
    jobResources[2]->idGPUs[0] = 0;
    jobResources[2]->idGPUs[1] = 4;


    jobResources[3] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    jobResources[3]->nGPUs = 4;
    jobResources[3]->idGPUs = (size_t*)calloc(4, sizeof(size_t));
    jobResources[3]->idGPUs[0] = 0;
    jobResources[3]->idGPUs[1] = 1;
    jobResources[3]->idGPUs[2] = 2;
    jobResources[3]->idGPUs[3] = 3;

    jobResources[4] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    jobResources[4]->nGPUs = 4;
    jobResources[4]->idGPUs = (size_t*)calloc(4, sizeof(size_t));
    jobResources[4]->idGPUs[0] = 0;
    jobResources[4]->idGPUs[1] = 1;
    jobResources[4]->idGPUs[2] = 4;
    jobResources[4]->idGPUs[3] = 5;

    jobResources[5] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    jobResources[5]->nGPUs = 4;
    jobResources[5]->idGPUs = (size_t*)calloc(4, sizeof(size_t));
    jobResources[5]->idGPUs[0] = 0;
    jobResources[5]->idGPUs[1] = 2;
    jobResources[5]->idGPUs[2] = 4;
    jobResources[5]->idGPUs[3] = 6;

    // argv for the input of the application
    void* jargs[12];

                
    for(size_t i = 3; i<nN; i++){

        for(size_t t = 0; t<nT; t++){
                
            for(size_t rec = 0; rec < nConfigurations; rec++){
    
                for(size_t rep = 0; rep<nRepetitions; rep++){
                                                    
                    // weak scalability
                    size_t ja = N[i]; // the amount of data is fixed, each partition changes depending on the number of GPUs and partitions
                    size_t jT = T[t];
                    size_t jS = 1; //arrPartitions[nP]; // number of partitions per GPU

                    // communication method
                    size_t jPinned = pinned[0];
                    size_t jAsync = async[0];
                    size_t jSteps = steps[0];
                    size_t jCores = cores[0];
                    int jReconfDir = 1;
                    size_t jmall = 1;

                    jargs[0] = &ja;
                    jargs[1] = &jT;
                    jargs[2] = &jS;
                    jargs[3] = &jPinned;
                    jargs[4] = &jAsync; // async
                    jargs[5] = &jSteps;
                    jargs[6] = &jCores;
                    jargs[7] = &jReconfDir;
                    jargs[8] = &jmall;

                    // launch job
                    jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));
                    jobLauncher->jobType = MALLEABLE;
                    jobLauncher->jobPriority = LOW;
                    jobLauncher->nReqGPUs = 8; // no matter
                    jobLauncher->nReqMinGPUs = 1;
                    jobLauncher->launchTimeStep = 1; // no matter
                    jobLauncher->appType = 2; // no matter
                    jobLauncher->argc = 8;
                    jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
                    jobLauncher->launchFunc = &launch_communication_bench;


                    printf(" [%zu, %zu, %zu]: ", ja, jT, rec);
                    fflush(stdout);
                    addPendingJob(jobLauncher); // add to pending list
                    launchJob(schInfo, getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources[rec]); // launch from pending list
                    
                    // notify the job to start running
                    notifyStartRunning(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl);

                    // wait until job finishes and finish job
                    while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                        sleep(1);
                    }

                    finishJob(schInfo, getJobFromQueue(&(schInfo->runningJobs), 0), 0);
                    removeJobFromQueueByIndex(&schInfo->finishedJobs, 0);
                    
                    fflush(stdout);
                }
            }
        }
    }
    return 1;
}

int jacobiReconf(schInfo_t *schInfo, size_t nGPUs){
    
    size_t nCommunicationModes = 1;
    const char *communicationModes[1] = {"OASYNC"};
    const size_t async[1] =  {1};
    const size_t pinned[1] = {0};
    const size_t steps[1] =  {0};
    const size_t cores[1] =  {0};

    size_t nRows = (size_t)512;
    size_t nCols = (size_t)131072;
    size_t T = 100000;
    
    size_t nRepetitions = 1;

    jobResources_t *jobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    jobResources->nGPUs = 2;
    jobResources->idGPUs = (size_t*)calloc(2, sizeof(size_t));
    jobResources->idGPUs[0] = 0;
    jobResources->idGPUs[1] = 1;
  

    // argv for the input of the application
    void* jargs[12];
                                                  
    // weak scalability
    size_t jRows = nRows; // the amount of data is fixed, each partition changes depending on the number of GPUs and partitions
    size_t jCols = nCols;
    size_t jT = T;
    size_t jS = 1; //arrPartitions[nP]; // number of partitions per GPU

    // communication method
    size_t jPinned = pinned[0];
    size_t jAsync = async[0];
    size_t jSteps = steps[0];
    size_t jCores = cores[0];
    int jReconfDir = 1;
    size_t jmall = 1;

    jargs[0] = &jRows;
    jargs[1] = &jCols;
    jargs[2] = &jT;
    jargs[3] = &jS;
    jargs[4] = &jPinned;
    jargs[5] = &jAsync; // async
    jargs[6] = &jSteps;
    jargs[7] = &jCores;
    jargs[8] = &jReconfDir;
    jargs[9] = &jmall;

    // launch job
    jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));
    jobLauncher->jobType = MALLEABLE;
    jobLauncher->jobPriority = LOW;
    jobLauncher->nReqGPUs = 8; // no matter
    jobLauncher->nReqMinGPUs = 1;
    jobLauncher->launchTimeStep = 1; // no matter
    jobLauncher->appType = 2; // no matter
    jobLauncher->argc = 9;
    jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
    jobLauncher->launchFunc = &launch_jacobi_stencil;


    printf(" [%zu, %zu, %zu]: ", jRows, jCols, jT);
    fflush(stdout);
    addPendingJob(jobLauncher); // add to pending list
    launchJob(schInfo, getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources); // launch from pending list
    

    // get job control
    jobControl_t *jobControl = getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl;
    
    // schedule reconfiguration
    jobResources_t *reconfJobResources = (jobResources_t*)calloc(1,sizeof(jobResources_t));
    reconfJobResources->nGPUs = 4;
    reconfJobResources->idGPUs = (size_t*)calloc(4, sizeof(size_t));
    reconfJobResources->idGPUs[0] = 0;
    reconfJobResources->idGPUs[1] = 2;
    reconfJobResources->idGPUs[2] = 4;
    reconfJobResources->idGPUs[3] = 6;


    sleep(20);

    // schedule
    scheduleReconfiguration(schInfo, getJobFromQueue(&(schInfo->runningJobs), 0), 0, reconfJobResources);

    // wait until the reconfiguration finishes and finish it

    while(checkReconfigurationDone(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl) == 1){
        sleep(1);
    }
    jobFinishedReconfiguration(schInfo, getJobFromQueue(&(schInfo->reconfiguringJobs), 0), 0);


    // wait until job finishes and finish job
    while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

        sleep(1);
    }

    finishJob(schInfo, getJobFromQueue(&(schInfo->runningJobs), 0), 0);
    removeJobFromQueueByIndex(&schInfo->finishedJobs, 0);
    
    fflush(stdout);

    return 1;
}


int nbodyReconf(schInfo_t *schInfo, size_t nGPUs){
    
    size_t nCommunicationModes = 1;
    const char *communicationModes[1] = {"OASYNC"};
    const size_t async[1] =  {1};
    const size_t pinned[1] = {0};
    const size_t steps[1] =  {0};
    const size_t cores[1] =  {0};

    size_t N = (size_t)131072;
    size_t T = 2500;
    
    jobResources_t *jobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    jobResources->nGPUs = 2;
    jobResources->idGPUs = (size_t*)calloc(2, sizeof(size_t));
    jobResources->idGPUs[0] = 0;
    jobResources->idGPUs[1] = 1;
  

    // argv for the input of the application
    void* jargs[12];
                       
    size_t jN = N; // the amount of data is fixed, each partition changes depending on the number of GPUs and partitions
    size_t jT = T;
    size_t jS = 1; //arrPartitions[nP]; // number of partitions per GPU

    // communication method
    size_t jPinned = pinned[0];
    size_t jAsync = async[0];
    size_t jSteps = steps[0];
    size_t jCores = cores[0];
    int jReconfDir = 1;
    size_t jmall = 1;

    jargs[0] = &jN;
    jargs[1] = &jT;
    jargs[2] = &jS;
    jargs[3] = &jPinned;
    jargs[4] = &jAsync; // async
    jargs[5] = &jSteps;
    jargs[6] = &jCores;
    jargs[7] = &jReconfDir;
    jargs[8] = &jmall;

    // launch job
    jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));
    jobLauncher->jobType = MALLEABLE;
    jobLauncher->jobPriority = LOW;
    jobLauncher->nReqGPUs = 8; // no matter
    jobLauncher->nReqMinGPUs = 1;
    jobLauncher->launchTimeStep = 1; // no matter
    jobLauncher->appType = 2; // no matter
    jobLauncher->argc = 8;
    jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
    jobLauncher->launchFunc = &launch_n_body;


    printf(" [%zu, %zu]: ", jN, jT);
    fflush(stdout);
    addPendingJob(jobLauncher); // add to pending list
    launchJob(schInfo, getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources); // launch from pending list
    

    // get job control
    jobControl_t *jobControl = getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl;
    
    // schedule reconfiguration
    jobResources_t *reconfJobResources = (jobResources_t*)calloc(1,sizeof(jobResources_t));
    reconfJobResources->nGPUs = 4;
    reconfJobResources->idGPUs = (size_t*)calloc(4, sizeof(size_t));
    reconfJobResources->idGPUs[0] = 0;
    reconfJobResources->idGPUs[1] = 2;
    reconfJobResources->idGPUs[2] = 4;
    reconfJobResources->idGPUs[3] = 6;

    sleep(20);

    // schedule
    scheduleReconfiguration(schInfo, getJobFromQueue(&(schInfo->runningJobs), 0), 0, reconfJobResources);

    // wait until the reconfiguration finishes and finish it

    while(checkReconfigurationDone(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl) == 1){
        sleep(1);
    }
    jobFinishedReconfiguration(schInfo, getJobFromQueue(&(schInfo->reconfiguringJobs), 0), 0);


    // wait until job finishes and finish job
    while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

        sleep(1);
    }

    finishJob(schInfo, getJobFromQueue(&(schInfo->runningJobs), 0), 0);
    removeJobFromQueueByIndex(&schInfo->finishedJobs, 0);
    
    fflush(stdout);

    return 1;
}





int main(int argc, char* argv[]){

    pthread_t thrTime, thrJobs;

    int finished = 0; // whether simulation finished or not 
    int nGPUs;

    srand(time(NULL));


    // [SCHEDULER INITIALIZATION]
    // allocate memory to store cuda data

    cudaGetDeviceCount(&nGPUs); // get number of available devices
    nGPUs = strtoul(argv[1], NULL, 10); // TMP: load number of GPUs from terminal input
    gNGPUs = (size_t)nGPUs;

    // file directories to store results
    char *recordFileName = argv[4];
    char *gpuUsageFileName = argv[5];
    char *resultsFileName = argv[6];


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
    schInfo->nvLinkCount = (unsigned int*)calloc(schInfo->nGPUs, sizeof(unsigned int));

    schInfo->sched = &greedy;
    schInfo->rconf = &utilization;

    // initialize lock
    pthread_mutex_init(&(schInfo->lockTimer), NULL);
    pthread_mutex_init(&(printLock), NULL);

    // initialize internal GPU topology information
    initializeTopology(schInfo, argv[3]);

    // tmp: store in global variable for being used by threads
    gpuTopology = schInfo->gpuTopology;
    gpuTopologyRank = schInfo->gpuTopologyRank;


    // init queues
    initQueue(&(schInfo->pendingJobs));
    initQueue(&(schInfo->runningJobs));
    initQueue(&(schInfo->finishedJobs));
    initQueue(&(schInfo->reconfiguringJobs));


    // initialize scheduler jobs control information
    pthread_mutex_lock(&(printLock));
    printf(" -- [RMS] Initialized!\n");
    fflush(stdout);
    pthread_mutex_unlock(&(printLock));


    // loads jobs timeline (jobs information and when they are launched)
    jobsTimeline_t *jobsTimeline = loadJobsFromFile(argv[2]);
    pthread_mutex_lock(&(printLock));
    printf(" -- [RMS] Jobs loaded!\n");
    fflush(stdout);
    pthread_mutex_unlock(&(printLock));


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

    // test jacobi
    testJacobi(schInfo, 8);

    testnbody(schInfo, 8);

    testfft(schInfo, 8);

    exit(0);
}