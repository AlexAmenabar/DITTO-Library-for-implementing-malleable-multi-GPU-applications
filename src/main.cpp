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

// comparation of malloc and pinned malloc
int malloc_test(schInfo_t *schInfo){
    
    size_t nGPUs = schInfo->nGPUs;

    size_t nMallocTypes = 2;
    
    const char *mallocTypesString[2] = {"NP", "P"}; // non-pinned / pinned
    const size_t mallocTypes[2] = {0, 1}; // non-pinned / pinned
    
    size_t nRepetitions = 25;
    
    void* jargs[8];


    // from 1KB to 20GB of flotas (4 bytes)
    size_t maxBytes = (size_t)20 * (size_t)1024 * (size_t)1024 * (size_t)1024 / (size_t)4; //   10.000.000.000 --> 10GB 

    // 1024 / 4
    for(size_t nBytes = 1024 / 4; nBytes < maxBytes; nBytes *= 64){
        
        for(size_t gpus = 1; gpus <= nGPUs; gpus*=2){                

            for(size_t mallocType = 0; mallocType < nMallocTypes; mallocType++){

                for(size_t rep = 0; rep<nRepetitions; rep++){

                    jobResources_t *jobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
                    jobResources->nGPUs = gpus;
                    jobResources->idGPUs = (size_t*)malloc(jobResources->nGPUs * sizeof(size_t));
                    for(size_t g = 0; g<gpus; g++){
                        jobResources->idGPUs[g] = g;
                    }
                    
                    // job arguments
                    size_t ja = nBytes;
                    size_t jS = 0;
                    size_t jPinned = mallocType;
                    size_t jAsync = 0;//async[commMode];
                    size_t jSteps = 0;
                    size_t jCores = 0;
                    size_t jmall = 1;


                    //void* jargs[3];
                    jargs[0] = &ja;
                    jargs[1] = &jS;
                    jargs[2] = &jPinned;
                    jargs[3] = &jAsync; // async
                    jargs[4] = &jSteps;
                    jargs[5] = &jCores;
                    jargs[6] = &jmall;
                    //jargs[5] = &(schInfo.activeJobsControl[0]);


                    jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));
                    jobLauncher->jobType = MALLEABLE;
                    jobLauncher->jobPriority = LOW;
                    jobLauncher->nReqGPUs = 8; // no matter
                    jobLauncher->nReqMinGPUs = 1;
                    jobLauncher->launchTimeStep = 1; // no matter
                    jobLauncher->appType = 2;
                    jobLauncher->argc = 6;
                    jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
                    jobLauncher->launchFunc = &launch_malloc_test_app;
                    
                    printf(" [%s, %zu, %zu, %zu, %zu]: ", mallocTypesString[mallocType], nBytes, nBytes * sizeof(float), gpus, gpus);

                    // add job to pending jobs
                    addPendingJob(jobLauncher);
                    launchJob(schInfo, getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources);

                    
                    // finish job
                    while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                        sleep(1);
                    }
                    

                    // finish job
                    finishJob(schInfo, getJobFromQueue(&(schInfo->runningJobs), 0), 0);
                    removeJobFromQueueByIndex(&schInfo->finishedJobs, 0);
          
                    free(jobResources->idGPUs);
                    free(jobResources);

                    printf("\n");
                    fflush(stdout);
                }
            }
        }
    }

    sleep(5);

    return 1;
}


int reconfs_workload(schInfo_t *schInfo){

    size_t nGPUs = schInfo->nGPUs;

    size_t nCommunicationModes = 18;
    const char *communicationModes[18] = {"OSYNC", "OPSYNC", "OCSYNC", "OPCSYNC", "TSYNC", "TPSYNC", "TCSYNC", "TPCSYNC", "SSYNC", "SPSYNC", "SCSYNC", "SPCSYNC",
                                          "OPASYNC", "OPCASYNC", "TPASYNC", "TPCASYNC", "SPASYNC", "SPCASYNC"};

    const size_t async[18] =  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    const size_t pinned[18] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1};
    const size_t steps[18] =  {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 1, 1, 2, 2};
    const size_t cores[18] =  {0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1};

    
    //const char conf_names = ["NPSYNC","PSYNC","PASYNC"];

    //size_t nRepetitions = 10;
    size_t nRepetitions = 10;
    void* jargs[8];


    // from 1KB to 20GB of flotas (4 bytes)
    //size_t maxBytes = (size_t)20 * (size_t)1024 * (size_t)1024 * (size_t)1024 / (size_t)4; //   10.000.000.000 --> 10GB 
    size_t maxBytes = (size_t)20 * (size_t)1024 * (size_t)1024 / (size_t)4; // * (size_t)1024 / (size_t)4; //   10.000.000.000 --> 10GB 

    // 1024 / 4
    for(size_t nBytes = 1024 / 4; nBytes < maxBytes; nBytes *= 64){
        
        for(size_t gpus = 1; gpus <= nGPUs; gpus*=2){                

            for(size_t commMode = 0; commMode < nCommunicationModes; commMode++){

                for(size_t rep = 0; rep<nRepetitions; rep++){

                    jobResources_t *jobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
                    jobResources->nGPUs = gpus;
                    jobResources->idGPUs = (size_t*)malloc(jobResources->nGPUs * sizeof(size_t));
                    for(size_t g = 0; g<gpus; g++){
                        jobResources->idGPUs[g] = g;
                    }
                    
                    // job arguments
                    size_t ja = nBytes;
                    size_t jS = 0;
                    size_t jPinned = pinned[commMode];
                    size_t jAsync = async[commMode];
                    size_t jSteps = steps[commMode];
                    size_t jCores = cores[commMode];
                    size_t jmall = 1;


                    //void* jargs[3];
                    jargs[0] = &ja;
                    jargs[1] = &jS;
                    jargs[2] = &jPinned;
                    jargs[3] = &jAsync; // async
                    jargs[4] = &jSteps;
                    jargs[5] = &jCores;
                    jargs[6] = &jmall;
                    //jargs[5] = &(schInfo.activeJobsControl[0]);


                    jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));
                    jobLauncher->jobType = MALLEABLE;
                    jobLauncher->jobPriority = LOW;
                    jobLauncher->nReqGPUs = 8; // no matter
                    jobLauncher->nReqMinGPUs = 1;
                    jobLauncher->launchTimeStep = 1; // no matter
                    jobLauncher->appType = 2;
                    jobLauncher->argc = 6;
                    jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
                    jobLauncher->launchFunc = &launch_reconfs_test_app_new;
                    
                    printf(" [%s, %zu, %zu, %zu, %zu]: ", communicationModes[commMode], nBytes, nBytes * sizeof(float), gpus, gpus);
                    fflush(stdout);

                    // add job to pending jobs
                    addPendingJob(jobLauncher);
                    launchJob(schInfo, getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources);

                    
                    // reconfiguration
                    jobResources_t *reconfJobResources = (jobResources_t*)calloc(1,sizeof(jobResources_t));
                    reconfJobResources->nGPUs = gpus;
                    reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));

                    for(size_t j = 0; j<gpus; j++){
                        reconfJobResources->idGPUs[j] = j;
                    }

                    scheduleReconfiguration(schInfo, getJobFromQueue(&(schInfo->runningJobs), 0), 0, reconfJobResources);

                    // check reconfiguraiton done
                    int done = 0;
                    while(checkReconfigurationDone(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){
                        sleep(0.1);
                    }

                    jobFinishedReconfiguration(schInfo, getJobFromQueue(&(schInfo->reconfiguringJobs), 0), 0);


                    // finish job
                    while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                        sleep(1);
                    }
                    

                    // finish job
                    finishJob(schInfo, getJobFromQueue(&(schInfo->runningJobs), 0), 0);
                    removeJobFromQueueByIndex(&schInfo->finishedJobs, 0);
                    
                    free(reconfJobResources->idGPUs);
                    free(reconfJobResources);

                    free(jobResources->idGPUs);
                    free(jobResources);

                    printf("\n");
                    fflush(stdout);
                }
            }
        }
    }

    sleep(5);

    return 1;
}

int coalescending_reconfs_workload(schInfo_t *schInfo){

    size_t nGPUs = schInfo->nGPUs;

    size_t nCommunicationModes = 18;
    const char *communicationModes[18] = {"OSYNC", "OPSYNC", "OCSYNC", "OPCSYNC", "TSYNC", "TPSYNC", "TCSYNC", "TPCSYNC", "SSYNC", "SPSYNC", "SCSYNC", "SPCSYNC",
                                          "OPASYNC", "OPCASYNC", "TPASYNC", "TPCASYNC", "SPASYNC", "SPCASYNC"};

    const size_t async[18] =  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    const size_t pinned[18] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1};
    const size_t steps[18] =  {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 1, 1, 2, 2};
    const size_t cores[18] =  {0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1};


    void* jargs[8];

    // partition sizes
    size_t nN = 2;
    //size_t arrN[4] = {1, 1024 / 4, 1024*1024 / 4, 1024*1024*1024 / 4};
    size_t arrN[2] = {1024 / 4, 1024*1024 / 4};

    // n partitions
    size_t nPartitions = 3;
    size_t arrPartitions[3] = {1,4,16};


    //size_t nRepetitions = 10;
    size_t nRepetitions = 10;

    // from 1KB to 20GB of flotas (4 bytes)
    //size_t N = (size_t)1024 * (size_t)1024 * (size_t)1024 / sizeof(float); // 1GB (sizeof(float)), 268.435.456

    for(size_t gpus = 2; gpus<= nGPUs; gpus *= 2){

        for(size_t nP = 0; nP<nPartitions; nP++){

            for(size_t N = 0; N<nN; N++){

                for(size_t commMode = 0; commMode < nCommunicationModes; commMode++){
                
                    for(size_t rep = 0; rep<nRepetitions; rep++){
                        jobResources_t *jobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
                        jobResources->nGPUs = gpus;
                        jobResources->idGPUs = (size_t*)malloc(jobResources->nGPUs * sizeof(size_t));
                        for(size_t g = 0; g<gpus; g++){
                            jobResources->idGPUs[g] = g;
                        }
                        

                        // job arguments
                        size_t ja = arrN[N] * arrPartitions[nP] * gpus;
                        size_t jS = arrN[N];
                        size_t jPinned = pinned[commMode];
                        size_t jAsync = async[commMode];
                        size_t jSteps = steps[commMode];
                        size_t jCores = cores[commMode];
                        size_t jmall = 1;

                        //void* jargs[3];
                        jargs[0] = &ja;
                        jargs[1] = &jS;
                        jargs[2] = &jPinned;
                        jargs[3] = &jAsync; // async
                        jargs[4] = &jSteps;
                        jargs[5] = &jCores;
                        jargs[6] = &jmall;
                        //jargs[5] = &(schInfo.activeJobsControl[0]);


                        jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));
                        jobLauncher->jobType = MALLEABLE;
                        jobLauncher->jobPriority = LOW;
                        jobLauncher->nReqGPUs = 8; // no matter
                        jobLauncher->nReqMinGPUs = 1;
                        jobLauncher->launchTimeStep = 1; // no matter
                        jobLauncher->appType = 2;
                        jobLauncher->argc = 6;
                        jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
                        jobLauncher->launchFunc = &launch_reconfs_test_app_new;
                        
                        printf(" [%s, %zu, %zu, %zu, %zu]: ", communicationModes[commMode], arrN[N], arrN[N] * sizeof(float), arrPartitions[nP], gpus);
                        fflush(stdout);

                        // add job to pending jobs
                        addPendingJob(jobLauncher);
                        launchJob(schInfo, getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources);

                        // reconfiguration
                        jobResources_t *reconfJobResources = (jobResources_t*)calloc(1,sizeof(jobResources_t));
                        reconfJobResources->nGPUs = gpus;
                        reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));

                        for(size_t j = 0; j<gpus; j++)
                            reconfJobResources->idGPUs[j] = j;

                        scheduleReconfiguration(schInfo, getJobFromQueue(&(schInfo->runningJobs), 0), 0, reconfJobResources);

                        // check reconfiguraiton done
                        int done = 0;
                        while(!checkReconfigurationDone(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){
                            sleep(0.1);
                        }

                        jobFinishedReconfiguration(schInfo, getJobFromQueue(&(schInfo->reconfiguringJobs), 0), 0);

                        // finish job
                        while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                            sleep(1);
                        }
                        
                        // finish job
                        finishJob(schInfo, getJobFromQueue(&(schInfo->runningJobs), 0), 0);
                        removeJobFromQueueByIndex(&schInfo->finishedJobs, 0);

                        free(reconfJobResources->idGPUs);
                        free(reconfJobResources);

                        free(jobResources->idGPUs);
                        free(jobResources);

                        printf("\n");
                        fflush(stdout);  
                    }
                }
            }
        }
    }

        /*size_t gN = N * gpus; // scale N for the number of GPUs
        size_t s = N; // s = N * gpus / (nPartitions * gpus) = N / nPartitions where nPartitions = 1 at the beginning

        for(size_t s = N; s >= 1; s /= 16){//s /= 16){

            size_t tmp_s = s;

            if(s == N) // only 1 partition
                tmp_s = 0;

            size_t nPartitionsPerGPU = N / s;


            // test all communication methods
            for(size_t commMode = 0; commMode < nCommunicationModes; commMode++){

                jobResources_t *jobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
                jobResources->nGPUs = gpus;
                jobResources->idGPUs = (size_t*)malloc(jobResources->nGPUs * sizeof(size_t));
                for(size_t g = 0; g<gpus; g++){
                    jobResources->idGPUs[g] = g;
                }
                
                // job arguments
                size_t ja = gN;
                size_t jS = tmp_s;
                size_t async = commMode;
                size_t jmall = 1;
                //void* jargs[3];
                jargs[0] = &ja;
                jargs[1] = &jS;
                jargs[2] = &async; // async
                jargs[3] = &jmall;
                //jargs[5] = &(schInfo.activeJobsControl[0]);


                jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));
                jobLauncher->jobType = MALLEABLE;
                jobLauncher->jobPriority = LOW;
                jobLauncher->nReqGPUs = 8; // no matter
                jobLauncher->nReqMinGPUs = 1;
                jobLauncher->launchTimeStep = 1; // no matter
                jobLauncher->appType = 2;
                jobLauncher->argc = 3;
                jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
                jobLauncher->launchFunc = &launch_reconf_test_app;
                
                printf(" [%s, %zu, %zu, %zu, %zu]: ", communicationModes[commMode], gN, nPartitionsPerGPU, s, gpus);
                fflush(stdout);

                // add job to pending jobs
                addPendingJob(jobLauncher);
                launchJob(getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources);

                // reconfiguration
                jobResources_t *reconfJobResources = (jobResources_t*)calloc(1,sizeof(jobResources_t));
                reconfJobResources->nGPUs = gpus;
                reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));

                for(size_t j = 0; j<gpus; j++)
                    reconfJobResources->idGPUs[j] = j;

                scheduleReconfiguration(getJobFromQueue(&(schInfo->runningJobs), 0), 0, reconfJobResources);

                // check reconfiguraiton done
                int done = 0;
                while(!checkReconfigurationDone(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){
                    sleep(0.1);
                }

                // finish job
                while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                    sleep(1);
                }
                
                // finish job
                finishJob(getJobFromQueue(&(schInfo->runningJobs), 0), 0);

                free(reconfJobResources->idGPUs);
                free(reconfJobResources);

                free(jobResources->idGPUs);
                free(jobResources);

                printf("\n");
                fflush(stdout);
            }
        }

        if(s==1){
            break;
        }
    }*/

    return 1;
}


int reconfs_workload_definitive(schInfo_t *schInfo){

    size_t nGPUs = schInfo->nGPUs;

    size_t nCommunicationModes = 8;
    const char *communicationModes[8] = {"OASYNC", "OCASYNC", "SASYNC", "SCASYNC","OPASYNC", "OPCASYNC", "SPASYNC", "SPCASYNC"};
    const size_t async[8] =  {1, 1, 1, 1, 1, 1, 1, 1};
    const size_t pinned[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    const size_t steps[8] =  {0, 0, 2, 2, 0, 0, 2, 2};
    const size_t cores[8] =  {0, 1, 0, 1, 0, 1, 0, 1};

    // base size for each partition
    //size_t initN = 1024 * 1024 / 4; // 1MB 
    //size_t maxN = (size_t)1024 * (size_t)1024 * (size_t)1024 / (size_t)4; // 1GB

    // weak scalability
    size_t initN = (size_t)1024 / (size_t)4; // 1 KB
    size_t maxN = (size_t)1024 * (size_t)1024 * (size_t)1024 / (size_t)4; // 1 GB

    // n partitions
    size_t nPartitions = 2;
    size_t arrPartitions[2] = {8, 64}; // number of partitions per GPU


    //initN = 1024 / 4; // 1KB
    //maxN = (size_t)32 * (size_t)1024 * (size_t)1024 / 4; // 16 MB   
    //size_t arrPartitions[3] = {64,128,256};

    // number of repetitions for each experiment
    size_t nRepetitions = 10;


    // argv for the input of the application
    void* jargs[8];

    for(size_t gpus = 1; gpus<= nGPUs; gpus *= 2){

        for(size_t nP = 0; nP<nPartitions; nP++){

            for(size_t N = initN; N<=maxN; N*=4){

                for(size_t commMode = 0; commMode < nCommunicationModes; commMode++){
                
                    for(size_t rep = 0; rep<nRepetitions; rep++){
                        
                        jobResources_t *jobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
                        jobResources->nGPUs = gpus;
                        jobResources->idGPUs = (size_t*)malloc(jobResources->nGPUs * sizeof(size_t));
                        for(size_t g = 0; g<gpus; g++){
                            jobResources->idGPUs[g] = g;
                        }
                        
                        // job arguments
                        size_t ja = N * arrPartitions[nP] * gpus; // strong scalability: base partition size * number of patitions per each GPU * number of GPUs
                        size_t jS = N;

                        // weak scalability
                        ja = N; // the amount of data is fixed, each partition changes depending on the number of GPUs and partitions
                        // each partition has a length of N, and each partition is divided into for all the GPUs
                        jS = arrPartitions[nP]; // number of partitions per GPU
                        
                        //N / (gpus * arrPartitions[nP]); // each piece/partition size

                        size_t jPinned = pinned[commMode];
                        size_t jAsync = async[commMode];
                        size_t jSteps = steps[commMode];
                        size_t jCores = cores[commMode];
                        size_t jmall = 1;

                        //void* jargs[3];
                        jargs[0] = &ja;
                        jargs[1] = &jS;
                        jargs[2] = &jPinned;
                        jargs[3] = &jAsync; // async
                        jargs[4] = &jSteps;
                        jargs[5] = &jCores;
                        jargs[6] = &jmall;
                        //jargs[5] = &(schInfo.activeJobsControl[0]);


                        jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));
                        jobLauncher->jobType = MALLEABLE;
                        jobLauncher->jobPriority = LOW;
                        jobLauncher->nReqGPUs = 8; // no matter
                        jobLauncher->nReqMinGPUs = 1;
                        jobLauncher->launchTimeStep = 1; // no matter
                        jobLauncher->appType = 2;
                        jobLauncher->argc = 6;
                        jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
                        jobLauncher->launchFunc = &launch_reconfs_test_app_new;
                        
                        // print: communicationMode, total elements in array, partition size, number of partitions, number of GPUs
                        printf(" [%s, %zu, %zu, %zu, %zu]: ", communicationModes[commMode], ja, jS, arrPartitions[nP], gpus);
                        fflush(stdout);

                        // add job to pending jobs
                        addPendingJob(jobLauncher);
                        launchJob(schInfo, getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources);

                        // reconfiguration
                        jobResources_t *reconfJobResources = (jobResources_t*)calloc(1,sizeof(jobResources_t));
                        reconfJobResources->nGPUs = gpus;
                        reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));

                        for(size_t j = 0; j<gpus; j++)
                            reconfJobResources->idGPUs[j] = j;

                        scheduleReconfiguration(schInfo, getJobFromQueue(&(schInfo->runningJobs), 0), 0, reconfJobResources);

                        // check reconfiguraiton done
                        int done = 0;
                        while(!checkReconfigurationDone(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){
                            sleep(0.1);
                        }

                        jobFinishedReconfiguration(schInfo, getJobFromQueue(&(schInfo->reconfiguringJobs), 0), 0);

                        // finish job
                        while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                            sleep(1);
                        }
                        
                        // finish job
                        finishJob(schInfo, getJobFromQueue(&(schInfo->runningJobs), 0), 0);
                        removeJobFromQueueByIndex(&schInfo->finishedJobs, 0);

                        free(reconfJobResources->idGPUs);
                        free(reconfJobResources);

                        free(jobResources->idGPUs);
                        free(jobResources);

                        printf("\n");
                        fflush(stdout);  
                    }
                }
            }
        }
    }

    return 1;
}



int reconfs_workload_GPU2GPU_expand(schInfo_t *schInfo){

    size_t nGPUs = schInfo->nGPUs;

    size_t nCommunicationModes = 1;
    const char *communicationModes[1] = {"OASYNC"};
    const size_t async[1] =  {1};
    const size_t pinned[1] = {0};
    const size_t steps[1] =  {0};
    const size_t cores[1] =  {0};

    // base size for each partition
    //size_t initN = 1024 * 1024 / 4; // 1MB 
    //size_t maxN = (size_t)1024 * (size_t)1024 * (size_t)1024 / (size_t)4; // 1GB

    // weak scalability
    size_t initN = 10 * 8; // 1 KB
    size_t maxN = 10* 8 + 1; // 1 GB

    // n partitions
    size_t nPartitions = 1;
    size_t arrPartitions[1] = {1}; // number of partitions per GPU


    //initN = 1024 / 4; // 1KB
    //maxN = (size_t)32 * (size_t)1024 * (size_t)1024 / 4; // 16 MB   
    //size_t arrPartitions[3] = {64,128,256};

    // number of repetitions for each experiment
    size_t nRepetitions = 10;


    // argv for the input of the application
    void* jargs[8];

    for(size_t gpus = 1; gpus<= nGPUs / 2; gpus *= 2){

        for(size_t dstgpus = gpus * 2; dstgpus <= nGPUs; dstgpus*=2){

            for(size_t nP = 0; nP<nPartitions; nP++){

                for(size_t N = initN; N<=maxN; N*=4){

                    for(size_t commMode = 0; commMode < nCommunicationModes; commMode++){
                    
                        for(size_t rep = 0; rep<nRepetitions; rep++){
                            
                            // set job resources for experiment
                            jobResources_t *jobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
                            jobResources->nGPUs = gpus;
                            jobResources->idGPUs = (size_t*)malloc(jobResources->nGPUs * sizeof(size_t));
                            for(size_t g = 0; g<gpus; g++){
                                jobResources->idGPUs[g] = g;
                            }
                            
                            // job arguments
                            size_t ja = N * arrPartitions[nP] * gpus; // strong scalability: base partition size * number of patitions per each GPU * number of GPUs
                            size_t jS = N;

                            // weak scalability
                            ja = N; // the amount of data is fixed, each partition changes depending on the number of GPUs and partitions
                            // each partition has a length of N, and each partition is divided into for all the GPUs
                            jS = 0;//arrPartitions[nP]; // number of partitions per GPU
                            
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
                            jobLauncher->launchFunc = &launch_reconfs_test_app_new;

                            printf(" -- [RMS]: [%s, %zu, %zu, %zu, %zu]: ", communicationModes[commMode], ja, jS, arrPartitions[nP], gpus);
                            fflush(stdout);
                            addPendingJob(jobLauncher); // add to pending list
                            launchJob(schInfo, getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources); // launch from pending list
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
                            scheduleReconfiguration(schInfo, getJobFromQueue(&(schInfo->runningJobs), 0), 0, reconfJobResources);

                            // wait until the reconfiguration finishes and finish it
                            while(checkReconfigurationDone(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl) == 1){
                                sleep(0.1);
                            }
                            jobFinishedReconfiguration(schInfo, getJobFromQueue(&(schInfo->reconfiguringJobs), 0), 0);


                            // wait until job finishes and finish job
                            while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                                sleep(1);
                            }
                            finishJob(schInfo, getJobFromQueue(&(schInfo->runningJobs), 0), 0);
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
        }
    }

    return 1;
}


int reconfs_workload_GPU2GPU_shrink(schInfo_t *schInfo){

    size_t nGPUs = schInfo->nGPUs;

    size_t nCommunicationModes = 1;
    const char *communicationModes[1] = {"OASYNC"};
    const size_t async[1] =  {1};
    const size_t pinned[1] = {0};
    const size_t steps[1] =  {0};
    const size_t cores[1] =  {0};

    // base size for each partition
    //size_t initN = 1024 * 1024 / 4; // 1MB 
    //size_t maxN = (size_t)1024 * (size_t)1024 * (size_t)1024 / (size_t)4; // 1GB

    // weak scalability
    size_t initN = 10 * 8; // 1 KB
    size_t maxN = 10* 8 + 1; // 1 GB

    // n partitions
    size_t nPartitions = 1;
    size_t arrPartitions[1] = {1}; // number of partitions per GPU


    //initN = 1024 / 4; // 1KB
    //maxN = (size_t)32 * (size_t)1024 * (size_t)1024 / 4; // 16 MB   
    //size_t arrPartitions[3] = {64,128,256};

    // number of repetitions for each experiment
    size_t nRepetitions = 10;


    // argv for the input of the application
    void* jargs[8];

    for(size_t gpus = 2; gpus <= nGPUs; gpus *= 2){

        for(size_t dstgpus = 1; dstgpus <= gpus / 2; dstgpus*=2){

            for(size_t nP = 0; nP<nPartitions; nP++){

                for(size_t N = initN; N<=maxN; N*=4){

                    for(size_t commMode = 0; commMode < nCommunicationModes; commMode++){
                    
                        for(size_t rep = 0; rep<nRepetitions; rep++){
                            
                            jobResources_t *jobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
                            jobResources->nGPUs = gpus;
                            jobResources->idGPUs = (size_t*)malloc(jobResources->nGPUs * sizeof(size_t));
                            for(size_t g = 0; g<gpus; g++){
                                jobResources->idGPUs[g] = g;
                            }
                            
                            // job arguments
                            size_t ja = N * arrPartitions[nP] * gpus; // strong scalability: base partition size * number of patitions per each GPU * number of GPUs
                            size_t jS = N;

                            // weak scalability
                            ja = N; // the amount of data is fixed, each partition changes depending on the number of GPUs and partitions
                            // each partition has a length of N, and each partition is divided into for all the GPUs
                            jS = 0;//arrPartitions[nP]; // number of partitions per GPU
                            
                            //N / (gpus * arrPartitions[nP]); // each piece/partition size

                            size_t jPinned = pinned[commMode];
                            size_t jAsync = async[commMode];
                            size_t jSteps = steps[commMode];
                            size_t jCores = cores[commMode];
                            size_t jmall = 1;

                            //void* jargs[3];
                            jargs[0] = &ja;
                            jargs[1] = &jS;
                            jargs[2] = &jPinned;
                            jargs[3] = &jAsync; // async
                            jargs[4] = &jSteps;
                            jargs[5] = &jCores;
                            jargs[6] = &jmall;
                            //jargs[5] = &(schInfo.activeJobsControl[0]);


                            jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));
                            jobLauncher->jobType = MALLEABLE;
                            jobLauncher->jobPriority = LOW;
                            jobLauncher->nReqGPUs = 8; // no matter
                            jobLauncher->nReqMinGPUs = 1;
                            jobLauncher->launchTimeStep = 1; // no matter
                            jobLauncher->appType = 2;
                            jobLauncher->argc = 6;
                            jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
                            jobLauncher->launchFunc = &launch_reconfs_test_app_new;
                            
                            // print: communicationMode, total elements in array, partition size, number of partitions, number of GPUs
                            printf(" -- [RMS]: [%s, %zu, %zu, %zu, %zu]: ", communicationModes[commMode], ja, jS, arrPartitions[nP], gpus);
                            fflush(stdout);

                            // add job to pending jobs
                            addPendingJob(jobLauncher);
                            launchJob(schInfo, getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources);

                            printf(" -- [RMS]: Job launched!\n");
                            fflush(stdout);

                            // reconfiguration
                            jobResources_t *reconfJobResources = (jobResources_t*)calloc(1,sizeof(jobResources_t));
                            reconfJobResources->nGPUs = dstgpus;
                            reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));

                            for(size_t j = 0; j<reconfJobResources->nGPUs; j++)
                                reconfJobResources->idGPUs[j] = j;

                            scheduleReconfiguration(schInfo, getJobFromQueue(&(schInfo->runningJobs), 0), 0, reconfJobResources);
                            printf(" -- [RMS]: Reconfiguration programmed!\n");
                            fflush(stdout);

                            // check reconfiguraiton done
                            int done = 0;
                            while(!checkReconfigurationDone(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){
                                sleep(0.1);
                            }

                            jobFinishedReconfiguration(schInfo, getJobFromQueue(&(schInfo->reconfiguringJobs), 0), 0);
                            printf(" -- [RMS]: Job finished reconfiguration!\n");
                            fflush(stdout);

                            // finish job
                            while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                                sleep(1);
                            }
                            
                            // finish job
                            finishJob(schInfo, getJobFromQueue(&(schInfo->runningJobs), 0), 0);
                            removeJobFromQueueByIndex(&schInfo->finishedJobs, 0);

                            free(reconfJobResources->idGPUs);
                            free(reconfJobResources);

                            free(jobResources->idGPUs);
                            free(jobResources);

                            printf("\n");
                            fflush(stdout);  
                        }
                    }
                }
            }
        }
    }

    return 1;
}



int reconfs_workload_N2N(schInfo_t *schInfo){

    size_t nGPUs = schInfo->nGPUs;
    //nGPUs = 4; // using 8 do not make sense, since 8-8 are the same GPU identifiers

    size_t nCommunicationModes = 1;
    const char *communicationModes[1] = {"OPASYNC"};//, "OPCASYNC"};
    const size_t async[1] =  {1};//, 1, 1, 1, 1, 1, 1, 1};
    const size_t pinned[1] = {1};//, 0, 0, 0, 1, 1, 1, 1};
    const size_t steps[1] =  {0};//, 0, 2, 2, 0, 0, 2, 2};
    const size_t cores[1] =  {0};//, 1, 0, 1, 0, 1, 0, 1};


    // base size for each partition
    size_t initN = 1024 * 1024 / 4; // 1MB 
    size_t maxN = (size_t)1024 * (size_t)1024 * (size_t)1024 / (size_t)4; // 1GB

    // n partitions
    size_t nPartitions = 3;
    size_t arrPartitions[3] = {1,4,16};

    // number of repetitions for each experiment
    size_t nRepetitions = 10;

    // argv for the input of the application
    void* jargs[8];


    // gpus means the number of GPUs changed from one configuration to the other
    for(size_t gpus = 8; gpus<= nGPUs; gpus *= 2){

        for(size_t reconfGPUs = 1; reconfGPUs < gpus; reconfGPUs++){
            
            //for(size_t nP = 0; nP<nPartitions; nP++){
                
                for(size_t N = initN; N<=maxN; N*=2){

                    for(size_t commMode = 0; commMode < nCommunicationModes; commMode++){
                    
                        for(size_t rep = 0; rep<nRepetitions; rep++){
                            
                            jobResources_t *jobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
                            jobResources->nGPUs = gpus;
                            jobResources->idGPUs = (size_t*)malloc(jobResources->nGPUs * sizeof(size_t));
                            for(size_t g = 0; g<gpus; g++){
                                jobResources->idGPUs[g] = g;
                            }
                            

                            // job arguments
                            size_t ja = 0;// N * arrPartitions[nP] * gpus; // strong scalability: base partition size * number of patitions per each GPU * number of GPUs
                            size_t jS = N;
                            size_t jPinned = pinned[commMode];
                            size_t jAsync = async[commMode];
                            size_t jSteps = steps[commMode];
                            size_t jCores = cores[commMode];
                            size_t jmall = 1;

                            //void* jargs[3];
                            jargs[0] = &ja;
                            jargs[1] = &jS;
                            jargs[2] = &jPinned;
                            jargs[3] = &jAsync; // async
                            jargs[4] = &jSteps;
                            jargs[5] = &jCores;
                            jargs[6] = &jmall;
                            //jargs[5] = &(schInfo.activeJobsControl[0]);


                            jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));
                            jobLauncher->jobType = MALLEABLE;
                            jobLauncher->jobPriority = LOW;
                            jobLauncher->nReqGPUs = 8; // no matter
                            jobLauncher->nReqMinGPUs = 1;
                            jobLauncher->launchTimeStep = 1; // no matter
                            jobLauncher->appType = 2;
                            jobLauncher->argc = 6;
                            jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
                            jobLauncher->launchFunc = &launch_reconfs_test_app_new;
                            
                            // print: communicationMode, total elements in array, partition size, number of partitions, number of GPUs
                            //printf(" [%s, %zu, %zu, %zu, %zu, %zu]: ", communicationModes[commMode], N * arrPartitions[nP] * gpus * sizeof(float), N * sizeof(float), arrPartitions[nP], gpus, reconfGPUs);
                            fflush(stdout);

                            // add job to pending jobs
                            addPendingJob(jobLauncher);
                            launchJob(schInfo, getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources);

                            // reconfiguration
                            jobResources_t *reconfJobResources = (jobResources_t*)calloc(1,sizeof(jobResources_t));
                            reconfJobResources->nGPUs = gpus;
                            reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));

          
                            // initialize GPU ids for the new configuration
                            for(size_t j = 0; j<reconfGPUs; j++)
                                reconfJobResources->idGPUs[j] = j + 4; // GPUs for reconfiguration
                            for(size_t j = reconfGPUs; j<gpus; j++)
                                reconfJobResources->idGPUs[j] = j;

                            // print original GPUs and reconfigured GPUs
                            /*printf(" Experiment: %zu source GPUs, %zu reconf GPUs:\n[", gpus, reconfGPUs);
                            for(size_t i = 0; i<gpus; i++){
                                printf("%zu ",jobResources->idGPUs[i]);
                            }
                            printf("]\n[");
                            for(size_t i = 0; i<gpus; i++){
                                printf("%zu ",reconfJobResources->idGPUs[i]);
                            }
                            printf("]\n");*/

                            scheduleReconfiguration(schInfo, getJobFromQueue(&(schInfo->runningJobs), 0), 0, reconfJobResources);

                            // check reconfiguraiton done
                            int done = 0;
                            while(!checkReconfigurationDone(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){
                                sleep(0.1);
                            }

                            jobFinishedReconfiguration(schInfo, getJobFromQueue(&(schInfo->reconfiguringJobs), 0), 0);

                            // finish job
                            while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                                sleep(1);
                            }
                            
                            // finish job
                            finishJob(schInfo, getJobFromQueue(&(schInfo->runningJobs), 0), 0);
                            removeJobFromQueueByIndex(&schInfo->finishedJobs, 0);

                            free(reconfJobResources->idGPUs);
                            free(reconfJobResources);

                            free(jobResources->idGPUs);
                            free(jobResources);

                            printf("\n");
                            fflush(stdout);  
                        }
                    }
                }
            }
        }
    //}

    return 1;
}


////// RECONFIGURATION TIME EVALUATION

// [CPU]
int reconfCPUEval(schInfo_t *schInfo){

    size_t nGPUs = schInfo->nGPUs;

    size_t nCommunicationModes = 8;
    const char *communicationModes[8] = {"OASYNC", "OCASYNC", "SASYNC", "SCASYNC","OPASYNC", "OPCASYNC", "SPASYNC", "SPCASYNC"};
    const size_t async[8] =  {1, 1, 1, 1, 1, 1, 1, 1};
    const size_t pinned[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    const size_t steps[8] =  {0, 0, 2, 2, 0, 0, 2, 2};
    const size_t cores[8] =  {0, 1, 0, 1, 0, 1, 0, 1};

    // weak scalability
    size_t minN = (size_t)1024 * (size_t)1024 / (size_t)4;
    size_t maxN = (size_t)20 * (size_t)1024 * (size_t)1024 * (size_t)1024 / (size_t)4; // 20GB

    size_t nN = 9;
    size_t arrayN[9] = {(size_t)1048576, (size_t)1048576 * (size_t)4, (size_t)1048576 * (size_t)16, (size_t)1048576 * (size_t)64, (size_t)1048576 * (size_t)256, (size_t)1048576 * (size_t)1024, (size_t)1048576 * (size_t)2048, (size_t)1048576 * (size_t)4096, (size_t)1048576 * (size_t)8192};


    // n partitions
    size_t nPartitions = 4;
    size_t arrPartitions[4] = {1, 8, 64, 512}; // number of partitions per GPU

    size_t nRepetitions = 10;

    // argv for the input of the application
    void* jargs[9];

    for(size_t gpus = 1; gpus<= nGPUs; gpus *= 2){

        for(size_t nP = 0; nP<nPartitions; nP++){

            for(size_t iN = 0; iN<nN; iN++){

                for(size_t commMode = 0; commMode < nCommunicationModes; commMode++){
                    
                    for(size_t rep = 0; rep<nRepetitions; rep++){
                    
                        size_t N = arrayN[iN] / (size_t)4;
                    
                        // set job resources for experiment
                        jobResources_t *jobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
                        jobResources->nGPUs = gpus;
                        jobResources->idGPUs = (size_t*)malloc(jobResources->nGPUs * sizeof(size_t));
                        for(size_t g = 0; g<gpus; g++){
                            jobResources->idGPUs[g] = g;
                        }
                                            
                        // weak scalability
                        size_t ja = N; // the amount of data is fixed, each partition changes depending on the number of GPUs and partitions
                        size_t jS = arrPartitions[nP]; // number of partitions per GPU

                        // communication method
                        size_t jPinned = pinned[commMode];
                        size_t jAsync = async[commMode];
                        size_t jSteps = steps[commMode];
                        size_t jCores = cores[commMode];
                        int jReconfDir = 3;
                        size_t jmall = 1;
                        jargs[0] = &ja;
                        jargs[1] = &jS;
                        jargs[2] = &jPinned;
                        jargs[3] = &jAsync; // async
                        jargs[4] = &jSteps;
                        jargs[5] = &jCores;
                        jargs[6] = &jReconfDir;
                        jargs[7] = &jmall;

                        // launch job
                        jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));
                        jobLauncher->jobType = MALLEABLE;
                        jobLauncher->jobPriority = LOW;
                        jobLauncher->nReqGPUs = 8; // no matter
                        jobLauncher->nReqMinGPUs = 1;
                        jobLauncher->launchTimeStep = 1; // no matter
                        jobLauncher->appType = 2; // no matter
                        jobLauncher->argc = 7;
                        jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
                        jobLauncher->launchFunc = &launch_reconfs_test_app_new;


                        printf(" [%s, %zu, %zu, %zu, %zu]: ", communicationModes[commMode], ja, jS, arrPartitions[nP], gpus);
                        addPendingJob(jobLauncher); // add to pending list
                        launchJob(schInfo, getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources); // launch from pending list

                        // get job control
                        jobControl_t *jobControl = getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl;
                        
                        // schedule reconfiguration
                        jobResources_t *reconfJobResources = (jobResources_t*)calloc(1,sizeof(jobResources_t));
                        reconfJobResources->nGPUs = gpus;
                        reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));
                        for(size_t j = 0; j<reconfJobResources->nGPUs; j++)
                            reconfJobResources->idGPUs[j] = j;

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


                        if(jobControl->jobResources){
                            free(jobControl->jobResources->idGPUs);
                            free(jobControl->jobResources);
                        }
                    }
                }
            }
        }
        fflush(stdout);
    }
    return 1;
}


// [Expand]
int reconfExpandEval(schInfo_t *schInfo, size_t nGPUs){

    size_t nCommunicationModes = 2;
    const char *communicationModes[2] = {"SASYNC", "SCASYNC"};
    const size_t async[2] =  {1, 1};
    const size_t pinned[2] = {0, 0};
    const size_t steps[2] =  {2, 2};
    const size_t cores[2] =  {0, 1};

    // weak scalability
    //size_t minN = (size_t)1024 * (size_t)1024 / (size_t)4;
    size_t minN = (size_t)16 * (size_t)1024 * (size_t)1024 * (size_t)1024 / (size_t)4;
    size_t maxN = (size_t)20 * (size_t)1024 * (size_t)1024 * (size_t)1024 / (size_t)4; // 20GB

    size_t nN = 9;
    size_t arrayN[9] = {(size_t)1048576, (size_t)1048576 * (size_t)4, (size_t)1048576 * (size_t)16, (size_t)1048576 * (size_t)64, (size_t)1048576 * (size_t)256, (size_t)1048576 * (size_t)1024, (size_t)1048576 * (size_t)2048, (size_t)1048576 * (size_t)4096, (size_t)1048576 * (size_t)8192};

    // n partitions
    size_t nPartitions = 4;
    size_t arrPartitions[4] = {1, 8, 64, 512}; // number of partitions per GPU

    size_t nRepetitions = 10;

    // argv for the input of the application
    void* jargs[9];

    for(size_t gpus = 1; gpus<= nGPUs / 2; gpus *= 2){

        for(size_t dstgpus = gpus * 2; dstgpus <= nGPUs; dstgpus *= 2){

            for(size_t nP = 0; nP<nPartitions; nP++){

                for(size_t iN = 0; iN<nN; iN++){

                    for(size_t commMode = 0; commMode < nCommunicationModes; commMode++){
                        
                        for(size_t rep = 0; rep<nRepetitions; rep++){
                        
                            size_t N = arrayN[iN] / (size_t)4;

                            // set job resources for experiment
                            jobResources_t *jobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
                            jobResources->nGPUs = gpus;
                            jobResources->idGPUs = (size_t*)malloc(jobResources->nGPUs * sizeof(size_t));
                            for(size_t g = 0; g<gpus; g++){
                                jobResources->idGPUs[g] = g;
                            }
                                                
                            // weak scalability
                            size_t ja = N; // the amount of data is fixed, each partition changes depending on the number of GPUs and partitions
                            size_t jS = arrPartitions[nP]; // number of partitions per GPU

                            // communication method
                            size_t jPinned = pinned[commMode];
                            size_t jAsync = async[commMode];
                            size_t jSteps = steps[commMode];
                            size_t jCores = cores[commMode];
                            int jReconfDir = 1;
                            size_t jmall = 1;
                            jargs[0] = &ja;
                            jargs[1] = &jS;
                            jargs[2] = &jPinned;
                            jargs[3] = &jAsync; // async
                            jargs[4] = &jSteps;
                            jargs[5] = &jCores;
                            jargs[6] = &jReconfDir;
                            jargs[7] = &jmall;

                            // launch job
                            jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));
                            jobLauncher->jobType = MALLEABLE;
                            jobLauncher->jobPriority = LOW;
                            jobLauncher->nReqGPUs = 8; // no matter
                            jobLauncher->nReqMinGPUs = 1;
                            jobLauncher->launchTimeStep = 1; // no matter
                            jobLauncher->appType = 2; // no matter
                            jobLauncher->argc = 7;
                            jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
                            jobLauncher->launchFunc = &launch_reconfs_test_app_new;


                            printf(" [%s, %zu, %zu, %zu, %zu, %zu]: ", communicationModes[commMode], ja, jS, arrPartitions[nP], gpus, dstgpus);
                            addPendingJob(jobLauncher); // add to pending list
                            launchJob(schInfo, getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources); // launch from pending list

                            // get job control
                            jobControl_t *jobControl = getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl;
                            
                            // schedule reconfiguration
                            jobResources_t *reconfJobResources = (jobResources_t*)calloc(1,sizeof(jobResources_t));
                            reconfJobResources->nGPUs = dstgpus;
                            reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));
                            for(size_t j = 0; j<reconfJobResources->nGPUs; j++)
                                reconfJobResources->idGPUs[j] = j;

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


                            if(jobControl->jobResources){
                                free(jobControl->jobResources->idGPUs);
                                free(jobControl->jobResources);
                            
                            }
                        }
                    }
                    fflush(stdout);
                }
            }
        }
    }
    return 1;
}

int reconfExpandEval2(schInfo_t *schInfo, size_t nGPUs){

    size_t nCommunicationModes = 2;
    const char *communicationModes[2] = {"SASYNC", "SCASYNC"};
    const size_t async[2] =  {1, 1};
    const size_t pinned[2] = {0, 0};
    const size_t steps[2] =  {2, 2};
    const size_t cores[2] =  {0, 1};

    // weak scalability
    size_t minN = (size_t)1024 * (size_t)1024 / (size_t)4;
    size_t maxN = (size_t)20 * (size_t)1024 * (size_t)1024 * (size_t)1024 / (size_t)4; // 20GB

    size_t nN = 9;
    size_t arrayN[9] = {(size_t)1048576, (size_t)1048576 * (size_t)4, (size_t)1048576 * (size_t)16, (size_t)1048576 * (size_t)64, (size_t)1048576 * (size_t)256, (size_t)1048576 * (size_t)1024, (size_t)1048576 * (size_t)2048, (size_t)1048576 * (size_t)4096, (size_t)1048576 * (size_t)8192};


    // n partitions
    size_t nPartitions = 4;
    size_t arrPartitions[4] = {1, 8, 64, 512}; // number of partitions per GPU

    size_t nRepetitions = 10;

    // argv for the input of the application
    void* jargs[9];

    for(size_t gpus = 1; gpus<= nGPUs / 2; gpus *= 2){

        for(size_t dstgpus = gpus * 2; dstgpus <= nGPUs; dstgpus *= 2){

            for(size_t nP = 0; nP<nPartitions; nP++){

                for(size_t iN = 0; iN<nN; iN++){

                    for(size_t commMode = 0; commMode < nCommunicationModes; commMode++){
                        
                        for(size_t rep = 0; rep<nRepetitions; rep++){
                        
                            size_t N = arrayN[iN] / (size_t)4;
                        
                            // set job resources for experiment
                            jobResources_t *jobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
                            jobResources->nGPUs = gpus;
                            jobResources->idGPUs = (size_t*)malloc(jobResources->nGPUs * sizeof(size_t));
                            for(size_t g = 0; g<gpus; g++){
                                jobResources->idGPUs[g] = g;
                            }
                                                
                            // weak scalability
                            size_t ja = N; // the amount of data is fixed, each partition changes depending on the number of GPUs and partitions
                            size_t jS = arrPartitions[nP]; // number of partitions per GPU

                            // communication method
                            size_t jPinned = pinned[commMode];
                            size_t jAsync = async[commMode];
                            size_t jSteps = steps[commMode];
                            size_t jCores = cores[commMode];
                            int jReconfDir = 1;
                            size_t jmall = 1;
                            jargs[0] = &ja;
                            jargs[1] = &jS;
                            jargs[2] = &jPinned;
                            jargs[3] = &jAsync; // async
                            jargs[4] = &jSteps;
                            jargs[5] = &jCores;
                            jargs[6] = &jReconfDir;
                            jargs[7] = &jmall;

                            // launch job
                            jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));
                            jobLauncher->jobType = MALLEABLE;
                            jobLauncher->jobPriority = LOW;
                            jobLauncher->nReqGPUs = 8; // no matter
                            jobLauncher->nReqMinGPUs = 1;
                            jobLauncher->launchTimeStep = 1; // no matter
                            jobLauncher->appType = 2; // no matter
                            jobLauncher->argc = 7;
                            jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
                            jobLauncher->launchFunc = &launch_reconfs_test_app_new;


                            printf(" [%s, %zu, %zu, %zu, %zu, %zu]: ", communicationModes[commMode], ja, jS, arrPartitions[nP], gpus, dstgpus);
                            addPendingJob(jobLauncher); // add to pending list
                            launchJob(schInfo, getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources); // launch from pending list

                            // get job control
                            jobControl_t *jobControl = getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl;
                            
                            // schedule reconfiguration
                            jobResources_t *reconfJobResources = (jobResources_t*)calloc(1,sizeof(jobResources_t));
                            reconfJobResources->nGPUs = dstgpus;
                            reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));
                            for(size_t j = 0; j<reconfJobResources->nGPUs; j++)
                                reconfJobResources->idGPUs[j] = nGPUs - 1 - j;

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


                            if(jobControl->jobResources){
                                free(jobControl->jobResources->idGPUs);
                                free(jobControl->jobResources);
                            }
                        }
                    }
                }
            }
            fflush(stdout);
        }
    }
    return 1;
}

// [Shrink]
int reconfShrinkEval(schInfo_t *schInfo, size_t nGPUs){

    size_t nCommunicationModes = 2;
    const char *communicationModes[2] = {"SASYNC", "SCASYNC"};
    const size_t async[2] =  {1, 1};
    const size_t pinned[2] = {0, 0};
    const size_t steps[2] =  {2, 2};
    const size_t cores[2] =  {0, 1};

    // weak scalability
    size_t minN = (size_t)1024 * (size_t)1024 / (size_t)4;
    size_t maxN = (size_t)20 * (size_t)1024 * (size_t)1024 * (size_t)1024 / (size_t)4; // 20GB

    size_t nN = 9;
    size_t arrayN[9] = {(size_t)1048576, (size_t)1048576 * (size_t)4, (size_t)1048576 * (size_t)16, (size_t)1048576 * (size_t)64, (size_t)1048576 * (size_t)256, (size_t)1048576 * (size_t)1024, (size_t)1048576 * (size_t)2048, (size_t)1048576 * (size_t)4096, (size_t)1048576 * (size_t)8192};


    // n partitions
    size_t nPartitions = 4;
    size_t arrPartitions[4] = {1, 8, 64, 512}; // number of partitions per GPU

    size_t nRepetitions = 10;

    // argv for the input of the application
    void* jargs[9];

    for(size_t gpus = 2; gpus <= nGPUs; gpus *= 2){

        for(size_t dstgpus = 1; dstgpus < gpus; dstgpus *= 2){

            for(size_t nP = 0; nP<nPartitions; nP++){

                for(size_t iN = 0; iN<nN; iN++){

                    for(size_t commMode = 0; commMode < nCommunicationModes; commMode++){
                        
                        for(size_t rep = 0; rep<nRepetitions; rep++){
                        
                            size_t N = arrayN[iN] / (size_t)4;
                        
                            // set job resources for experiment
                            jobResources_t *jobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
                            jobResources->nGPUs = gpus;
                            jobResources->idGPUs = (size_t*)malloc(jobResources->nGPUs * sizeof(size_t));
                            for(size_t g = 0; g<gpus; g++){
                                jobResources->idGPUs[g] = g;
                            }
                                                
                            // weak scalability
                            size_t ja = N; // the amount of data is fixed, each partition changes depending on the number of GPUs and partitions
                            size_t jS = arrPartitions[nP]; // number of partitions per GPU

                            // communication method
                            size_t jPinned = pinned[commMode];
                            size_t jAsync = async[commMode];
                            size_t jSteps = steps[commMode];
                            size_t jCores = cores[commMode];
                            int jReconfDir = 1;
                            size_t jmall = 1;
                            jargs[0] = &ja;
                            jargs[1] = &jS;
                            jargs[2] = &jPinned;
                            jargs[3] = &jAsync; // async
                            jargs[4] = &jSteps;
                            jargs[5] = &jCores;
                            jargs[6] = &jReconfDir;
                            jargs[7] = &jmall;

                            // launch job
                            jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));
                            jobLauncher->jobType = MALLEABLE;
                            jobLauncher->jobPriority = LOW;
                            jobLauncher->nReqGPUs = 8; // no matter
                            jobLauncher->nReqMinGPUs = 1;
                            jobLauncher->launchTimeStep = 1; // no matter
                            jobLauncher->appType = 2; // no matter
                            jobLauncher->argc = 7;
                            jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
                            jobLauncher->launchFunc = &launch_reconfs_test_app_new;


                            printf(" [%s, %zu, %zu, %zu, %zu, %zu]: ", communicationModes[commMode], ja, jS, arrPartitions[nP], gpus, dstgpus);
                            addPendingJob(jobLauncher); // add to pending list
                            launchJob(schInfo, getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources); // launch from pending list

                            // get job control
                            jobControl_t *jobControl = getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl;
                            
                            // schedule reconfiguration
                            jobResources_t *reconfJobResources = (jobResources_t*)calloc(1,sizeof(jobResources_t));
                            reconfJobResources->nGPUs = dstgpus;
                            reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));
                            for(size_t j = 0; j<reconfJobResources->nGPUs; j++)
                                reconfJobResources->idGPUs[j] = j;

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


                            if(jobControl->jobResources){
                                free(jobControl->jobResources->idGPUs);
                                free(jobControl->jobResources);
                            }
                            fflush(stdout);
                        }
                    }
                }
            }
        }
    }
    return 1;
}

int reconfShrinkEval2(schInfo_t *schInfo, size_t nGPUs){

    size_t nCommunicationModes = 2;
    const char *communicationModes[2] = {"SASYNC", "SCASYNC"};
    const size_t async[2] =  {1, 1};
    const size_t pinned[2] = {0, 0};
    const size_t steps[2] =  {2, 2};
    const size_t cores[2] =  {0, 1};

    // weak scalability
    size_t minN = (size_t)1024 * (size_t)1024 / (size_t)4;
    size_t maxN = (size_t)20 * (size_t)1024 * (size_t)1024 * (size_t)1024 / (size_t)4; // 20GB

    size_t nN = 9;
    size_t arrayN[9] = {(size_t)1048576, (size_t)1048576 * (size_t)4, (size_t)1048576 * (size_t)16, (size_t)1048576 * (size_t)64, (size_t)1048576 * (size_t)256, (size_t)1048576 * (size_t)1024, (size_t)1048576 * (size_t)2048, (size_t)1048576 * (size_t)4096, (size_t)1048576 * (size_t)8192};


    // n partitions
    size_t nPartitions = 4;
    size_t arrPartitions[4] = {1, 8, 64, 512}; // number of partitions per GPU

    size_t nRepetitions = 10;

    // argv for the input of the application
    void* jargs[9];

    for(size_t gpus = 2; gpus <= nGPUs; gpus *= 2){

        for(size_t dstgpus = 1; dstgpus < gpus; dstgpus *= 2){

            for(size_t nP = 0; nP<nPartitions; nP++){

                for(size_t iN = 0; iN<nN; iN++){

                    for(size_t commMode = 0; commMode < nCommunicationModes; commMode++){
                        
                        for(size_t rep = 0; rep<nRepetitions; rep++){
                        
                            size_t N = arrayN[iN] / (size_t)4;
                        
                            // set job resources for experiment
                            jobResources_t *jobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
                            jobResources->nGPUs = gpus;
                            jobResources->idGPUs = (size_t*)malloc(jobResources->nGPUs * sizeof(size_t));
                            for(size_t g = 0; g<gpus; g++){
                                jobResources->idGPUs[g] = g;
                            }
                                                
                            // weak scalability
                            size_t ja = N; // the amount of data is fixed, each partition changes depending on the number of GPUs and partitions
                            size_t jS = arrPartitions[nP]; // number of partitions per GPU

                            // communication method
                            size_t jPinned = pinned[commMode];
                            size_t jAsync = async[commMode];
                            size_t jSteps = steps[commMode];
                            size_t jCores = cores[commMode];
                            int jReconfDir = 1;
                            size_t jmall = 1;
                            jargs[0] = &ja;
                            jargs[1] = &jS;
                            jargs[2] = &jPinned;
                            jargs[3] = &jAsync; // async
                            jargs[4] = &jSteps;
                            jargs[5] = &jCores;
                            jargs[6] = &jReconfDir;
                            jargs[7] = &jmall;

                            // launch job
                            jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));
                            jobLauncher->jobType = MALLEABLE;
                            jobLauncher->jobPriority = LOW;
                            jobLauncher->nReqGPUs = 8; // no matter
                            jobLauncher->nReqMinGPUs = 1;
                            jobLauncher->launchTimeStep = 1; // no matter
                            jobLauncher->appType = 2; // no matter
                            jobLauncher->argc = 7;
                            jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
                            jobLauncher->launchFunc = &launch_reconfs_test_app_new;


                            printf(" [%s, %zu, %zu, %zu, %zu, %zu]: ", communicationModes[commMode], ja, jS, arrPartitions[nP], gpus, dstgpus);
                            addPendingJob(jobLauncher); // add to pending list
                            launchJob(schInfo, getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources); // launch from pending list

                            // get job control
                            jobControl_t *jobControl = getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl;
                            
                            // schedule reconfiguration
                            jobResources_t *reconfJobResources = (jobResources_t*)calloc(1,sizeof(jobResources_t));
                            reconfJobResources->nGPUs = dstgpus;
                            reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));
                            for(size_t j = 0; j<reconfJobResources->nGPUs; j++)
                                reconfJobResources->idGPUs[j] = nGPUs - 1 - j;

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


                            if(jobControl->jobResources){
                                free(jobControl->jobResources->idGPUs);
                                free(jobControl->jobResources);
                            }
                        }
                    }
                    fflush(stdout);
                }
            }
        }
    }
    return 1;
}


// [Keep]
int reconfKeepEval(schInfo_t *schInfo, size_t nGPUs){

    size_t nCommunicationModes = 2;
    const char *communicationModes[2] = {"SASYNC", "SCASYNC"};
    const size_t async[2] =  {1, 1};
    const size_t pinned[2] = {0, 0};
    const size_t steps[2] =  {2, 2};
    const size_t cores[2] =  {0, 1};

    // weak scalability
    size_t minN = (size_t)1024 * (size_t)1024 / (size_t)4;
    size_t maxN = (size_t)20 * (size_t)1024 * (size_t)1024 * (size_t)1024 / (size_t)4; // 20GB

    size_t nN = 9;
    size_t arrayN[9] = {(size_t)1048576, (size_t)1048576 * (size_t)4, (size_t)1048576 * (size_t)16, (size_t)1048576 * (size_t)64, (size_t)1048576 * (size_t)256, (size_t)1048576 * (size_t)1024, (size_t)1048576 * (size_t)2048, (size_t)1048576 * (size_t)4096, (size_t)1048576 * (size_t)8192};


    // n partitions
    size_t nPartitions = 4;
    size_t arrPartitions[4] = {1, 8, 64, 512}; // number of partitions per GPU

    size_t nRepetitions = 10;

    // argv for the input of the application
    void* jargs[9];

    for(size_t gpus = 1; gpus <= nGPUs; gpus *= 2){

        for(size_t nP = 0; nP<nPartitions; nP++){

            for(size_t iN = 0; iN<nN; iN++){

                for(size_t commMode = 0; commMode < nCommunicationModes; commMode++){
                    
                    for(size_t rep = 0; rep<nRepetitions; rep++){
                    
                        size_t N = arrayN[iN] / (size_t)4;
                    
                        // set job resources for experiment
                        jobResources_t *jobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
                        jobResources->nGPUs = gpus;
                        jobResources->idGPUs = (size_t*)malloc(jobResources->nGPUs * sizeof(size_t));
                        for(size_t g = 0; g<gpus; g++){
                            jobResources->idGPUs[g] = g;
                        }
                                            
                        // weak scalability
                        size_t ja = N; // the amount of data is fixed, each partition changes depending on the number of GPUs and partitions
                        size_t jS = arrPartitions[nP]; // number of partitions per GPU

                        // communication method
                        size_t jPinned = pinned[commMode];
                        size_t jAsync = async[commMode];
                        size_t jSteps = steps[commMode];
                        size_t jCores = cores[commMode];
                        int jReconfDir = 1;
                        size_t jmall = 1;
                        jargs[0] = &ja;
                        jargs[1] = &jS;
                        jargs[2] = &jPinned;
                        jargs[3] = &jAsync; // async
                        jargs[4] = &jSteps;
                        jargs[5] = &jCores;
                        jargs[6] = &jReconfDir;
                        jargs[7] = &jmall;

                        // launch job
                        jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));
                        jobLauncher->jobType = MALLEABLE;
                        jobLauncher->jobPriority = LOW;
                        jobLauncher->nReqGPUs = 8; // no matter
                        jobLauncher->nReqMinGPUs = 1;
                        jobLauncher->launchTimeStep = 1; // no matter
                        jobLauncher->appType = 2; // no matter
                        jobLauncher->argc = 7;
                        jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
                        jobLauncher->launchFunc = &launch_reconfs_test_app_new;


                        printf(" [%s, %zu, %zu, %zu, %zu]: ", communicationModes[commMode], ja, jS, arrPartitions[nP], gpus);
                        addPendingJob(jobLauncher); // add to pending list
                        launchJob(schInfo, getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources); // launch from pending list

                        // get job control
                        jobControl_t *jobControl = getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl;
                        
                        // schedule reconfiguration
                        jobResources_t *reconfJobResources = (jobResources_t*)calloc(1,sizeof(jobResources_t));
                        reconfJobResources->nGPUs = gpus;
                        reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));
                        for(size_t j = 0; j<reconfJobResources->nGPUs; j++)
                            reconfJobResources->idGPUs[j] = j;

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


                        if(jobControl->jobResources){
                            free(jobControl->jobResources->idGPUs);
                            free(jobControl->jobResources);
                        }
                    }
                }
                fflush(stdout);
            }
        }
    }
    return 1;
}

int reconfKeepEval2(schInfo_t *schInfo, size_t nGPUs){

    size_t nCommunicationModes = 2;
    const char *communicationModes[2] = {"SASYNC", "SCASYNC"};
    const size_t async[2] =  {1, 1};
    const size_t pinned[2] = {0, 0};
    const size_t steps[2] =  {2, 2};
    const size_t cores[2] =  {0, 1};

    // weak scalability
    size_t minN = (size_t)1024 * (size_t)1024 / (size_t)4;
    size_t maxN = (size_t)20 * (size_t)1024 * (size_t)1024 * (size_t)1024 / (size_t)4; // 20GB

    size_t nN = 9;
    size_t arrayN[9] = {(size_t)1048576, (size_t)1048576 * (size_t)4, (size_t)1048576 * (size_t)16, (size_t)1048576 * (size_t)64, (size_t)1048576 * (size_t)256, (size_t)1048576 * (size_t)1024, (size_t)1048576 * (size_t)2048, (size_t)1048576 * (size_t)4096, (size_t)1048576 * (size_t)8192};


    // n partitions
    size_t nPartitions = 4;
    size_t arrPartitions[4] = {1, 8, 64, 512}; // number of partitions per GPU

    size_t nRepetitions = 10;

    // argv for the input of the application
    void* jargs[9];

    for(size_t gpus = 1; gpus <= nGPUs; gpus *= 2){

        for(size_t nP = 0; nP<nPartitions; nP++){

            for(size_t iN = 0; iN<nN; iN++){

                for(size_t commMode = 0; commMode < nCommunicationModes; commMode++){
                    
                    for(size_t rep = 0; rep<nRepetitions; rep++){
                    
                        size_t N = arrayN[iN] / (size_t)4;

                        // set job resources for experiment
                        jobResources_t *jobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
                        jobResources->nGPUs = gpus;
                        jobResources->idGPUs = (size_t*)malloc(jobResources->nGPUs * sizeof(size_t));
                        for(size_t g = 0; g<gpus; g++){
                            jobResources->idGPUs[g] = g;
                        }
                                            
                        // weak scalability
                        size_t ja = N; // the amount of data is fixed, each partition changes depending on the number of GPUs and partitions
                        size_t jS = arrPartitions[nP]; // number of partitions per GPU

                        // communication method
                        size_t jPinned = pinned[commMode];
                        size_t jAsync = async[commMode];
                        size_t jSteps = steps[commMode];
                        size_t jCores = cores[commMode];
                        int jReconfDir = 1;
                        size_t jmall = 1;
                        jargs[0] = &ja;
                        jargs[1] = &jS;
                        jargs[2] = &jPinned;
                        jargs[3] = &jAsync; // async
                        jargs[4] = &jSteps;
                        jargs[5] = &jCores;
                        jargs[6] = &jReconfDir;
                        jargs[7] = &jmall;

                        // launch job
                        jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));
                        jobLauncher->jobType = MALLEABLE;
                        jobLauncher->jobPriority = LOW;
                        jobLauncher->nReqGPUs = 8; // no matter
                        jobLauncher->nReqMinGPUs = 1;
                        jobLauncher->launchTimeStep = 1; // no matter
                        jobLauncher->appType = 2; // no matter
                        jobLauncher->argc = 7;
                        jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
                        jobLauncher->launchFunc = &launch_reconfs_test_app_new;



                        printf(" [%s, %zu, %zu, %zu, %zu]: ", communicationModes[commMode], ja, jS, arrPartitions[nP], gpus);
                        addPendingJob(jobLauncher); // add to pending list
                        launchJob(schInfo, getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources); // launch from pending list
       
                        // get job control
                        jobControl_t *jobControl = getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl;
                        
                        // schedule reconfiguration
                        jobResources_t *reconfJobResources = (jobResources_t*)calloc(1,sizeof(jobResources_t));
                        reconfJobResources->nGPUs = gpus;
                        reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));
                        for(size_t j = 0; j<reconfJobResources->nGPUs; j++)
                            reconfJobResources->idGPUs[j] = nGPUs - 1 - j;

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


                        if(jobControl->jobResources){
                            free(jobControl->jobResources->idGPUs);
                            free(jobControl->jobResources);
                        }
                    }
                }
                fflush(stdout);
            }
        }
    }
    return 1;
}


// TOPOLOGY-AWARE

int reconfExpandEvalTopologyAware(schInfo_t *schInfo, size_t nGPUs){

    size_t nCommunicationModes = 2;
    const char *communicationModes[2] = {"SASYNC", "SCASYNC"};
    const size_t async[2] =  {1, 1};
    const size_t pinned[2] = {0, 0};
    const size_t steps[2] =  {2, 2};
    const size_t cores[2] =  {0, 1};

    // weak scalability
    size_t minN = (size_t)1024 * (size_t)1024 / (size_t)4;
    size_t maxN = (size_t)20 * (size_t)1024 * (size_t)1024 * (size_t)1024 / (size_t)4; // 20GB

    size_t nN = 9;
    size_t arrayN[9] = {(size_t)1048576, (size_t)1048576 * (size_t)4, (size_t)1048576 * (size_t)16, (size_t)1048576 * (size_t)64, (size_t)1048576 * (size_t)256, (size_t)1048576 * (size_t)1024, (size_t)1048576 * (size_t)2048, (size_t)1048576 * (size_t)4096, (size_t)1048576 * (size_t)8192};


    // n partitions
    size_t nPartitions = 4;
    size_t arrPartitions[4] = {1, 8, 64, 512}; // number of partitions per GPU

    size_t nRepetitions = 10;

    // argv for the input of the application
    void* jargs[9];


    size_t nReconfigurations = 4;
    jobResources_t **reconfigurations = (jobResources_t**)calloc(4, sizeof(jobResources_t*));
    
    reconfigurations[0] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    reconfigurations[0]->nGPUs = 4;
    reconfigurations[0]->idGPUs = (size_t*)calloc(4, sizeof(size_t));
    reconfigurations[0]->idGPUs[0] = 0;
    reconfigurations[0]->idGPUs[1] = 4;
    reconfigurations[0]->idGPUs[2] = 1;
    reconfigurations[0]->idGPUs[3] = 5;

    reconfigurations[1] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    reconfigurations[1]->nGPUs = 4;
    reconfigurations[1]->idGPUs = (size_t*)calloc(4, sizeof(size_t));
    reconfigurations[1]->idGPUs[0] = 0;
    reconfigurations[1]->idGPUs[1] = 4;
    reconfigurations[1]->idGPUs[2] = 5;
    reconfigurations[1]->idGPUs[3] = 1;
    
    reconfigurations[2] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    reconfigurations[2]->nGPUs = 4;
    reconfigurations[2]->idGPUs = (size_t*)calloc(4, sizeof(size_t));
    reconfigurations[2]->idGPUs[0] = 0;
    reconfigurations[2]->idGPUs[1] = 4;
    reconfigurations[2]->idGPUs[2] = 2;
    reconfigurations[2]->idGPUs[3] = 6;

    reconfigurations[3] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    reconfigurations[3]->nGPUs = 4;
    reconfigurations[3]->idGPUs = (size_t*)calloc(4, sizeof(size_t));
    reconfigurations[3]->idGPUs[0] = 0;
    reconfigurations[3]->idGPUs[1] = 4;
    reconfigurations[3]->idGPUs[2] = 6;
    reconfigurations[3]->idGPUs[3] = 2;

    for(size_t rec = 0; rec < nReconfigurations; rec++){

        for(size_t nP = 0; nP<nPartitions; nP++){

            for(size_t iN = 0; iN<nN; iN++){

                for(size_t commMode = 0; commMode < nCommunicationModes; commMode++){
                    
                    for(size_t rep = 0; rep<nRepetitions; rep++){
                    
                        size_t N = arrayN[iN] / (size_t)4;
                    
                        // set job resources for experiment
                        jobResources_t *jobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
                        jobResources->nGPUs = 2;
                        jobResources->idGPUs = (size_t*)malloc(jobResources->nGPUs * sizeof(size_t));
                        jobResources->idGPUs[0] = 0;
                        jobResources->idGPUs[1] = 4;
                                            
                        // weak scalability
                        size_t ja = N; // the amount of data is fixed, each partition changes depending on the number of GPUs and partitions
                        size_t jS = arrPartitions[nP]; // number of partitions per GPU

                        // communication method
                        size_t jPinned = pinned[commMode];
                        size_t jAsync = async[commMode];
                        size_t jSteps = steps[commMode];
                        size_t jCores = cores[commMode];
                        int jReconfDir = 1;
                        size_t jmall = 1;
                        jargs[0] = &ja;
                        jargs[1] = &jS;
                        jargs[2] = &jPinned;
                        jargs[3] = &jAsync; // async
                        jargs[4] = &jSteps;
                        jargs[5] = &jCores;
                        jargs[6] = &jReconfDir;
                        jargs[7] = &jmall;

                        // launch job
                        jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));
                        jobLauncher->jobType = MALLEABLE;
                        jobLauncher->jobPriority = LOW;
                        jobLauncher->nReqGPUs = 8; // no matter
                        jobLauncher->nReqMinGPUs = 1;
                        jobLauncher->launchTimeStep = 1; // no matter
                        jobLauncher->appType = 2; // no matter
                        jobLauncher->argc = 7;
                        jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
                        jobLauncher->launchFunc = &launch_reconfs_test_app_new;


                        printf(" [%s, %zu, %zu, %zu, %zu]: ", communicationModes[commMode], ja, jS, arrPartitions[nP], rec);

                        addPendingJob(jobLauncher); // add to pending list                        
                        launchJob(schInfo, getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources); // launch from pending list

                        // get job control
                        jobControl_t *jobControl = getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl;
                        
                        // schedule reconfiguration
                        jobResources_t *reconfJobResources = reconfigurations[rec];

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


                        //if(jobControl->jobResources){
                        //    free(jobControl->jobResources->idGPUs);
                        //    free(jobControl->jobResources);
                        //}
                    }
                }
            }
        }
    }
    return 1;
}


int reconfShrinkEvalTopologyAware(schInfo_t *schInfo, size_t nGPUs){

    size_t nCommunicationModes = 2;
    const char *communicationModes[2] = {"SASYNC", "SCASYNC"};
    const size_t async[2] =  {1, 1};
    const size_t pinned[2] = {0, 0};
    const size_t steps[2] =  {2, 2};
    const size_t cores[2] =  {0, 1};

    // weak scalability
    size_t minN = (size_t)1024 * (size_t)1024 / (size_t)4;
    size_t maxN = (size_t)20 * (size_t)1024 * (size_t)1024 * (size_t)1024 / (size_t)4; // 20GB

    size_t nN = 9;
    size_t arrayN[9] = {(size_t)1048576, (size_t)1048576 * (size_t)4, (size_t)1048576 * (size_t)16, (size_t)1048576 * (size_t)64, (size_t)1048576 * (size_t)256, (size_t)1048576 * (size_t)1024, (size_t)1048576 * (size_t)2048, (size_t)1048576 * (size_t)4096, (size_t)1048576 * (size_t)8192};


    // n partitions
    size_t nPartitions = 4;
    size_t arrPartitions[4] = {1, 8, 64, 512}; // number of partitions per GPU

    size_t nRepetitions = 10;

    // argv for the input of the application
    void* jargs[9];


    size_t nReconfigurations = 2;
    jobResources_t **reconfigurations = (jobResources_t**)calloc(2, sizeof(jobResources_t*));
    
    reconfigurations[0] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    reconfigurations[0]->nGPUs = 2;
    reconfigurations[0]->idGPUs = (size_t*)calloc(2, sizeof(size_t));
    reconfigurations[0]->idGPUs[0] = 0;
    reconfigurations[0]->idGPUs[1] = 2;

    reconfigurations[1] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    reconfigurations[1]->nGPUs = 2;
    reconfigurations[1]->idGPUs = (size_t*)calloc(2, sizeof(size_t));
    reconfigurations[1]->idGPUs[0] = 4;
    reconfigurations[1]->idGPUs[1] = 5;

    for(size_t rec = 0; rec < nReconfigurations; rec++){

        for(size_t nP = 0; nP<nPartitions; nP++){

            for(size_t iN = 0; iN<nN; iN++){

                for(size_t commMode = 0; commMode < nCommunicationModes; commMode++){
                    
                    for(size_t rep = 0; rep<nRepetitions; rep++){
                    
                        size_t N = arrayN[iN] / (size_t)4;
                    
                        // set job resources for experiment
                        jobResources_t *jobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
                        jobResources->nGPUs = 4;
                        jobResources->idGPUs = (size_t*)malloc(jobResources->nGPUs * sizeof(size_t));
                        jobResources->idGPUs[0] = 0;
                        jobResources->idGPUs[1] = 1;
                        jobResources->idGPUs[2] = 2;
                        jobResources->idGPUs[3] = 3;

                        // weak scalability
                        size_t ja = N; // the amount of data is fixed, each partition changes depending on the number of GPUs and partitions
                        size_t jS = arrPartitions[nP]; // number of partitions per GPU

                        // communication method
                        size_t jPinned = pinned[commMode];
                        size_t jAsync = async[commMode];
                        size_t jSteps = steps[commMode];
                        size_t jCores = cores[commMode];
                        int jReconfDir = 1;
                        size_t jmall = 1;
                        jargs[0] = &ja;
                        jargs[1] = &jS;
                        jargs[2] = &jPinned;
                        jargs[3] = &jAsync; // async
                        jargs[4] = &jSteps;
                        jargs[5] = &jCores;
                        jargs[6] = &jReconfDir;
                        jargs[7] = &jmall;

                        // launch job
                        jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));
                        jobLauncher->jobType = MALLEABLE;
                        jobLauncher->jobPriority = LOW;
                        jobLauncher->nReqGPUs = 8; // no matter
                        jobLauncher->nReqMinGPUs = 1;
                        jobLauncher->launchTimeStep = 1; // no matter
                        jobLauncher->appType = 2; // no matter
                        jobLauncher->argc = 7;
                        jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
                        jobLauncher->launchFunc = &launch_reconfs_test_app_new;


                        printf(" [%s, %zu, %zu, %zu, %zu]: ", communicationModes[commMode], ja, jS, arrPartitions[nP], rec);
                        addPendingJob(jobLauncher); // add to pending list
                        launchJob(schInfo, getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources); // launch from pending list

                        // get job control
                        jobControl_t *jobControl = getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl;
                        
                        // schedule reconfiguration
                        jobResources_t *reconfJobResources = reconfigurations[rec];
                        
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


                        //if(jobControl->jobResources){
                        //    free(jobControl->jobResources->idGPUs);
                        //    free(jobControl->jobResources);
                        //}
                    }
                }
                fflush(stdout);
            }
        }
    }
    return 1;
}


int reconfKeepEvalTopologyAware(schInfo_t *schInfo, size_t nGPUs){

    size_t nCommunicationModes = 2;
    const char *communicationModes[2] = {"SASYNC", "SCASYNC"};
    const size_t async[2] =  {1, 1};
    const size_t pinned[2] = {0, 0};
    const size_t steps[2] =  {2, 2};
    const size_t cores[2] =  {0, 1};

    // weak scalability
    size_t minN = (size_t)1024 * (size_t)1024 / (size_t)4;
    size_t maxN = (size_t)20 * (size_t)1024 * (size_t)1024 * (size_t)1024 / (size_t)4; // 20GB

    size_t nN = 9;
    size_t arrayN[9] = {(size_t)1048576, (size_t)1048576 * (size_t)4, (size_t)1048576 * (size_t)16, (size_t)1048576 * (size_t)64, (size_t)1048576 * (size_t)256, (size_t)1048576 * (size_t)1024, (size_t)1048576 * (size_t)2048, (size_t)1048576 * (size_t)4096, (size_t)1048576 * (size_t)8192};


    // n partitions
    size_t nPartitions = 4;
    size_t arrPartitions[4] = {1, 8, 64, 512}; // number of partitions per GPU

    size_t nRepetitions = 10;

    // argv for the input of the application
    void* jargs[9];


    size_t nReconfigurations = 3;
    jobResources_t **reconfigurations = (jobResources_t**)calloc(3, sizeof(jobResources_t*));
    
    reconfigurations[0] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    reconfigurations[0]->nGPUs = 4;
    reconfigurations[0]->idGPUs = (size_t*)calloc(4, sizeof(size_t));
    reconfigurations[0]->idGPUs[0] = 1;
    reconfigurations[0]->idGPUs[1] = 3;
    reconfigurations[0]->idGPUs[2] = 5;
    reconfigurations[0]->idGPUs[3] = 7;

    reconfigurations[1] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    reconfigurations[1]->nGPUs = 4;
    reconfigurations[1]->idGPUs = (size_t*)calloc(4, sizeof(size_t));
    reconfigurations[1]->idGPUs[0] = 3;
    reconfigurations[1]->idGPUs[1] = 1;
    reconfigurations[1]->idGPUs[2] = 7;
    reconfigurations[1]->idGPUs[3] = 5;

    reconfigurations[2] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    reconfigurations[2]->nGPUs = 4;
    reconfigurations[2]->idGPUs = (size_t*)calloc(4, sizeof(size_t));
    reconfigurations[2]->idGPUs[0] = 5;
    reconfigurations[2]->idGPUs[1] = 7;
    reconfigurations[2]->idGPUs[2] = 1;
    reconfigurations[2]->idGPUs[3] = 3;

    for(size_t rec = 0; rec < nReconfigurations; rec++){

        for(size_t nP = 0; nP<nPartitions; nP++){

            for(size_t iN = 0; iN<nN; iN++){

                for(size_t commMode = 0; commMode < nCommunicationModes; commMode++){
                    
                    for(size_t rep = 0; rep<nRepetitions; rep++){
                    
                        size_t N = arrayN[iN] / (size_t)4;
                    
                        // set job resources for experiment
                        jobResources_t *jobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
                        jobResources->nGPUs = 4;
                        jobResources->idGPUs = (size_t*)malloc(jobResources->nGPUs * sizeof(size_t));
                        jobResources->idGPUs[0] = 0;
                        jobResources->idGPUs[1] = 2;
                        jobResources->idGPUs[2] = 4;
                        jobResources->idGPUs[3] = 6;

                        // weak scalability
                        size_t ja = N; // the amount of data is fixed, each partition changes depending on the number of GPUs and partitions
                        size_t jS = arrPartitions[nP]; // number of partitions per GPU

                        // communication method
                        size_t jPinned = pinned[commMode];
                        size_t jAsync = async[commMode];
                        size_t jSteps = steps[commMode];
                        size_t jCores = cores[commMode];
                        int jReconfDir = 1;
                        size_t jmall = 1;
                        jargs[0] = &ja;
                        jargs[1] = &jS;
                        jargs[2] = &jPinned;
                        jargs[3] = &jAsync; // async
                        jargs[4] = &jSteps;
                        jargs[5] = &jCores;
                        jargs[6] = &jReconfDir;
                        jargs[7] = &jmall;

                        // launch job
                        jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));
                        jobLauncher->jobType = MALLEABLE;
                        jobLauncher->jobPriority = LOW;
                        jobLauncher->nReqGPUs = 8; // no matter
                        jobLauncher->nReqMinGPUs = 1;
                        jobLauncher->launchTimeStep = 1; // no matter
                        jobLauncher->appType = 2; // no matter
                        jobLauncher->argc = 7;
                        jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
                        jobLauncher->launchFunc = &launch_reconfs_test_app_new;


                        printf(" [%s, %zu, %zu, %zu, %zu]: ", communicationModes[commMode], ja, jS, arrPartitions[nP], rec);
                        addPendingJob(jobLauncher); // add to pending list
                        launchJob(schInfo, getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources); // launch from pending list

                        // get job control
                        jobControl_t *jobControl = getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl;
                        
                        // schedule reconfiguration
                        jobResources_t *reconfJobResources = reconfigurations[rec];
                        
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


                        //if(jobControl->jobResources){
                        //    free(jobControl->jobResources->idGPUs);
                        //    free(jobControl->jobResources);
                        //}
                    }
                }
                fflush(stdout);
            }
        }
    }
    return 1;
}


int ncclCommunicationOverhead(schInfo_t *schInfo, size_t nGPUs){

    size_t nCommunicationModes = 1;
    const char *communicationModes[1] = {"OASYNC"};
    const size_t async[1] =  {1};
    const size_t pinned[1] = {0};
    const size_t steps[1] =  {0};
    const size_t cores[1] =  {0};


    size_t P = 5;
    size_t nK = 5;
    size_t nN = 3;
    size_t T = 1000;

    size_t N[nN] = {(size_t)1024, (size_t)1048576, (size_t)1048576 * (size_t)1024};
    size_t p[P] = {10000, 1000, 100, 10, 1};
    size_t K[nK] = {512, 1024, 2048, 4096, 8192};


    size_t nRepetitions = 10;

    size_t nConfigurations = 7;
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

    jobResources[6] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    jobResources[6]->nGPUs = 8;
    jobResources[6]->idGPUs = (size_t*)calloc(8, sizeof(size_t));
    jobResources[6]->idGPUs[0] = 0;
    jobResources[6]->idGPUs[1] = 1;
    jobResources[6]->idGPUs[2] = 2;
    jobResources[6]->idGPUs[3] = 3;
    jobResources[6]->idGPUs[4] = 4;
    jobResources[6]->idGPUs[5] = 5;
    jobResources[6]->idGPUs[6] = 6;
    jobResources[6]->idGPUs[7] = 7;

    // argv for the input of the application
    void* jargs[12];

    for(size_t rec = 0; rec < nConfigurations; rec++){
                
        for(size_t i = 0; i<nN; i++){

            for(size_t k = 0; k<nK; k++){

                for(size_t iP = 0; iP < P; iP++){
                    
                    for(size_t rep = 0; rep<nRepetitions; rep++){
                                                        
                        // weak scalability
                        size_t ja = N[i] / sizeof(float); // the amount of data is fixed, each partition changes depending on the number of GPUs and partitions
                        size_t jT = T;
                        size_t jK = K[k];
                        size_t jP = p[iP];
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
                        jargs[2] = &jK;
                        jargs[3] = &jP;
                        jargs[4] = &jS;
                        jargs[5] = &jPinned;
                        jargs[6] = &jAsync; // async
                        jargs[7] = &jSteps;
                        jargs[8] = &jCores;
                        jargs[9] = &jReconfDir;
                        jargs[10] = &jmall;

                        // launch job
                        jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));
                        jobLauncher->jobType = MALLEABLE;
                        jobLauncher->jobPriority = LOW;
                        jobLauncher->nReqGPUs = 8; // no matter
                        jobLauncher->nReqMinGPUs = 1;
                        jobLauncher->launchTimeStep = 1; // no matter
                        jobLauncher->appType = 2; // no matter
                        jobLauncher->argc = 10;
                        jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
                        jobLauncher->launchFunc = &launch_NCCL_communications_app;


                        printf(" [%zu, %zu, %zu, %zu, %zu]: ", ja, jT, jK, jP, rec);
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


int unifiedMemoryTest(schInfo_t *schInfo, size_t nGPUs){


    size_t nCommunicationModes = 1;
    const char *communicationModes[1] = {"OASYNC"};
    const size_t async[1] =  {1};
    const size_t pinned[1] = {0};
    const size_t steps[1] =  {0};
    const size_t cores[1] =  {0};


    size_t nN = 1;
    size_t arrayN[nN] = {(size_t)1048576 * (size_t)1}; // 1GB

    size_t N = (size_t)1048576 / 4;// * (size_t)1024 * (size_t)8;
    N = (size_t)1024 * (size_t)512 / (size_t)4;    
    size_t T = 10;
    size_t K = 1;
    size_t nRepetitions = 10;


    size_t nG = 5;
    size_t arrayGIn[nG] = {50, 40, 30, 20, 10};
    size_t arrayGOut[nG] = {0, 10, 20, 30, 40};


    size_t nConfigurations = 7;
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

    jobResources[6] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    jobResources[6]->nGPUs = 8;
    jobResources[6]->idGPUs = (size_t*)calloc(8, sizeof(size_t));
    jobResources[6]->idGPUs[0] = 0;
    jobResources[6]->idGPUs[1] = 1;
    jobResources[6]->idGPUs[2] = 2;
    jobResources[6]->idGPUs[3] = 3;
    jobResources[6]->idGPUs[4] = 4;
    jobResources[6]->idGPUs[5] = 5;
    jobResources[6]->idGPUs[6] = 6;
    jobResources[6]->idGPUs[7] = 7;

    // argv for the input of the application

    for(size_t rec = 0; rec < nConfigurations; rec++){
        
        for(size_t g = 0; g < nG; g++){
        
            for(size_t rep = 0; rep<nRepetitions; rep++){
                                                
                // weak scalability
                size_t ja = N; // the amount of data is fixed, each partition changes depending on the number of GPUs and partitions
                size_t jT = T;
                size_t jK = K;

                // communication method
                size_t jPinned = pinned[0];
                size_t jAsync = async[0];
                size_t jSteps = steps[0];
                size_t jCores = cores[0];
                int jReconfDir = 1;
                size_t jmall = 1;
                
                size_t pIn = arrayGIn[g];
                size_t pOut = arrayGOut[g];

                void* jargs[12];

                jargs[0] = &ja;
                jargs[1] = &jT;
                jargs[2] = &jK;
                jargs[3] = &jPinned;
                jargs[4] = &jAsync; // async
                jargs[5] = &jSteps;
                jargs[6] = &jCores;
                jargs[7] = &jReconfDir;
                jargs[8] = &pIn;
                jargs[9] = &pOut;
                jargs[10] = &jmall;

                // launch job
                jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));
                jobLauncher->jobType = MALLEABLE;
                jobLauncher->jobPriority = LOW;
                jobLauncher->nReqGPUs = 8; // no matter
                jobLauncher->nReqMinGPUs = 1;
                jobLauncher->launchTimeStep = 1; // no matter
                jobLauncher->appType = 2; // no matter
                jobLauncher->argc = 10;
                jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
                jobLauncher->launchFunc = &launch_unified_memory_app;


                printf(" [%zu, %zu, %zu, %zu, %zu, %zu]: ", ja, jT, jK, pIn, pOut, rec);
                addPendingJob(jobLauncher); // add to pending list
                launchJob(schInfo, getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources[rec]); // launch from pending list
                
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

    // thread for monitorization
    //pthread_create(&thrTime, NULL, resourceMonitoring, (void*)(&finished));

    // EXPERIMENTS

    //printf("\n\n\n[CPU]\n");
    //fflush(stdout);
    //reconfCPUEval(schInfo);

#ifdef RECONFEXP
    
    printf("\n\n\n[EXPAND]\n");
    fflush(stdout);
    reconfExpandEval(schInfo, schInfo->nGPUs);

    printf("\n\n\n[EXPAND DIFF]\n");
    fflush(stdout);
    reconfExpandEval2(schInfo, 4);

    printf("\n\n\n[SHRINK]\n");
    fflush(stdout);
    reconfShrinkEval(schInfo, schInfo->nGPUs);

    printf("\n\n[SHRINK DIFF]\n");
    fflush(stdout);
    reconfShrinkEval2(schInfo, 4);

    // EXECUTE AFTER THIS
    printf("\n\n[KEEP DIFF]\n");
    fflush(stdout);
    reconfKeepEval2(schInfo, 8);

    // TOPOLOGY_AWARE
    printf("\n\n\n[EXPAND TOPO]\n");
    fflush(stdout);
    reconfExpandEvalTopologyAware(schInfo, 4);

    printf("\n\n\n[SHRINK TOPO]\n");
    fflush(stdout);
    reconfShrinkEvalTopologyAware(schInfo, 4);

    printf("\n\n\n[KEEP TOPO]\n");
    fflush(stdout);
    reconfKeepEvalTopologyAware(schInfo, 4);


#elif NCCLEXP

    // COMMUNICATION OVERHEAD ANALYSIS
    printf("\n\n\n[COMMUNICATION OVERHEAD]\n");
    fflush(stdout);
    ncclCommunicationOverhead(schInfo, 8);

#elif UNIFIED
    // UNIFIED MEMORY ANALYSIS
    printf("\n\n\n[UNIFIED MEMORY]\n");
    fflush(stdout);
    unifiedMemoryTest(schInfo, 8);
#endif

    exit(0);
}