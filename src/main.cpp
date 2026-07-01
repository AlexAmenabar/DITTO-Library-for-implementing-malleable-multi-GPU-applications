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
                    launchJob(getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources);

                    
                    // finish job
                    while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                        sleep(1);
                    }
                    

                    // finish job
                    finishJob(getJobFromQueue(&(schInfo->runningJobs), 0), 0);
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
    size_t nRepetitions = 1;
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
                    launchJob(getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources);

                    
                    // reconfiguration
                    jobResources_t *reconfJobResources = (jobResources_t*)calloc(1,sizeof(jobResources_t));
                    reconfJobResources->nGPUs = gpus;
                    reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));

                    for(size_t j = 0; j<gpus; j++){
                        reconfJobResources->idGPUs[j] = j;
                    }

                    scheduleReconfiguration(getJobFromQueue(&(schInfo->runningJobs), 0), 0, reconfJobResources);

                    // check reconfiguraiton done
                    int done = 0;
                    while(checkReconfigurationDone(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){
                        sleep(0.1);
                    }

                    jobFinishedReconfiguration(getJobFromQueue(&(schInfo->reconfiguringJobs), 0), 0);


                    // finish job
                    while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                        sleep(1);
                    }
                    

                    // finish job
                    finishJob(getJobFromQueue(&(schInfo->runningJobs), 0), 0);
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
    size_t nRepetitions = 1;

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

                        jobFinishedReconfiguration(getJobFromQueue(&(schInfo->reconfiguringJobs), 0), 0);

                        // finish job
                        while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                            sleep(1);
                        }
                        
                        // finish job
                        finishJob(getJobFromQueue(&(schInfo->runningJobs), 0), 0);
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

                        jobFinishedReconfiguration(getJobFromQueue(&(schInfo->reconfiguringJobs), 0), 0);

                        // finish job
                        while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                            sleep(1);
                        }
                        
                        // finish job
                        finishJob(getJobFromQueue(&(schInfo->runningJobs), 0), 0);
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
    size_t nRepetitions = 1;


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
                            while(checkReconfigurationDone(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl) == 1){
                                sleep(0.1);
                            }
                            jobFinishedReconfiguration(getJobFromQueue(&(schInfo->reconfiguringJobs), 0), 0);


                            // wait until job finishes and finish job
                            while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                                sleep(1);
                            }
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
    size_t nRepetitions = 1;


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
                            launchJob(getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources);

                            printf(" -- [RMS]: Job launched!\n");
                            fflush(stdout);

                            // reconfiguration
                            jobResources_t *reconfJobResources = (jobResources_t*)calloc(1,sizeof(jobResources_t));
                            reconfJobResources->nGPUs = dstgpus;
                            reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));

                            for(size_t j = 0; j<reconfJobResources->nGPUs; j++)
                                reconfJobResources->idGPUs[j] = j;

                            scheduleReconfiguration(getJobFromQueue(&(schInfo->runningJobs), 0), 0, reconfJobResources);
                            printf(" -- [RMS]: Reconfiguration programmed!\n");
                            fflush(stdout);

                            // check reconfiguraiton done
                            int done = 0;
                            while(!checkReconfigurationDone(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){
                                sleep(0.1);
                            }

                            jobFinishedReconfiguration(getJobFromQueue(&(schInfo->reconfiguringJobs), 0), 0);
                            printf(" -- [RMS]: Job finished reconfiguration!\n");
                            fflush(stdout);

                            // finish job
                            while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                                sleep(1);
                            }
                            
                            // finish job
                            finishJob(getJobFromQueue(&(schInfo->runningJobs), 0), 0);
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
                            launchJob(getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources);

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

                            scheduleReconfiguration(getJobFromQueue(&(schInfo->runningJobs), 0), 0, reconfJobResources);

                            // check reconfiguraiton done
                            int done = 0;
                            while(!checkReconfigurationDone(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){
                                sleep(0.1);
                            }

                            jobFinishedReconfiguration(getJobFromQueue(&(schInfo->reconfiguringJobs), 0), 0);

                            // finish job
                            while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                                sleep(1);
                            }
                            
                            // finish job
                            finishJob(getJobFromQueue(&(schInfo->runningJobs), 0), 0);
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


void communications_workload(schInfo_t *schInfo){

    jobResources_t **jobResources = (jobResources_t**)calloc(2, sizeof(jobResources_t*));

    // all configurations
    jobResources[0] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    jobResources[0]->nGPUs = 2;
    jobResources[0]->idGPUs = (size_t*)malloc(jobResources[0]->nGPUs * sizeof(size_t));
    jobResources[0]->idGPUs[0] = 0;
    jobResources[0]->idGPUs[1] = 1;


    jobResources[1] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    jobResources[1]->nGPUs = 2;
    jobResources[1]->idGPUs = (size_t*)malloc(jobResources[1]->nGPUs * sizeof(size_t));
    jobResources[1]->idGPUs[0] = 0;
    jobResources[1]->idGPUs[1] = 4;

    size_t nConfigurations = 2;
    size_t maxBytes = (size_t)20 * (size_t)1024 * (size_t)1024 * (size_t)1024 / 4; // 10.000.000.000 --> 10GB 


    // from 1KB to 20GB of flotas (4 bytes)
    for(size_t nBytes = 1024 / 4; nBytes < maxBytes; nBytes *= 2){
        for(size_t conf = 0; conf<nConfigurations; conf++){

            // job arguments
            size_t ja = nBytes;
            size_t jb = 10;
            size_t jc = 1;
            size_t jd = 1;
            size_t jmall = 1;
            void* jargs[6];
            jargs[0] = &ja;
            jargs[1] = &jb;
            jargs[2] = &jc;
            jargs[3] = &jd;
            jargs[4] = &jmall;
            //jargs[5] = &(schInfo.activeJobsControl[0]);


            jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));
            jobLauncher->jobType = RIGID;
            jobLauncher->jobPriority = LOW;
            jobLauncher->nReqGPUs = 2; // no matter
            jobLauncher->launchTimeStep = 1; // no matter
            jobLauncher->appType = 2;
            jobLauncher->argc = 4;
            jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
            jobLauncher->launchFunc = &launch_communications_app;
            
            printf(" Job nBytes = %zu, conf = %zu launched!\n",nBytes,conf);
            fflush(stdout);

            // add job to pending jobs
            addPendingJob(jobLauncher);
            launchJob(getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources[conf]);

            sleep(10);
            notifyStartRunning(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl);
            
            while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                sleep(1);
            }
            
            // finish job
            finishJob(getJobFromQueue(&(schInfo->runningJobs), 0), 0);
        }
    }
}


void communications_workload_multipleJobs(schInfo_t *schInfo){


    jobResources_t **jobResources = (jobResources_t**)calloc(8, sizeof(jobResources_t*));

    jobResources[0] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    jobResources[0]->nGPUs = 2;
    jobResources[0]->idGPUs = (size_t*)malloc(jobResources[0]->nGPUs * sizeof(size_t));
    jobResources[0]->idGPUs[0] = 0;
    jobResources[0]->idGPUs[1] = 1;

    jobResources[1] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    jobResources[1]->nGPUs = 2;
    jobResources[1]->idGPUs = (size_t*)malloc(jobResources[1]->nGPUs * sizeof(size_t));
    jobResources[1]->idGPUs[0] = 2;
    jobResources[1]->idGPUs[1] = 3;

    jobResources[2] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    jobResources[2]->nGPUs = 2;
    jobResources[2]->idGPUs = (size_t*)malloc(jobResources[2]->nGPUs * sizeof(size_t));
    jobResources[2]->idGPUs[0] = 4;
    jobResources[2]->idGPUs[1] = 5;

    jobResources[3] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    jobResources[3]->nGPUs = 2;
    jobResources[3]->idGPUs = (size_t*)malloc(jobResources[3]->nGPUs * sizeof(size_t));
    jobResources[3]->idGPUs[0] = 6;
    jobResources[3]->idGPUs[1] = 7;



    jobResources[4] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    jobResources[4]->nGPUs = 2;
    jobResources[4]->idGPUs = (size_t*)malloc(jobResources[4]->nGPUs * sizeof(size_t));
    jobResources[4]->idGPUs[0] = 0;
    jobResources[4]->idGPUs[1] = 7;

    jobResources[5] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    jobResources[5]->nGPUs = 2;
    jobResources[5]->idGPUs = (size_t*)malloc(jobResources[5]->nGPUs * sizeof(size_t));
    jobResources[5]->idGPUs[0] = 1;
    jobResources[5]->idGPUs[1] = 6;

    jobResources[6] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    jobResources[6]->nGPUs = 2;
    jobResources[6]->idGPUs = (size_t*)malloc(jobResources[6]->nGPUs * sizeof(size_t));
    jobResources[6]->idGPUs[0] = 2;
    jobResources[6]->idGPUs[1] = 5;

    jobResources[7] = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    jobResources[7]->nGPUs = 2;
    jobResources[7]->idGPUs = (size_t*)malloc(jobResources[7]->nGPUs * sizeof(size_t));
    jobResources[7]->idGPUs[0] = 3;
    jobResources[7]->idGPUs[1] = 4;


    size_t nConfigurations = 2;
    size_t maxBytes = (size_t)20 * (size_t)1024 * (size_t)1024 * (size_t)1024 / 4; // 10.000.000.000 --> 10GB 

    for(size_t nBytes = 1024; nBytes < maxBytes; nBytes *= 10){
        for(size_t conf = 0; conf<nConfigurations; conf++){

            // job arguments
            size_t ja = nBytes;
            size_t jb = 10;
            size_t jc = 1;
            size_t jd = 1;
            size_t jmall = 1;
            void* jargs[6];
            jargs[0] = &ja;
            jargs[1] = &jb;
            jargs[2] = &jc;
            jargs[3] = &jd;
            jargs[4] = &jmall;
            //jargs[5] = &(schInfo.activeJobsControl[0]);

            printf(" Job nBytes = %zu, conf = %zu launched!\n",nBytes,conf);
            fflush(stdout);

            for(size_t job = 0; job<4; job++){
                
                jobLauncher_t *jobLauncher = (jobLauncher_t*)calloc(1, sizeof(jobLauncher_t));

                jobLauncher->jobType = RIGID;
                jobLauncher->jobPriority = LOW;
                jobLauncher->nReqGPUs = 2; // no matter
                jobLauncher->launchTimeStep = 1; // no matter
                jobLauncher->appType = 2;
                jobLauncher->argc = 4;
                jobLauncher->argv = jargs; // 2 extra arguments: whether the job is malleable and the job control (latter set)
                jobLauncher->launchFunc = &launch_communications_app;

                addPendingJob(jobLauncher);
                launchJob(getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources[conf * 4 + job]);

                printf(" -- Job %zu launched with GPUs ", job);
                for(size_t ngpu = 0; ngpu<jobResources[conf * 4 + job]->nGPUs; ngpu++){
                    printf(" %zu", jobResources[conf * 4 + job]->idGPUs[ngpu]);
                }
                printf(" \n");
                fflush(stdout);
            }

            sleep(10);

            // indicate jobs that they can start
            for(size_t job = 0; job<4; job++){

                notifyStartRunning(getJobFromQueue(&(schInfo->runningJobs), job)->jobControl);
            }

            // finish jobs
            while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                sleep(1);
            }
            
            // finish job
            finishJob(getJobFromQueue(&(schInfo->runningJobs), 0), 0);

            while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                sleep(1);
            }
            
            // finish job
            finishJob(getJobFromQueue(&(schInfo->runningJobs), 0), 0);

            while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                sleep(1);
            }
            
            // finish job
            finishJob(getJobFromQueue(&(schInfo->runningJobs), 0), 0);

            while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                sleep(1);
            }
            
            // finish job
            finishJob(getJobFromQueue(&(schInfo->runningJobs), 0), 0);
        }
    }
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
                        launchJob(getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources); // launch from pending list

                        // get job control
                        jobControl_t *jobControl = getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl;
                        
                        // schedule reconfiguration
                        jobResources_t *reconfJobResources = (jobResources_t*)calloc(1,sizeof(jobResources_t));
                        reconfJobResources->nGPUs = gpus;
                        reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));
                        for(size_t j = 0; j<reconfJobResources->nGPUs; j++)
                            reconfJobResources->idGPUs[j] = j;

                        // schedule
                        scheduleReconfiguration(getJobFromQueue(&(schInfo->runningJobs), 0), 0, reconfJobResources);

                        // wait until the reconfiguration finishes and finish it

                        while(checkReconfigurationDone(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl) == 1){
                            sleep(0.1);
                        }
                        jobFinishedReconfiguration(getJobFromQueue(&(schInfo->reconfiguringJobs), 0), 0);


                        // wait until job finishes and finish job
                        while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                            sleep(1);
                        }

                        finishJob(getJobFromQueue(&(schInfo->runningJobs), 0), 0);
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

    size_t nCommunicationModes = 4;
    const char *communicationModes[4] = {"OASYNC", "OCASYNC", "SASYNC", "SCASYNC"};
    const size_t async[4] =  {1, 1, 1, 1};
    const size_t pinned[4] = {0, 0, 0, 0};
    const size_t steps[4] =  {0, 0, 2, 2};
    const size_t cores[4] =  {0, 1, 0, 1};

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
                            launchJob(getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources); // launch from pending list

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

                            while(checkReconfigurationDone(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl) == 1){
                                sleep(0.1);
                            }
                            jobFinishedReconfiguration(getJobFromQueue(&(schInfo->reconfiguringJobs), 0), 0);


                            // wait until job finishes and finish job
                            while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                                sleep(1);
                            }

                            finishJob(getJobFromQueue(&(schInfo->runningJobs), 0), 0);
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

    size_t nCommunicationModes = 4;
    const char *communicationModes[4] = {"OASYNC", "OCASYNC", "SASYNC", "SCASYNC"};
    const size_t async[4] =  {1, 1, 1, 1};
    const size_t pinned[4] = {0, 0, 0, 0};
    const size_t steps[4] =  {0, 0, 2, 2};
    const size_t cores[4] =  {0, 1, 0, 1};

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
                            launchJob(getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources); // launch from pending list

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

                            while(checkReconfigurationDone(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl) == 1){
                                sleep(0.1);
                            }
                            jobFinishedReconfiguration(getJobFromQueue(&(schInfo->reconfiguringJobs), 0), 0);


                            // wait until job finishes and finish job
                            while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                                sleep(1);
                            }

                            finishJob(getJobFromQueue(&(schInfo->runningJobs), 0), 0);
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

// [Shrink]
int reconfShrinkEval(schInfo_t *schInfo, size_t nGPUs){

    size_t nCommunicationModes = 4;
    const char *communicationModes[4] = {"OASYNC", "OCASYNC", "SASYNC", "SCASYNC"};
    const size_t async[4] =  {1, 1, 1, 1};
    const size_t pinned[4] = {0, 0, 0, 0};
    const size_t steps[4] =  {0, 0, 2, 2};
    const size_t cores[4] =  {0, 1, 0, 1};

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
                            launchJob(getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources); // launch from pending list

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

                            while(checkReconfigurationDone(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl) == 1){
                                sleep(0.1);
                            }
                            jobFinishedReconfiguration(getJobFromQueue(&(schInfo->reconfiguringJobs), 0), 0);


                            // wait until job finishes and finish job
                            while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                                sleep(1);
                            }

                            finishJob(getJobFromQueue(&(schInfo->runningJobs), 0), 0);
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

int reconfShrinkEval2(schInfo_t *schInfo, size_t nGPUs){

    size_t nCommunicationModes = 4;
    const char *communicationModes[4] = {"OASYNC", "OCASYNC", "SASYNC", "SCASYNC"};
    const size_t async[4] =  {1, 1, 1, 1};
    const size_t pinned[4] = {0, 0, 0, 0};
    const size_t steps[4] =  {0, 0, 2, 2};
    const size_t cores[4] =  {0, 1, 0, 1};

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
                            launchJob(getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources); // launch from pending list

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

                            while(checkReconfigurationDone(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl) == 1){
                                sleep(0.1);
                            }
                            jobFinishedReconfiguration(getJobFromQueue(&(schInfo->reconfiguringJobs), 0), 0);


                            // wait until job finishes and finish job
                            while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                                sleep(1);
                            }

                            finishJob(getJobFromQueue(&(schInfo->runningJobs), 0), 0);
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

    size_t nCommunicationModes = 4;
    const char *communicationModes[4] = {"OASYNC", "OCASYNC", "SASYNC", "SCASYNC"};
    const size_t async[4] =  {1, 1, 1, 1};
    const size_t pinned[4] = {0, 0, 0, 0};
    const size_t steps[4] =  {0, 0, 2, 2};
    const size_t cores[4] =  {0, 1, 0, 1};

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
                        launchJob(getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources); // launch from pending list

                        // get job control
                        jobControl_t *jobControl = getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl;
                        
                        // schedule reconfiguration
                        jobResources_t *reconfJobResources = (jobResources_t*)calloc(1,sizeof(jobResources_t));
                        reconfJobResources->nGPUs = gpus;
                        reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));
                        for(size_t j = 0; j<reconfJobResources->nGPUs; j++)
                            reconfJobResources->idGPUs[j] = j;

                        // schedule
                        scheduleReconfiguration(getJobFromQueue(&(schInfo->runningJobs), 0), 0, reconfJobResources);

                        // wait until the reconfiguration finishes and finish it

                        while(checkReconfigurationDone(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl) == 1){
                            sleep(0.1);
                        }
                        jobFinishedReconfiguration(getJobFromQueue(&(schInfo->reconfiguringJobs), 0), 0);


                        // wait until job finishes and finish job
                        while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                            sleep(1);
                        }

                        finishJob(getJobFromQueue(&(schInfo->runningJobs), 0), 0);
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

    size_t nCommunicationModes = 4;
    const char *communicationModes[4] = {"OASYNC", "OCASYNC", "SASYNC", "SCASYNC"};
    const size_t async[4] =  {1, 1, 1, 1};
    const size_t pinned[4] = {0, 0, 0, 0};
    const size_t steps[4] =  {0, 0, 2, 2};
    const size_t cores[4] =  {0, 1, 0, 1};

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
                        launchJob(getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources); // launch from pending list
       
                        // get job control
                        jobControl_t *jobControl = getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl;
                        
                        // schedule reconfiguration
                        jobResources_t *reconfJobResources = (jobResources_t*)calloc(1,sizeof(jobResources_t));
                        reconfJobResources->nGPUs = gpus;
                        reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));
                        for(size_t j = 0; j<reconfJobResources->nGPUs; j++)
                            reconfJobResources->idGPUs[j] = nGPUs - 1 - j;

                        // schedule
                        scheduleReconfiguration(getJobFromQueue(&(schInfo->runningJobs), 0), 0, reconfJobResources);

                        // wait until the reconfiguration finishes and finish it

                        while(checkReconfigurationDone(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl) == 1){
                            sleep(0.1);
                        }
                        jobFinishedReconfiguration(getJobFromQueue(&(schInfo->reconfiguringJobs), 0), 0);


                        // wait until job finishes and finish job
                        while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                            sleep(1);
                        }

                        finishJob(getJobFromQueue(&(schInfo->runningJobs), 0), 0);
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

    size_t nCommunicationModes = 4;
    const char *communicationModes[4] = {"OASYNC", "OCASYNC", "SASYNC", "SCASYNC"};
    const size_t async[4] =  {1, 1, 1, 1};
    const size_t pinned[4] = {0, 0, 0, 0};
    const size_t steps[4] =  {0, 0, 2, 2};
    const size_t cores[4] =  {0, 1, 0, 1};

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
                        launchJob(getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources); // launch from pending list

                        // get job control
                        jobControl_t *jobControl = getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl;
                        
                        // schedule reconfiguration
                        jobResources_t *reconfJobResources = reconfigurations[rec];

                        // schedule
                        scheduleReconfiguration(getJobFromQueue(&(schInfo->runningJobs), 0), 0, reconfJobResources);

                        // wait until the reconfiguration finishes and finish it

                        while(checkReconfigurationDone(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl) == 1){
                            sleep(0.1);
                        }
                        jobFinishedReconfiguration(getJobFromQueue(&(schInfo->reconfiguringJobs), 0), 0);


                        // wait until job finishes and finish job
                        while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                            sleep(1);
                        }

                        finishJob(getJobFromQueue(&(schInfo->runningJobs), 0), 0);
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


int reconfShrinkEvalTopologyAware(schInfo_t *schInfo, size_t nGPUs){

    size_t nCommunicationModes = 4;
    const char *communicationModes[4] = {"OASYNC", "OCASYNC", "SASYNC", "SCASYNC"};
    const size_t async[4] =  {1, 1, 1, 1};
    const size_t pinned[4] = {0, 0, 0, 0};
    const size_t steps[4] =  {0, 0, 2, 2};
    const size_t cores[4] =  {0, 1, 0, 1};

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
                        launchJob(getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources); // launch from pending list

                        // get job control
                        jobControl_t *jobControl = getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl;
                        
                        // schedule reconfiguration
                        jobResources_t *reconfJobResources = reconfigurations[rec];
                        
                        // schedule
                        scheduleReconfiguration(getJobFromQueue(&(schInfo->runningJobs), 0), 0, reconfJobResources);

                        // wait until the reconfiguration finishes and finish it

                        while(checkReconfigurationDone(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl) == 1){
                            sleep(0.1);
                        }
                        jobFinishedReconfiguration(getJobFromQueue(&(schInfo->reconfiguringJobs), 0), 0);


                        // wait until job finishes and finish job
                        while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                            sleep(1);
                        }

                        finishJob(getJobFromQueue(&(schInfo->runningJobs), 0), 0);
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


int reconfKeepEvalTopologyAware(schInfo_t *schInfo, size_t nGPUs){

    size_t nCommunicationModes = 4;
    const char *communicationModes[4] = {"OASYNC", "OCASYNC", "SASYNC", "SCASYNC"};
    const size_t async[4] =  {1, 1, 1, 1};
    const size_t pinned[4] = {0, 0, 0, 0};
    const size_t steps[4] =  {0, 0, 2, 2};
    const size_t cores[4] =  {0, 1, 0, 1};

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
                        launchJob(getJobFromQueue(&(schInfo->pendingJobs), 0), 0, jobResources); // launch from pending list

                        // get job control
                        jobControl_t *jobControl = getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl;
                        
                        // schedule reconfiguration
                        jobResources_t *reconfJobResources = reconfigurations[rec];
                        
                        // schedule
                        scheduleReconfiguration(getJobFromQueue(&(schInfo->runningJobs), 0), 0, reconfJobResources);

                        // wait until the reconfiguration finishes and finish it

                        while(checkReconfigurationDone(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl) == 1){
                            sleep(0.1);
                        }
                        jobFinishedReconfiguration(getJobFromQueue(&(schInfo->reconfiguringJobs), 0), 0);


                        // wait until job finishes and finish job
                        while(!checkJobFinished(getJobFromQueue(&(schInfo->runningJobs), 0)->jobControl)){

                            sleep(1);
                        }

                        finishJob(getJobFromQueue(&(schInfo->runningJobs), 0), 0);
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

    // thread for adding jobs to the system simulating they arrival to the system
    //pthread_create(&thrJobs, NULL, jobManager, (void*)(jobsTimeline));

    // EXPERIMENTS

    //printf("\n\n\n[CPU]\n");
    //reconfCPUEval(schInfo);

    //printf("\n\n\n[EXPAND]\n");
    //reconfExpandEval(schInfo, schInfo->nGPUs);

    //printf("\n\n\n[EXPAND DIFF]\n");
    //reconfExpandEval2(schInfo, 4);

    //printf("\n\n\n[SHRINK]\n");
    //reconfShrinkEval(schInfo, schInfo->nGPUs);

    //printf("\n\n[SHRINK DIFF]\n");
    //reconfShrinkEval2(schInfo, 4);

    printf("\n\n[KEEP DIFF]\n");
    reconfKeepEval2(schInfo, 8);

    // TOPOLOGY_AWARE
    printf("\n\n\n[EXPAND TOPO]\n");
    reconfExpandEvalTopologyAware(schInfo, 4);

    printf("\n\n\n[SHRINK TOPO]\n");
    reconfShrinkEvalTopologyAware(schInfo, 4);

    printf("\n\n\n[KEEP TOPO]\n");
    reconfKeepEvalTopologyAware(schInfo, 4);

    // especialized experiments (require to comment the previous line, the jobManager thread)
    //malloc_test(schInfo); // memory allocation vs pinned memory allocation
    //reconfs_workload(schInfo); // reconfiguration costs
    //coalescending_reconfs_workload(schInfo); // reconfiguration costs with several partitions
    //reconfs_workload_definitive(schInfo);
    //reconfs_workload_N2N(schInfo);

    // TODO: jobs with communication
    //communications_workload(schInfo);
    //communications_workload_multipleJobs(schInfo);
    exit(0);


    // Revise all the code below
    /*
    Important notes:
        -- JobReources is copied by the job, but there is no direct synchronization mechanism
        for that property. If the reconfiguration policy just respects the fact of waiting for the job
        to finish its programmed reconfiguration before programming a new one, the synchronization is
        correct since it is the unique points in which race conditions can occur for that structure
            -- RMS program reconfiguration
            -- Job reconfigure
            -- Job notify RMS
    */


    // [DECLARE VARIABLES USED DURING WORKLOAD SIMULATION]

    // get queues
    jobQueue_t *pendingQueue = &(schInfo->pendingJobs);
    jobQueue_t *runningQueue = &(schInfo->runningJobs);
    jobQueue_t *finishedQueue = &(schInfo->finishedJobs);
    jobQueue_t *reconfiguringQueue = &(schInfo->reconfiguringJobs);

    // helper variables
    size_t nJobsFinished; // number of jobs that already finished their execution
    size_t iJob, nPendingJobs, nRunningJobs, nFinishedJobs, nReconfiguringJobs, nReqGPUs, nReqMinGPUs, nLaunchGPUs; // helper variables
    jobTypeEnum jobType;

    // helper variables
    job_t *job; // links all job properties
    jobLauncher_t *jobLauncher; // information for launching the job
    jobControl_t *jobControl; // job control for communicating RMS and application

    // initialize structure for job resources
    jobResources_t *jobResources; // structure with job resources information
    jobMonitoring_t *jobMonitor; // structure with job monitorization information



    // [SCHEDULING LOOP]
    printf(" -- [RMS] Starting simulation (%zu jobs for completion)\n", jobsTimeline->nJobs);
    fflush(stdout);

    struct timespec startTimer, endTimer;
    clock_gettime(CLOCK_MONOTONIC, &(startTimer)); 

    // loop until all jobs finish their execution
    nJobsFinished = 0; 
    while(nJobsFinished < jobsTimeline->nJobs){

        // wait one second
        sleep(1);

        // get number of jobs on each queue
        nPendingJobs = getNumberOfJobsInQueue(&(schInfo->pendingJobs));
        nRunningJobs = getNumberOfJobsInQueue(&(schInfo->runningJobs));
        nFinishedJobs = getNumberOfJobsInQueue(&(schInfo->finishedJobs));
        nReconfiguringJobs = getNumberOfJobsInQueue(&(schInfo->reconfiguringJobs));


        // print relevant information each 5 seconds
        //if(timer % 5 == 0){

            // Number of jobs on each queue
        printf(" -- [RMS] State:\n ---- Pending jobs = %zu\n ---- Running jobs = %zu\n ---- Finished jobs = %zu\n ---- Reconfiguring jobs = %zu\n ---- Free gpus = %zu (", 
            nPendingJobs, nRunningJobs, nFinishedJobs, nReconfiguringJobs, schInfo->nAvGPUs);
        
        // Availability of GPUs 
        for(size_t gpu = 0; gpu<schInfo->nGPUs; gpu ++){
            if(schInfo->avGPUs[gpu]){
                printf(" %zu", gpu);
            }
        }
        printf(")\n");

        // Running jobs information (GPUs they are using)
        for(iJob = 0; iJob < nRunningJobs; iJob++){

            job = getJobFromQueue(runningQueue, iJob);
            printf(" ---- ---- Job %zu is using %zu GPUs: (", job->jobId, job->jobControl->jobResources->nGPUs);
            for(size_t gpu = 0; gpu<job->jobControl->jobResources->nGPUs; gpu++){

                printf(" %zu", job->jobControl->jobResources->idGPUs[gpu]);
            }
            printf(")\n");
        }
        if(timer % 5 == 0){
            printf("\n");
            fflush(stdout);
        }
        //}

        // Invoque scheduler 
        //printf(" --- Invoquing scheduler!\n");
        //fflush(stdout);
        sched(schInfo); 
        //printf(" --- Scheduler invoqued!\n");
        //fflush(stdout);

        // check whether jobs finished reconfigurations and manage them
        manageReconfigurations(schInfo); 
        //printf(" --- Reconfiugrations managed!\n");
        //fflush(stdout);

        // manage jobs that finished their execution
        nFinishedJobs += manageJobsFinish(schInfo); 
        //printf(" --- Finished jobs managed!\n");
        //fflush(stdout);

        // Invoque reconfigurator
        reconf(schInfo); 
        //printf(" --- Reconfiguration scheduler invoqued!\n");
        //fflush(stdout);

        // [SCHEDULING ALGORITHM]: scheduling, reconfigurations...
        // loop over pending jobs and check whether any job can be scheduled
        //sched(schInfo);
        
        /*nPendingJobs = getNumberOfJobsInQueue(&(schInfo->pendingJobs));
        for(iJob = 0; iJob<nPendingJobs; iJob++){

            // get job from pending queue (execute any job)
            job = getJobFromQueue(pendingQueue, iJob);
            
            // get job type and general job information
            jobLauncher = job->jobLauncher;
            jobType = jobLauncher->jobType;

            // get required number of GPUs
            nReqGPUs = jobLauncher->nReqGPUs;
            nReqMinGPUs = nReqGPUs; // [FIXED or MALLEABLE]

            // if job is MOLDABLE or FLEXIBLE, store the minimum number of GPUs too
            if(jobType == MOLDABLE || jobType == FLEXIBLE){
                nReqMinGPUs = jobLauncher->nReqMinGPUs;
            }


            // [SCHEDULING POLICY] : launch, if possible, using all the GPUs the job wants

            // launch using the maximum number of GPUs possible
            nLaunchGPUs = 0;
            if(schInfo->nAvGPUs >= nReqGPUs){
                nLaunchGPUs = nReqGPUs;
            }
            // if job is MOLDABLE or FLEXIBLE, enable there are [avGPUs >= minGPUs]
            else if(schInfo->nAvGPUs >= nReqMinGPUs){
                nLaunchGPUs = schInfo->nAvGPUs;
            }

            // launch job if possible
            if(nLaunchGPUs){

                // job resources for the job
                jobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
                jobResources->nGPUs = nLaunchGPUs;
                jobResources->idGPUs = (size_t*)malloc(nLaunchGPUs * sizeof(size_t));
                

                // decide where the job will be executed
                // find available GPUs and update system state // [THIS IS A POLICY TOO]
                size_t gpuId = 0;
                size_t jobGPU = 0;
                
                while(jobGPU < nLaunchGPUs){

                    // allocate the gpu for the job
                    if(schInfo->avGPUs[gpuId] == 1){
                        
                        // set the GPU id
                        jobResources->idGPUs[jobGPU] = gpuId; 

                        // update index
                        jobGPU ++;
                    }

                    gpuId ++;
                }

                // allocate resources for the job
                allocateResources(schInfo, jobResources);

                
                // [LAUNCH JOB]
                launchJob(job, iJob, jobResources);

                // print information
                printf(" -- [RMS] Launching job %zu (id %zu) with %zu GPUs (", iJob, job->jobId, nLaunchGPUs);
                
                for(size_t g = 0; g<job->jobControl->jobResources->nGPUs; g++){
                    printf(" %zu", job->jobControl->jobResources->idGPUs[g]);
                }

                printf(")\n");
                fflush(stdout);

                // update number of pending jobs
                nPendingJobs--;
                iJob--;
            }
        }*/

        // loop over jobs that are being reconfigured and check if they finished
        /*nReconfiguringJobs = getNumberOfJobsInQueue(&(schInfo->reconfiguringJobs)); 
        for(iJob = 0; iJob<nReconfiguringJobs; iJob ++){

            job = getJobFromQueue(reconfiguringQueue, iJob);

            // check whether job finished the reconfiguration
            char reconfigurationDone = checkReconfigurationDone(job->jobControl);

            if(reconfigurationDone == 0){

                // get job information
                jobControl = job->jobControl;

                // finish job reconfiguration: leave reconfiguration queue
                jobFinishedReconfiguration(job, iJob);


                // deallocate job old resources
                jobResources_t *oldJobResources = (jobResources_t*)calloc(1,sizeof(jobResources_t));
                oldJobResources->nGPUs = jobControl->prevJobResources->nGPUs - jobControl->jobResources->nGPUs;
                oldJobResources->idGPUs = (size_t*)malloc(oldJobResources->nGPUs * sizeof(size_t));

                // find the GPUs to be deallocaated
                // TODO (I need to test this) // TESTED?
                int next = 0;
                for(size_t i = 0; i<jobControl->prevJobResources->nGPUs; i++){
                    
                    int founded = 0;
                    for(size_t j = 0; j<jobControl->jobResources->nGPUs; j++){

                        if(jobControl->prevJobResources->idGPUs[i] == jobControl->jobResources->idGPUs[j]){

                            founded = 1;
                        }
                    }

                    if(!founded){

                        oldJobResources->idGPUs[next] = jobControl->prevJobResources->idGPUs[i];
                        next++;
                    }
                }

                // deallocate resources
                deallocateResources(schInfo, oldJobResources);

                // deallocate structure of oldJobResources
                deallocateJobResourcesStruct(&oldJobResources);


                // update values for the loop indexing
                nReconfiguringJobs --;
                iJob --;
            }
        }*/





        // manage running jobs: whether it finished or it needs a reconfiguration
        /*nRunningJobs = getNumberOfJobsInQueue(&(schInfo->runningJobs));
        for(iJob = 0; iJob<nRunningJobs; iJob++){

            // get job information
            job = getJobFromQueue(runningQueue, iJob);
            jobLauncher = job->jobLauncher;
            jobResources = job->jobControl->jobResources;
            jobType = jobLauncher->jobType;


            // JOB FINISH MANAGEMENT
            // check whether the job finished
            char jobFinished = checkJobFinished(job->jobControl);

            // check if job finished
            if(jobFinished){
            
                // finish job
                finishJob(job, iJob);

                // deallocate resources
                deallocateResources(schInfo, jobResources);

                 
                // free job resources (should be a function?) [TODO]
                //free(jobResources->idGPUs);
                //free(jobResources);
                //free(jobControl->prevJobResources->idGPUs);
                //free(jobControl->prevJobResources);
//
                //free(job->jobControl);
//
                //free(job->jobLauncher->argv);
                //free(job->jobLauncher);
                
                // [TODO]: free job pointers (jobControl, job, jobLauncher...)

                printf(" -- [RMS] Job %zu finished\n", job->jobId);
                fflush(stdout);


                nJobsFinished ++;
                nRunningJobs --;
                iJob--;
            }

            // if job not finished and job is malleable or flexible, check for reconfigurations [RECONFIGURATION POLICY]
            else if (jobType > 1){

                // update GPU usage
                jobMonitor = job->jobMonitor;  
                for(size_t jobGPU = 0; jobGPU < job->jobControl->jobResources->nGPUs; jobGPU++){

                    jobMonitor->gpuUsage[jobGPU] += schInfo->gpuUtilization[job->jobControl->jobResources->idGPUs[jobGPU]][0];
                }
                jobMonitor->step ++;


                // [ RECONFIGURATION POLICY ]
                if(jobMonitor->step > 10){ // wait 5 seconds before taking decisions about what to do

                    // get mean usage of the GPUs
                    double meanUsage = 0.0;
                    for(size_t jobGPU = 0; jobGPU < job->jobControl->jobResources->nGPUs; jobGPU++){

                        meanUsage += jobMonitor->gpuUsage[jobGPU]; 
                    }  
                    meanUsage /= (jobMonitor->step * job->jobControl->jobResources->nGPUs);


                    // take decisions depending on the mean usage

                    if(jobResources->nGPUs > 1 && (meanUsage < 70.0)){ // shrink
                                        
                        // new job resources
                        jobResources_t *reconfJobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));

                        reconfJobResources->nGPUs = jobResources->nGPUs / 2;
                        reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));

                        // set old GPUs as new ones for this job
                        for(size_t i = 0; i<reconfJobResources->nGPUs; i++){

                            reconfJobResources->idGPUs[i] = jobResources->idGPUs[i];
                        }

                        // allocate new resources and deallocate WHEN reconfiguration is FINISHED
                        // do not call allocate, since the resources are the same as the previous ones
                        //allocateResources(schInfo, reconfJobResources);

                        scheduleReconfiguration(job, iJob, reconfJobResources);
                    
                
                        printf(" -- [RMS] Mean GPU usage of job %zu is %lf, so it will be shrinked (from %zu to %zu)\n", 
                            job->jobId, meanUsage, jobResources->nGPUs, reconfJobResources->nGPUs);
                    }
                    else if(meanUsage > 90.0){ // expand


                        // if nGPUs % 2 == 0, nGPUs * 2, else, + 1 (module of 2?)
                        //jobResources_t *reconfJobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
                        
                        //reconfJobResources->nGPUs = jobResources->nGPUs * 2;

                        //reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));

                        //for(size_t i = 0; i<jobResources->nGPUs; i++){

                        //    reconfJobResources->idGPUs[i] = jobResources->idGPUs[i]; 
                        //}

                        // find available GPUs
                        //size_t gpuId = 0;
                        //size_t jobGPU = jobResources->nGPUs;
                        
                        //while(jobGPU < reconfJobResources->nGPUs){

                            // allocate the gpu for the job
                        //    if(schInfo->avGPUs[gpuId] == 1){
                                
                                // set the GPU id
                        //        jobResources->idGPUs[jobGPU] = gpuId; 

                                // update index
                        //        jobGPU ++;
                        //    }

                        //    gpuId ++;
                        //}

                        //printf(" -- Mean GPU usage of job %zu is %lf, so it will be expanded\n", job->jobId, meanUsage);
                    //}


                    // reinitialize monitor (steps and GPU usage) : [TODO]: refactorize to a function?
                    jobMonitor->step = 0; 
                    for(size_t jobGPU = 0; jobGPU < job->jobControl->jobResources->nGPUs; jobGPU++){

                        jobMonitor->gpuUsage[jobGPU] = 0;
                    }
                }
            }


            // reconfiguration policy
            //printf(" -- No reconfiguration policy yet\n");

            // For each reconfiguration
            // - Reconfigure application
            // - set new resource availability
            // - update monitoring...
            // All this should be done in a function for modularity
        }*/
    }

    
    // inform monitor and the rest threads that the entire workload has been executed
    finished = 1;


    /* comptue statistics */
    // -- Mean wait time
    // -- Mean execution time
    // -- Mean total time

    /*double meanWait = 0.0, meanRun = 0.0, meanTotal = 0.0;
    
    nFinishedJobs = getNumberOfJobsInQueue(&(schInfo->finishedJobs));
    for(iJob = 0; iJob < nFinishedJobs; iJob ++){

        // get job
        job = getJobFromQueue(finishedQueue, iJob);

        meanWait += (job->jobEndPending.tv_sec - job->jobStartPending.tv_sec) + (job->jobEndPending.tv_nsec - job->jobStartPending.tv_nsec) / 1e9;
        meanRun += (job->jobEndRunning.tv_sec - job->jobStartRunning.tv_sec) + (job->jobEndRunning.tv_nsec - job->jobStartRunning.tv_nsec) / 1e9;
    }

    meanTotal = (meanWait + meanRun) / nFinishedJobs;
    meanWait = meanWait / nFinishedJobs;
    meanRun = meanRun / nFinishedJobs;


    clock_gettime(CLOCK_MONOTONIC, &(endTimer));
    double totalTime = (endTimer.tv_sec - startTimer.tv_sec) + (endTimer.tv_nsec - startTimer.tv_nsec) / 1e9;



    printf(" -- [RMS] Mean total time = %lf, mean total wait time = %lf, mean total run time = %lf\n", meanTotal, meanWait, meanRun);


    // mean usage of active GPUs and energy consumption or something like that?
    fprintf(fOutput, "%zu %lf %lf %lf %lf %lf %lf %d\n", 
            jobsTimeline->nJobs, totalTime, meanTotal, meanWait, meanRun, (double)((double)usageGPUs / (double)registeredUsages), usagePower, timer);
    fflush(fOutput);*/
}


// TODO: rethink code organization
// - Which will encapsulate each function? When a job is launche