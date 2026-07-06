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



// Workload simulation
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
    pthread_create(&thrJobs, NULL, jobManager, (void*)(jobsTimeline));


    // Revise all the code below
    /*
    Important notes:
        -- JobReources is copied by the job, but there is no direct synchronization mechanism
        for that property. If the reconfiguration policy just respects the fact of waiting for the job
        to finish its programmed reconfiguration before programming a new one, the synchronization is
        correct since it is the unique point in which race conditions can occur for that structure
            -- RMS program reconfiguration
            -- Job reconfigure
            -- Job notify RMS
    */


    // [DECLARE VARIABLES USED DURING WORKLOAD SIMULATION]

    // get queues

    // helper variables
    size_t nJobsFinished; // number of jobs that already finished their execution
    size_t iJob, nPendingJobs, nRunningJobs, nFinishedJobs, nReconfiguringJobs; // helper variables

    // helper variables
    job_t *job; // links all job properties


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


        // print relevant information each 30 seconds
        if(timer % 30 == 0){

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

                job = getJobFromQueue(&(schInfo->runningJobs), iJob);
                printf(" ---- ---- Job %zu is using %zu GPUs: (", job->jobId, job->jobControl->jobResources->nGPUs);
                for(size_t gpu = 0; gpu<job->jobControl->jobResources->nGPUs; gpu++){

                    printf(" %zu", job->jobControl->jobResources->idGPUs[gpu]);
                }
                printf(")\n");
            }
        }

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


    clock_gettime(CLOCK_MONOTONIC, &(endTimer));
    double totalTime = (endTimer.tv_sec - startTimer.tv_sec) + (endTimer.tv_nsec - startTimer.tv_nsec) / 1e9;

    printf("Summarize of the workload execution:\n");
    printf(" - Makespan: %lf\n", totalTime);
}