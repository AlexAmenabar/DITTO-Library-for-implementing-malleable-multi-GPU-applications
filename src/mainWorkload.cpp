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

size_t usageGPUs = 0;
double usagePower = 0.0;
size_t registeredUsages = 0;

// temp
int *gpuTopology;
int *gpuTopologyRank;
size_t gNGPUs;

pthread_mutex_t printLock;



void storeBatchResultsInFile(schInfo_t *schInfo, FILE *fOutput, size_t batch, size_t tBatch, size_t iPrevJob){

    size_t i;
    size_t nFinishedJobs = getNumberOfJobsInQueue(&(schInfo->finishedJobs));
    size_t batchFinishedJobs = nFinishedJobs - iPrevJob;
    size_t nGPUs = schInfo->nGPUs;
    job_t *job;

    //pthread_mutex_lock(&(printLock));
    fprintf(fOutput, "BATCH=%zu\n", batch);

    fprintf(fOutput, "[System]\n");
    
    fprintf(fOutput, " - Throughput = %zu\n", batchFinishedJobs);
    schInfo->finalThroughput[batch] = (double)batchFinishedJobs;
    fprintf(fOutput, " - Utilization = %lf\n", (double)schInfo->totalUtilization / ((double)schInfo->nGPUs * (double)tBatch));
    schInfo->finalUtilization[batch] = (double)schInfo->totalUtilization / ((double)schInfo->nGPUs * (double)tBatch);
    fprintf(fOutput, " - Utilization per GPU = [");
    for(i = 0; i < schInfo->nGPUs-1; i++){

        fprintf(fOutput, "%lf ", (double)schInfo->totalUtilizationPerGPU[i] / ((double)tBatch));
        schInfo->finalUtilizationPerGPU[i][batch] = (double)schInfo->totalUtilizationPerGPU[i] / ((double)tBatch);

    }
    fprintf(fOutput, "%lf]\n", (double)schInfo->totalUtilizationPerGPU[i] / ((double)tBatch));
    schInfo->finalUtilizationPerGPU[i][batch] = (double)schInfo->totalUtilizationPerGPU[i] / ((double)tBatch);


    // allocation time
    fprintf(fOutput, " - Allocation area = %lf\n", (double)schInfo->totalAllocationArea / ((double)nGPUs * (double)tBatch));
    schInfo->finalAllocationArea[batch] = (double)schInfo->totalAllocationArea / ((double)nGPUs * (double)tBatch);
    fprintf(fOutput, " - Allocation time per GPU = [");    
    for(i = 0; i < schInfo->nGPUs-1; i++){

        fprintf(fOutput, "%lf ", (double)schInfo->allocationTimePerGPU[i] / ((double)tBatch));
        schInfo->finalAllocationTimePerGPU[i][batch] = (double)schInfo->allocationTimePerGPU[i] / ((double)tBatch);

    }
    fprintf(fOutput, "%lf]\n", (double)schInfo->allocationTimePerGPU[i] / ((double)tBatch));
    schInfo->finalAllocationTimePerGPU[i][batch] = (double)schInfo->allocationTimePerGPU[i] / ((double)tBatch);


    // bandwidth time
    fprintf(fOutput, " - PCIe throug. = %lf\n", (double)(schInfo->totalThroughputPCIe) / ((double)nGPUs * (double)tBatch));
    schInfo->finalThroughputPCIe[batch] = (double)(schInfo->totalThroughputPCIe) / ((double)nGPUs * (double)tBatch);
    fprintf(fOutput, " - PCIe throug. per GPU = [");    
    for(i = 0; i < schInfo->nGPUs-1; i++){

        fprintf(fOutput, "%lf ", (double)(schInfo->totalThroughputPCIePerGPU[i]) / (double)tBatch);
        schInfo->finalThroughputPCIePerGPU[i][batch] = (double)(schInfo->totalThroughputPCIePerGPU[i]) / ((double)tBatch);

    }
    fprintf(fOutput, "%lf]\n", (double)(schInfo->totalThroughputPCIePerGPU[i]) / ((double)tBatch));
    schInfo->finalThroughputPCIePerGPU[i][batch] = (double)(schInfo->totalThroughputPCIePerGPU[i]) / ((double)tBatch);

    
    fprintf(fOutput, " - NVLink band. = %lf\n", (double)(schInfo->totalThroughputNVLink) / ((double)nGPUs * (double)tBatch));
    schInfo->finalThroughputNVLink[batch] = (double)(schInfo->totalThroughputNVLink) / ((double)nGPUs * (double)tBatch);
    fprintf(fOutput, " - NVLink band. per GPU = [");
    for(i = 0; i < schInfo->nGPUs-1; i++){

        fprintf(fOutput, "%lf ", (double)(schInfo->totalThroughputNVLinkPerGPU[i]) / ((double)tBatch));
        schInfo->finalThroughputNVLinkPerGPU[i][batch] = (double)(schInfo->totalThroughputNVLinkPerGPU[i]) / ((double)tBatch);
    }
    fprintf(fOutput, "%lf]\n", (double)(schInfo->totalThroughputNVLinkPerGPU[i]) / ((double)tBatch));
    schInfo->finalThroughputNVLinkPerGPU[i][batch] = (double)(schInfo->totalThroughputNVLinkPerGPU[i]) / ((double)tBatch);


    // Jobs information
    fprintf(fOutput, "\n[Jobs]\n");

    // Job information
    double meanRunTime = 0.0, meanWaitTime = 0.0;
    for(i = iPrevJob; i < nFinishedJobs; i++){

        // get job
        job = getJobFromQueue(&(schInfo->finishedJobs), i);

        meanRunTime += (job->jobEndRunning.tv_sec - job->jobStartRunning.tv_sec) + (job->jobEndRunning.tv_nsec - job->jobStartRunning.tv_nsec) / 1e9;
        meanWaitTime += (job->jobEndPending.tv_sec - job->jobStartPending.tv_sec) + (job->jobEndPending.tv_nsec - job->jobStartPending.tv_nsec) / 1e9;
    }

    fprintf(fOutput, " - Mean execution time = %lf\n", meanRunTime / (double)(batchFinishedJobs));
    fprintf(fOutput, " - Mean wait time = %lf\n", meanWaitTime / (double)(batchFinishedJobs));

    schInfo->finalMeanExecutionTime[batch] = meanRunTime / (double)(batchFinishedJobs);
    schInfo->finalMeanWaitTime[batch] = meanWaitTime / (double)(batchFinishedJobs);

    // Data per job
    for(i = iPrevJob; i < nFinishedJobs; i++){

        // get job
        job = getJobFromQueue(&(schInfo->finishedJobs), i);

        meanRunTime = (job->jobEndRunning.tv_sec - job->jobStartRunning.tv_sec) + (job->jobEndRunning.tv_nsec - job->jobStartRunning.tv_nsec) / 1e9;
        meanWaitTime = (job->jobEndPending.tv_sec - job->jobStartPending.tv_sec) + (job->jobEndPending.tv_nsec - job->jobStartPending.tv_nsec) / 1e9;
        
        fprintf(fOutput, " - Job %zu execution time = %lf, wait time = %lf\n", i, meanRunTime, meanWaitTime);
    }
    
    fprintf(fOutput, " - Reconfigurations = %u\n", schInfo->nReconfigurations);
    fprintf(fOutput, " - Expands = %u\n", schInfo->nExpands);
    fprintf(fOutput, " - Shrinks = %u\n", schInfo->nShrinks);
    fprintf(fOutput, " - Keeps = %u\n", schInfo->nKeeps);
    fprintf(fOutput, "\n\n");


    // store results
    schInfo->finalNReconfigurations[batch] = (double)schInfo->nReconfigurations;
    schInfo->finalNExpands[batch] = (double)schInfo->nExpands;
    schInfo->finalNShrinks[batch] = (double)schInfo->nShrinks;
    schInfo->finalNKeeps[batch] = (double)schInfo->nKeeps;

    fflush(fOutput);
    //pthread_mutex_unlock(&(printLock));
}


void storeFinalResultsInFile(schInfo_t *schInfo, FILE *fOutput, size_t nBatches){

    size_t i;

    //pthread_mutex_lock(&(printLock));
    fprintf(fOutput, "Final results\n");

    fprintf(fOutput, "[System]\n");
    
    double meanThroughput = 0.0;
    double meanFinalUtilization = 0.0;
    double *meanFinalUtilizationPerGPU = (double*)calloc(schInfo->nGPUs, sizeof(double));
    double meanFinalPCIeThroughput = 0.0;
    double *meanFinalPCIeThroughputPerGPU = (double*)calloc(schInfo->nGPUs, sizeof(double));
    double meanFinalNVLinkThroughput = 0.0;
    double *meanFinalNVLinkThroughputPerGPU = (double*)calloc(schInfo->nGPUs, sizeof(double));
    double meanFinalAllocationArea = 0.0;
    double *meanFinalAllocationAreaPerGPU = (double*)calloc(schInfo->nGPUs, sizeof(double));
   
    double nReconf = 0.0;
    double nExpands = 0.0;
    double nShrinks = 0.0;
    double nKeeps = 0.0;
    double meanFinalExecutionTime = 0.0;
    double meanFinalWaitTime = 0.0;

    for(size_t batch = 0; batch<nBatches; batch++){

        meanThroughput += schInfo->finalThroughput[batch];
        meanFinalUtilization += schInfo->finalUtilization[batch];
        meanFinalPCIeThroughput += schInfo->finalThroughputPCIe[batch];
        meanFinalNVLinkThroughput += schInfo->finalThroughputNVLink[batch];
        meanFinalAllocationArea += schInfo->finalAllocationArea[batch];
        nReconf += schInfo->finalNReconfigurations[batch];
        nExpands += schInfo->finalNExpands[batch];
        nShrinks += schInfo->finalNShrinks[batch];
        nKeeps += schInfo->finalNKeeps[batch];

        meanFinalExecutionTime += schInfo->finalMeanExecutionTime[batch];
        meanFinalWaitTime += schInfo->finalMeanWaitTime[batch];

        for(size_t g = 0; g<schInfo->nGPUs; g++){

            meanFinalUtilizationPerGPU[g] += schInfo->finalUtilizationPerGPU[g][batch];
            meanFinalPCIeThroughputPerGPU[g] += schInfo->finalThroughputPCIePerGPU[g][batch];
            meanFinalNVLinkThroughputPerGPU[g] += schInfo->finalThroughputNVLinkPerGPU[g][batch];
            meanFinalAllocationAreaPerGPU[g] += schInfo->finalAllocationTimePerGPU[g][batch];
        }
    }

    fprintf(fOutput, " - Throughput = %lf\n", meanThroughput / (double)nBatches);
    fprintf(fOutput, " - Utilization = %lf\n", meanFinalUtilization / ((double)nBatches));
    fprintf(fOutput, " - Utilization per GPU = [");
    for(i = 0; i < schInfo->nGPUs-1; i++){

        fprintf(fOutput, "%lf ", meanFinalUtilizationPerGPU[i] / ((double)nBatches));

    }
    fprintf(fOutput, "%lf]\n", meanFinalUtilizationPerGPU[i] / ((double)nBatches));

    // allocation time
    fprintf(fOutput, " - Allocation area = %lf\n", meanFinalAllocationArea / ((double)nBatches));
    fprintf(fOutput, " - Allocation time per GPU = [");    
    for(i = 0; i < schInfo->nGPUs-1; i++){

        fprintf(fOutput, "%lf ", meanFinalAllocationAreaPerGPU[i] / ((double)nBatches));

    }
    fprintf(fOutput, "%lf]\n", meanFinalAllocationAreaPerGPU[i] / ((double)nBatches));


    // throughput time
    fprintf(fOutput, " - PCIe throug. = %lf\n", meanFinalPCIeThroughput / ((double)nBatches));
    fprintf(fOutput, " - PCIe throug. per GPU = [");    
    for(i = 0; i < schInfo->nGPUs-1; i++){

        fprintf(fOutput, "%lf ", meanFinalPCIeThroughputPerGPU[i] / ((double)nBatches));

    }
    fprintf(fOutput, "%lf]\n", meanFinalPCIeThroughputPerGPU[i] / ((double)nBatches));

    
    fprintf(fOutput, " - NVLink band. = %lf\n", meanFinalNVLinkThroughput / ((double)nBatches));
    fprintf(fOutput, " - NVLink band. per GPU = [");
    for(i = 0; i < schInfo->nGPUs-1; i++){

        fprintf(fOutput, "%lf ", meanFinalNVLinkThroughputPerGPU[i] / ((double)nBatches));
    }
    fprintf(fOutput, "%lf]\n", meanFinalNVLinkThroughputPerGPU[i] / ((double)nBatches));


    // Jobs information
    fprintf(fOutput, "\n[Jobs]\n");

    fprintf(fOutput, " - Mean execution time = %lf\n", meanFinalExecutionTime / (double)(nBatches));
    fprintf(fOutput, " - Mean wait time = %lf\n", meanFinalWaitTime / (double)(nBatches));
    fprintf(fOutput, " - Reconfigurations = %lf\n", nReconf);
    fprintf(fOutput, " - Expands = %lf\n", nExpands);
    fprintf(fOutput, " - Shrinks = %lf\n", nShrinks);
    fprintf(fOutput, " - Keeps = %lf\n", nKeeps);
    fprintf(fOutput, "\n\n");
    fflush(fOutput);
    //pthread_mutex_unlock(&(printLock));
}


// Workload simulation
int main(int argc, char* argv[]){

    pthread_t thrJobs;
    int nGPUs;

    srand(time(NULL));


    // [SCHEDULER INITIALIZATION]
    // allocate memory to store cuda data

    //cudaGetDeviceCount(&nGPUs); // get number of available devices
    nGPUs = strtoul(argv[1], NULL, 10); // TMP: load number of GPUs from terminal input
    gNGPUs = (size_t)nGPUs;

    // file directories to store results
    char *recordFileName = argv[4];
    char *gpuUsageFileName = argv[5];
    char *resultsFileName = argv[6];

    int intNBatches = strtoul(argv[7], NULL, 10); // TMP: load number of GPUs from terminal input
    int intTBatch = strtoul(argv[8], NULL, 10); // TMP: load number of GPUs from terminal input

    size_t nBatches = (size_t)(intNBatches);
    size_t tBatch = (size_t)(intTBatch);
    size_t batch = 0;

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

#ifdef UTILIZATION
    schInfo->sched = &greedy;
    schInfo->rconf = &utilization;
#elif TOPO
    schInfo->sched = &greedy;
    schInfo->rconf = &topology;
#endif

    schInfo->timeout = 60 * 10;
    schInfo->timeoutCurrent = 0;

    // initialize lock
    pthread_mutex_init(&(schInfo->lockTimer), NULL);
    pthread_mutex_init(&(schInfo->invoqueSchedulerLock), NULL);
    schInfo->invoqueScheduler = 0;

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
    initResourceMonitor(schInfo, nBatches);

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
    size_t iJob, nPendingJobs, nRunningJobs, nFinishedJobs, nReconfiguringJobs; // helper variables

    // helper variables
    job_t *job; // links all job properties


    double et = 0.0;
    struct timespec start, end;
    double sleepTime = INTERVAL_US;

    // [SCHEDULING LOOP]
    pthread_mutex_lock(&(printLock));
    printf(" -- [RMS] Starting simulation (%zu jobs for completion)\n", jobsTimeline->nJobs);
    fflush(stdout);
    pthread_mutex_unlock(&(printLock));

    struct timespec startTimer, endTimer;
    clock_gettime(CLOCK_MONOTONIC, &(startTimer)); 

    // loop until all jobs finish their execution
    nFinishedJobs = 0; 
    //while(nFinishedJobs < jobsTimeline->nJobs){
    
    batch = 0;
    size_t iPrevJob = 0;

    while(batch < nBatches){

        // wait one second between scheduler calls
        usleep(sleepTime);
        int tmpTimer;
        pthread_mutex_lock(&(schInfo->lockTimer));
        timer ++;    
        tmpTimer = timer;
        pthread_mutex_unlock(&(schInfo->lockTimer));

        // update timeout counter only if the warm-up time finished
        schInfo->timeoutCurrent ++;

        // batch finished
        if((tmpTimer) % tBatch == 0 && tmpTimer > 60 * 20){ // 10 minutes of warm-up
            
            storeBatchResultsInFile(schInfo, fOutput, batch, tBatch, iPrevJob);
            iPrevJob = getNumberOfJobsInQueue(&(schInfo->finishedJobs));

            // reinit accumulated data for the following batch
            reinitMonitorAcc(schInfo);

            // update batch
            batch++;
        }


        //pthread_mutex_lock(&(printLock));
        //printf(" %d\n", tmpTimer);
        //fflush(stdout);
        //pthread_mutex_unlock(&(printLock));

        // start step timer
        clock_gettime(CLOCK_MONOTONIC, &start);

        // monitor resources
        stepResourceMonitor(schInfo);

        // get number of jobs on each queue
        nPendingJobs = getNumberOfJobsInQueue(&(schInfo->pendingJobs));
        nRunningJobs = getNumberOfJobsInQueue(&(schInfo->runningJobs));
        nFinishedJobs = getNumberOfJobsInQueue(&(schInfo->finishedJobs));
        nReconfiguringJobs = getNumberOfJobsInQueue(&(schInfo->reconfiguringJobs));

        // print relevant information each 30 seconds
        if(tmpTimer % 10 == 0){
        
            pthread_mutex_lock(&(printLock));
        
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

            fflush(stdout);

            pthread_mutex_unlock(&(printLock));
        }

        // check whether jobs finished reconfigurations and manage them
        manageReconfigurations(schInfo); 

        // manage jobs that finished their execution
        manageJobsFinish(schInfo); 

        // Invoque reconfigurator (first reconfiugration, then sched)
        if(tmpTimer % 5 == 0){
            reconf(schInfo); 
        }

        // Invoque scheduler when: a job is added to the pending list, a reconfiguration finishes, a job finishes
        int invoque = 0;
        pthread_mutex_lock(&(schInfo->invoqueSchedulerLock));
        if(schInfo->invoqueScheduler){
            
            invoque = 1;
            schInfo->invoqueScheduler = 0;    
        }
        pthread_mutex_unlock(&(schInfo->invoqueSchedulerLock));

        if(invoque)
            sched(schInfo);


        clock_gettime(CLOCK_MONOTONIC, &end);
        et = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        sleepTime = INTERVAL_US - (et * 1000000); // 1 second between calls

        if (sleepTime < 0.0)
            sleepTime = 0.0;

        if(tmpTimer % 10 == 0){
        
            printf("Scheduler step = %.3f s, sleep = %.0f us\n", et, sleepTime);
            fflush(stdout);
        }
    }

    storeFinalResultsInFile(schInfo, fOutput, nBatches);


    // destroy resource monitor
    destroyResourceMonitor(schInfo);

    printf(" Finished!\n");
    fflush(stdout);


    clock_gettime(CLOCK_MONOTONIC, &(endTimer));
    double totalTime = (endTimer.tv_sec - startTimer.tv_sec) + (endTimer.tv_nsec - startTimer.tv_nsec) / 1e9;
}