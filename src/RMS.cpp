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


//#define INTERVAL_US 1000000  // 0.1 seconds


// [TODO]: in the future they should be private?
/*schInfo_t *schInfo;
int timer = 0;
FILE *fEventRecord = NULL, *fUsage = NULL, *fOutput = NULL;
size_t nextJobId = 1;

int invoqueScheduler = 0;

size_t usageGPUs = 0;
double usagePower = 0.0;
size_t registeredUsages = 0;

// temp
int *gpuTopology;
int *gpuTopologyRank;*/


// TODO: ADAPT THIS TO NEW IMPLEMEMNTATION
jobsTimeline_t* loadJobsFromFile(const char* jobsFileName){

    // declare variables
    FILE *jobF; // file to load jobs timeline from
    jobsTimeline_t *jobsTimeline; // jobs timeline structure to store jobs data
    size_t jobType, jobPriority;

    // open file and allocate memory for timeline data
    jobF = fopen(jobsFileName, "r");
    jobsTimeline = (jobsTimeline_t*)calloc(1, sizeof(jobsTimeline_t));


    // read the number of jobs
    fscanf(jobF, "%zu", &(jobsTimeline->nJobs));

    // allocate memory for job launchers
    jobsTimeline->jobLaunchers = (jobLauncher_t*)calloc(jobsTimeline->nJobs, sizeof(jobLauncher_t));

    // loop over jobs and load their information in jobLaunchers
    for(size_t job=0; job < jobsTimeline->nJobs; job++){

        // get job launcher
        jobLauncher_t *jobLauncher = &(jobsTimeline->jobLaunchers[job]);

        // load job type and set enum 
        fscanf(jobF, "%zu", &(jobType));

        switch (jobType){
            case 0:
            jobLauncher->jobType = RIGID;
            break;

            case 1:
            jobLauncher->jobType = MOLDABLE;
            break;
            
            case 2:
            jobLauncher->jobType = MALLEABLE;
            break;
            
            case 3:
            jobLauncher->jobType = FLEXIBLE;
            break;
        }


        // load job priority and set enum 
        fscanf(jobF, "%zu", &(jobPriority));

        switch (jobPriority){
            case 0:
            jobLauncher->jobPriority = LOW;
            break;

            case 1:
            jobLauncher->jobPriority = MEDIUM;
            break;
            
            case 2:
            jobLauncher->jobPriority = HIGH;
            break;
            
            case 3:
            jobLauncher->jobPriority = DEADLINE;
            break;
        }

        // load requested number of GPUs by the job
        fscanf(jobF, "%zu", &(jobLauncher->nReqGPUs));
        fscanf(jobF, "%zu", &(jobLauncher->nReqMinGPUs));

        // if job is moldable, malleable or flexible, load also the minimum number of GPUs
        if(jobType == 0)
            jobLauncher->nReqMinGPUs = jobLauncher->nReqGPUs;


        // load the time step in which the user launches the job to the system
        fscanf(jobF, "%zu", &(jobLauncher->launchTimeStep));

        // estimated job duration
        fscanf(jobF, "%zu", &(jobLauncher->estimatedDuration));

        // load the application type
        fscanf(jobF, "%zu", &(jobLauncher->appType));

        // load the arguments for launching the application
        fscanf(jobF, "%d", &(jobLauncher->argc)); // number of arguments

        // allocate memory for argv
        jobLauncher->argv = (void**)malloc((jobLauncher->argc + 2) * sizeof(void*)); // 2 extra arguments: whether the job is malleable and the job control (latter set)
        
        pthread_mutex_lock(&(printLock));

        //printf(" Printing argc");

        // load argv parameters: application related parameters
        for(int arg = 0; arg<jobLauncher->argc; arg++){
            
            // if not pointer will disappear after function ends (would be stored in the stack)
            size_t *val = (size_t*)malloc(sizeof(size_t));
            fscanf(jobF, "%zu", val);
            jobLauncher->argv[arg] = (void*)val;

            //printf(" %d = %zu\n", arg, (*val));
        }
        
        pthread_mutex_unlock(&(printLock));


        // load whether application is or not malleable or flexible for allowing reconfiugrations during execution
        size_t *jmall = (size_t*)malloc(sizeof(size_t)); 
        (*jmall) = 0;
        if(jobType == 2 || jobType == 3){
            (*jmall) = 1;
        }



        jobLauncher->argv[jobLauncher->argc] = (void*)jmall;

        // load application main function call pointer
        if(jobLauncher->appType == 0){

            jobLauncher->launchFunc = &launch_iterative_app;
        }
        else if(jobLauncher->appType == 1){

            jobLauncher->launchFunc = &launch_phases_app;
        }
        else if(jobLauncher->appType == 2){
            jobLauncher->launchFunc = &launch_NCCL_communications_app;
        }

        pthread_mutex_lock(&(printLock));
        printf(" [RMS]: Loaded job with: \n");
        printf("  -- Job type = %d, app type = %zu, min gpus = %zu, max gpus = %zu\n", jobLauncher->jobType, jobLauncher->appType, jobLauncher->nReqGPUs, jobLauncher->nReqMinGPUs);
        fflush(stdout);
        pthread_mutex_unlock(&(printLock));
    }

    return jobsTimeline;
}


// TODO
void initSystem(){

    printf(" -- Not implemented yet\n");
    fflush(stdout);
    exit(1); 
}

void initializeTopology(schInfo_t *schInfo, char *topoFile){

    // get the number of GPUs
    size_t N = schInfo->nGPUs;

    // topology file
    FILE *fTopo; // file to load jobs timeline from

    // open file and allocate memory for timeline data
    fTopo = fopen(topoFile, "r");


    // matrix of N * N
    int *gpuTopology = (int*)calloc(N * N, sizeof(int));
    int *gpuTopologyRank = (int*)calloc(N * N, sizeof(int));

    // initialize NVML init for obtaining topology information
    // level for storing the connection level between two GPUs
    //nvmlGpuTopologyLevel_t level;
    nvmlReturn_t ret = nvmlInit();
    
    if (ret != NVML_SUCCESS) {
        
        fprintf(stderr, "NVML init failed: %s\n", nvmlErrorString(ret));
        fflush(stdout);
        exit(1);
    }

    // loop over GPUs get the topology information P2P
    for(size_t i = 0; i<N; i++){

        for(size_t j = 0; j<N; j++){

            fscanf(fTopo, "%d", &(gpuTopology[i * N + j]));
            gpuTopologyRank[i * N + j] = 0;


            /*if(i!=j){

                nvmlDevice_t dev0, dev1;
                nvmlDeviceGetHandleByIndex(i, &dev0);
                nvmlDeviceGetHandleByIndex(j, &dev1);
                nvmlDeviceGetTopologyCommonAncestor(dev0, dev1, &level);
                
                cudaDeviceGetP2PAttribute(&rank, cudaDevP2PAttrPerformanceRank, (int)i, (int)j);

                switch (level) {
                    case NVML_TOPOLOGY_INTERNAL:
                    gpuTopology[i * N + j] = 1;
                    break;
                    
                    case NVML_TOPOLOGY_SINGLE:
                    gpuTopology[i * N + j] = 2;
                    break;
                    
                    case NVML_TOPOLOGY_MULTIPLE:
                    gpuTopology[i * N + j] = 3;    
                    break;
                    
                    case NVML_TOPOLOGY_HOSTBRIDGE:
                    gpuTopology[i * N + j] = 4;    
                    break;

                    case NVML_TOPOLOGY_NODE:
                    gpuTopology[i * N + j] = 5;    
                    break;
                    
                    case NVML_TOPOLOGY_SYSTEM:
                    gpuTopology[i * N + j] = 6;    
                    break;
                }

                gpuTopologyRank[i * N + j] = rank;
            }
            else{

                gpuTopology[i * N + j] = 0;
                gpuTopologyRank[i * N + j] = 0;

            }*/
        }
    }

    // store the topology matrix
    schInfo->gpuTopology = gpuTopology;
    schInfo->gpuTopologyRank = gpuTopologyRank;

    // shut down NVML
    nvmlShutdown();

    
    pthread_mutex_lock(&(printLock));

    // print topology information
    printf(" -- Printing topology:\n");
    for(size_t i = 0; i<N; i++){
        for(size_t j = 0; j<N; j++){

            printf("%d (%d)  ", gpuTopology[i*N+j], gpuTopologyRank[i*N + j]);
        }
        printf("\n");
    }
    printf("\n");

    pthread_mutex_unlock(&(printLock));
}

void addPendingJob(jobLauncher_t *jobLauncher){

    // initialize job structure (jobLauncher in timeline stucture)
    job_t *job = initJob(jobLauncher);
    
    // change job state
    job->jobState = PENDING; // job is pending
    job->jobId = nextJobId; // set job id
    nextJobId ++; // update id for the next job

    // start pending timer
    clock_gettime(CLOCK_MONOTONIC, &(job->jobStartPending));

    // add job to pending queue
    addJobToQueue(&(schInfo->pendingJobs), job);
}

void launchJob(schInfo_t *schInfo, job_t *job, size_t pendingIndex, jobResources_t *jobResources){

    // allocate resources
    allocateResources(schInfo, jobResources);

    // remove job from pending queue
    removeJobFromQueueByIndex(&(schInfo->pendingJobs), pendingIndex);
  
    // update job state
    job->jobState = RUNNING;

    // complete job information
    job->jobControl->jobResources = jobResources; // set job resources
    job->jobControl->jobId = job->jobId; // copy job id to jobControl (TODO: revise, repeated info)

    // initialize job monitor structure 
    job->jobMonitor = initJobMonitor(job->jobMonitor, jobResources);

    // launch time model
    // wait (simulate launch time) (TODO: should be improved?)
    //int sleepTime = (int)rand() % 5;
    //sleep(sleepTime);

    // launch job (thread)
    pthread_create(&(job->jobThread), NULL, runJob, (void*)(job->jobLauncher));

    // manage timers
    clock_gettime(CLOCK_MONOTONIC, &(job->jobEndPending)); // end pending
    clock_gettime(CLOCK_MONOTONIC, &(job->jobStartRunning)); // start running

    // record event (visualization purposes)
    recordEvent(JOBSTARTED, job, jobResources);

    // add job to running queue
    addJobToQueue(&(schInfo->runningJobs), job);
}

void scheduleReconfiguration(schInfo_t *schInfo, job_t *job, size_t jobIndex, jobResources_t *reconfJobResources){

    // get job control
    jobControl_t *jobControl = job->jobControl;    

    // update job state
    job->jobState = RECONFIGURING;

    // model time required for allocating new resources    
    // wait (simulate allocation time) (TODO: should be improved?)
    //int sleepTime = (int)rand() % 5;
    //sleep(sleepTime);

    // add job to reconfiguration queue
    addJobToQueue(&(schInfo->reconfiguringJobs), job);

    // set job resources for reconfiguration
    jobControl->reconfJobResources = reconfJobResources;
    jobResources_t *prevJobResources = jobControl->jobResources;

    // find GPUs that in reconf and not in prev for allocating
    jobResources_t* diffJobResources = findDiffResources(reconfJobResources, prevJobResources);

    // allocate new GPUs and record event, if there are new resources
    if(diffJobResources->nGPUs > 0){
    
        allocateResources(schInfo, diffJobResources);

        // record event
        recordEvent(RECONFSTARTED, job, diffJobResources);
    }

    // deallocate temporal structure
    deallocateJobResourcesStruct(&(diffJobResources));

    // notify reconfiguration to the job
    notifyReconfiguration(jobControl);
}

void jobFinishedReconfiguration(schInfo_t *schInfo, job_t *job, size_t jobIndex){

    // get job control
    jobControl_t *jobControl = job->jobControl;
    jobResources_t *jobResources = jobControl->jobResources;
    jobResources_t *reconfJobResources = jobControl->reconfJobResources;

    // update job state
    job->jobState = RUNNING;

    // remove job from reconfiguring queue
    removeJobFromQueueByIndex(&(schInfo->reconfiguringJobs), jobIndex);
  
    // update job monitor to the new resources
    job->jobMonitor = initJobMonitor(job->jobMonitor, jobControl->reconfJobResources);

    // find GPUs that are in the previous configuration but not in the reconfiguration for deallocation
    jobResources_t* diffJobResources = findDiffResources(jobResources, reconfJobResources);
    recordEvent(RECONFFINISHED, job, diffJobResources);
    deallocateResources(schInfo, diffJobResources);
    deallocateJobResourcesStruct(&(diffJobResources));

    // update job resources
    // 1. deallocate old resources, reconfiguration is done, they are not relevant anymore
    deallocateJobResourcesStruct(&(jobControl->jobResources));
    
    // reconf job resources are now current job resources
    jobControl->jobResources = jobControl->reconfJobResources;
    jobControl->reconfJobResources = NULL;
}

void finishJob(schInfo_t *schInfo, job_t *job, size_t runningJobIndex){

    // update queues
    removeJobFromQueueByIndex(&(schInfo->runningJobs), runningJobIndex); // remove from running jobs queue
    addJobToQueue(&(schInfo->finishedJobs), job); // add to finished jobs queue

    // kill job thread
    pthread_join(job->jobThread, NULL);

    // manage timers
    clock_gettime(CLOCK_MONOTONIC, &(job->jobEndRunning));

    // record event
    recordEvent(JOBFINISHED, job, job->jobControl->jobResources);
    
    // deallocate resources (RMS level)
    deallocateResources(schInfo, job->jobControl->jobResources);

    // deallocate resource structure // TODO: delete comment
    //deallocateJobResourcesStruct(&(job->jobControl->jobResources));

    // update job state
    job->jobState = FINISHED;
}

jobMonitoring_t* initJobMonitor(jobMonitoring_t *jobMonitor, jobResources_t *jobResources){

    // deallocate previous monitorization memory
    if(jobMonitor->gpuUsage)             free(jobMonitor->gpuUsage);
    if(jobMonitor->gpuTemperature)       free(jobMonitor->gpuTemperature);
    if(jobMonitor->gpuEnergyConsumption) free(jobMonitor->gpuEnergyConsumption);
    if(jobMonitor->gpuPCIeThroughput)    free(jobMonitor->gpuPCIeThroughput);
    if(jobMonitor->gpuNVLinkThroughput)    free(jobMonitor->gpuNVLinkThroughput);


    jobMonitor->steps = DECISION_JOB_MONITOR_STEPS;

    // allocate memory for monitoring new GPUs
    jobMonitor->gpuUsage = (unsigned int (*)[DECISION_JOB_MONITOR_STEPS])calloc(jobResources->nGPUs, sizeof(*jobMonitor->gpuUsage));
    jobMonitor->gpuTemperature = (unsigned int (*)[DECISION_JOB_MONITOR_STEPS])calloc(jobResources->nGPUs, sizeof(*jobMonitor->gpuTemperature));
    jobMonitor->gpuEnergyConsumption = (unsigned int (*)[DECISION_JOB_MONITOR_STEPS])calloc(jobResources->nGPUs, sizeof(*jobMonitor->gpuEnergyConsumption));
    jobMonitor->gpuPCIeThroughput = (unsigned int (*)[DECISION_JOB_MONITOR_STEPS])calloc(jobResources->nGPUs, sizeof(*jobMonitor->gpuPCIeThroughput));
    jobMonitor->gpuNVLinkThroughput = (unsigned int (*)[DECISION_JOB_MONITOR_STEPS])calloc(jobResources->nGPUs, sizeof(*jobMonitor->gpuNVLinkThroughput));


    // we are again in the first step
    jobMonitor->step = 0;

    // return job monitor
    return jobMonitor;
}

job_t* initJob(jobLauncher_t* jobLauncher){

    // allocate memory for the job structure
    job_t *job = (job_t*)calloc(1, sizeof(job_t));

    // job not launched until it is in the pending list 
    job->jobState = NOTLAUNCHED;

    // set job launcher
    job->jobLauncher = jobLauncher;

    // allocate memory for the job control structure and initialize
    jobControl_t *jobControl = (jobControl_t*)calloc(1, sizeof(jobControl_t));
    job->jobControl = jobControl; // set job control

    // set jobLauncher just parameter
    jobLauncher->argv[jobLauncher->argc+1] = (void*)(jobControl);

    // initialize locks and notification variables
    jobControl->pendingReconf = 0; // signal to indicate there is a pending reconfiguration (RMS --> job)
    pthread_mutex_init(&(jobControl->lockPendingReconf), NULL);

    jobControl->sigGPUs = 0; // TODO: I don't remember what this signal is (job --> RMS)
    pthread_mutex_init(&(jobControl->lockSigGPUs), NULL);

    jobControl->reqGPUs = 0; // signal for requesting GPUs (job --> RMS)
    pthread_mutex_init(&(jobControl->lockReqGPUs), NULL);
    pthread_cond_init(&(jobControl->condReqGPUs), NULL);

    jobControl->startRunning = 0;
    pthread_mutex_init(&(jobControl->lockStartRunning), NULL);

    jobControl->finished = 0; // signal indicating that job finished (job --> RMS)
    pthread_mutex_init(&(jobControl->lockFinished), NULL);

    // job resources are set when it is running
    jobControl->jobResources = NULL; 
    jobControl->reconfJobResources = NULL;


    // initialize job monitor
    jobMonitoring_t *jobMonitor = (jobMonitoring_t*)calloc(1,sizeof(jobMonitoring_t));
    jobMonitor->gpuUsage = NULL; // we need to know how much resources are allocated for the job
    jobMonitor->step = 0;
    job->jobMonitor = jobMonitor;


    // return job structure
    return job;
}

void* runJob(void *vJobLauncher){

    jobLauncher_t *jobLauncher = (jobLauncher_t*)(vJobLauncher);
    jobLauncher->launchFunc(jobLauncher->argc, jobLauncher->argv);
    return NULL;
}

void recordEvent(eventsEnum event, job_t *job, jobResources_t *jobResources){

    size_t gpu;
    size_t jobId = job->jobId;

    int tmpTimer;
    pthread_mutex_lock(&(schInfo->lockTimer));
    tmpTimer = timer;    
    pthread_mutex_unlock(&(schInfo->lockTimer));
    
    switch(event){
        case JOBSTARTED:
            for(gpu = 0; gpu < jobResources->nGPUs; gpu++){
                fprintf(fEventRecord, "%zu,%zu,%d,start\n", jobResources->idGPUs[gpu], jobId, tmpTimer);
            }
        break;
        case RECONFSTARTED:
            for(gpu = 0; gpu < jobResources->nGPUs; gpu++){
                fprintf(fEventRecord, "%zu,%zu,%d,start\n", jobResources->idGPUs[gpu], jobId, tmpTimer);
            }
        break;
        case RECONFFINISHED:
            for(gpu = 0; gpu < jobResources->nGPUs; gpu++){
                fprintf(fEventRecord, "%zu,%zu,%d,end\n", jobResources->idGPUs[gpu], jobId, tmpTimer);
            }
        break;
        case JOBFINISHED:
            for(gpu = 0; gpu < jobResources->nGPUs; gpu++){
                fprintf(fEventRecord, "%zu,%zu,%d,end\n", jobResources->idGPUs[gpu], jobId, tmpTimer);
            }
        break;
    }
    fflush(fEventRecord);
}

void allocateResources(schInfo_t *schInfo, jobResources_t *jobResources){

    size_t i;

    // allocate selected resources for job
    for(i = 0; i<jobResources->nGPUs; i++){

        size_t gpu = jobResources->idGPUs[i];
        schInfo->avGPUs[gpu] = 0; // set resource as not available
    }

    schInfo->nAvGPUs -= jobResources->nGPUs; // update the number of available GPUs
}

void deallocateResources(schInfo_t *schInfo, jobResources_t *jobResources){

    size_t i;
    
    for(i = 0; i<jobResources->nGPUs; i++){

        size_t gpu = jobResources->idGPUs[i];
        schInfo->avGPUs[gpu] = 1; // not available
    }

    schInfo->nAvGPUs += jobResources->nGPUs;
}

void deallocateJobResourcesStruct(jobResources_t **jobResources){

    if((*jobResources)->idGPUs)
        free((*jobResources)->idGPUs);
    free((*jobResources));
}

int checkJobFinished(jobControl_t *jControl){
    
    // Check if job activated the finished flag
    int finished = 0;
    pthread_mutex_lock(&(jControl->lockFinished));
    finished = jControl->finished;    
    pthread_mutex_unlock(&jControl->lockFinished);

    return finished;
}

void notifyReconfiguration(jobControl_t *jobControl){

    // Activate pending reconf flag
    pthread_mutex_lock(&(jobControl->lockPendingReconf));
    jobControl->pendingReconf = 1;
    pthread_mutex_unlock(&(jobControl->lockPendingReconf));
}

int checkSignalNoGPUs(jobControl_t *jobControl){

    int signal = 0;

    // lock
    pthread_mutex_lock(&(jobControl->lockSigGPUs));

    // get signal (not GPUs required)
    signal = jobControl->sigGPUs;

    // deactivate signal
    jobControl->sigGPUs = 0;

    // unlock
    pthread_mutex_unlock(&(jobControl->lockSigGPUs));

    // return signal
    return signal;
}

int checkSignalReqGPUs(jobControl_t *jobControl){

    int req = 0;

    // lock
    pthread_mutex_lock(&(jobControl->lockReqGPUs));

    // get signal value
    req = jobControl->reqGPUs;
    
    // deactivate signal (if it was active)
    jobControl->reqGPUs = 0;

    // unlock
    pthread_mutex_unlock(&(jobControl->lockReqGPUs));

    // return signal value
    return req;
}

int checkReconfigurationDone(jobControl_t *jobControl){

    int pending;
    
    // lock
    pthread_mutex_lock(&(jobControl->lockPendingReconf));

    // get signal
    pending = jobControl->pendingReconf;

    // unlock
    pthread_mutex_unlock(&(jobControl->lockPendingReconf));

    // return signal (if 0, reconfiguration is finished)
    return pending;
}



// [JOB SCOPE: called by the job]
int checkIfReconfiguration(jobControl_t *jobControl){

    int localPendingReconf;

    // check whether there is any pending reconfiguration
    pthread_mutex_lock(&(jobControl->lockPendingReconf));

    localPendingReconf = jobControl->pendingReconf;

    pthread_mutex_unlock(&jobControl->lockPendingReconf);

    return localPendingReconf;
}

void jobFinished(jobControl_t* jControl){

    // activate flag to indicate that job finished
    pthread_mutex_lock(&(jControl->lockFinished));
    jControl->finished = 1;
    pthread_mutex_unlock(&(jControl->lockFinished));
}

void notifyReconfigurationDone(jobControl_t *jobControl){

    // deactivate the flag indicating that there is a pending reconfiguration
    pthread_mutex_lock(&(jobControl->lockPendingReconf));
    jobControl->pendingReconf = 0;
    pthread_mutex_unlock(&(jobControl->lockPendingReconf));
}

void notifySigGPUs(jobControl_t *jobControl){

    // no GPUs required
    pthread_mutex_lock(&(jobControl->lockSigGPUs));
    jobControl->sigGPUs = 1; 
    pthread_mutex_unlock(&(jobControl->lockSigGPUs));
}

void notifyReqGPUs(jobControl_t *jobControl){
    
    // GPUs required
    pthread_mutex_lock(&(jobControl->lockReqGPUs));
    jobControl->reqGPUs = 1; 
    pthread_mutex_unlock(&(jobControl->lockReqGPUs));
}


// Only for experiments
void notifyStartRunning(jobControl_t *jobControl){

    // Indicate to the job that it can start
    pthread_mutex_lock(&(jobControl->lockStartRunning));
    jobControl->startRunning = 1; 
    pthread_mutex_unlock(&(jobControl->lockStartRunning));
}

// TODO
// collect resource usage information each second. 
// Utilization, energy consumption, temperature, PCIe throughput.. usage on each second
// store results 
void* resourceMonitoring(void *finished){
 
    int *finish = (int*)finished;


    fprintf(fUsage, "time_step,gpu_id,utilization_%%,power_W,temperature,tx,rx\n");
    fflush(fUsage);

    // Initialize NVML
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));
        exit(1);
    }

    unsigned int device_count;
    nvmlDeviceGetCount(&device_count);


    // [MONITOR INITIALIZATION]
    schInfo->gpuUtilization = (unsigned int (*)[JOB_MONITOR_STEPS])calloc(device_count, sizeof(*schInfo->gpuUtilization));
    schInfo->gpuPower = (double (*)[JOB_MONITOR_STEPS])calloc(device_count, sizeof(*schInfo->gpuPower));
    schInfo->gpuTemperature = (unsigned int (*)[JOB_MONITOR_STEPS])calloc(device_count, sizeof(*schInfo->gpuTemperature));
    schInfo->gpuPCIeThroughput = (unsigned int (*)[JOB_MONITOR_STEPS])calloc(device_count, sizeof(*schInfo->gpuPCIeThroughput));
    schInfo->nMonitored = 0;
    schInfo->monitorIndex = 0;
    
    // allocate memory for final monitored data
    schInfo->totalPowerConsumption = 0.0;
    schInfo->totalPowerConsumptionPerGPU = (double*)malloc(device_count * sizeof(double));
    schInfo->totalUtilization = 0;
    schInfo->totalUtilizationPerGPU = (unsigned int*)malloc(device_count * sizeof(unsigned int));
    schInfo->temperatureSum = 0;
    schInfo->temperatureSumPerGPU = (unsigned int*)malloc(device_count * sizeof(unsigned int));
    schInfo->totalThroughputPCIe = 0;
    schInfo->totalThroughputPCIePerGPU = (unsigned int*)malloc(device_count * sizeof(unsigned int));
    schInfo->totalAllocationArea = 0;
    schInfo->allocationTimePerGPU = (unsigned int*)malloc(device_count * sizeof(unsigned int));

    for(unsigned int i = 0; i < device_count; i++){

        schInfo->totalPowerConsumptionPerGPU[i] = 0.0;
        schInfo->totalUtilizationPerGPU[i] = 0;
        schInfo->temperatureSumPerGPU[i] = 0;
        schInfo->totalThroughputPCIePerGPU[i] = 0;
        schInfo->allocationTimePerGPU[i] = 0;

    }

    while((*finish) == 0){

        usleep(INTERVAL_US); // 1 seconds
        
        pthread_mutex_lock(&(schInfo->lockTimer));
        timer ++;    
        pthread_mutex_unlock(&(schInfo->lockTimer));
        
        // store info obtained from nvidia-smi
        for (unsigned int i = 0; i < device_count; i++) {

            nvmlDevice_t device;
            nvmlDeviceGetHandleByIndex(i, &device);

            // Utilization
            nvmlUtilization_t util;
            nvmlDeviceGetUtilizationRates(device, &util);

            // Power (in milliwatts)
            unsigned int power;
            nvmlDeviceGetPowerUsage(device, &power);

            // Temperature
            unsigned int temperature;
            nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature);

            // PCIe throughput // https://amirsojoodi.github.io/posts/NVML/
            unsigned int tx_throughput;
            unsigned int rx_throughput;
            nvmlDeviceGetPcieThroughput(device, NVML_PCIE_UTIL_TX_BYTES, &tx_throughput);
            nvmlDeviceGetPcieThroughput(device, NVML_PCIE_UTIL_RX_BYTES, &rx_throughput);

            //int valuesCount = 10;
            //nvmlFieldValue_t *values;
            //nvmlDeviceGetFieldValues(device, );

            // store results in file
            int tmpTimer;
            pthread_mutex_lock(&(schInfo->lockTimer));
            tmpTimer = timer;    
            pthread_mutex_unlock(&(schInfo->lockTimer));

            fprintf(fUsage, "%d,%u,%u,%.2f,%u,%u,%u\n", tmpTimer, i, util.gpu, power / 1000.0, temperature, tx_throughput, rx_throughput);


            // store utilization in sch info and update step
            schInfo->gpuUtilization[i][schInfo->monitorIndex] = util.gpu;
            schInfo->gpuPower[i][schInfo->monitorIndex] = power / 1000.0;
            schInfo->gpuTemperature[i][schInfo->monitorIndex] = temperature;
            schInfo->gpuPCIeThroughput[i][schInfo->monitorIndex] = tx_throughput + rx_throughput;

            /* Interesting data to compute:
                - Mean utilization per each GPU
                - Mean utilization of all GPUs

                - Mean utilization of each GPU when allocated
                - Mean utilization of all GPUs when allocated

                - Total energy consumtpion
                - Total bandwitdh usage
                
                - Mean temperature
            
                Standard deviation for each value too
            */
            schInfo->totalUtilization += util.gpu;
            schInfo->totalUtilizationPerGPU[i] += util.gpu;

            schInfo->totalPowerConsumption += power / 1000.0;
            schInfo->totalPowerConsumptionPerGPU[i] += power / 1000.0;
            
            schInfo->temperatureSum += temperature;
            schInfo->temperatureSumPerGPU[i] += temperature;

            schInfo->totalAllocationArea += (int)schInfo->avGPUs[i];
            schInfo->allocationTimePerGPU[i] += (int)schInfo->avGPUs[i];
        }

        fflush(fUsage);  // ensure data is written

        // update monitorization data
        schInfo->nMonitored ++;
        schInfo->monitorIndex ++;
        schInfo->monitorIndex = schInfo->monitorIndex % JOB_MONITOR_STEPS;
    }

    // shutdown NVML and close file
    nvmlShutdown();
    fclose(fUsage);

    return NULL;
}

void initResourceMonitor(schInfo_t *schInfo, size_t nBatches){

    // write first line in the file
    fprintf(fUsage, "time_step,gpu_id,utilization_%%,power_W,temperature,tx,rx\n");
    fflush(fUsage);

    // Initialize NVML
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));
        exit(1);
    }

    unsigned int device_count;
    nvmlDeviceGetCount(&device_count);

    // [MONITOR INITIALIZATION]
    // real time monitorization
    schInfo->gpuUtilization = (unsigned int (*)[JOB_MONITOR_STEPS])calloc(device_count, sizeof(*schInfo->gpuUtilization));
    schInfo->gpuPower = (double (*)[JOB_MONITOR_STEPS])calloc(device_count, sizeof(*schInfo->gpuPower));
    schInfo->gpuTemperature = (unsigned int (*)[JOB_MONITOR_STEPS])calloc(device_count, sizeof(*schInfo->gpuTemperature));
    schInfo->gpuPCIeThroughput = (unsigned int (*)[JOB_MONITOR_STEPS])calloc(device_count, sizeof(*schInfo->gpuPCIeThroughput));
    schInfo->gpuNVLinkThroughput = (unsigned int (*)[JOB_MONITOR_STEPS])calloc(device_count, sizeof(*schInfo->gpuNVLinkThroughput));
    
    schInfo->nMonitored = 0;
    schInfo->monitorIndex = 0;
    

    // accumulated monitored data
    // energy
    schInfo->totalPowerConsumption = 0.0;
    schInfo->totalPowerConsumptionPerGPU = (double*)malloc(device_count * sizeof(double));
    // utilization
    schInfo->totalUtilization = 0;
    schInfo->totalUtilizationPerGPU = (unsigned int*)malloc(device_count * sizeof(unsigned int));
    // tempretaure
    schInfo->temperatureSum = 0;
    schInfo->temperatureSumPerGPU = (unsigned int*)malloc(device_count * sizeof(unsigned int));
    // PCIe
    schInfo->totalThroughputPCIe = 0;
    schInfo->totalThroughputPCIePerGPU = (unsigned int*)malloc(device_count * sizeof(unsigned int));
    // NVLink
    schInfo->totalThroughputNVLink = 0;
    schInfo->totalThroughputNVLinkPerGPU = (unsigned int*)malloc(device_count * sizeof(unsigned int));
    // allocation
    schInfo->totalAllocationArea = 0;
    schInfo->allocationTimePerGPU = (unsigned int*)malloc(device_count * sizeof(unsigned int));
    // number of reconfigurations
    schInfo->nReconfigurations = 0;
    schInfo->nExpands = 0;
    schInfo->nShrinks = 0;
    schInfo->nKeeps = 0;

    // initialize final values
    // throughput
    schInfo->finalThroughput = (double*)calloc(nBatches, sizeof(double));
    // utilization
    schInfo->finalUtilization = (double*)calloc(nBatches, sizeof(double));
    schInfo->finalUtilizationPerGPU = (double**)calloc(device_count, sizeof(double*));
    // power
    schInfo->finalPowerConsumption = (double*)calloc(nBatches, sizeof(double));
    schInfo->finalPowerConsumptionPerGPU = (double**)calloc(device_count, sizeof(double*));
    // temperature
    schInfo->finalTemperatureSum = (double*)calloc(nBatches, sizeof(double));
    schInfo->finalTemperatureSumPerGPU = (double**)calloc(device_count, sizeof(double*));
    // PCIe
    schInfo->finalThroughputPCIe = (double*)calloc(nBatches, sizeof(double));
    schInfo->finalThroughputPCIePerGPU = (double**)calloc(device_count, sizeof(double*));
    // NVLink
    schInfo->finalThroughputNVLink = (double*)calloc(nBatches, sizeof(double));
    schInfo->finalThroughputNVLinkPerGPU = (double**)calloc(device_count, sizeof(double*));
    // allocation
    schInfo->finalAllocationArea = (double*)calloc(nBatches, sizeof(double));
    schInfo->finalAllocationTimePerGPU = (double**)calloc(device_count, sizeof(double)); 
    // number of reconfigurations
    schInfo->finalNReconfigurations = (double*)calloc(nBatches, sizeof(double));
    schInfo->finalNExpands = (double*)calloc(nBatches, sizeof(double));
    schInfo->finalNShrinks = (double*)calloc(nBatches, sizeof(double));
    schInfo->finalNKeeps = (double*)calloc(nBatches, sizeof(double));
    // job execution and wait times
    schInfo->finalMeanExecutionTime = (double*)calloc(nBatches, sizeof(double));
    schInfo->finalMeanWaitTime = (double*)calloc(nBatches, sizeof(double));

    for(unsigned int i = 0; i < device_count; i++){

        // power
        schInfo->totalPowerConsumptionPerGPU[i] = 0.0;
        // utilization
        schInfo->totalUtilizationPerGPU[i] = 0;
        // temperature
        schInfo->temperatureSumPerGPU[i] = 0;
        // PCIe
        schInfo->totalThroughputPCIePerGPU[i] = 0;
        // allocation
        schInfo->allocationTimePerGPU[i] = 0;
        // NVLink
        schInfo->totalThroughputNVLinkPerGPU[i] = 0;

        nvmlDevice_t device;
        nvmlDeviceGetHandleByIndex(i, &device);

        for (unsigned int link = 0; link < NVML_NVLINK_MAX_LINKS; link++) {

            nvmlEnableState_t state;
            nvmlReturn_t ret = nvmlDeviceGetNvLinkState(device, link, &state);

            if (ret != NVML_SUCCESS)
                continue;

            if (state == NVML_FEATURE_ENABLED)
                schInfo->nvLinkCount[i]++;
        }

        // final results
        // utilization
        schInfo->finalUtilizationPerGPU[i] = (double*)calloc(nBatches, sizeof(double));
        // power
        schInfo->finalPowerConsumptionPerGPU[i] = (double*)calloc(nBatches, sizeof(double));
        // temperature
        schInfo->finalTemperatureSumPerGPU[i] = (double*)calloc(nBatches, sizeof(double));
        // PCIe
        schInfo->finalThroughputPCIePerGPU[i] = (double*)calloc(nBatches, sizeof(double));
        // allocation
        schInfo->finalAllocationTimePerGPU[i] = (double*)calloc(nBatches, sizeof(double));
        // NVLink
        schInfo->finalThroughputNVLinkPerGPU[i] = (double*)calloc(nBatches, sizeof(double));
    }
}

void stepResourceMonitor(schInfo_t *schInfo){

    unsigned int device_count;
    nvmlDeviceGetCount(&device_count);

    // store info obtained from nvidia-smi
    for (unsigned int i = 0; i < device_count; i++) {

        nvmlDevice_t device;
        nvmlDeviceGetHandleByIndex(i, &device);

        // Utilization
        nvmlUtilization_t util;
        nvmlDeviceGetUtilizationRates(device, &util);

        // Power (in milliwatts)
        unsigned int power;
        nvmlDeviceGetPowerUsage(device, &power);

        // Temperature
        unsigned int temperature;
        nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature);

        // PCIe throughput // https://amirsojoodi.github.io/posts/NVML/
        unsigned int tx_throughput;
        unsigned int rx_throughput;
        nvmlDeviceGetPcieThroughput(device, NVML_PCIE_UTIL_TX_BYTES, &tx_throughput); // throughput estimate KB / s
        nvmlDeviceGetPcieThroughput(device, NVML_PCIE_UTIL_RX_BYTES, &rx_throughput);

        //int valuesCount = 10;
        //nvmlFieldValue_t *values;
        //nvmlDeviceGetFieldValues(device, );

        // store results in file
        if(util.gpu > 100.0) 
            util.gpu = 100.0;

        int tmpTimer;
        pthread_mutex_lock(&(schInfo->lockTimer));
        tmpTimer = timer;    
        pthread_mutex_unlock(&(schInfo->lockTimer));

        fprintf(fUsage, "%d,%u,%u,%.2f,%u,%u,%u\n", tmpTimer, i, util.gpu, power / 1000.0, temperature, tx_throughput, rx_throughput);


        // store utilization in sch info and update step
        schInfo->gpuUtilization[i][schInfo->monitorIndex] = util.gpu;
        schInfo->gpuPower[i][schInfo->monitorIndex] = power / 1000.0;
        schInfo->gpuTemperature[i][schInfo->monitorIndex] = temperature;
        schInfo->gpuPCIeThroughput[i][schInfo->monitorIndex] = tx_throughput + rx_throughput; // throughput in KB/s

        // monitor NVLink communication data
        schInfo->gpuNVLinkThroughput[i][schInfo->monitorIndex] = 0;
        for (unsigned int link = 0; link < schInfo->nvLinkCount[i]; link++) {
        
            nvmlEnableState_t isActive;
            
            if(nvmlDeviceGetNvLinkState(device, link, &isActive) == NVML_SUCCESS && isActive == NVML_FEATURE_ENABLED){
                
                // Try Field Values API first
                nvmlFieldValue_t fieldValues[2];
                fieldValues[0].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX;
                fieldValues[0].scopeId = link;
                fieldValues[1].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX;
                fieldValues[1].scopeId = link;

                nvmlReturn_t fieldResult = nvmlDeviceGetFieldValues(device, 2, fieldValues);
                unsigned long long rxCounter = 0, txCounter = 0;

                if (fieldResult == NVML_SUCCESS) {
                    
                    // Extract TX counter
                    if (fieldValues[0].nvmlReturn == NVML_SUCCESS && fieldValues[0].valueType == NVML_VALUE_TYPE_UNSIGNED_LONG_LONG) {
                        txCounter = fieldValues[0].value.ullVal;
                    }

                    // Extract RX counter
                    if (fieldValues[1].nvmlReturn == NVML_SUCCESS && fieldValues[1].valueType == NVML_VALUE_TYPE_UNSIGNED_LONG_LONG) {
                        rxCounter = fieldValues[1].value.ullVal;
                    }
                } 
                else {
                    
                    // Fallback to traditional API
                    nvmlDeviceGetNvLinkUtilizationCounter(device, link, 0, &rxCounter, &txCounter);
                }

                // NVLink: update total data
                schInfo->gpuNVLinkThroughput[i][schInfo->monitorIndex] += rxCounter + txCounter;
            }
        }

        if(schInfo->nvLinkCount[i]>0){
            
            // compute throughput
            size_t index1 = schInfo->monitorIndex;
            size_t index2 = schInfo->monitorIndex > 0 ? schInfo->monitorIndex - 1 : JOB_MONITOR_STEPS - 1; 

            schInfo->gpuNVLinkThroughput[i][index1] = schInfo->gpuNVLinkThroughput[i][index1] - schInfo->gpuNVLinkThroughput[i][index2]; // KB/s
        }
           

        // store in accumulated values
        schInfo->totalUtilization += util.gpu;
        schInfo->totalUtilizationPerGPU[i] += util.gpu;

        schInfo->totalPowerConsumption += power / 1000.0;
        schInfo->totalPowerConsumptionPerGPU[i] += power / 1000.0;
        
        schInfo->temperatureSum += temperature;
        schInfo->temperatureSumPerGPU[i] += temperature;

        schInfo->totalAllocationArea += 1 - (unsigned int)schInfo->avGPUs[i];
        schInfo->allocationTimePerGPU[i] += 1 - (unsigned int)schInfo->avGPUs[i];

        schInfo->totalThroughputPCIe += tx_throughput + rx_throughput; // accumulate throughput
        schInfo->totalThroughputPCIePerGPU[i] += tx_throughput + rx_throughput;

        schInfo->totalThroughputNVLink += schInfo->gpuNVLinkThroughput[i][schInfo->monitorIndex];
        schInfo->totalThroughputNVLinkPerGPU[i] += schInfo->gpuNVLinkThroughput[i][schInfo->monitorIndex];
    }

    fflush(fUsage);  // ensure data is written

    // update monitorization data
    schInfo->nMonitored ++;
    schInfo->monitorIndex ++;
    schInfo->monitorIndex = schInfo->monitorIndex % JOB_MONITOR_STEPS;
}

void reinitMonitorAcc(schInfo_t *schInfo){

    // reinit
    schInfo->totalUtilization = 0;
    schInfo->totalPowerConsumption = 0.0;
    schInfo->temperatureSum = 0;
    schInfo->totalAllocationArea = 0;
    schInfo->totalThroughputPCIe = 0;
    schInfo->totalThroughputNVLink = 0;
    schInfo->nReconfigurations = 0;
    schInfo->nExpands = 0;
    schInfo->nShrinks = 0;
    schInfo->nKeeps = 0;

    for(size_t i = 0; i<schInfo->nGPUs; i++){
       
        schInfo->totalUtilizationPerGPU[i] = 0;
        schInfo->totalPowerConsumptionPerGPU[i] = 0.0;
        schInfo->temperatureSumPerGPU[i] = 0;
        schInfo->allocationTimePerGPU[i] = 0;
        schInfo->totalThroughputPCIePerGPU[i] = 0;
        schInfo->totalThroughputNVLinkPerGPU[i] = 0;
    }

    /*for(size_t i = 0; i<schInfo->nGPUs; i++){

        for(size_t m = 0; m<JOB_MONITOR_STEPS; m++){

            size_t monitorIndex = (schInfo->monitorIndex + m) % JOB_MONITOR_STEPS;

            schInfo->totalUtilization += schInfo->gpuUtilization[i][monitorIndex];
            schInfo->totalPowerConsumption += schInfo->gpuPower[i][monitorIndex];
            schInfo->temperatureSum += schInfo->gpuTemperature[i][monitorIndex];
            schInfo->totalAllocationArea += 1 - (unsigned int)schInfo->avGPUs[i];
            schInfo->bandPCIe += schInfo->gpuPCIeThroughput[i][monitorIndex];
            schInfo->bandNVLink += schInfo->gpuNVLinkThroughput[i][monitorIndex];

            schInfo->totalUtilizationPerGPU[i] += schInfo->gpuUtilization[i][monitorIndex];
            schInfo->totalPowerConsumptionPerGPU[i] += schInfo->gpuPower[i][monitorIndex];
            schInfo->temperatureSumPerGPU[i] += schInfo->gpuTemperature[i][monitorIndex];
            schInfo->allocationTimePerGPU[i] += 1 - (unsigned int)schInfo->avGPUs[i];
            schInfo->bandPCIePerGPU[i] += schInfo->gpuPCIeThroughput[i][monitorIndex];
            schInfo->bandNVLinkPerGPU[i] += schInfo->gpuNVLinkThroughput[i][monitorIndex];
        }
    }*/
}

void destroyResourceMonitor(schInfo_t *schInfo){

    // shutdown NVML and close file
    nvmlShutdown();
    fclose(fUsage);
}


/// [Thread] Function for simulating jobs launching from jobs timeline
void* jobManager(void *voidJobsTimeline){

    jobsTimeline_t *jobsTimeline = (jobsTimeline_t*)voidJobsTimeline;

    // get fist job launcher
    jobLauncher_t *jobLauncher;

    // launch jobs
    size_t job = 0;
    while(job < (size_t)(jobsTimeline->nJobs)){

        // get job launcher
        jobLauncher = &(jobsTimeline->jobLaunchers[job]);

        int tmpTimer;
        pthread_mutex_lock(&(schInfo->lockTimer));
        tmpTimer = timer;    
        pthread_mutex_unlock(&(schInfo->lockTimer));


        //pthread_mutex_lock(&(printLock));
        //printf(" [JOB MANAGER]: %d\n", tmpTimer);
        //fflush(stdout);
        //pthread_mutex_unlock(&(printLock));


        while((size_t)tmpTimer <= (size_t)jobLauncher->launchTimeStep){

            usleep(INTERVAL_US); // 0.5 second?

            pthread_mutex_lock(&(schInfo->lockTimer));
            tmpTimer = timer;    
            pthread_mutex_unlock(&(schInfo->lockTimer));
        }

        // add job to the pending queue
        addPendingJob(jobLauncher);

        // indicate to invoque the scheduler        
        pthread_mutex_lock(&(schInfo->invoqueSchedulerLock));
        schInfo->invoqueScheduler = 1;    
        pthread_mutex_unlock(&(schInfo->invoqueSchedulerLock));

        pthread_mutex_lock(&(printLock));
        printf(" -- [RMS] Job added to pending queue at time step %d\n",tmpTimer);
        fflush(stdout);
        pthread_mutex_unlock(&(printLock));

        
        job ++;
    }

    return NULL;
}



void selectFirstAvailableGPUs(size_t *selectedGPUs, size_t nLaunchGPUs, schInfo_t *schInfo){

    size_t gpuId = 0;
    size_t jobGPU = 0;
    
    // loop over necessary number of GPUs, and select available ones in order
    while(jobGPU < nLaunchGPUs){

        // allocate the gpu for the job
        if(schInfo->avGPUs[gpuId] == 1){
            
            // set the GPU id
            selectedGPUs[jobGPU] = gpuId; 

            // update index
            jobGPU ++;
        }

        gpuId ++;
    }
}

// Find which GPUs in jobResourcesBase are not in jobResourcesComp
jobResources_t* findDiffResources(jobResources_t *jobResourcesBase, jobResources_t *jobResourcesComp){

    jobResources_t *diffJobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));

    size_t nGPUs = jobResourcesBase->nGPUs;

    diffJobResources->idGPUs = (size_t*)calloc(nGPUs, sizeof(size_t));
    diffJobResources->nGPUs = 0; // we don't know how much GPUs will be deallocated
    
    // find the GPUs that appear in jobResourcesBase and no in jobResourcesComp
    for(size_t i = 0; i<jobResourcesBase->nGPUs; i++){
        
        // loop over reconf GPUs and check if GPU appears
        int found = 0;
        for(size_t j = 0; j<jobResourcesComp->nGPUs; j++){

            // check if appears
            if(jobResourcesBase->idGPUs[i] == jobResourcesComp->idGPUs[j]) 
                found = 1;
        }

        // if not found, store GPU to be deallocated
        if(!found){

            diffJobResources->idGPUs[diffJobResources->nGPUs] = jobResourcesBase->idGPUs[i];
            diffJobResources->nGPUs ++; // add one gpu to be deallocated
        }
    }

    return diffJobResources;
}

// TODO: revise
void manageReconfigurations(schInfo_t *schInfo){

    // helper variables
    job_t *job;

    // get info in the queue of jobs being reconfigured
    jobQueue_t *reconfiguringQueue = &(schInfo->reconfiguringJobs);
    size_t nReconfiguringJobs = getNumberOfJobsInQueue(reconfiguringQueue); 
    size_t iJob;
    int invoque = 0;
    
    // loop over jobs that are being reconfigured
    for(iJob = 0; iJob<nReconfiguringJobs; iJob ++){

        // get job
        job = getJobFromQueue(reconfiguringQueue, iJob);

        // check whether job finished the reconfiguration
        int reconfigurationDone = checkReconfigurationDone(job->jobControl);
        if(reconfigurationDone == 0){ // reconfiguration finished
                        
            // finish job reconfiguration (leave reconfiguration queue)
            jobFinishedReconfiguration(schInfo, job, iJob);

            // update values for the loop indexing
            nReconfiguringJobs --;
            iJob --;
            invoque = 1;
        }
    }

    // indicate to invoque the scheduler
    if(invoque){
    
        pthread_mutex_lock(&(schInfo->invoqueSchedulerLock));
        schInfo->invoqueScheduler = 1;    
        pthread_mutex_unlock(&(schInfo->invoqueSchedulerLock));
    }
}

// TODO: revise
void manageJobsFinish(schInfo_t *schInfo){

    // helper variables
    job_t *job;
    size_t iJob, nRunningJobs;
    int invoque = 0;

    // get queue of running jobs
    jobQueue_t *runningQueue = &(schInfo->runningJobs);


    // manage running jobs: whether it finished or it needs a reconfiguration
    nRunningJobs = getNumberOfJobsInQueue(runningQueue);
    for(iJob = 0; iJob<nRunningJobs; iJob++){

        // get job information
        job = getJobFromQueue(runningQueue, iJob);

        // JOB FINISH MANAGEMENT
        // check whether the job finished
        int jobFinished = checkJobFinished(job->jobControl);

        // check if job finished
        if(jobFinished){
        
            // finish job
            finishJob(schInfo, job, iJob);

            // update values
            nRunningJobs --;
            iJob--;
            invoque = 1;
        }
    }

    // indicate to invoque the scheduler
    if(invoque){
    
        pthread_mutex_lock(&(schInfo->invoqueSchedulerLock));
        schInfo->invoqueScheduler = 1;    
        pthread_mutex_unlock(&(schInfo->invoqueSchedulerLock));
    }
}