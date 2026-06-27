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
    void** argv; // arguments variable to initialize job launchers
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
        

        // load argv parameters: application related parameters
        for(int arg = 0; arg<jobLauncher->argc; arg++){
            
            // if not pointer will disappear after function ends (would be stored in the stack)
            size_t *val = (size_t*)malloc(sizeof(size_t));
            fscanf(jobF, "%zu", val);
            jobLauncher->argv[arg] = (void*)val;
        }
        
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
            
            jobLauncher->launchFunc = &launch_communications_app;
        }
        fflush(stdout);
    }

    return jobsTimeline;
}


// TODO
void initSystem(){

    printf(" -- Not implemented yet\n");
    fflush(stdout);
    exit(1); 
}

void initializeTopology(schInfo_t *schInfo){

    // get the number of GPUs
    size_t N = schInfo->nGPUs;

    // matrix of N * N
    int *gpuTopology = (int*)calloc(N * N, sizeof(int));
    int *gpuTopologyRank = (int*)calloc(N * N, sizeof(int));
    int rank;

    // initialize NVML init for obtaining topology information
    // level for storing the connection level between two GPUs
    nvmlGpuTopologyLevel_t level;
    nvmlReturn_t ret = nvmlInit();
    
    if (ret != NVML_SUCCESS) {
        
        fprintf(stderr, "NVML init failed: %s\n", nvmlErrorString(ret));
        fflush(stdout);
        exit(1);
    }

    // loop over GPUs get the topology information P2P
    for(size_t i = 0; i<N; i++){

        for(size_t j = 0; j<N; j++){

            if(i!=j){

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

            }
        }
    }

    // store the topology matrix
    schInfo->gpuTopology = gpuTopology;
    schInfo->gpuTopologyRank = gpuTopologyRank;

    // shut down NVML
    nvmlShutdown();

    // print topology information
    printf(" -- Printing topology:\n");
    for(size_t i = 0; i<N; i++){
        for(size_t j = 0; j<N; j++){

            printf("%d (%d)  ", gpuTopology[i*N+j], gpuTopologyRank[i*N + j]);
        }
        printf("\n");
    }
    printf("\n");

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

void launchJob(job_t *job, size_t pendingIndex, jobResources_t *jobResources){

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
    recordEvent(JOBSTARTED, job);

    // add job to running queue
    addJobToQueue(&(schInfo->runningJobs), job);
}

void scheduleReconfiguration(job_t *job, size_t jobIndex, jobResources_t *jobResources){

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
    jobControl->reconfJobResources = jobResources;

    // notify reconfiguration to the job
    notifyReconfiguration(jobControl);
}

void jobFinishedReconfiguration(job_t *job, size_t jobIndex){

    // get job control
    jobControl_t *jobControl = job->jobControl;    

    // update job state
    job->jobState = RUNNING;

    // remove job from reconfiguring queue
    removeJobFromQueueByIndex(&(schInfo->reconfiguringJobs), jobIndex);
  
    // update job monitor to the new resources
    job->jobMonitor = initJobMonitor(job->jobMonitor, jobControl->reconfJobResources);

    // record event
    recordEvent(JOBRECONFIGURED, job);

    // update job resources
    // 1. deallocate old resources, reconfiguration is done, they are not relevant anymore
    free(jobControl->jobResources->idGPUs);
    free(jobControl->jobResources);

    // reconf job resources are now current job resources
    jobControl->jobResources = jobControl->reconfJobResources;
    jobControl->reconfJobResources = NULL;
}

void finishJob(job_t *job, size_t runningJobIndex){

    // update queues
    removeJobFromQueueByIndex(&(schInfo->runningJobs), runningJobIndex); // remove from running jobs queue
    addJobToQueue(&(schInfo->finishedJobs), job); // add to finished jobs queue

    // kill job thread
    pthread_join(job->jobThread, NULL);

    // manage timers
    clock_gettime(CLOCK_MONOTONIC, &(job->jobEndRunning));

    // record event
    recordEvent(JOBFINISHED, job);

    // update job state
    job->jobState = FINISHED;

    // TODO: deallocate job information that is not necessary anymore?
}

jobMonitoring_t* initJobMonitor(jobMonitoring_t *jobMonitor, jobResources_t *jobResources){

    // deallocate previous monitorization memory
    if(jobMonitor->gpuUsage)             free(jobMonitor->gpuUsage);
    if(jobMonitor->gpuTemperature)       free(jobMonitor->gpuTemperature);
    if(jobMonitor->gpuEnergyConsumption) free(jobMonitor->gpuEnergyConsumption);
    if(jobMonitor->gpuPCIeThroughput)    free(jobMonitor->gpuPCIeThroughput);


    jobMonitor->steps = JOB_MONITOR_STEPS;

    // allocate memory for monitoring new GPUs
    jobMonitor->gpuUsage = (unsigned int (*)[100])calloc(jobResources->nGPUs, sizeof(*jobMonitor->gpuUsage));
    jobMonitor->gpuTemperature = (unsigned int (*)[100])calloc(jobResources->nGPUs, sizeof(*jobMonitor->gpuTemperature));
    jobMonitor->gpuEnergyConsumption = (unsigned int (*)[100])calloc(jobResources->nGPUs, sizeof(*jobMonitor->gpuEnergyConsumption));
    jobMonitor->gpuPCIeThroughput = (unsigned int (*)[100])calloc(jobResources->nGPUs, sizeof(*jobMonitor->gpuPCIeThroughput));
    

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

void recordEvent(eventsEnum event, job_t *job){

    size_t gpu;
    size_t jobId = job->jobId;

    jobControl_t *jobControl = job->jobControl;

    switch(event){
        case JOBSTARTED:
            for(gpu = 0; gpu<jobControl->jobResources->nGPUs; gpu++){
                fprintf(fEventRecord, "%zu,%zu,%d,start\n", jobControl->jobResources->idGPUs[gpu], jobId, timer);
            }
        break;
        case JOBRECONFIGURED:
            for(gpu = 0; gpu<jobControl->reconfJobResources->nGPUs; gpu++){
                fprintf(fEventRecord, "%zu,%zu,%d,start\n", jobControl->reconfJobResources->idGPUs[gpu], jobId, timer);
            }
            for(gpu = 0; gpu<jobControl->jobResources->nGPUs; gpu++){
                fprintf(fEventRecord, "%zu,%zu,%d,end\n", jobControl->jobResources->idGPUs[gpu], jobId, timer);
            }
        break;
        case JOBFINISHED:
            for(gpu = 0; gpu<jobControl->jobResources->nGPUs; gpu++){
                fprintf(fEventRecord, "%zu,%zu,%d,end\n", jobControl->jobResources->idGPUs[gpu], jobId, timer);
            }
        break;
    }
    fflush(fEventRecord);
}

void allocateResources(schInfo_t *schInfo, jobResources_t *jobResources){

    size_t i;

    // current implementation allocates resources in order
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

    /*FILE *fUsage = fopen("gpu_log.csv", "w");
    if (!fUsage) {
        perror("fopen");
        exit(1);
    }*/

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
    schInfo->gpuUtilization = (unsigned int (*)[MONITOR_STEPS])calloc(device_count, sizeof(*schInfo->gpuUtilization));
    schInfo->gpuPower = (double (*)[MONITOR_STEPS])calloc(device_count, sizeof(*schInfo->gpuPower));
    schInfo->gpuTemperature = (unsigned int (*)[MONITOR_STEPS])calloc(device_count, sizeof(*schInfo->gpuTemperature));
    schInfo->gpuPCIeThroughput = (unsigned int (*)[MONITOR_STEPS])calloc(device_count, sizeof(*schInfo->gpuPCIeThroughput));


    while((*finish) == 0){

        usleep(INTERVAL_US); // 1 seconds
        timer += 1; // update timer

        size_t tmpUsageGPUs = 0;
        int workingGPUs = 0;

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
            fprintf(fUsage, "%d,%u,%u,%.2f,%u,%u,%u\n", timer, i, util.gpu, power / 1000.0, temperature, tx_throughput, rx_throughput);


            // store utilization in sch info and update step
            schInfo->gpuUtilization[i][0] = util.gpu;
            schInfo->gpuPower[i][0] = power / 1000.0;
            schInfo->gpuTemperature[i][0] = temperature;
            schInfo->gpuPCIeThroughput[i][0] = tx_throughput + rx_throughput;

            /*// if gpu is being utilized, compute current mean usage
            if(util.gpu > 0){
            
                tmpUsageGPUs += util.gpu; 
                workingGPUs ++;
            }


            // energy consumption
            usagePower += schInfo->gpuPower[i][0];*/
        }

        // update mean value
        /*if(workingGPUs > 0){
            tmpUsageGPUs /= workingGPUs;
            usageGPUs += tmpUsageGPUs;
            registeredUsages ++;
        }*/

        fflush(fUsage);  // ensure data is written
    }

    // 
    nvmlShutdown();
    fclose(fUsage);

    return NULL;
}

/// [Thread] Function for simulating jobs launching from jobs timeline
void* jobManager(void *voidJobsTimeline){

    jobsTimeline_t *jobsTimeline = (jobsTimeline_t*)voidJobsTimeline;

    // get fist job launcher
    jobLauncher_t *jobLauncher;

    // launch jobs
    size_t job = 0;
    while(job < jobsTimeline->nJobs){

        // get job launcher
        jobLauncher = &(jobsTimeline->jobLaunchers[job]);

        while(timer <= jobLauncher->launchTimeStep){

            usleep(INTERVAL_US); // 0.5 second?
        }

        // add job to the pending queue
        addPendingJob(jobLauncher);
        if(invoqueScheduler == 0)
            invoqueScheduler = 1; // lock?

        printf(" -- [RMS] Job added to pending queue at time step %d\n",timer);

        job ++;
    }

    return NULL;
}



void selectFirstAvailableGPUs(size_t *selectedGPUs, size_t nLaunchGPUs, schInfo_t *schInfo){

    size_t gpuId = 0;
    size_t jobGPU = 0;
    
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

// TODO: revise
void manageReconfigurations(schInfo_t *schInfo){

    // helper variables
    job_t *job;
    jobControl_t *jobControl;

    // get info in the queue of jobs being reconfigured
    jobQueue_t *reconfiguringQueue = &(schInfo->reconfiguringJobs);
    size_t nReconfiguringJobs = getNumberOfJobsInQueue(reconfiguringQueue); 

    size_t iJob;

    // loop over jobs that are being reconfigured
    for(iJob = 0; iJob<nReconfiguringJobs; iJob ++){

        job = getJobFromQueue(reconfiguringQueue, iJob);

        // check whether job finished the reconfiguration
        int reconfigurationDone = checkReconfigurationDone(job->jobControl);
        if(reconfigurationDone == 0){ // reconfiguration finished

            // get job information
            jobControl = job->jobControl;

            // get previous and current job resources
            jobResources_t *reconfJobResources, *jobResources;
            reconfJobResources = jobControl->reconfJobResources;
            jobResources = jobControl->jobResources;

            // deallocate job old resources
            jobResources_t *oldJobResources = (jobResources_t*)calloc(1,sizeof(jobResources_t));
            
            // as maximum, the number of GPUs to be deallocated is the previously allocated number of GPUs
            oldJobResources->idGPUs = (size_t*)malloc(jobResources->nGPUs * sizeof(size_t));
            oldJobResources->nGPUs = 0; // we don't know how much GPUs will be deallocated
            
            // find the GPUs to be deallocaated (jobs which are not part of the new resources)
            // TODO (I need to test this) // TESTED?
            int next = 0; // index to store the next GPU
            for(size_t i = 0; i<jobResources->nGPUs; i++){
                
                // for each GPU in the previous allocated ones, check whether it is in the new ones
                int founded = 0;
                for(size_t j = 0; j<reconfJobResources->nGPUs; j++){

                    if(jobResources->idGPUs[i] == reconfJobResources->idGPUs[j]) founded = 1;
                }

                // if not founded, store GPU to be deallocated
                if(!founded){

                    oldJobResources->idGPUs[next] = jobResources->idGPUs[i];
                    oldJobResources->nGPUs ++; // add one gpu to be deallocated
                    next++;
                }
            }

            // deallocate resources
            deallocateResources(schInfo, oldJobResources);

            // deallocate structure of oldJobResources
            deallocateJobResourcesStruct(&oldJobResources);

            // finish job reconfiguration (leave reconfiguration queue)
            jobFinishedReconfiguration(job, iJob);

            // update values for the loop indexing
            nReconfiguringJobs --;
            iJob --;
        }
    }
}

// TODO: revise
size_t manageJobsFinish(schInfo_t *schInfo){

    // helper variables
    job_t *job;
    jobLauncher_t *jobLauncher;
    jobResources_t *jobResources;
    jobMonitoring_t *jobMonitor;
    size_t iJob, nRunningJobs, nJobsFinished = 0;

    // get queue of running jobs
    jobQueue_t *runningQueue = &(schInfo->runningJobs);


    // manage running jobs: whether it finished or it needs a reconfiguration
    nRunningJobs = getNumberOfJobsInQueue(runningQueue);
    for(iJob = 0; iJob<nRunningJobs; iJob++){

        // get job information
        job = getJobFromQueue(runningQueue, iJob);
        jobLauncher = job->jobLauncher;
        jobResources = job->jobControl->jobResources;


        // JOB FINISH MANAGEMENT
        // check whether the job finished
        int jobFinished = checkJobFinished(job->jobControl);

        // check if job finished
        if(jobFinished){
        
            // finish job
            finishJob(job, iJob);

            // deallocate resources
            deallocateResources(schInfo, jobResources);

            // TODO
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

            // update values
            nJobsFinished ++;
            nRunningJobs --;
            iJob--;
        }
    }

    // return the number of jobs that finished
    return nJobsFinished;
}
