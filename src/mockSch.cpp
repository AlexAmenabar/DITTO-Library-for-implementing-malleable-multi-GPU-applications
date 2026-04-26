#include <cstddef>
#include <omp.h>
#include <unistd.h>
#include <stdio.h>      
#include <stdlib.h>     
#include <pthread.h>
#include <time.h>

#include "mockSch.hpp"
#include "jobQueue.hpp"


#ifndef TESTRMS

// include CUDA related libraries
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvml.h>

// include apps that use CUDA
#include "../testApps/toy_app_malleable.hpp"

#endif


#define INTERVAL_US 1000000  // 0.1 seconds


// [TODO]: in the future they should be private?
schInfo_t *schInfo;
int timer = 0;
FILE *fEventRecord = NULL, *fUsage = NULL, *fOutput = NULL;
size_t nextJobId = 1;

size_t usageGPUs = 0;
double usagePower = 0.0;
size_t registeredUsages = 0;


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

        // if job is moldable, malleable or flexible, load also the minimum number of GPUs
        if(jobType >= 1){
        
            fscanf(jobF, "%zu", &(jobLauncher->nReqMinGPUs));
        }

        // load the time step in which the user launches the job to the system
        fscanf(jobF, "%zu", &(jobLauncher->launchTimeStep));

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

        // print job information
        /*printf(" -- Printing job %zu information\n", job);
        printf(" ---- Job type %zu\n", jobType);
        printf(" ---- Required gpus %zu\n", jobLauncher->nReqGPUs);
        printf(" ---- Required min. gpus %zu\n", jobLauncher->nReqMinGPUs);
        printf(" ---- Job launch time step %zu\n", jobLauncher->launchTimeStep);
        printf(" ---- Job application type %zu\n", jobLauncher->appType);*/
        fflush(stdout);
    }

    return jobsTimeline;
}

void initSystem(){
    printf(" -- Not implemented yet\n");
    fflush(stdout);
    exit(-1); 
}

void schedulingPolicy(){
    printf(" -- Not implemented yet\n");
    fflush(stdout);
    exit(-1); 
}

void addPendingJob(jobLauncher_t *jobLauncher){

    // initialize job structure
    job_t *job = initJob(jobLauncher);
    
    // change job state
    job->jobState = PENDING;
    job->jobId = nextJobId;
    nextJobId ++;

    // start pending timer
    clock_gettime(CLOCK_MONOTONIC, &(job->jobStartPending));

    // add job to pending queue
    addJobToQueue(&(schInfo->pendingJobs), job);
}

void launchJob(job_t *job, size_t pendingIndex, jobResources_t *jobResources){

    // update job state
    job->jobState = RUNNING;

    // remove job from pending queue
    removeJobFromQueueByIndex(&(schInfo->pendingJobs), pendingIndex);
  
    // complete job information
    job->jobControl->jobResources = jobResources; // set job resources
    job->jobControl->jobId = job->jobId;

    // initialize job monitor (already allocated)
    job->jobMonitor = initJobMonitor(job->jobMonitor, jobResources);

    // wait (simulate launch time) (TODO: should be improved?)
    int sleepTime = (int)rand() % 5;
    sleep(sleepTime);

    // launch job (thread)
    pthread_create(&(job->jobThread), NULL, runJob, (void*)(job->jobLauncher));

    // manage timers
    clock_gettime(CLOCK_MONOTONIC, &(job->jobEndPending));
    clock_gettime(CLOCK_MONOTONIC, &(job->jobStartRunning));

    // record event
    recordEvent(JOBSTARTED, job);

    // add job to running queue
    addJobToQueue(&(schInfo->runningJobs), job);
}

void scheduleReconfiguration(job_t *job, size_t jobIndex, jobResources_t *jobResources){

    // get job control
    jobControl_t *jobControl = job->jobControl;    

    // update job state
    job->jobState = RECONFIGURING;

    
    // wait (simulate allocation time) (TODO: should be improved?)
    int sleepTime = (int)rand() % 5;
    sleep(sleepTime);

    // add job to reconfiguring queue
    addJobToQueue(&(schInfo->reconfiguringJobs), job);


    // set new job resources
    if(jobControl->prevJobResources != NULL){
        
        // deallocate old job resources
        free(jobControl->prevJobResources->idGPUs);
        free(jobControl->prevJobResources);
    }

    // for events recording
    jobControl->prevJobResources = jobControl->jobResources;

    // set new job resources
    jobControl->jobResources = jobResources;

    // notify reconfiguration to job
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
    job->jobMonitor = initJobMonitor(job->jobMonitor, jobControl->jobResources);

    // record event
    recordEvent(JOBRECONFIGURED, job);
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
}

jobMonitoring_t* initJobMonitor(jobMonitoring_t *jobMonitor, jobResources_t *jobResources){

    if(jobMonitor->gpuUsage) free(jobMonitor->gpuUsage);

    jobMonitor->gpuUsage = (double*)calloc(jobResources->nGPUs, sizeof(double));
    jobMonitor->step = 0;

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

    jobControl->finished = 0; // signal indicating that job finished (job --> RMS)
    pthread_mutex_init(&(jobControl->lockFinished), NULL);

    // job resources are set when it is running
    jobControl->jobResources = NULL; 
    jobControl->prevJobResources = NULL;


    // initialize job monitor
    jobMonitoring_t *jobMonitor = (jobMonitoring_t*)calloc(1,sizeof(jobMonitoring_t));
    jobMonitor->gpuUsage = NULL; // we need to know how much resources are allocated for the job
    jobMonitor->step = 0;
    job->jobMonitor = jobMonitor;

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
            for(gpu = 0; gpu<jobControl->jobResources->nGPUs; gpu++){
                fprintf(fEventRecord, "%zu,%zu,%d,start\n", jobControl->jobResources->idGPUs[gpu], jobId, timer);
            }
            for(gpu = 0; gpu<jobControl->prevJobResources->nGPUs; gpu++){
                fprintf(fEventRecord, "%zu,%zu,%d,end\n", jobControl->jobResources->idGPUs[gpu], jobId, timer);
            }
        break;
        case JOBFINISHED:
            for(gpu = 0; gpu<jobControl->jobResources->nGPUs; gpu++){
                fprintf(fEventRecord, "%zu,%zu,%d,end\n", jobControl->jobResources->idGPUs[gpu], jobId, timer);
            }
        break;
    }
}


// allocate GPUs
void allocateResources(schInfo_t *schInfo, jobResources_t *jobResources){

    size_t i;
    
    for(i = 0; i<jobResources->nGPUs; i++){

        size_t gpu = jobResources->idGPUs[i];
        schInfo->avGPUs[gpu] = 0; // not available
    }

    schInfo->nAvGPUs -= jobResources->nGPUs;
}

// deallocate GPUs
void deallocateResources(schInfo_t *schInfo, jobResources_t *jobResources){

    size_t i;
    
    for(i = 0; i<jobResources->nGPUs; i++){

        size_t gpu = jobResources->idGPUs[i];
        schInfo->avGPUs[gpu] = 1; // not available
    }

    schInfo->nAvGPUs += jobResources->nGPUs;
}


/* [NOTIFICATIONS / SIGNALS] */

// [RMS SCOPE: called by the RMS]

char checkJobFinished(jobControl_t *jControl){
    
    // check whether there is any pending reconfiguration
    char finished = 0;
    pthread_mutex_lock(&(jControl->lockFinished));
    finished = jControl->finished;    
    pthread_mutex_unlock(&jControl->lockFinished);

    return finished;
}

void notifyReconfiguration(jobControl_t *jobControl){

    size_t i;

    // lock
    pthread_mutex_lock(&(jobControl->lockPendingReconf));

    // new pending reconfiguration
    jobControl->pendingReconf = 1;
    
    // unlock
    pthread_mutex_unlock(&(jobControl->lockPendingReconf));
}

char checkSignalNoGPUs(jobControl_t *jobControl){

    char signal = 0;

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

char checkSignalReqGPUs(jobControl_t *jobControl){

    char req = 0;

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

char checkReconfigurationDone(jobControl_t *jobControl){

    char pending;
    
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

void jobFinished(jobControl_t* jControl){

    // check whether there is any pending reconfiguration
    pthread_mutex_lock(&(jControl->lockFinished));
    jControl->finished = 1;
    pthread_mutex_unlock(&(jControl->lockFinished));
}

char checkIfReconfiguration(jobControl_t *jobControl){

    char localPendingReconf;

    // check whether there is any pending reconfiguration
    pthread_mutex_lock(&(jobControl->lockPendingReconf));
    localPendingReconf = jobControl->pendingReconf;

    if(localPendingReconf == 1){
        jobControl->pendingReconf = 2; // performing reconfiguration
    }

    pthread_mutex_unlock(&jobControl->lockPendingReconf);

    return localPendingReconf;
}

void notifyReconfigurationDone(jobControl_t *jobControl){

    pthread_mutex_lock(&(jobControl->lockPendingReconf));
    // reconfiguration done
    jobControl->pendingReconf = 0;
    pthread_mutex_unlock(&(jobControl->lockPendingReconf));
}

void notifySigGPUs(jobControl_t *jobControl){

    pthread_mutex_lock(&(jobControl->lockSigGPUs));
    // no GPUs required
    jobControl->sigGPUs = 1; 
    pthread_mutex_unlock(&(jobControl->lockSigGPUs));
}

void notifyReqGPUs(jobControl_t *jobControl){
    
    pthread_mutex_lock(&(jobControl->lockReqGPUs));
    // GPUs required
    jobControl->reqGPUs = 1; 
    pthread_mutex_unlock(&(jobControl->lockReqGPUs));
}


void* resourceMonitoring(void *finished){
 
    int *finish = (int*)finished;

    /*FILE *fUsage = fopen("gpu_log.csv", "w");
    if (!fUsage) {
        perror("fopen");
        exit(1);
    }*/

    fprintf(fUsage, "time_step,gpu_id,utilization_%%,power_W\n");
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
    int next_monitor_step = 0;
    schInfo->gpuUtilization = (unsigned int (*)[MONITOR_STEPS])calloc(device_count, sizeof(*schInfo->gpuUtilization));
    schInfo->gpuPower = (double (*)[MONITOR_STEPS])calloc(device_count, sizeof(*schInfo->gpuPower));


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

            fprintf(fUsage, "%d,%u,%u,%.2f\n", timer, i, util.gpu, power / 1000.0);


            // store utilization in sch info and update step
            schInfo->gpuUtilization[i][0] = util.gpu;
            schInfo->gpuPower[i][0] = power / 1000.0;

            // if gpu is being utilized, compute current mean usage
            if(util.gpu > 0){
            
                tmpUsageGPUs += util.gpu; 
                workingGPUs ++;
            }


            // energy consumption
            usagePower += schInfo->gpuPower[i][0];
        }

        // update mean value
        if(workingGPUs > 0){
            tmpUsageGPUs /= workingGPUs;
            usageGPUs += tmpUsageGPUs;
            registeredUsages ++;
        }

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
            //printf(" -- Job managet in job %zu, time step %d (%zu)\n", job, timer, jobLauncher->launchTimeStep);
        }

        // add job to the pending queue
        addPendingJob(jobLauncher);
        printf(" -- [RMS] Job added to pending queue at time step %d\n",timer);

        job ++;
    }

    return NULL;
}



int main(int argc, char* argv[]){

    pthread_t thrTime, thrJobs;
    int finished = 0, nGPUs;

    srand(time(NULL));

    // [SCHEDULER INITIALIZATION]
    // allocate memory to store cuda data

    //cudaGetDeviceCount(&nGPUs); // get number of available devices
    
    // TMP: load number of GPUs from terminal input
    nGPUs = strtoul(argv[1], NULL, 10);

    // initialize "scheduler" resource availability
    schInfo = (schInfo_t *)calloc(1, sizeof(schInfo_t));

    // initialize system resources
    schInfo->nGPUs = (size_t)nGPUs; // number of GPUs
    schInfo->nAvGPUs = schInfo->nGPUs; // available gpus = number of gpus
    schInfo->avGPUs = (char*)malloc(schInfo->nGPUs * sizeof(char)); 
    for(size_t gpu = 0; gpu < (size_t)nGPUs; gpu++){
        schInfo->avGPUs[gpu] = 1; // available
    }
    //schInfo->gpuJob = (unsigned int*)calloc(schInfo->nGPUs, sizeof(unsigned int)); 


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


    // files to store results
    char *recordFileName = argv[3];
    char *gpuUsageFileName = argv[4];
    char *resultsFileName = argv[5];


    // [TIMER AND RESOURCE MONITOR]
    // File for recording events: jobs pending, jobs running, reconfigurations...
    //f = fopen("eventRecord.csv", "w");
    fEventRecord = fopen(recordFileName, "w");
    if (fEventRecord == NULL) {
        perror("fopen\n");
        printf(" -- Error opening the file\n");
        return -1;  // or handle error appropriately
    }

    fUsage = fopen(gpuUsageFileName, "w");
    if (fUsage == NULL) {
        perror("fopen\n");
        printf(" -- Error opening the file\n");
        return -1;  // or handle error appropriately
    }

    fOutput = fopen(resultsFileName, "w");
    if (fUsage == NULL) {
        perror("fopen\n");
        printf(" -- Error opening the file\n");
        return -1;  // or handle error appropriately
    }


    fprintf(fEventRecord, "GPU,Job,time,event\n");
    fflush(fEventRecord);
    printf(" -- [RMS] Files opened!\n");
    fflush(stdout);

    // thread for counting time
    pthread_create(&thrTime, NULL, resourceMonitoring, (void*)(&finished)); // thread for updating timer
    pthread_create(&thrJobs, NULL, jobManager, (void*)(jobsTimeline)); // thread for simulating jobs launching to the system



    // [DECLARE VARIABLES USED DURING SCHEDULING PROCESS]
    size_t nJobsFinished;
    size_t iJob, nPendingJobs, nRunningJobs, nFinishedJobs, nReconfiguringJobs, nReqGPUs, nReqMinGPUs, nLaunchGPUs;
    jobTypeEnum jobType;

    // get queues
    jobQueue_t *pendingQueue = &(schInfo->pendingJobs);
    jobQueue_t *runningQueue = &(schInfo->runningJobs);
    jobQueue_t *finishedQueue = &(schInfo->finishedJobs);
    jobQueue_t *reconfiguringQueue = &(schInfo->reconfiguringJobs);

    job_t *job; 
    jobLauncher_t *jobLauncher;
    jobControl_t *jobControl;

    // initialize structure for job resources
    jobResources_t *jobResources;
    jobMonitoring_t *jobMonitor;


    // [SCHEDULING LOOP]
    printf(" -- [RMS] Entering the scheduling loop (%zu jobs for completion)\n", jobsTimeline->nJobs);
    fflush(stdout);

    nJobsFinished = 0; // loop until all jobs are finished



    struct timespec startTimer, endTimer;
    clock_gettime(CLOCK_MONOTONIC, &(startTimer));



    // loop until all jobs finish their execution
    while(nJobsFinished < jobsTimeline->nJobs){

        // scheduler sleep
        sleep(1);

        // get number of elements on each GPU
        nPendingJobs = getNumberOfJobsInQueue(&(schInfo->pendingJobs));
        nRunningJobs = getNumberOfJobsInQueue(&(schInfo->runningJobs));
        nFinishedJobs = getNumberOfJobsInQueue(&(schInfo->finishedJobs));
        nReconfiguringJobs = getNumberOfJobsInQueue(&(schInfo->reconfiguringJobs));

        if(timer % 5 == 0){
            printf(" -- [RMS] State:\n ---- Pending jobs = %zu\n ---- Running jobs = %zu\n ---- Finished jobs = %zu\n ---- Reconfiguring jobs = %zu\n ---- Free gpus = %zu (", 
                nPendingJobs, nRunningJobs, nFinishedJobs, nReconfiguringJobs, schInfo->nAvGPUs);
            
            for(size_t gpu = 0; gpu<schInfo->nGPUs; gpu ++){

                if(schInfo->avGPUs[gpu]){
                    printf(" %zu", gpu);
                }
            }

            printf(")\n");

            for(iJob = 0; iJob < nRunningJobs; iJob++){

                job = getJobFromQueue(runningQueue, iJob);

                printf(" ---- ---- Job %zu is using %zu GPUs: (", job->jobId, job->jobControl->jobResources->nGPUs);

                for(size_t gpu = 0; gpu<job->jobControl->jobResources->nGPUs; gpu++){

                    printf(" %zu", job->jobControl->jobResources->idGPUs[gpu]);
                }
                printf(")\n");
            }

            fflush(stdout);
        }

        // [SCHEDULING ALGORITHM]: scheduling, reconfigurations...

        // loop over pending jobs and check whether any job can be launched
        nPendingJobs = getNumberOfJobsInQueue(&(schInfo->pendingJobs));
        for(iJob = 0; iJob<nPendingJobs; iJob++){

            // get job
            job = getJobFromQueue(pendingQueue, iJob);
            
            // get job type
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

                // allocate resources
                allocateResources(schInfo, jobResources);

                
                // [LAUNCH JOB]
                launchJob(job, iJob, jobResources);

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
        }

        // loop over jobs that are being reconfigured and check if they finished
        nReconfiguringJobs = getNumberOfJobsInQueue(&(schInfo->reconfiguringJobs)); 
        for(iJob = 0; iJob<nReconfiguringJobs; iJob ++){
            
            job = getJobFromQueue(reconfiguringQueue, iJob);

            // check whether it finished the reconfiguration
            char reconfigurationDone = checkReconfigurationDone(job->jobControl);

            if(reconfigurationDone == 0){

                // get job control
                jobControl = job->jobControl;

                // finish job
                jobFinishedReconfiguration(job, iJob);
             

                // deallocate old resources
                jobResources_t *oldJobResources = (jobResources_t*)calloc(1,sizeof(jobResources_t));
                oldJobResources->nGPUs = jobControl->prevJobResources->nGPUs - jobControl->jobResources->nGPUs;
                oldJobResources->idGPUs = (size_t*)malloc(oldJobResources->nGPUs * sizeof(size_t));

                // find the GPUs to be deallocaated
                // TODO (I need to test this)
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


                // update flow values if job finished reconfiguring
                nReconfiguringJobs --;
                iJob --;
            }
        }

        // manage reconfigurations for running jobs
        nRunningJobs = getNumberOfJobsInQueue(&(schInfo->runningJobs));
        for(iJob = 0; iJob<nRunningJobs; iJob++){

            // get GPUs needed by job
            job = getJobFromQueue(runningQueue, iJob);
            jobLauncher = job->jobLauncher;
            jobResources = job->jobControl->jobResources;
            jobType = jobLauncher->jobType;

            // check whether the job finished
            char jobFinished = checkJobFinished(job->jobControl);

            // check if job finished
            if(jobFinished){
            
                // finish job
                finishJob(job, iJob);

                // deallocate resources
                deallocateResources(schInfo, jobResources);

                 
                // free job resources (should be a function?) [TODO]
                /*free(jobResources->idGPUs);
                free(jobResources);
                free(jobControl->prevJobResources->idGPUs);
                free(jobControl->prevJobResources);

                free(job->jobControl);

                free(job->jobLauncher->argv);
                free(job->jobLauncher);*/
                
                // [TODO]: free job pointers (jobControl, job, jobLauncher...)

                printf(" -- [RMS] Job %zu finished\n", job->jobId);
                fflush(stdout);


                nJobsFinished ++;
                nRunningJobs --;
                iJob--;
            }

            // if job not finished and job is malleable (malleable or flexible), check for reconfigurations
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
                        /*jobResources_t *reconfJobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
                        
                        reconfJobResources->nGPUs = jobResources->nGPUs * 2;

                        reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));

                        for(size_t i = 0; i<jobResources->nGPUs; i++){

                            reconfJobResources->idGPUs[i] = jobResources->idGPUs[i]; 
                        }

                        // find available GPUs
                        size_t gpuId = 0;
                        size_t jobGPU = jobResources->nGPUs;
                        
                        while(jobGPU < reconfJobResources->nGPUs){

                            // allocate the gpu for the job
                            if(schInfo->avGPUs[gpuId] == 1){
                                
                                // set the GPU id
                                jobResources->idGPUs[jobGPU] = gpuId; 

                                // update index
                                jobGPU ++;
                            }

                            gpuId ++;
                        }*/

                        //printf(" -- Mean GPU usage of job %zu is %lf, so it will be expanded\n", job->jobId, meanUsage);
                    }


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
        }
    }

    // inform monitor that execution ended
    finished = 1;


    /* comptue statistics */
    // -- Mean wait time
    // -- Mean execution time
    // -- Mean total time

    double meanWait = 0.0, meanRun = 0.0, meanTotal = 0.0;
    
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
    fflush(fOutput);
}


// TODO: rethink code organization
// - Which will encapsulate each function? When a job is launche