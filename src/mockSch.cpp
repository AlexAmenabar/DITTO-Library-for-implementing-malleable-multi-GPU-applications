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
FILE *f = NULL;
size_t nextJobId = 0;


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
        
        // if job is moldable or flexible, load also the minimum number of GPUs
        if(jobType == 1 || jobType == 3){
        
            fscanf(jobF, "%zu", &(jobLauncher->nReqMinGPUs));
        }

        // load the time step in which the user launches the job to the system
        fscanf(jobF, "%zu", &(jobLauncher->launchTimeStep));

        // load the application type
        fscanf(jobF, "%zu", &(jobLauncher->appType));

        // load the arguments for launching the application
        fscanf(jobF, "%d", &(jobLauncher->argc)); // number of arguments

        // allocate memory for argv
        jobLauncher->argv = (void**)calloc(jobLauncher->argc + 2, sizeof(void*)); // 2 extra arguments: whether the job is malleable and the job control (latter set)
        

        // load argv parameters: application related parameters
        for(size_t arg = 0; arg<jobLauncher->argc; arg++){
            
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
        jobLauncher->argv[jobLauncher->argc-2] = (void*)jmall;

        // load application main function call pointer
        if(jobLauncher->appType == 0){

            jobLauncher->launchFunc = &launch_iterative_app;
        }
        else if(jobLauncher->appType == 1){

            jobLauncher->launchFunc = &launch_phases_app;
        }

        // print job information
        printf(" -- Printing job %zu information\n", job);
        printf(" ---- Job type %zu\n", jobType);
        printf(" ---- Required gpus %zu\n", jobLauncher->nReqGPUs);
        printf(" ---- Required min. gpus %zu\n", jobLauncher->nReqMinGPUs);
        printf(" ---- Job launch time step %zu\n", jobLauncher->launchTimeStep);
        printf(" ---- Job application type %zu\n", jobLauncher->appType);
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
    printf(" Updating job control...\n");
    fflush(stdout);
    job->jobControl->jobResources = jobResources; // set job resources
    job->jobControl->jobId = job->jobId;

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
  
    // set new job resources
    if(jobControl->prevJobResources != NULL){
        
        // deallocate old job resources
        free(jobControl->prevJobResources->idGPUs);
        free(jobControl->prevJobResources);
    }

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
    jobLauncher->argv[jobLauncher->argc-1] = (void*)(jobControl);

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
                fprintf(f, "%zu,%zu,%d,start\n", jobControl->jobResources->idGPUs[gpu], jobId, timer);
            }
        break;
        case JOBRECONFIGURED:
            for(gpu = 0; gpu<jobControl->jobResources->nGPUs; gpu++){
                fprintf(f, "%zu,%zu,%d,start\n", jobControl->jobResources->idGPUs[gpu], jobId, timer);
            }
            for(gpu = 0; gpu<jobControl->prevJobResources->nGPUs; gpu++){
                fprintf(f, "%zu,%zu,%d,end\n", jobControl->jobResources->idGPUs[gpu], jobId, timer);
            }
        break;
        case JOBFINISHED:
            for(gpu = 0; gpu<jobControl->jobResources->nGPUs; gpu++){
                fprintf(f, "%zu,%zu,%d,end\n", jobControl->jobResources->idGPUs[gpu], jobId, timer);
            }
        break;
    }
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


// [WORKLOADS]
/*
int workload1(int argc, char* argv[]){

    // [Launch job 0]
    // job resources
    size_t gpus = 4;
    size_t *ids = (size_t*)calloc(gpus, sizeof(size_t));
    ids[0] = 0;
    ids[1] = 1;
    ids[2] = 2;
    ids[3] = 3;
    schInfo.activeJobsControl[0].jobId = 0;
    initJobControl(&(schInfo.activeJobsControl[0]), gpus, ids); // job-sched communication

    // job arguments
    size_t j0a = 50000;
    size_t j0b = 2000000;
    size_t j0c = 1000;
    size_t j0mall = 0;
    void* j0args[5];
    j0args[0] = &j0a;
    j0args[1] = &j0b;
    j0args[2] = &j0c;
    j0args[3] = &j0mall;
    j0args[4] = &(schInfo.activeJobsControl[0]);

    // define job
    jobLauncher_t job0;
    job0.argc = 5;
    job0.argv = j0args;
    job0.launchFunc = &launch_iterative_app;

    // [record event]: GPUs allocated for job
    fprintf(f, "0,0,%d,start\n1,0,%d,start\n2,0,%d,start\n3,0,%d,start\n", timer,timer,timer,timer);
    fflush(f);
    // [record event]

    pthread_t thr0;
    pthread_create(&thr0, NULL, runJob, (void*)(&job0));

    // wait until job finishes
    pthread_join(thr0, NULL);


    // [record event]: deallocate resources
    fprintf(f, "0,0,%d,end\n1,0,%d,end\n2,0,%d,end\n3,0,%d,end\n", timer,timer,timer,timer);
    fflush(f);
    // [record event]


    // [launch job 1]
    gpus = 2;
    ids = (size_t*)calloc(gpus, sizeof(size_t));
    ids[0] = 0;
    ids[1] = 1;
    schInfo.activeJobsControl[1].jobId = 1;
    initJobControl(&(schInfo.activeJobsControl[1]), gpus, ids);

    size_t j1a = 25000000; 
    size_t j1b = 800;
    size_t j1c = 10000;
    size_t j1mall = 0;
    void* j1args[5];
    j1args[0] = &j1a;
    j1args[1] = &j1b;
    j1args[2] = &j1c;
    j1args[3] = &j1mall;
    j1args[4] = &(schInfo.activeJobsControl[1]);

    // define job
    jobLauncher_t job1;
    job1.argc = 5;
    job1.argv = j1args;
    job1.launchFunc = &launch_iterative_app;


    // [record event]: allocate GPUs for job
    fprintf(f, "0,1,%d,start\n1,1,%d,start\n", timer,timer);
    fflush(f);
    // [record event]

    pthread_t thr1;
    pthread_create(&thr1, NULL, runJob, (void*)(&job1));


    // [launch job 2]
    size_t j2a = 10000000;
    size_t j2b = 2000;
    size_t j2c = 15000;
    size_t j2mall = 0;

    gpus = 2;
    ids = (size_t*)calloc(gpus, sizeof(size_t));
    ids[0] = 2;
    ids[1] = 3;
    initJobControl(&(schInfo.activeJobsControl[2]), gpus, ids);

    void* argsj2[5];
    argsj2[0] = &j2a;
    argsj2[1] = &j2b;
    argsj2[2] = &j2c;
    argsj2[3] = &j2mall;
    argsj2[4] = &(schInfo.activeJobsControl[2]);

    // define job
    jobLauncher_t job2;
    job2.argc = 5;
    job2.argv = argsj2;
    job2.launchFunc = &launch_iterative_app;

    // [record event]: allocate two GPUs for job
    fprintf(f, "2,2,%d,start\n3,2,%d,start\n", timer,timer);
    fflush(f);
    // [record event]

    pthread_t thr2;
    pthread_create(&thr2, NULL, runJob, (void*)(&job2));


    // job finished, deallocate resources
    pthread_join(thr1, NULL);
    // [record event]
    fprintf(f, "0,1,%d,end\n1,1,%d,end\n", timer,timer);
    fflush(f);
    // [record event]

    // job finished, deallocate resources
    pthread_join(thr2, NULL);
    // [record event]
    fprintf(f, "2,2,%d,end\n3,2,%d,end\n", timer,timer);
    fflush(f);
    // [record event]


    // wait 5 seconds before ending for finishing information storage
    sleep(5);

    return 1;
}


int workload1_malleable(int argc, char* argv[]){

    // [launch job 1]
    size_t gpus = 4;
    size_t *ids = (size_t*)calloc(gpus, sizeof(size_t));
    ids[0] = 0;
    ids[1] = 1;
    ids[2] = 2;
    ids[3] = 3;
    schInfo.activeJobsControl[0].jobId = 0;
    initJobControl(&(schInfo.activeJobsControl[0]), gpus, ids);

    // job arguments
    size_t j0a = 50000;
    size_t j0b = 2000000;
    size_t j0c = 1000;
    size_t j0mall = 1;
    void* j0args[5];
    j0args[0] = &j0a;
    j0args[1] = &j0b;
    j0args[2] = &j0c;
    j0args[3] = &j0mall;
    j0args[4] = &(schInfo.activeJobsControl[0]);

    // define job
    jobLauncher_t job0;
    job0.argc = 5;
    job0.argv = j0args;
    job0.launchFunc = &launch_iterative_app;

    // [record event]
    fprintf(f, "0,0,%d,start\n1,0,%d,start\n2,0,%d,start\n3,0,%d,start\n", timer,timer,timer,timer);
    fflush(f);
    // [record event]

    pthread_t thr0;
    pthread_create(&thr0, NULL, runJob, (void*)(&job0));

    printf(" Job 0 launched!\n");
    fflush(stdout);


    // reconfigure app 0
    sleep(4);

    free(ids);
    gpus = 2;
    ids = (size_t*)calloc(gpus, sizeof(size_t));
    ids[0] = 0;
    ids[1] = 1;
    notifyReconfiguration(&(schInfo.activeJobsControl[0]), gpus, ids);

    // check if the job finished the reconfiguration
    int done = 0;
    while(done == 0){
        done = !checkReconfigurationDone(&(schInfo.activeJobsControl[0]));
        sleep(1);
    }

    // [record event]
    fprintf(f, "2,0,%d,end\n3,0,%d,end\n", timer,timer);
    fflush(f);
    // [record event]


    // [launch job 2]
    gpus = 2;
    ids = (size_t*)calloc(gpus, sizeof(size_t));
    ids[0] = 2;
    ids[1] = 3;

    schInfo.activeJobsControl[1].jobId = 1;
    initJobControl(&(schInfo.activeJobsControl[1]), gpus, ids);

    size_t j1a = 25000000; //1000000;
    size_t j1b = 800;
    size_t j1c = 10000;
    size_t j1mall = 1;
    void* j1args[5];
    j1args[0] = &j1a;
    j1args[1] = &j1b;
    j1args[2] = &j1c;
    j1args[3] = &j1mall;
    j1args[4] = &(schInfo.activeJobsControl[1]);

    // define job
    jobLauncher_t job1;
    job1.argc = 5;
    job1.argv = j1args;
    job1.launchFunc = &launch_iterative_app;

    // [record event]
    fprintf(f, "2,1,%d,start\n3,1,%d,start\n", timer,timer);
    fflush(f);
    // [record event]

    pthread_t thr1;
    pthread_create(&thr1, NULL, runJob, (void*)(&job1));
    printf(" Job 1 launched!\n");
    fflush(stdout);


    // reconfigure app
    sleep(2);

    free(ids);
    gpus = 1;
    ids = (size_t*)calloc(gpus, sizeof(size_t));
    ids[0] = 0;

    notifyReconfiguration(&(schInfo.activeJobsControl[0]), gpus, ids);

    // check if the job finished the reconfiguration
    done = 0;
    while(done == 0){
        done = !checkReconfigurationDone(&(schInfo.activeJobsControl[0]));
        sleep(1);
    }

    // [record event]
    fprintf(f, "1,0,%d,end\n", timer);
    fflush(f);
    // [record event]


    // [launch job 3]
    size_t j2a = 10000000; //1000000;
    size_t j2b = 2000;
    size_t j2c = 15000;
    size_t j2mall = 0;

    gpus = 1;
    ids = (size_t*)calloc(gpus, sizeof(size_t));
    ids[0] = 1;
    initJobControl(&(schInfo.activeJobsControl[2]), gpus, ids);

    void* argsj2[5];
    argsj2[0] = &j2a;
    argsj2[1] = &j2b;
    argsj2[2] = &j2c;
    argsj2[3] = &j2mall;
    argsj2[4] = &(schInfo.activeJobsControl[2]);

    // define job
    jobLauncher_t job2;
    job2.argc = 5;
    job2.argv = argsj2;
    job2.launchFunc = &launch_iterative_app;

    // [record event]
    fprintf(f, "1,2,%d,start\n", timer);
    fflush(f);
    // [record event]

    pthread_t thr2;
    pthread_create(&thr2, NULL, runJob, (void*)(&job2));

    printf(" Job 2 launched!\n");
    fflush(stdout);


    pthread_join(thr0, NULL);
    printf(" Job 0 finished!\n");
    fflush(stdout);

    // [record event]
    fprintf(f, "0,0,%d,end\n", timer);
    fflush(f);
    // [record event]

    // reconf app 2
    free(ids);
    gpus = 2;
    ids = (size_t*)calloc(gpus, sizeof(size_t));
    ids[0] = 0;
    ids[1] = 1;
    notifyReconfiguration(&(schInfo.activeJobsControl[2]), gpus, ids);

    // check if the job finished the reconfiguration
    done = 0;
    while(done == 0){
        done = !checkReconfigurationDone(&(schInfo.activeJobsControl[2]));
        sleep(1);
    }

    // [record event]
    fprintf(f, "0,2,%d,start\n", timer);
    fflush(f);
    // [record event]


    pthread_join(thr1, NULL);
    printf(" Job 1 finished!\n");
    fflush(stdout);

    // [record event]
    fprintf(f, "2,1,%d,end\n3,1,%d,end\n", timer,timer);
    fflush(f);
    // [record event]


    // reconf app 1
    free(ids);
    gpus = 4;
    ids = (size_t*)calloc(gpus, sizeof(size_t));
    ids[0] = 0;
    ids[1] = 1;
    ids[2] = 2;
    ids[3] = 3;
    notifyReconfiguration(&(schInfo.activeJobsControl[2]), gpus, ids);

    // [record event]
    fprintf(f, "0,2,%d,start\n1,2,%d,start\n2,2,%d,start\n3,2,%d,start\n", timer,timer,timer,timer);
    fflush(f);
    // [record event]

    // check if the job finished the reconfiguration
    done = 0;
    while(done == 0){
        done = !checkReconfigurationDone(&(schInfo.activeJobsControl[0]));
        sleep(1);
    }


    pthread_join(thr2, NULL);
    printf(" Job 2 finished!\n");
    fflush(stdout);

    // [record event]
    fprintf(f, "0,2,%d,end\n1,2,%d,end\n2,2,%d,end\n3,2,%d,end\n", timer,timer,timer,timer);
    fflush(f);
    // [record event]

    sleep(5);

    return 1;
}



int workload2(int argc, char* argv[]){

    // [launch job 1]
    size_t gpus = 4;
    size_t *ids = (size_t*)calloc(gpus, sizeof(size_t));
    ids[0] = 0;
    ids[1] = 1;
    ids[2] = 2;
    ids[3] = 3;
    schInfo.activeJobsControl[0].jobId = 0;
    initJobControl(&(schInfo.activeJobsControl[0]), gpus, ids);


    // job arguments
    size_t j0a = 50000000;
    size_t j0b = 5;
    size_t j0c = 500000;
    size_t j0d = 100;
    size_t j0e = 2;
    size_t j0p1 = 0;
    size_t j0p2 = 1;
    size_t j0mall = 0;

    void* j0args[9];
    j0args[0] = &j0a;
    j0args[1] = &j0b;
    j0args[2] = &j0c;
    j0args[3] = &j0d;
    j0args[4] = &j0e;
    j0args[5] = &j0p1;
    j0args[6] = &j0p2;
    j0args[7] = &j0mall;
    j0args[8] = &(schInfo.activeJobsControl[0]);

    // define job
    jobLauncher_t job0;
    job0.argc = 9;
    job0.argv = j0args;
    job0.launchFunc = &launch_phases_app;

    // [record event]
    fprintf(f, "0,0,%d,start\n1,0,%d,start\n2,0,%d,start\n3,0,%d,start\n", timer,timer,timer,timer);
    fflush(f);
    // [record event]

    pthread_t thr0;
    pthread_create(&thr0, NULL, runJob, (void*)(&job0));

    printf(" Job 0 launched!\n");
    fflush(stdout);


    pthread_join(thr0, NULL);

    // [record event]
    fprintf(f, "0,0,%d,end\n1,0,%d,end\n2,0,%d,end\n3,0,%d,end\n", timer,timer,timer,timer);
    fflush(f);
    // [record event]


    // [launch job 1]
    gpus = 4;
    ids = (size_t*)calloc(gpus, sizeof(size_t));
    ids[0] = 0;
    ids[1] = 1;
    ids[2] = 2;
    ids[3] = 3;
    schInfo.activeJobsControl[1].jobId = 1;
    initJobControl(&(schInfo.activeJobsControl[1]), gpus, ids);


    // job arguments
    size_t j1a = 10000000;
    size_t j1b = 100;
    size_t j1c = 500000;
    size_t j1mall = 0;

    void* j1args[4];
    j1args[0] = &j1a;
    j1args[1] = &j1b;
    j1args[2] = &j1c;
    j0args[3] = &j1mall;
    j1args[4] = &(schInfo.activeJobsControl[1]);

    // define job
    jobLauncher_t job1;
    job1.argc = 5;
    job1.argv = j1args;
    job1.launchFunc = &launch_iterative_app;

    // [record event]
    fprintf(f, "0,1,%d,start\n1,1,%d,start\n2,1,%d,start\n3,1,%d,start\n", timer,timer,timer,timer);
    fflush(f);
    // [record event]

    pthread_t thr1;
    pthread_create(&thr1, NULL, runJob, (void*)(&job1));

    printf(" Job 1 launched!\n");
    fflush(stdout);

    pthread_join(thr1, NULL);

        
    // [record event]
    fprintf(f, "0,1,%d,end\n1,1,%d,end\n2,1,%d,end\n3,1,%d,end\n", timer,timer,timer,timer);
    fflush(f);
    // [record event]

    sleep(4);

    return 1;
}


int workload2_malleable(int argc, char* argv[]){

    // [launch job 1]
    size_t gpus = 4;
    size_t *ids = (size_t*)calloc(gpus, sizeof(size_t));
    ids[0] = 0;
    ids[1] = 1;
    ids[2] = 2;
    ids[3] = 3;
    schInfo.activeJobsControl[0].jobId = 0;
    initJobControl(&(schInfo.activeJobsControl[0]), gpus, ids);


    // job arguments
    size_t j0a = 50000000;
    size_t j0b = 5;
    size_t j0c = 500000;
    size_t j0d = 100;
    size_t j0e = 2;
    size_t j0p1 = 0;
    size_t j0p2 = 1;
    size_t j0mall = 1;

    void* j0args[9];
    j0args[0] = &j0a;
    j0args[1] = &j0b;
    j0args[2] = &j0c;
    j0args[3] = &j0d;
    j0args[4] = &j0e;
    j0args[5] = &j0p1;
    j0args[6] = &j0p2;
    j0args[7] = &j0mall;
    j0args[8] = &(schInfo.activeJobsControl[0]);

    // define job
    jobLauncher_t job0;
    job0.argc = 9;
    job0.argv = j0args;
    job0.launchFunc = &launch_phases_app;


    // [record event]
    fprintf(f, "0,0,%d,start\n1,0,%d,start\n2,0,%d,start\n3,0,%d,start\n", timer,timer,timer,timer);
    fflush(f);
    // [record event]

    pthread_t thr0, thr1;
    pthread_create(&thr0, NULL, runJob, (void*)(&job0));

    
    printf(" Job 0 launched!\n");
    fflush(stdout);

   
    int counter = 0, done, signal;
    while(checkJobFinished(&(schInfo.activeJobsControl[0])) == 0){

        // check for signals
        signal = checkSignalNoGPUs(&(schInfo.activeJobsControl[0]));

        if(signal){

            // reconfigure app
            gpus = 0;
            ids = NULL;
            notifyReconfiguration(&(schInfo.activeJobsControl[0]), gpus, ids);


            // check if the job finished the reconfiguration
            done = 0;
            while(!done){
                done = !checkReconfigurationDone(&(schInfo.activeJobsControl[0]));
                sleep(0);
            }

            // GPUs deallocated for job 0
            if(counter > 2){
                // [record event]
                fprintf(f, "0,0,%d,end\n1,0,%d,end\n2,0,%d,end\n", timer,timer,timer);
                fflush(f);
                // [record event]
            }
            else{
                // [record event]
                fprintf(f, "0,0,%d,end\n1,0,%d,end\n2,0,%d,end\n3,0,%d,end\n", timer,timer,timer,timer);
                fflush(f);
                // [record event]
            }


            if(counter == 2){
                
                sleep(1);

                // [launch job 1]
                gpus = 4;
                ids = (size_t*)calloc(gpus, sizeof(size_t));
                ids[0] = 0;
                ids[1] = 1;
                ids[2] = 2;
                ids[3] = 3;

                schInfo.activeJobsControl[1].jobId = 1;
                initJobControl(&(schInfo.activeJobsControl[1]), gpus, ids);


                // job arguments
                size_t j1a = 10000000;
                size_t j1b = 100;
                size_t j1c = 500000;
                size_t j1mall = 1;

                void* j1args[4];
                j1args[0] = &j1a;
                j1args[1] = &j1b;
                j1args[2] = &j1c;
                j0args[3] = &j1mall;
                j1args[4] = &(schInfo.activeJobsControl[1]);

                // define job
                jobLauncher_t job1;
                job1.argc = 5;
                job1.argv = j1args;
                job1.launchFunc = &launch_iterative_app;

                // GPUs allocated for job 1
                // [record event]
                fprintf(f, "0,1,%d,start\n1,1,%d,start\n2,1,%d,start\n3,1,%d,start\n", timer,timer,timer,timer);
                fflush(f);
                // [record event]

                pthread_create(&thr1, NULL, runJob, (void*)(&job1));

                printf(" Job 1 launched!\n");
                fflush(stdout);
            }
            if(counter > 2){

                gpus = 4;
                ids = (size_t*)calloc(gpus, sizeof(size_t));
                ids[0] = 0;
                ids[1] = 1;
                ids[2] = 2;
                ids[3] = 3;
                

                // [record event]
                fprintf(f, "0,1,%d,start\n1,1,%d,start\n2,1,%d,start\n", timer,timer,timer);
                fflush(f);
                // [record event]


                notifyReconfiguration(&(schInfo.activeJobsControl[1]), gpus, ids);

                // check if the job finished the reconfiguration
                int done = 0;
                while(done == 0){
                    done = !checkReconfigurationDone(&(schInfo.activeJobsControl[1]));
                    sleep(0.1);
                }
            }
        }

        signal = checkSignalReqGPUs(&(schInfo.activeJobsControl[0]));

        if(signal){

            if(counter >= 2){

                // Job 1 deallocates resources
                // [record event]
                fprintf(f, "0,1,%d,end\n1,1,%d,end\n2,1,%d,end\n", timer,timer,timer);
                fflush(f);
                // [record event]

                gpus = 1;
                ids = (size_t*)calloc(gpus, sizeof(size_t));
                ids[0] = 3;
                
                notifyReconfiguration(&(schInfo.activeJobsControl[1]), gpus, ids);
                
                // check if the job finished the reconfiguration
                int done = 0;
                while(done == 0){
                    done = !checkReconfigurationDone(&(schInfo.activeJobsControl[1]));
                    sleep(0.1);
                }
            }


            // reconfigure app 0
            if(counter >= 2){
                gpus = 3;
                ids = (size_t*)calloc(gpus, sizeof(size_t));
                ids[0] = 0;
                ids[1] = 1;
                ids[2] = 2;

                // [record event]
                fprintf(f, "0,0,%d,start\n1,0,%d,start\n2,0,%d,start\n", timer,timer,timer);
                fflush(f);
                // [record event]
            }
            else{
                gpus = 4;
                ids = (size_t*)calloc(gpus, sizeof(size_t));
                ids[0] = 0;
                ids[1] = 1;
                ids[2] = 2;
                ids[3] = 3;

                // [record event]
                fprintf(f, "0,0,%d,start\n1,0,%d,start\n2,0,%d,start\n3,0,%d,start\n", timer,timer,timer,timer);
                fflush(f);
                // [record event]
            }

            notifyReconfiguration(&(schInfo.activeJobsControl[0]), gpus, ids);

            // check if the job finished the reconfiguration
            int done = 0;
            while(done == 0){
                done = !checkReconfigurationDone(&(schInfo.activeJobsControl[0]));
                sleep(0.1);
            }

            counter ++;
        }

        sleep(0.1);
    }

    pthread_join(thr0, NULL);

    // [record event]
    fprintf(f, "0,0,%d,end\n1,0,%d,end\n2,0,%d,end\n", timer,timer,timer);
    fflush(f);
    // [record event]
    

    gpus = 4;
    ids = (size_t*)calloc(gpus, sizeof(size_t));
    ids[0] = 0;
    ids[1] = 1;
    ids[2] = 2;
    ids[3] = 3;

    fprintf(f, "0,1,%d,start\n1,1,%d,start\n2,1,%d,start\n", timer,timer,timer);
    fflush(f);


    notifyReconfiguration(&(schInfo.activeJobsControl[1]), gpus, ids);

    pthread_join(thr1, NULL);

    fprintf(f, "0,1,%d,end\n1,1,%d,end\n2,1,%d,end\n3,1,%d,end\n", timer,timer,timer,timer);
    fflush(f);

    sleep(5);

    return 1;
}


int reconfs_workload(int argc, char* argv[]){

    // [initialize scheduler]

    // allocate memory to store cuda data
    int nGPUs;
    cudaGetDeviceCount(&nGPUs); // get number of available devices
    
    // initialize "scheduler" resource availability
    schInfo_t schInfo;
    schInfo.nGPUs = (size_t)nGPUs;
    schInfo.avGPUs = (char*)calloc(schInfo.nGPUs, sizeof(char));

    // initialize scheduler jobs control information
    schInfo.nJobs = 0;
    schInfo.nMaxJobs = 100;
    schInfo.lastJob = 0;
    schInfo.jobThreads = (pthread_t*)calloc(schInfo.nMaxJobs, sizeof(pthread_t));
    schInfo.activeJobsControl = (jobControl_t*)calloc(schInfo.nMaxJobs, sizeof(jobControl_t));
    schInfo.maskActiveJobs = (char*)calloc(schInfo.nMaxJobs, sizeof(char));

    schInfo.nPendingJobs = 0;
    schInfo.nMaxPendingJobs = 100;
    schInfo.pendingJobsControl = (jobControl_t*)calloc(schInfo.nMaxPendingJobs, sizeof(jobControl_t));
    schInfo.maskPendingJobs = (char*)calloc(schInfo.nMaxPendingJobs, sizeof(char));

    printf(" Scheduler initialized!\n");
    fflush(stdout);

    pthread_t thr;
    jobLauncher_t job;


    size_t maxBytes = 10000000000; // 10.000.000.000 --> 10GB 

    for(size_t nBytes = 1; nBytes < maxBytes; nBytes *= 10){
        
        for(size_t gpus = 1; gpus <= nGPUs; gpus++){

            size_t *ids = (size_t*)calloc(gpus, sizeof(size_t));
            for(size_t i = 0; i<gpus; i++){
                ids[i] = i;
            }

            // notify reconfiguration and measure time
            // check if the job finished the reconfiguration
            for(size_t recGPUs = 1; recGPUs <= nGPUs; recGPUs++){
                
                // [launch job 1]    
                schInfo.activeJobsControl[0].jobId = 0;
                initJobControl(&(schInfo.activeJobsControl[0]), gpus, ids);

                // job arguments
                size_t ja = nBytes;
                size_t jb = 1;
                size_t jc = 1;
                size_t jmall = 1;
                void* jargs[5];
                jargs[0] = &ja;
                jargs[1] = &jb;
                jargs[2] = &jc;
                jargs[3] = &jmall;
                jargs[4] = &(schInfo.activeJobsControl[0]);

                // define job
                jobLauncher_t job;
                job.argc = 5;
                job.argv = jargs;
                job.launchFunc = &launch_reconf_test_app;

                pthread_create(&thr, NULL, runJob, (void*)(&job));

                printf(" Job nBytes = %zu, nGPUs = %u launched!\n",nBytes,nGPUs);
                fflush(stdout);



                size_t *recGPUids = (size_t*)malloc(recGPUs * sizeof(size_t));
                for(size_t j = 0; j<recGPUs; j++){
                    recGPUids[j] = j;
                }

                notifyReconfiguration(&(schInfo.activeJobsControl[0]), recGPUs, recGPUids);

                int done = 0;
                while(done == 0){
                    done = !checkReconfigurationDone(&(schInfo.activeJobsControl[0]));
                    sleep(0.1);
                }


                pthread_join(thr, NULL);

                free(recGPUids);
            }

            free(ids);
        }
    }

    sleep(5);

    return 1;
}*/


void* timeCounter(void *finished){
 
    int *finish = (int*)finished;

    FILE *f = fopen("gpu_log.csv", "w");
    if (!f) {
        perror("fopen");
        exit(1);
    }

    fprintf(f, "time_step,gpu_id,utilization_%%,power_W\n");
    fflush(f);

    // Initialize NVML
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));
        exit(1);
    }

    unsigned int device_count;
    nvmlDeviceGetCount(&device_count);

    while((*finish) == 0){

        usleep(INTERVAL_US); // 1 seconds
        timer += 1; // update timer

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

            fprintf(f, "%d,%u,%u,%.2f\n", timer, i, util.gpu, power / 1000.0);
        }
        fflush(f);  // ensure data is written
    }

    nvmlShutdown();
    fclose(f);

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
            printf(" -- Job managet in job %zu, time step %d (%zu)\n", job, timer, jobLauncher->launchTimeStep);
        }

        // add job to the pending queue
        addPendingJob(jobLauncher);
        printf(" -- Job added to pending queue at time step %d\n",timer);

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

    // init queues
    initQueue(&(schInfo->pendingJobs));
    initQueue(&(schInfo->runningJobs));
    initQueue(&(schInfo->finishedJobs));
    initQueue(&(schInfo->reconfiguringJobs));


    // initialize scheduler jobs control information
    printf(" -- RMS initialized!\n");
    fflush(stdout);

    // loads jobs timeline (jobs information and when they are launched)
    jobsTimeline_t *jobsTimeline = loadJobsFromFile(argv[2]);
    printf(" -- Jobs loaded!\n");
    fflush(stdout);


    // [TIMER AND RESOURCE MONITOR]
    // File for recording events: jobs pending, jobs running, reconfigurations...
    f = fopen("eventRecord.csv", "w");
    if (f == NULL) {
        perror("fopen\n");
        printf(" -- Error opening the file\n");
        return -1;  // or handle error appropriately
    }

    fprintf(f, "GPU,Job,time,event\n");
    fflush(f);
    printf(" -- File for recording events opened!\n");
    fflush(stdout);

    // thread for counting time
    pthread_create(&thrTime, NULL, timeCounter, (void*)(&finished)); // thread for updating timer
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


    // [SCHEDULING LOOP]
    printf(" -- Entering the scheduling loop\n");
    fflush(stdout);

    nJobsFinished = 0; // loop until all jobs are finished

    // loop until all jobs finish their execution
    while(nJobsFinished < jobsTimeline->nJobs){

        // scheduler sleep
        sleep(1);

        // get number of elements on each GPU
        nPendingJobs = getNumberOfJobsInQueue(&(schInfo->pendingJobs));
        nRunningJobs = getNumberOfJobsInQueue(&(schInfo->runningJobs));
        nFinishedJobs = getNumberOfJobsInQueue(&(schInfo->finishedJobs));
        nReconfiguringJobs = getNumberOfJobsInQueue(&(schInfo->reconfiguringJobs));

        //if(timer % 5 == 0){
            printf(" -- Scheduling:\n ---- Pending jobs = %zu\n ---- Running jobs = %zu\n ---- Finished jobs = %zu\n ---- Reconfiguring jobs = %zu\n", 
                nPendingJobs, nRunningJobs, nFinishedJobs, nReconfiguringJobs);
            fflush(stdout);
        //}

        // [SCHEDULING ALGORITHM]: scheduling, reconfigurations...

        // loop over pending jobs and check whether any job can be launched
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

                // store number of GPUs for the job
                jobResources->nGPUs = nLaunchGPUs;

                // allocate memory for GPU ids
                jobResources->idGPUs = (size_t*)malloc(nLaunchGPUs * sizeof(size_t));
                
                // find available GPUs and update system state
                size_t gpuId = 0;
                size_t jobGPU = 0;
                while(jobGPU < nLaunchGPUs){

                    if(schInfo->avGPUs[gpuId]){
                        
                        jobResources->idGPUs[jobGPU] = gpuId; // allocate for the job
                        schInfo->avGPUs[gpuId] = 0; // no available
                        jobGPU ++;
                    }

                    gpuId ++;
                }

                schInfo->nAvGPUs -= jobResources->nGPUs;

                // [LAUNCH JOB]
                printf(" -- Launching job %zu with %zu GPUs\n", iJob, nLaunchGPUs);
                launchJob(job, iJob, jobResources);
                printf(" -- Job %zu launched!\n", iJob);
            }
        }

        for(iJob = 0; iJob<nReconfiguringJobs; iJob ++){
            
            job = getJobFromQueue(reconfiguringQueue, iJob);

            // check whether it finished the reconfiguration
            char reconfigurationDone = checkReconfigurationDone(job->jobControl);

            if(reconfigurationDone){

                jobFinishedReconfiguration(job, iJob);
            }
        }

        // manage reconfigurations for running jobs
        for(iJob = 0; iJob<nRunningJobs; iJob++){

            // get GPUs needed by job
            job = getJobFromQueue(runningQueue, iJob);
            jobLauncher = job->jobLauncher;

            // check whether the job finished
            char jobFinished = checkJobFinished(job->jobControl);

            // if job finished, end thread
            if(jobFinished){
            
                // finish job
                finishJob(job, iJob);


                // [TODO: refactorize to a function]

                // deallocate resources
                jobResources = job->jobControl->jobResources;

                schInfo->nAvGPUs += jobResources->nGPUs;
                
                // find available GPUs
                size_t jobGPU = 0;
                for(size_t gpuId = 0; gpuId<jobResources->nGPUs; gpuId++){
                        
                    schInfo->avGPUs[jobResources->idGPUs[gpuId]] = 1; // GPU is avalaible again
                }

                // free job resources
                free(jobResources->idGPUs);
                free(jobResources);
                free(jobControl->prevJobResources->idGPUs);
                free(jobControl->prevJobResources);

                free(job->jobControl);

                free(job->jobLauncher->argv);
                free(job->jobLauncher);
                
                // [TODO]: free job pointers (jobControl, job, jobLauncher...)

                printf(" -- Job %zu finished\n", iJob);
                fflush(stdout);
            }

            // reconfiguration policy
            printf(" -- No reconfiguration policy yet\n");
        }
    }


    // [workloads]
    //workload1(argc, argv);
    //workload1_malleable(argc, argv);

    //workload2(argc, argv);
    //workload2_malleable(argc, argv);

    //reconfs_workload(argc, argv);

    // inform monitor that execution ended
    finished = 1;
}


// TODO: rethink code organization
// - Which will encapsulate each function? When a job is launche