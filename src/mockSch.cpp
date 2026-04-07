#include <cstddef>
#include <omp.h>
#include <unistd.h>
#include <stdio.h>      
#include <stdlib.h>     
#include <pthread.h>
#include <time.h>

// TODO: activate if CUDA [umcomment]
/*#include <cuda.h>
#include <cuda_runtime.h>
#include <nvml.h>*/

#include "mockSch.hpp"
#include "jobQueue.hpp"

// include matrix summation 
// TODO: [unccomment]
//#include "../testApps/toy_app_malleable.hpp"



#define INTERVAL_US 1000000  // 0.1 seconds

// TODO: schInfo not a global variable???

schInfo_t *schInfo;
int timer = 0;
FILE *f = NULL;


jobsTimeline_t* loadJobsFromFile(const char* jobsFileName){

    FILE *jobF;
    void** argv;
    jobsTimeline_t *jobsTimeline;
    
    jobF = fopen(jobsFileName, "r");
    jobsTimeline = (jobsTimeline_t*)calloc(1, sizeof(jobsTimeline_t));

    // read number of jobs
    fscanf(jobF, "%zu", &(jobsTimeline->nJobs));

    // allocate memory for job launchers
    jobsTimeline->jobLaunchers = (jobLauncher_t*)calloc(jobsTimeline->nJobs, sizeof(jobLauncher_t));

    // loop over jobs and load their information in jobLaunchers
    for(size_t job=0; job < jobsTimeline->nJobs; job++){

        // get job launcher
        jobLauncher_t *jobLauncher = &(jobsTimeline->jobLaunchers[job]);

        // load job information: job type, requested resources, launch time and app time
        size_t jobType;
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

        fscanf(jobF, "%zu", &(jobLauncher->nReqGPUs));
        
        // if job is moldable or flexible, load also the minimum number of resources
        if(jobType == 1 || jobType == 3){
        
            fscanf(jobF, "%zu", &(jobLauncher->nReqMinGPUs));
        }

        fscanf(jobF, "%zu", &(jobLauncher->launchTimeStep));
        fscanf(jobF, "%zu", &(jobLauncher->appType));


        // load argc and argv and the function pointer for launching the job later
        fscanf(jobF, "%d", &(jobLauncher->argc));

        // allocate memory for argv
        jobLauncher->argv = (void**)calloc(jobLauncher->argc + 2, sizeof(void*));
        

        // load argv parameters: application related parameters
        for(size_t arg = 0; arg<jobLauncher->argc; arg++){
            
            // if not pointer will disappear after function ends
            size_t *val = (size_t*)malloc(sizeof(size_t));
            fscanf(jobF, "%zu", val);
            jobLauncher->argv[arg] = (void*)val;
        }
        
        // load whether application is or not malleable (or flexible) for allowing reconfiugrations during execution
        size_t *jmall = (size_t*)malloc(sizeof(size_t)); 
        if(jobType == 2 || jobType == 3){
            (*jmall) = 1;
        }
        else{
            (*jmall) = 0;
        }
        jobLauncher->argv[jobLauncher->argc-2] = (void*)jmall;

        // [TODO] quit comment
        // load application function pointer
        /*if(jobLauncher->appType == 0){

            jobLauncher->launchFunc = &launch_iterative_app;
        }
        else if(jobLauncher->appType == 1){

            jobLauncher->launchFunc = &launch_phases_app;
        }*/

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


void addPendingJob(jobLauncher_t *jobLauncher){

    // init job control
    job_t *job = initJob(jobLauncher);
    job->jobState = PENDING; // change state

    // start timer
    clock_gettime(CLOCK_MONOTONIC, &(job->jobStartPending));

    // add job to pending queue
    addJobToQueue(&(schInfo->pendingJobs), job);
}

// remove job from pending queue and add to running job. Update system state too
void launchJob(job_t *job, size_t pendingIndex, jobResources_t *jobResources){

    // update job state
    job->jobState = RUNNING;

    // remove from pending queue
    removeJobFromQueueByIndex(&(schInfo->pendingJobs), pendingIndex);
  
    // TODO: complete job information
    job->jobControl->jobResources = jobResources; // set job resources: Revise this, I am updating a value that other thread can write too


    // TODO: wait (simulate launch time) (should be improved?)
    int sleepTime = (int)rand() % 5;
    sleep(sleepTime);


    // TODO: launch job (thread)


    // manage timers
    clock_gettime(CLOCK_MONOTONIC, &(job->jobEndPending));
    clock_gettime(CLOCK_MONOTONIC, &(job->jobStartRunning));

    // add job to running queue
    addJobToQueue(&(schInfo->runningJobs), job);
}


void finishJob(job_t *job, size_t runningJobIndex){

    // update queues
    removeJobFromQueueByIndex(&(schInfo->runningJobs), runningJobIndex); // remove from running jobs queue
    addJobToQueue(&(schInfo->finishedJobs), job); // add to finished jobs queue

    // kill job thread
    pthread_join(job->jobThread, NULL);

    // manage timers
    clock_gettime(CLOCK_MONOTONIC, &(job->jobEndRunning));

    // update job state
    job->jobState = FINISHED;
}


job_t* initJob(jobLauncher_t* jobLauncher){

    job_t *job = (job_t*)calloc(1, sizeof(job_t));

    // set job launcher
    job->jobLauncher = jobLauncher;


    // initialize job control
    jobControl_t *jobControl = (jobControl_t*)calloc(1, sizeof(jobControl_t));
    job->jobControl = jobControl; // set job control

    jobControl->pendingReconf = 0;
    pthread_mutex_init(&(jobControl->lockPendingReconf), NULL);

    jobControl->sigGPUs = 0;
    pthread_mutex_init(&(jobControl->lockSigGPUs), NULL);

    jobControl->reqGPUs = 0;
    pthread_mutex_init(&(jobControl->lockReqGPUs), NULL);
    pthread_cond_init(&(jobControl->condReqGPUs), NULL);

    jobControl->finished = 0;
    pthread_mutex_init(&(jobControl->lockFinished), NULL);

    // resources not initialized because they are set when process starts running

    // 
    job->jobState = NOTLAUNCHED;


    return job;
}

void* runJob(void *jobLauncherVoid){

    jobLauncher_t *jobLauncher = (jobLauncher_t*)jobLauncherVoid;
    jobLauncher->launchFunc(jobLauncher->argc, jobLauncher->argv);
    return NULL;
}



/* [NOTIFICATIONS] */

// [SCHEDULER SCOPE]

int checkJobFinished(jobControl_t *jControl){
    
    // check whether there is any pending reconfiguration
    int finished = 0;
    pthread_mutex_lock(&(jControl->lockFinished));
    finished = jControl->finished;

    if(finished == 1)
        jControl->finished = 2;
    
    pthread_mutex_unlock(&jControl->lockFinished);

    return finished;
}


// scheduler notifies to the app that there is a pending reconfiguration
void notifyReconfiguration(jobControl_t *jobControl, size_t nGPUs, size_t *idGPUs){

    size_t i;
    jobResources_t *jobResources = jobControl->jobResources;

    // lock
    pthread_mutex_lock(&(jobControl->lockPendingReconf));

    // program the reconfiguration if there are no pending reconfigurations
    if(jobControl->pendingReconf == 0){

        jobResources->nGPUs = nGPUs;

        if(jobResources->idGPUs) 
            free(jobResources->idGPUs);
 
        jobResources->idGPUs = (size_t*)calloc(nGPUs, sizeof(size_t));
        for(i = 0; i<nGPUs; i++)
            jobResources->idGPUs[i] = idGPUs[i];
        
        // new pending reconfiguration
        jobControl->pendingReconf = 1;
    }

    // unlock
    pthread_mutex_unlock(&(jobControl->lockPendingReconf));
}

int checkSignalNoGPUs(jobControl_t *jobControl){

    int signal = 0;

    // lock
    pthread_mutex_lock(&(jobControl->lockSigGPUs));

    signal = jobControl->sigGPUs;
    jobControl->sigGPUs = 0;

    // unlock
    pthread_mutex_unlock(&(jobControl->lockSigGPUs));

    return signal;

}

int checkSignalReqGPUs(jobControl_t *jobControl){

    size_t i;
    int req = 0;

    // lock
    pthread_mutex_lock(&(jobControl->lockReqGPUs));

    req = jobControl->reqGPUs;
    jobControl->reqGPUs = 0;

    // unlock
    pthread_mutex_unlock(&(jobControl->lockReqGPUs));

    return req;

}

int checkReconfigurationDone(jobControl_t *jobControl){

    int pending;
    // lock
    pthread_mutex_lock(&(jobControl->lockPendingReconf));

    pending = jobControl->pendingReconf;

    // unlock
    pthread_mutex_unlock(&(jobControl->lockPendingReconf));

    return pending;
}



// [JOB SCOPE]


void jobFinished(jobControl_t* jControl){

    // check whether there is any pending reconfiguration
    pthread_mutex_lock(&(jControl->lockFinished));
    jControl->finished = 1;
    pthread_mutex_unlock(&(jControl->lockFinished));
}

int checkIfReconfiguration(jobControl_t *jobControl){

    int localPendingReconf;

    // check whether there is any pending reconfiguration
    pthread_mutex_lock(&(jobControl->lockPendingReconf));
    localPendingReconf = jobControl->pendingReconf;
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
    /*nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));
        exit(1);
    }

    unsigned int device_count;
    nvmlDeviceGetCount(&device_count);*/

    while((*finish) == 0){

        usleep(INTERVAL_US); // 1 seconds
        timer += 1; // update timer

        /*// store info obtained from nvidia-smi
        for (unsigned int i = 0; i < device_count; i++) {

            nvmlDevice_t device;
            nvmlDeviceGetHandleByIndex(i, &device);

            // Utilization
            nvmlUtilization_t util;
            nvmlDeviceGetUtilizationRates(device, &util);

            // Power (in milliwatts)
            unsigned int power;
            nvmlDeviceGetPowerUsage(device, &power);

            fprintf(f, "%d,%u,%u,%.2f\n",
                    timer,
                    i,
                    util.gpu,
                    power / 1000.0);
        }
        fflush(f);  // ensure data is written*/
    }

    /*nvmlShutdown();
    fclose(f);*/

    return NULL;
}

// function for simulating jobs launching
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
    size_t iJob, nPendingJobs, nRunningJobs, nFinishedJobs, nReqGPUs, nReqMinGPUs, nLaunchGPUs;
    jobTypeEnum jobType;

    // get queues
    jobQueue_t *pendingQueue = &(schInfo->pendingJobs);
    jobQueue_t *runninggQueue = &(schInfo->runningJobs);
    jobQueue_t *finishedQueue = &(schInfo->finishedJobs);
    job_t *job; 
    jobLauncher_t *jobLauncher;
    jobControl_t *jobControl;

    // initialize structure for job resources
    jobResources_t *jobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
    jobResources->nGPUs = 0;
    jobResources->idGPUs = NULL;


    // [SCHEDULING LOOP]

    nJobsFinished = 0; // loop until all jobs are finished

    // loop until all jobs finish their execution
    while(nJobsFinished < jobsTimeline->nJobs){

        // scheduler sleep
        sleep(1);

        // get number of elements on each GPU
        nPendingJobs = getNumberOfJobsInQueue(&(schInfo->pendingJobs));
        nRunningJobs = getNumberOfJobsInQueue(&(schInfo->runningJobs));
        nFinishedJobs = getNumberOfJobsInQueue(&(schInfo->finishedJobs));

        if(timer % 5 == 0)
            printf(" -- Scheduling:\n ---- Pending jobs = %zu\n ---- Running jobs = %zu\n ---- Finished jobs = %zu\n", 
                nPendingJobs, nRunningJobs, nFinishedJobs);



        // [SCHEDULING ALGORITHM]: scheduling, reconfigurations...

        // loop over pending jobs and check whether any job can be launched
        for(iJob = 0; iJob<nPendingJobs; iJob++){
        
            // get GPUs needed by job
            job = getJobFromQueue(pendingQueue, iJob);
            jobLauncher = job->jobLauncher;
            jobType = jobLauncher->jobType;

            // get required number of GPUs
            nReqGPUs = jobLauncher->nReqGPUs;
            nReqMinGPUs = nReqGPUs; // [FIXED or MALLEABLE]

            // if job is moldable or flexible, store the minimum number of GPUs too
            if(jobType == MOLDABLE || jobType == FLEXIBLE){
                nReqMinGPUs = jobLauncher->nReqMinGPUs;
            }


            // launch using the maximum number of GPUs possible
            nLaunchGPUs = 0;
            if(schInfo->nAvGPUs >= nReqGPUs){
                nLaunchGPUs = nReqGPUs;
            }
            // if job is moldable or flexible, enable there are [avGPUs >= minGPUs]
            else if(schInfo->nAvGPUs >= nReqMinGPUs){
                nLaunchGPUs = schInfo->nAvGPUs;
            }

            // launch job if possible
            if(nLaunchGPUs){

                // TODO: Revise this, I'm not sure whether I should copy values instead of doing this, is confusing
                jobResources->nGPUs = nLaunchGPUs;


                // [ALLOCATE RESOURCES FOR JOB]: allocation policy, ORDERED

                // deallocate old memory for idGPUs
                if(jobResources->idGPUs) free(jobResources->idGPUs);
                
                // allocate memory for GPU ids
                jobResources->idGPUs = (size_t*)malloc(nLaunchGPUs * sizeof(size_t));
                
                // find available GPUs
                size_t jobGPU = 0;
                for(size_t gpuId = 0; gpuId<nLaunchGPUs; gpuId++){
                    
                    if(schInfo->avGPUs[gpuId]){ // available
                        
                        jobResources->idGPUs[jobGPU] = gpuId; // allocate for the job
                        schInfo->avGPUs[gpuId] = 0; // no available
                    }
                }
                schInfo->nAvGPUs -= jobResources->nGPUs;


                // [LAUNCH JOB]
                printf(" -- Launching job %zu with %zu GPUs\n", iJob, nLaunchGPUs);
                launchJob(job, iJob, jobResources);
                printf(" -- Job %zu launched!\n", iJob);
            }
        }


        // manage reconfigurations for running jobs
        for(iJob = 0; iJob<nRunningJobs; iJob++){


        }

        // look for finished jobs
        for(iJob = 0; iJob<nFinishedJobs; iJob++){
        
            // get GPUs needed by job
            job = getJobFromQueue(finishedQueue, iJob);
            jobLauncher = job->jobLauncher;

            // check whether job finished (lock, since job can be writing that it finished)
            pthread_mutex_lock(&(jobControl->lockFinished));
            char jobFinished = jobControl->finished;
            pthread_mutex_unlock(&(jobControl->lockFinished));

            // if job finished, end thread
            if(jobFinished){
            
                // finish job
                finishJob(job, iJob);

                // deallocate resources
                jobResources = job->jobControl->jobResources;

                schInfo->nAvGPUs += jobResources->nGPUs;
                
                // find available GPUs
                size_t jobGPU = 0;
                for(size_t gpuId = 0; gpuId<jobResources->nGPUs; gpuId++){
                        
                    schInfo->avGPUs[jobResources->idGPUs[gpuId]] = 1; // is not avalaible again
                }

                // free job resources
                free(jobResources->idGPUs);
                free(jobResources);
                // [TODO]: free job pointers (jobControl, job, jobLauncher...)

                printf(" -- Job %zu finished\n", iJob);
                fflush(stdout);
            }
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