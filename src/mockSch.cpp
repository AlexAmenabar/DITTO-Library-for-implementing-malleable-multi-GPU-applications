#include <cstddef>
#include <omp.h>
#include <unistd.h>
#include <stdio.h>      
#include <stdlib.h>     
#include <pthread.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "mockSch.hpp"

// include matrix summation 
#include "../testApps/toy_app_malleable.hpp"


schInfo_t *schInfo;



void initJobControl(jobControl_t* jControl, size_t nGPUs, size_t* idGPUs){

    jControl->nGPUs = nGPUs;

    jControl->idGPUs = (size_t*)calloc(nGPUs, sizeof(size_t));
    for(size_t i = 0; i<nGPUs; i++)
        jControl->idGPUs[i] = idGPUs[i];

    // initialize reconfiguration related variables
    jControl->pendingReconf = 0; // indicates whether there is any pending reconfiguration
    pthread_mutex_init(&(jControl->lockPendingReconf), NULL);

    jControl->sigGPUs = 0;
    pthread_mutex_init(&(jControl->lockSigGPUs), NULL);

    jControl->reqGPUs = 0;
    pthread_mutex_init(&(jControl->lockReqGPUs), NULL);
    pthread_cond_init(&(jControl->condReqGPUs), NULL);

    jControl->finished = 0;
    pthread_mutex_init(&(jControl->lockFinished), NULL);
}

void* runJob(void *jobLauncherVoid){

    job_t *jobLauncher = (job_t*)jobLauncherVoid;
    jobLauncher->launchFunc(jobLauncher->argc, jobLauncher->argv);
    return NULL;
}

void jobFinished(jobControl_t* jControl){

    // check whether there is any pending reconfiguration
    pthread_mutex_lock(&(jControl->lockFinished));
    jControl->finished = 1;
    pthread_mutex_unlock(&(jControl->lockFinished));
}

int checkJobFinished(jobControl_t *jControl){
    
    // check whether there is any pending reconfiguration
    int finished = 0;
    pthread_mutex_lock(&(jControl->lockFinished));
    finished = jControl->finished;
    pthread_mutex_unlock(&jControl->lockFinished);

    return finished;
}


// scheduler notifies to the app that there is a pending reconfiguration
void notifyReconfiguration(jobControl_t *jobControl, size_t nGPUs, size_t *idGPUs){

    size_t i;

    // lock
    pthread_mutex_lock(&(jobControl->lockPendingReconf));

    // program the reconfiguration if there are no pending reconfigurations
    if(jobControl->pendingReconf == 0){

        jobControl->nGPUs = nGPUs;

        if(jobControl->idGPUs) 
            free(jobControl->idGPUs);
 
        jobControl->idGPUs = (size_t*)calloc(nGPUs, sizeof(size_t));
        for(i = 0; i<nGPUs; i++)
            jobControl->idGPUs[i] = idGPUs[i];
        
        // new pending reconfiguration
        jobControl->pendingReconf = 1;
    }

    // unlock
    pthread_mutex_unlock(&(jobControl->lockPendingReconf));
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

// add pending job
void submitJob(schInfo_t *sched, job_t *job){

    // TODO
}

void launchJob(schInfo_t *sched, job_t *job){

    // TODO
}

int main(int argc, char* argv[]){

    
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

    

    // [launch job 1]
    size_t *ids = (size_t*)calloc(4, sizeof(size_t));
    ids[0] = 0;
    ids[1] = 1;
    ids[2] = 2;
    ids[3] = 3;
    schInfo.activeJobsControl[0].jobId = 0;
    initJobControl(&(schInfo.activeJobsControl[0]), 4, ids);

    // job arguments
    size_t j0a = 500000;
    size_t j0b = 50000;
    size_t j0c = 10000;
    void* j0args[4];
    j0args[0] = &j0a;
    j0args[1] = &j0b;
    j0args[2] = &j0c;
    j0args[3] = &(schInfo.activeJobsControl[0]);

    // define job
    job_t job0;
    job0.argc = 4;
    job0.argv = j0args;
    job0.launchFunc = &launch_iterative_app;

    pthread_t thr0;
    pthread_create(&thr0, NULL, runJob, (void*)(&job0));

    printf(" Job 0 launched!\n");
    fflush(stdout);

    // Reconfigure job 1 from 4 GPUs to 2

    sleep(5);

    free(ids);
    ids = (size_t*)calloc(2, sizeof(size_t));
    ids[0] = 0;
    ids[1] = 1;

    printf(" -- Notifying to job 0 reconfiguration\n");
    fflush(stdout);
    notifyReconfiguration(&(schInfo.activeJobsControl[0]), 2, ids);
    printf(" -- Reconfiguration notified to job 0\n");
    fflush(stdout);

    // check if the job finished the reconfiguration
    int done = 0;
    while(done == 0){
        done = !checkReconfigurationDone(&(schInfo.activeJobsControl[0]));
        sleep(1);
    }


    // [launch job 2]

    ids = (size_t*)calloc(1, sizeof(size_t));
    ids[0] = 2;

    schInfo.activeJobsControl[1].jobId = 1;
    initJobControl(&(schInfo.activeJobsControl[1]), 1, ids);

    size_t j1a = 50000; //1000000;
    size_t j1b = 5000;
    size_t j1c = 25000;
    void* j1args[4];
    j1args[0] = &j1a;
    j1args[1] = &j1b;
    j1args[2] = &j1c;
    j1args[3] = &(schInfo.activeJobsControl[1]);

    // define job
    job_t job1;
    job1.argc = 4;
    job1.argv = j1args;
    job1.launchFunc = &launch_iterative_app;

    pthread_t thr1;
    pthread_create(&thr1, NULL, runJob, (void*)(&job1));
    printf(" Job 1 launched!\n");
    fflush(stdout);


    sleep(100);

    // [launch job 3]
    /*size_t j2a = 500000; //1000000;
    size_t j2b = 5000;
    size_t j2c = 15000;

    ids = (size_t*)calloc(1, sizeof(size_t));
    ids[0] = 3;
    initJobControl(&(schInfo.activeJobsControl[2]), 1, ids);

    void* argsj2[4];
    argsj2[0] = &j2a;
    argsj2[1] = &j2b;
    argsj2[2] = &j2c;
    argsj2[3] = &(schInfo.activeJobsControl[2]);

    // define job
    job_t job2;
    job2.argc = 4;
    job2.argv = argsj2;
    job2.launchFunc = &launch_iterative_app;

    pthread_t thr2;
    pthread_create(&thr2, NULL, runJob, (void*)(&job2));

    printf(" Job 2 launched!\n");
    fflush(stdout);

    /*int allJobsFinished = 0;
    int nJobs = 1;
    while(allJobsFinished == 0){

        allJobsFinished = 1;

        for(size_t j = 0; j<nJobs; j++){

            allJobsFinished = allJobsFinished & checkJobFinished(&(schInfo.activeJobsControl[j]))
        }
    }*/

    pthread_join(thr0, NULL);
    pthread_join(thr1, NULL);
    //pthread_join(thr2, NULL);
}