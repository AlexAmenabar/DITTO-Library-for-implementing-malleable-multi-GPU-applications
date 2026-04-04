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


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <nvml.h>
#include <time.h>

#define INTERVAL_US 1000000  // 0.1 seconds


schInfo_t *schInfo;
int timer = 0;
FILE *f = NULL;



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
int workload1(int argc, char* argv[]){

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
    job_t job0;
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
    job_t job1;
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
    job_t job2;
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
    job_t job0;
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
    job_t job1;
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
    job_t job2;
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
    job_t job0;
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


    /*while(checkJobFinished(&(schInfo.activeJobsControl[0])) == 0){

        // check for signals
        int signal = checkSignalNoGPUs(&(schInfo.activeJobsControl[0]));

        if(signal){

            // reconfigure app
            gpus = 0;
            ids = NULL;

            notifyReconfiguration(&(schInfo.activeJobsControl[0]), gpus, ids);

            // check if the job finished the reconfiguration
            int done = 0;
            while(done == 0){
                done = !checkReconfigurationDone(&(schInfo.activeJobsControl[0]));
                sleep(0.1);
            }
        }

        signal = checkSignalReqGPUs(&(schInfo.activeJobsControl[0]));

        if(signal){

            // reconfigure app
            gpus = 4;
            ids = (size_t*)calloc(gpus, sizeof(size_t));
            ids[0] = 0;
            ids[1] = 1;
            ids[2] = 2;
            ids[3] = 3;

            notifyReconfiguration(&(schInfo.activeJobsControl[0]), gpus, ids);

            // check if the job finished the reconfiguration
            int done = 0;
            while(done == 0){
                done = !checkReconfigurationDone(&(schInfo.activeJobsControl[0]));
                sleep(0.1);
            }
        }

        sleep(0.1);
    }*/



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
    job_t job1;
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

    
    printf(" In workload 2\n");
    fflush(stdout);


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
    job_t job0;
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
                job_t job1;
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
    job_t job;


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
                job_t job;
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
}


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

            fprintf(f, "%d,%u,%u,%.2f\n",
                    timer,
                    i,
                    util.gpu,
                    power / 1000.0);
        }
        fflush(f);  // ensure data is written
    }

    nvmlShutdown();
    fclose(f);

    return NULL;
}


int main(int argc, char* argv[]){

    // [timer and resource monitor]
    f = fopen("eventRecord.csv", "w");
    if (f == NULL) {
        perror("fopen\n");
        printf(" Error opening the file\n");
        return -1;  // or handle error appropriately
    }

    fprintf(f, "GPU,Job,time,event\n");
    fflush(f);

    pthread_t thrTime;
    int finished = 0;

    // thread for counting time
    pthread_create(&thrTime, NULL, timeCounter, (void*)(&finished));

    // [workloads]
    //workload1(argc, argv);
    //workload1_malleable(argc, argv);

    //workload2(argc, argv);
    //workload2_malleable(argc, argv);

    reconfs_workload(argc, argv);

    // inform monitor that execution ended
    finished = 1;
}
