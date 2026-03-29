#ifndef MOCKSCH_HPP
#define MOCKSCH_HPP

#include <pthread.h>

// Structure to control job and inform reconfigurations
typedef struct jobControl_t {

    // dedicated resources
    size_t jobId;
    size_t nGPUs;
    size_t *idGPUs;

    int pendingReconf = 0; // whether there is any pending reconfiguration
    pthread_mutex_t lockPendingReconf; // lock for synchronization

    // application info
    char sigGPUs; // 0, not GPUs needed; 1, GPU needed
    pthread_mutex_t lockSigGPUs; // lock for synchronization

    char reqGPUs; // 0, not GPU request; 1, GPU request
    pthread_mutex_t lockReqGPUs; // lock for synchronization
    pthread_cond_t condReqGPUs; // cond 

    char finished; // 0, not GPU request; 1, GPU request
    pthread_mutex_t lockFinished; // lock for synchronization

    // monitoring
    double *gpuUsage;

} jobControl_t;


// launcher associated to a pending job
typedef struct job_t {

    int argc;
    void** argv;
    void (*launchFunc)(int, void* []);

    size_t reqGPUs;

} job_t;

// scheduler data structure
typedef struct schInfo_t {

    size_t nGPUs;
    char *avGPUs; // 0 | 1

    size_t nJobs;
    size_t nMaxJobs;
    size_t lastJob;
    pthread_t *jobThreads;
    jobControl_t *activeJobsControl;
    char *maskActiveJobs;

    size_t nPendingJobs;
    size_t nMaxPendingJobs;
    jobControl_t *pendingJobsControl;
    char *maskPendingJobs;

} schInfo_t;

void jobFinished(jobControl_t* jControl);


#endif