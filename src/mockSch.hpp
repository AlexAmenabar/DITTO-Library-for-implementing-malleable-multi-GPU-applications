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

    // launch
    size_t reqMinGPUs; // minimum number of required GPUs
    size_t reqMaxGPUs; // maximum number of required GPUs

    size_t t_launch; // time step in which it has been launched

} jobControl_t;


// launcher associated to a pending job
typedef struct job_t {

    int argc;
    void** argv;
    void (*launchFunc)(int, void* []);

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


/// Init job control for communicating the scheduler and the job
void initJobControl(jobControl_t* jControl, size_t nGPUs, size_t* idGPUs);

/// Launch a job
void* runJob(void *jobLauncherVoid);


// [NOTIFICATIONS]

// [Scheduler scope]
/// Scheduler checks whether the job finished its execution
int checkJobFinished(jobControl_t *jControl);

/// Scheduler notifies a reconfiguration to a job
void notifyReconfiguration(jobControl_t *jobControl, size_t nGPUs, size_t *idGPUs);

/// Scheduler checks whether the job indicated that it does not need GPUs
int checkSignalNoGPUs(jobControl_t *jobControl);

/// Scheduler check wheter the job indicated that it need GPUs
int checkSignalReqGPUs(jobControl_t *jobControl);

/// Scheduler checks if the job finished its reconfiguration
int checkReconfigurationDone(jobControl_t *jobControl);


// [Job scope]

/// Job checks if there are pending reconfigurations
int checkIfReconfiguration(jobControl_t *jobControl);

/// Notify to the scheduler that the job finished
void jobFinished(jobControl_t* jControl);

/// Job notifies that it finished the reconfiguration
void notifyReconfigurationDone(jobControl_t *jobControl);

/// Notify to the scheduler that no GPUs are required by the job
void notifySigGPUs(jobControl_t *jobControl);

/// Notify to the scheduler that the job require GPU(s)
void notifyReqGPUs(jobControl_t *jobControl);


#endif