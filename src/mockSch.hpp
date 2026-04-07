#ifndef MOCKSCH_HPP
#define MOCKSCH_HPP

#include <pthread.h>
#include <time.h>


#include "jobQueue.hpp"


typedef struct user_t {

    const char *userName;

} user_t;

enum jobStateEnum {
    
    PENDING,
    RUNNING,
    FINISHED,
    NOTLAUNCHED
};

enum jobTypeEnum {

    RIGID,
    MOLDABLE,
    MALLEABLE,
    FLEXIBLE
};

// launcher associated to a pending job
typedef struct jobLauncher_t {

    int argc;
    void** argv;
    void (*launchFunc)(int, void* []);

    // Information for launching the job
    size_t nReqMinGPUs; // minimum number of required GPUs
    size_t nReqGPUs; // requested GPUs or maximum number of GPUs
    jobTypeEnum jobType; // rigid, moldable, malleable, flexible (0, 1, 2, 3)
    size_t appType;
    size_t launchTimeStep; // time step in which it has been launched

} jobLauncher_t;


// Resouces dedicated for running a job
typedef struct jobResources_t {

    size_t nGPUs;
    size_t *idGPUs;

} jobResources_t;


// Structure to control job and inform reconfigurations
typedef struct jobControl_t {

    // "Signals" for communicating the job and the scheduler

    int pendingReconf = 0; // whether there is any pending reconfiguration
    pthread_mutex_t lockPendingReconf; // lock for synchronization

    char sigGPUs; // 0, not GPUs needed; 1, GPU needed
    pthread_mutex_t lockSigGPUs; // lock for synchronization

    char reqGPUs; // 0, not GPU request; 1, GPU request
    pthread_mutex_t lockReqGPUs; // lock for synchronization
    pthread_cond_t condReqGPUs; // cond 

    char finished; // 0, not GPU request; 1, GPU request
    pthread_mutex_t lockFinished; // lock for synchronization

    // resources dedicated to the job
    jobResources_t *jobResources;

} jobControl_t;


typedef struct job_t{

    // job identifier
    size_t jobId;

    // job state
    jobStateEnum jobState;

    // monitoring information (move to another structure)
    struct timespec jobStartPending, jobEndPending, jobStartRunning, jobEndRunning;
    double *gpuUsage;

    // job thread
    pthread_t jobThread;

    // user that launched the job
    user_t *user;

    // information for launching the job
    jobLauncher_t *jobLauncher;

    // job controller for communicating the RMS and the applicaiton
    jobControl_t *jobControl;

} job_t;

// scheduler data structure
typedef struct schInfo_t {

    // System resources and availability
    size_t nGPUs;
    size_t nAvGPUs;
    char *avGPUs; // 0 | 1

    // Users
    user_t *users;
    size_t nUsers;

    // System state (running jobs)
    jobQueue_t runningJobs;

    // System state (pending jobs)
    jobQueue_t pendingJobs;

    // Jobs already finished 
    jobQueue_t finishedJobs;

} schInfo_t;


typedef struct jobsTimeline_t {

    size_t nJobs;
    jobLauncher_t *jobLaunchers;

} jobsTimeline_t;

// [JOB MANAGEMENT]

jobsTimeline_t* loadJobsFromFile(const char* jobsFileName);

// initialize system: users, resources...
void initSystem();

// job scheduler
void schedule(); 


// add job to pending queue
void addPendingJob(jobLauncher_t *jobLauncher);

// move job from pending to running queue, and launch
void launchJob(job_t *job, size_t pendingIndex, jobResources_t *jobResources);

// move job from running job to finished job
void finishJob();


/// Init job control for communicating the scheduler and the job
job_t* initJob(jobLauncher_t* jobLauncher);

/// Function to be called by pthreads that executes the application associated to the job
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


// [Job scope notifications]

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