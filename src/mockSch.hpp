#ifndef MOCKSCH_HPP
#define MOCKSCH_HPP

#include <pthread.h>
#include <time.h>

#include "jobQueue.hpp"

#define MONITOR_STEPS 20

enum eventsEnum {

    JOBSTARTED,
    JOBRECONFIGURED,
    JOBFINISHED
};

/// Job priority for scheduling decisions
enum jobPriorityEnum {

    LOW,
    MEDIUM,
    HIGH,
    DEADLINE
};

/// Job state
enum jobStateEnum {
    
    PENDING,
    RUNNING,
    RECONFIGURING,
    FINISHED,
    NOTLAUNCHED
};


/// Job type
enum jobTypeEnum {

    RIGID,
    MOLDABLE,
    MALLEABLE,
    FLEXIBLE
};


/// User structure. Name and priority
typedef struct user_t {

    const char *userName;
    size_t userPriority;

} user_t;


/// Launcher associated to job. Is used to launch the job
typedef struct jobLauncher_t {

    int argc;
    void** argv;
    void (*launchFunc)(int, void* []);

    // Information for launching the job
    size_t nReqMinGPUs; // minimum number of required GPUs
    size_t nReqGPUs; // requested GPUs or maximum number of GPUs
    jobTypeEnum jobType; // rigid, moldable, malleable, flexible (0, 1, 2, 3)
    jobPriorityEnum jobPriority;
    size_t appType;
    size_t launchTimeStep; // time step in which it has been launched
    


} jobLauncher_t;


/// Resouces dedicated to a job
typedef struct jobResources_t {

    size_t nGPUs;
    size_t *idGPUs;

} jobResources_t;


/// Structure to control job and inform reconfigurations (shared by the RMS and the job threads)
typedef struct jobControl_t {

    // job identifier
    size_t jobId;

    // "Signals" for communicating the job and the scheduler

    char pendingReconf = 0; // whether there is any pending reconfiguration
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
    jobResources_t *prevJobResources;

} jobControl_t;


typedef struct jobMonitoring_t {

    double *gpuUsage; // usage per GPU (mean): sum, and after N steps, compute mean
    size_t step; // steps with this configuration

    //
    size_t finalUsageGPUs;
    double finalEnergyConsumption;

} jobMonitoring_t;


typedef struct job_t{

    // job identifier
    size_t jobId;

    // job state
    jobStateEnum jobState;

    // monitoring information (move to another structure)
    struct timespec jobStartPending, jobEndPending, jobStartRunning, jobEndRunning;

    // job thread
    pthread_t jobThread;

    // user that launched the job
    user_t *user;

    // information for launching the job
    jobLauncher_t *jobLauncher;

    // job controller for communicating the RMS and the applicaiton
    jobControl_t *jobControl;

    // job monitoring information
    jobMonitoring_t *jobMonitor;

} job_t;

/// Scheduler structure
typedef struct schInfo_t {

    // System resources and availability
    size_t nGPUs;
    size_t nAvGPUs;
    char *avGPUs; // 0 | 1
    unsigned int *gpuJob; // job associated to the GPU
    
    // Monitoring
    unsigned int (*gpuUtilization) [MONITOR_STEPS]; // (nGPUs)? arrays of MONITOR_STEPS
    double (*gpuPower) [MONITOR_STEPS];

    // Users
    user_t *users;
    size_t nUsers;

    // System state (running jobs)
    jobQueue_t runningJobs;

    // System state (pending jobs)
    jobQueue_t pendingJobs;

    // Jobs already finished 
    jobQueue_t finishedJobs;

    // Jobs with pending reconfigurations
    jobQueue_t reconfiguringJobs;

} schInfo_t;

/// Timeline that stores jobs for adding them to the system
typedef struct jobsTimeline_t {

    size_t nJobs;
    jobLauncher_t *jobLaunchers;

} jobsTimeline_t;



// [JOB MANAGEMENT]

/// Load job timeline from file and initialize job launchers
jobsTimeline_t* loadJobsFromFile(const char* jobsFileName);

// NOT IMPLEMENTED YET
void initSystem();

// NOT IMPLEMENTED YET
void schedulingPolicy(); 

/// Add job to pending queue
void addPendingJob(jobLauncher_t *jobLauncher);

/// Start running a job
void launchJob(job_t *job, size_t pendingIndex, jobResources_t *jobResources);

/// Schedule a reconfiguration to a running job
void scheduleReconfiguration(job_t *job, size_t jobIndex, jobResources_t *jobResources);

// Manage the reconfiguration finished
void jobFinishedReconfiguration(job_t *job, size_t jobIndex);

/// Job finished
void finishJob();

jobMonitoring_t* initJobMonitor(jobMonitoring_t *jobMonitor, jobResources_t *jobResources);

/// Initialize Job structure from the job launcher
job_t* initJob(jobLauncher_t* jobLauncher);

/// Function called by pthreads to start running the application in another thread (job)
void* runJob(void *jobLauncherVoid);

/// Recod an event related to a job
void recordEvent(eventsEnum event, job_t *job);

/// Allocate GPUs and update system availability state
void allocateResources(schInfo_t *schInfo, jobResources_t *jobResources);

/// Deallocate GPUs and update system availability state
void deallocateResources(schInfo_t *schInfo, jobResources_t *jobResources);


// [NOTIFICATIONS]

// [RMS scope]
/// Scheduler checks whether the job finished its execution
char checkJobFinished(jobControl_t *jControl);

/// Scheduler notifies a reconfiguration to a job
void notifyReconfiguration(jobControl_t *jobControl);

/// Scheduler checks whether the job indicated that it does not need GPUs
char checkSignalNoGPUs(jobControl_t *jobControl);

/// Scheduler check wheter the job indicated that it need GPUs
char checkSignalReqGPUs(jobControl_t *jobControl);

/// Scheduler checks if the job finished its reconfiguration
char checkReconfigurationDone(jobControl_t *jobControl);


// [Job scope notifications]

/// Job checks if there are pending reconfigurations
char checkIfReconfiguration(jobControl_t *jobControl);

/// Notify to the scheduler that the job finished
void jobFinished(jobControl_t* jControl);

/// Job notifies that it finished the reconfiguration
void notifyReconfigurationDone(jobControl_t *jobControl);

/// Notify to the scheduler that no GPUs are required by the job
void notifySigGPUs(jobControl_t *jobControl);

/// Notify to the scheduler that the job require GPU(s)
void notifyReqGPUs(jobControl_t *jobControl);


#endif