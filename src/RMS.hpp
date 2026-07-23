#ifndef RMS_HPP
#define RMS_HPP

#include <pthread.h>
#include <time.h>

#include "jobQueue.hpp"
#include "eventQueue.hpp"

#define JOB_MONITOR_STEPS 10
#define DECISION_JOB_MONITOR_STEPS 5
#define INTERVAL_US 1000000  // 0.1 seconds


// [Enumerations]

// Enumeration: manage monitorization events
enum eventsEnum {

    JOBSTARTED,
    RECONFSTARTED,
    RECONFFINISHED,
    JOBFINISHED
};

/// Enumeration: job priorities
enum jobPriorityEnum {

    LOW,
    MEDIUM,
    HIGH,
    DEADLINE
};

/// Enumeration: possible job states
enum jobStateEnum {
    
    PENDING,
    RUNNING,
    RECONFIGURING,
    FINISHED,
    NOTLAUNCHED
};

/// Enumeration: job type
enum jobTypeEnum {

    RIGID,
    MOLDABLE,
    MALLEABLE,
    FLEXIBLE
};

// TODO: this will be used in the future for implementing event-driven scheduling 
// Enumeration: job related events
enum EVENT_ENUM {

    JOB_ARRIVAL, // job arrived to the system
    JOB_FINISHED, // job finished its execution
    RECONFIGURATION_REQUEST, // reconfiguration requested by the job
    RECONFIGURATION_DECISION, // reconfiguration decided by the RMS
    RECONFIGURATION_FINISHED // job finished reconfiguration
};

// [Data structures]

/// Structure: users
typedef struct user_t {

    const char *userName;
    size_t userPriority;

} user_t;


// Job related structures
/// Structure used to launch the job
typedef struct jobLauncher_t {

    // job input arguments
    int argc;
    void** argv;
    // job main function pointer
    void (*launchFunc)(int, void* []);

    // information for launching the job
    size_t nReqMinGPUs; // requested min GPUs
    size_t nReqGPUs; // requested (max) GPUs
    size_t estimatedDuration; // estimated job duration
    jobTypeEnum jobType; // job type
    jobPriorityEnum jobPriority; // job priority
    size_t appType; // synthetic application model
    size_t launchTimeStep; // launch time step
    
} jobLauncher_t;

/// Structure to store the GPUs allocated for a job
typedef struct jobResources_t {

    size_t nGPUs;
    size_t *idGPUs;

} jobResources_t;

/// Structure to communicate the job and the RMS
typedef struct jobControl_t {

    // job identifier
    size_t jobId;

    // "Signals" for communicating the job and the scheduler

    // there is a pending reconfiguration
    int pendingReconf = 0; 
    pthread_mutex_t lockPendingReconf;

    // hint: no GPUs needed (job->RMS)
    int sigGPUs; 
    pthread_mutex_t lockSigGPUs;

    // hint: GPUs needed
    int reqGPUs; 
    pthread_mutex_t lockReqGPUs;
    pthread_cond_t condReqGPUs; 

    // temporal
    int startRunning;
    pthread_mutex_t lockStartRunning;
    int finished; // 0, not GPU request; 1, GPU request
    pthread_mutex_t lockFinished; // lock for synchronization

    // resources dedicated to the job
    jobResources_t *jobResources; 
    // resources for reconfiguring the job
    jobResources_t *reconfJobResources;
    // not used?? TODO
    pthread_mutex_t lockJobResources; 

} jobControl_t;

/// Structure to monitor resource utilization by each job
typedef struct jobMonitoring_t {

    // arrays to store different parameters related to job allocated GPUs
    // [nGPUs x 100 time steps]
    unsigned int (*gpuUsage)[DECISION_JOB_MONITOR_STEPS]; 
    unsigned int (*gpuTemperature)[DECISION_JOB_MONITOR_STEPS];
    unsigned int (*gpuEnergyConsumption)[DECISION_JOB_MONITOR_STEPS];

    unsigned int (*gpuPCIeThroughput)[DECISION_JOB_MONITOR_STEPS];
    unsigned int (*gpuNVLinkThroughput)[DECISION_JOB_MONITOR_STEPS];

    unsigned int (*gpuBandPCIe)[DECISION_JOB_MONITOR_STEPS];
    unsigned int (*gpuBandNVLink)[DECISION_JOB_MONITOR_STEPS];
    

    // current step and the array length
    size_t step; 
    size_t steps;

    // usage summarized
    size_t finalUsageGPUs;
    double finalEnergyConsumption;
    double meanTemperature;

} jobMonitoring_t;


/// Structure for merging all job related structures
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


typedef struct schInfo_t schInfo_t;

/// Structure of the RMS (TODO: change the name)
struct schInfo_t {

    // Pointers to the scheduling and reconfiguration policies
    void (*sched)(schInfo_t *schInfo);
    void (*rconf)(schInfo_t *schInfo);
    int invoqueScheduler;
    pthread_mutex_t invoqueSchedulerLock;

    // System resources and availability
    size_t nGPUs; // total number of GPUs
    size_t nAvGPUs; // number of available GPUs
    char *avGPUs; // GPU available (binary)
    unsigned int *gpuJob; // job that has the allocation of the GPU 
    unsigned int *nvLinkCount;

    // GPU topology info
    int *gpuTopology; // information by topo -m
    int *gpuTopologyRank; // information by rank (more accurate) TODO: not working

    // real time monitor data
    unsigned int (*gpuUtilization) [JOB_MONITOR_STEPS]; // (nGPUs)? arrays of MONITOR_STEPS
    double (*gpuPower) [JOB_MONITOR_STEPS];
    unsigned int (*gpuTemperature) [JOB_MONITOR_STEPS];
    unsigned int (*gpuPCIeThroughput) [JOB_MONITOR_STEPS];
    unsigned int (*gpuNVLinkThroughput) [JOB_MONITOR_STEPS];

    size_t nMonitored;
    size_t monitorIndex;

    size_t timeout;
    size_t timeoutCurrent;


    // accumulated monitored data
    //power consumption
    double totalPowerConsumption;
    double *totalPowerConsumptionPerGPU;
    // utilization
    unsigned int totalUtilization;
    unsigned int *totalUtilizationPerGPU;
    // temperature
    unsigned int temperatureSum;
    unsigned int *temperatureSumPerGPU;
    // PCIe throughput
    unsigned int totalThroughputPCIe; 
    unsigned int *totalThroughputPCIePerGPU;
    // NVLink throughput
    unsigned int totalThroughputNVLink;
    unsigned int *totalThroughputNVLinkPerGPU;
    // allocation time
    unsigned int totalAllocationArea;
    unsigned int *allocationTimePerGPU;
    // number of reconfigurations
    unsigned int nReconfigurations;
    unsigned int nExpands;
    unsigned int nShrinks;
    unsigned int nKeeps;


    // final results per batch
    double *finalThroughput;
    // utilization
    double *finalUtilization;
    double **finalUtilizationPerGPU;
    // power consumption
    double *finalPowerConsumption;
    double **finalPowerConsumptionPerGPU;
    // temperature
    double *finalTemperatureSum;
    double **finalTemperatureSumPerGPU;
    // PCIe throughput
    double *finalThroughputPCIe; 
    double **finalThroughputPCIePerGPU;
    // NVLink throughput
    double *finalThroughputNVLink; 
    double **finalThroughputNVLinkPerGPU;
    // allocation time
    double *finalAllocationArea;
    double **finalAllocationTimePerGPU;  
    // number of reconfiugration
    double *finalNReconfigurations;
    double *finalNExpands;
    double *finalNShrinks;
    double *finalNKeeps;
    // job execution and wait time
    double *finalMeanExecutionTime;
    double *finalMeanWaitTime;


    // Users in the system
    user_t *users;
    size_t nUsers;

    // Job in the execution queue 
    jobQueue_t runningJobs;

    // Jobs in the pending queue
    jobQueue_t pendingJobs;

    // Job already finished
    jobQueue_t finishedJobs;

    // Jobs being reconfigured
    jobQueue_t reconfiguringJobs;

    // TODO: not implemented yet
    // queue of events
    eventQueue_t eventQueue;

    // locks

    // hint: GPUs needed 
    pthread_mutex_t lockTimer;
};

/// Structure previous to enter in the pending queue
typedef struct jobsTimeline_t {

    // number of jobs to be introduced and job laucnhers
    size_t nJobs;
    jobLauncher_t *jobLaunchers;

} jobsTimeline_t;


// TODO
/// Structure of system events for developing event-driven scheduling
typedef struct event_t {

    EVENT_ENUM eventEnum; // description of the vent
    
    job_t *job; // pointer to the job realted to the event
    jobQueue_t *jobQueue; // queue of the job
    size_t jobIndex; // job index in the queue

} event_t;



// global variables

// [TODO]: in the future they should be private?
extern schInfo_t *schInfo;
extern int timer;
extern FILE *fEventRecord, *fUsage, *fOutput;
extern size_t nextJobId;

extern int invoqueScheduler;

extern size_t usageGPUs;
extern double usagePower;
extern size_t registeredUsages;

// temp
extern int *gpuTopology;
extern int *gpuTopologyRank;
extern size_t gNGPUs;

// lock for printf
extern pthread_mutex_t printLock;




// [Job management]

/// Load job timeline from file and initialize job launchers
jobsTimeline_t* loadJobsFromFile(const char* jobsFileName);

/// TODO
void initSystem();

/// Initialize the intra-node GPU topology matrix  
void initializeTopology(schInfo_t *schInfo, char *topoFile);

/// Add job to pending queue
void addPendingJob(jobLauncher_t *jobLauncher);

/// Start running a job
void launchJob(schInfo_t *schInfo, job_t *job, size_t pendingIndex, jobResources_t *jobResources);

/// Schedule a reconfiguration to a running job
void scheduleReconfiguration(schInfo_t *schInfo, job_t *job, size_t jobIndex, jobResources_t *jobResources);

/// Manage the finalization of the job reconfiguration
void jobFinishedReconfiguration(schInfo_t *schInfo, job_t *job, size_t jobIndex);

/// Job finished
void finishJob(schInfo_t *schInfo, job_t *job, size_t runningJobIndex);

/// Initialize the job monitor
jobMonitoring_t* initJobMonitor(jobMonitoring_t *jobMonitor, jobResources_t *jobResources);

/// Initialize Job structure from the job launcher
job_t* initJob(jobLauncher_t* jobLauncher);

/// Function called by pthreads to start running the application in another thread (job)
void* runJob(void *jobLauncherVoid);

/// Recod an event related to a job (visualization purposes)
void recordEvent(eventsEnum event, job_t *job, jobResources_t *jobResources);

/// Allocate GPUs and update system availability state
void allocateResources(schInfo_t *schInfo, jobResources_t *jobResources);

/// Deallocate GPUs and update system availability state
void deallocateResources(schInfo_t *schInfo, jobResources_t *jobResources);

// Deallocate job resources structure
void deallocateJobResourcesStruct(jobResources_t **jobResources);


// [NOTIFICATIONS]

// [RMS scope]
/// Scheduler checks whether the job finished its execution
int checkJobFinished(jobControl_t *jControl);

/// Scheduler notifies a reconfiguration to a job
void notifyReconfiguration(jobControl_t *jobControl);

/// Scheduler checks if jobs can free GPUs
int checkSignalNoGPUs(jobControl_t *jobControl);

/// Scheduler check wheter job needs GPUs
int checkSignalReqGPUs(jobControl_t *jobControl);

/// Scheduler checks if the job finished reconfiguration
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

void notifyStartRunning(jobControl_t *jobControl);


// General

/// Monitor resource data
void* resourceMonitoring(void *finished);

/// 
void initResourceMonitor(schInfo_t *schInfo, size_t nBatch);

/// 
void stepResourceMonitor(schInfo_t *schInfo);

void reinitMonitorAcc(schInfo_t *schInfo);

///
void destroyResourceMonitor(schInfo_t *schInfo);

/// Add jobs to the pending jobs queue after their arrival time
void* jobManager(void *voidJobsTimeline);

/// This method finds the first available n GPUs
void selectFirstAvailableGPUs(size_t *selectedGPUs, size_t nLaunchGPUs, schInfo_t *schInfo);

jobResources_t* findDiffResources(jobResources_t *jobResources, jobResources_t *reconfJobResources);


/// Manage reconfigurations
void manageReconfigurations(schInfo_t *schInfo);

/// Manage jobs finish
void manageJobsFinish(schInfo_t *schInfo);



#endif