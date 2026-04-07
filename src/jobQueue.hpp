# ifndef JOBQUEUE_HPP
# define JOBQUEUE_HPP

#include <pthread.h>


#define MAX_JOBS 1000



typedef struct job_t job_t;

typedef struct jobQueue_t {

    job_t *jobs[MAX_JOBS];
    size_t nJobsInQueue;
    pthread_mutex_t queueLock; // lock for synchronization

} jobQueue_t;

void initQueue(jobQueue_t *queue);
void addJobToQueue(jobQueue_t *queue, job_t *job);
job_t* removeJobFromQueueByIndex(jobQueue_t *queue, size_t index);

job_t* getJobFromQueue(jobQueue_t *queue, size_t index);
size_t getNumberOfJobsInQueue(jobQueue_t *queue);


#endif