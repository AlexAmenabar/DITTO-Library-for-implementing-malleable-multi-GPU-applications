#include <cstddef>
#include <stdio.h>      
#include <stdlib.h>    

# include "jobQueue.hpp"
# include "mockSch.hpp"

// queue operations: lock queue, perform operations, unlock queue

void initQueue(jobQueue_t *queue){

    queue->nJobsInQueue = 0;
    pthread_mutex_init(&(queue->queueLock), NULL);
}

void addJobToQueue(jobQueue_t *queue, job_t *job){

    pthread_mutex_lock(&(queue->queueLock));
    queue->jobs[queue->nJobsInQueue] = job;
    queue->nJobsInQueue ++;
    pthread_mutex_unlock(&(queue->queueLock));
}

job_t* removeJobFromQueueByIndex(jobQueue_t *queue, size_t index){

    job_t *jobCtrl = queue->jobs[index];

    pthread_mutex_lock(&(queue->queueLock));
    for(size_t i = index; i<queue->nJobsInQueue-1; i++){

        queue->jobs[i] = queue->jobs[i + 1];
    }
    queue->jobs[queue->nJobsInQueue-1] = NULL;
    queue->nJobsInQueue --;
    pthread_mutex_unlock(&queue->queueLock);

    return jobCtrl;
}

job_t* getJobFromQueue(jobQueue_t *queue, size_t index){

    pthread_mutex_lock(&(queue->queueLock));
    job_t *job = queue->jobs[index]; 
    pthread_mutex_unlock(&queue->queueLock);

    return job;
}

size_t getNumberOfJobsInQueue(jobQueue_t *queue){

    pthread_mutex_lock(&(queue->queueLock));
    size_t nJobs = queue->nJobsInQueue; 
    pthread_mutex_unlock(&queue->queueLock);

    return nJobs;
}
