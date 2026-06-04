#include <cstddef>
#include <stdio.h>      
#include <stdlib.h>    

# include "eventQueue.hpp"
# include "RMS.hpp"

// queue operations: lock queue, perform operations, unlock queue

void initQueue(eventQueue_t *queue){

    queue->nEventsInQueue = 0;
    pthread_mutex_init(&(queue->queueLock), NULL);
}

void addJobToQueue(eventQueue_t *queue, event_t *event){

    pthread_mutex_lock(&(queue->queueLock));
    queue->events[queue->nEventsInQueue] = event;
    queue->nEventsInQueue ++;
    pthread_mutex_unlock(&(queue->queueLock));
}

event_t* removeJobFromQueueByIndex(eventQueue_t *queue, size_t index){

    event_t *jobCtrl = queue->events[index];

    pthread_mutex_lock(&(queue->queueLock));
    for(size_t i = index; i<queue->nEventsInQueue-1; i++){

        queue->events[i] = queue->events[i + 1];
    }
    queue->events[queue->nEventsInQueue-1] = NULL;
    queue->nEventsInQueue --;
    pthread_mutex_unlock(&queue->queueLock);

    return jobCtrl;
}

event_t* getJobFromQueue(eventQueue_t *queue, size_t index){

    pthread_mutex_lock(&(queue->queueLock));
    event_t *events = queue->events[index]; 
    pthread_mutex_unlock(&queue->queueLock);

    return events;
}

size_t getNumberOfJobsInQueue(eventQueue_t *queue){

    pthread_mutex_lock(&(queue->queueLock));
    size_t nEvents = queue->nEventsInQueue; 
    pthread_mutex_unlock(&queue->queueLock);

    return nEvents;
}
