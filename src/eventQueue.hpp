# ifndef EVENTQUEUE_HPP
# define EVENTQUEUE_HPP

#include <pthread.h>


#define MAX_EVENTS 10000



typedef struct event_t event_t;

typedef struct eventQueue_t {

    event_t *events[MAX_EVENTS];
    size_t nEventsInQueue;
    pthread_mutex_t queueLock; // lock for synchronization

} eventQueue_t;

void initQueue(eventQueue_t *queue);
void addJobToQueue(eventQueue_t *queue, event_t *job);
event_t* removeJobFromQueueByIndex(eventQueue_t *queue, size_t index);

event_t* getJobFromQueue(eventQueue_t *queue, size_t index);
size_t getNumberOfJobsInQueue(eventQueue_t *queue);


#endif