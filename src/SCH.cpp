#include <cstddef>
#include <omp.h>
#include <unistd.h>
#include <stdio.h>      
#include <stdlib.h>     
#include <pthread.h>
#include <time.h>

#include "RMS.hpp"
#include "SCH.hpp"
#include "jobQueue.hpp"
#include "eventQueue.hpp"


// This algorithm launches a job if possible with the maximum number of available resources
void greedy(schInfo_t *schInfo){

    // helper variables
    job_t *job;
    jobLauncher_t *jobLauncher;
    jobResources_t *jobResources;
    size_t jobType, nReqGPUs, nReqMinGPUs, iJob, nLaunchGPUs, estimatedDuration;

    // get pending jobs queue information
    jobQueue_t *pendingQueue = &(schInfo->pendingJobs);
    size_t nPendingJobs = getNumberOfJobsInQueue(pendingQueue);
    

    // loop over pending jobs and check whether any job can be scheduled
    for(iJob = 0; iJob<nPendingJobs; iJob++){

        // get job from pending queue (execute any job)
        job = getJobFromQueue(pendingQueue, iJob);
        
        // get job information
        jobLauncher = job->jobLauncher;
        jobType = jobLauncher->jobType;

        // get required number of GPUs
        nReqGPUs = jobLauncher->nReqGPUs;
        nReqMinGPUs = nReqGPUs; // [FIXED or MALLEABLE]

        // if job is MOLDABLE or FLEXIBLE, store the minimum number of GPUs too
        if(jobType == MOLDABLE || jobType == FLEXIBLE){
            nReqMinGPUs = jobLauncher->nReqMinGPUs;
        }

        // estimated job duration
        estimatedDuration = jobLauncher->estimatedDuration;


        // [SCHEDULING POLICY] : launch, if possible, using all the GPUs the job wants
        // launch using the maximum number of GPUs possible
        nLaunchGPUs = 0;
        if(schInfo->nAvGPUs >= nReqGPUs){
            nLaunchGPUs = nReqGPUs;
        }
        // if job is MOLDABLE or FLEXIBLE, enable there are [avGPUs >= minGPUs]
        else if(schInfo->nAvGPUs >= nReqMinGPUs){
            nLaunchGPUs = schInfo->nAvGPUs;
        }

        // launch job if possible
        if(nLaunchGPUs){

            // allocate resources for the job
            jobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
            jobResources->nGPUs = nLaunchGPUs;
            jobResources->idGPUs = (size_t*)malloc(nLaunchGPUs * sizeof(size_t));
            
            // decide where the job will be executed
            // find available GPUs and update system state // [THIS IS A POLICY TOO] -- refactorize?
            selectFirstAvailableGPUs(jobResources->idGPUs, nLaunchGPUs, schInfo);

            // allocate resources for the job
            allocateResources(schInfo, jobResources);

            // launch job
            launchJob(job, iJob, jobResources);

            // print information of the job
            printf(" -- [RMS] Launching job %zu (id %zu) with %zu GPUs (", iJob, job->jobId, nLaunchGPUs);
            for(size_t g = 0; g<job->jobControl->jobResources->nGPUs; g++)
                printf(" %zu", job->jobControl->jobResources->idGPUs[g]);
            printf(")\n");
            fflush(stdout);

            // update number of pending jobs
            nPendingJobs--;
            iJob--;
        }
    }

}


void sched(schInfo_t *schInfo){

    // call the scheduling algorithm set for the workload simulation
    schInfo->sched(schInfo);
}