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

// Scheduling policy: launch any possible job
// Allocation policy: select first available GPUs 
void greedy(schInfo_t *schInfo){

    // helper variables
    job_t *job;
    jobLauncher_t *jobLauncher;
    jobResources_t *jobResources;
    size_t jobType, nReqGPUs, nReqMinGPUs, iJob, nLaunchGPUs;

    // get pending jobs queue information
    jobQueue_t *pendingQueue = &(schInfo->pendingJobs);
    size_t nPendingJobs = getNumberOfJobsInQueue(pendingQueue);
    
    // loop over pending jobs and check whether any job can be scheduled
    iJob = 0;
    int keep = 1;
    while(nPendingJobs > 0 && keep){

        keep = 0;
        
        // get job from pending queue
        job = getJobFromQueue(pendingQueue, iJob);
        
        // get job structures
        jobLauncher = job->jobLauncher;
        jobType = jobLauncher->jobType;

        // get required number of GPUs
        nReqGPUs = jobLauncher->nReqGPUs;
        nReqMinGPUs = nReqGPUs; // [FIXED or MALLEABLE]

        // if job is MOLDABLE or FLEXIBLE, store the minimum number of GPUs too
        if(jobType == MOLDABLE || jobType == FLEXIBLE){
            nReqMinGPUs = jobLauncher->nReqMinGPUs;
        }

        // [SCHEDULING POLICY] : launch, if possible, using all the GPUs the job wants
        // launch using the maximum number of GPUs possible
        nLaunchGPUs = 0;    

        // if the job is not the first in the pending queue, and the timeout has been surpassed
        /*if(iJob > 0 && schInfo->timeoutCurrent >= schInfo->timeout){

            // running jobs
            jobQueue_t *runningQueue = &(schInfo->runningJobs);
            size_t nRunningJobs = getNumberOfJobsInQueue(runningQueue);

            // GPUs required by pending job 0
            job_t *job0 = getJobFromQueue(pendingQueue, 0);
            size_t nReqGPUsJob0 = job0->jobLauncher->nReqGPUs;
            size_t nReqMinGPUsJob0 = nReqGPUsJob0; // [FIXED or MALLEABLE]

            // if job is MOLDABLE or FLEXIBLE, store the minimum number of GPUs too
            if(job0->jobLauncher->jobType == MOLDABLE || job0->jobLauncher->jobType == FLEXIBLE){
                nReqMinGPUsJob0 = job0->jobLauncher->nReqMinGPUs;
            }

            // if job 0 is in timeout, then, launch a job only if the first job depends only on 1 job to execute and this will no longer starvate the first one
            size_t irnJob = 0;
            int found = 0;
            while(irnJob < nRunningJobs && !found){

                // get job from queue
                job_t *rnJob = getJobFromQueue(runningQueue, irnJob);

                // check if the job uses more GPUs or the same amount of GPUs that the first job requires
                if(rnJob->jobControl->jobResources->nGPUs >= nReqGPUsJob0){

                    // found
                    found = 1;

                    // compute the number of GPUs to launch the job
                    size_t avGPUs = ;
                    if(schInfo->nAvGPUs >= nReqGPUs){
                        nLaunchGPUs = nReqGPUs;
                    }
                    // if job is MOLDABLE or FLEXIBLE, enable there are [avGPUs >= minGPUs]
                    else if(schInfo->nAvGPUs >= nReqMinGPUs){
                        nLaunchGPUs = schInfo->nAvGPUs;
                    }
                }

                irnJob ++;
            }
        }
        // else, work as always
        else{

            if(schInfo->nAvGPUs >= nReqGPUs){
                nLaunchGPUs = nReqGPUs;
            }
            // if job is MOLDABLE or FLEXIBLE, enable there are [avGPUs >= minGPUs]
            else if(schInfo->nAvGPUs >= nReqMinGPUs){
                nLaunchGPUs = schInfo->nAvGPUs;
            }
        }*/


        // compute if job can be launched
        if(schInfo->nAvGPUs >= nReqGPUs){
            nLaunchGPUs = nReqGPUs;
        }
        // if job is MOLDABLE or FLEXIBLE, enable there are [avGPUs >= minGPUs]
        else if(schInfo->nAvGPUs >= nReqMinGPUs){
            nLaunchGPUs = schInfo->nAvGPUs;
        }
        

        // launch job if possible
        if(nLaunchGPUs){

            // if the first job is launched, reinit the timeout
            if(iJob == 0){

                schInfo->timeoutCurrent = 0;
            }

            // allocate resources for the job (refactorize?)
            jobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));
            jobResources->nGPUs = nLaunchGPUs;
            jobResources->idGPUs = (size_t*)malloc(nLaunchGPUs * sizeof(size_t));
            
            // select the GPUs for the job
            selectFirstAvailableGPUs(jobResources->idGPUs, nLaunchGPUs, schInfo);

            // launch job
            launchJob(schInfo, job, iJob, jobResources);

            pthread_mutex_lock(&printLock);
            printf(" -- [RMS] Launching job %zu (id %zu) with %zu GPUs (", iJob, job->jobId, nLaunchGPUs);
            for(size_t g = 0; g<job->jobControl->jobResources->nGPUs; g++)
                printf(" %zu", job->jobControl->jobResources->idGPUs[g]);
            printf(")\n");
            fflush(stdout);
            pthread_mutex_unlock(&printLock);

            // update number of pending jobs
            nPendingJobs = getNumberOfJobsInQueue(pendingQueue);
            keep = 1;

            //if(nPendingJobs > 0) nPendingJobs = 1;
            //if(iJob > 0)
            //    iJob--;
        }
    }

}


void sched(schInfo_t *schInfo){

    // call the scheduling algorithm set for the workload simulation
    schInfo->sched(schInfo);
}