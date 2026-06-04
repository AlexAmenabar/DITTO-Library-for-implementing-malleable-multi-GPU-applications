#include <cstddef>
#include <omp.h>
#include <unistd.h>
#include <stdio.h>      
#include <stdlib.h>     
#include <pthread.h>
#include <time.h>

#include "RMS.hpp"
#include "RCF.hpp"
#include "jobQueue.hpp"
#include "eventQueue.hpp"


void utilization(schInfo_t *schInfo){

    // helper variables
    job_t *job;
    jobLauncher_t *jobLauncher;
    jobResources_t *jobResources;
    jobMonitoring_t *jobMonitor;
    size_t jobType, iJob, nRunningJobs, nJobsFinished = 0;

    // get queue of running jobs
    jobQueue_t *runningQueue = &(schInfo->runningJobs);


    // manage running jobs: whether it finished or it needs a reconfiguration
    nRunningJobs = getNumberOfJobsInQueue(&(schInfo->runningJobs));
    for(iJob = 0; iJob<nRunningJobs; iJob++){

        // get job information
        job = getJobFromQueue(runningQueue, iJob);
        jobLauncher = job->jobLauncher;
        jobResources = job->jobControl->jobResources;
        jobType = jobLauncher->jobType;


        // JOB FINISH MANAGEMENT
        // check whether the job finished
        /*char jobFinished = checkJobFinished(job->jobControl);

        // check if job finished
        if(jobFinished){
        
            // finish job
            finishJob(job, iJob);

            // deallocate resources
            deallocateResources(schInfo, jobResources);

                
            // free job resources (should be a function?) [TODO]
            //free(jobResources->idGPUs);
            //free(jobResources);
            //free(jobControl->prevJobResources->idGPUs);
            //free(jobControl->prevJobResources);
            //
            //free(job->jobControl);
            //
            //free(job->jobLauncher->argv);
            //free(job->jobLauncher);
            
            // [TODO]: free job pointers (jobControl, job, jobLauncher...)

            printf(" -- [RMS] Job %zu finished\n", job->jobId);
            fflush(stdout);


            nJobsFinished ++;
            nRunningJobs --;
            iJob--;
        }*/

        // if job not finished and job is malleable or flexible, check for reconfigurations [RECONFIGURATION POLICY]
        //else 
        


        // TODO: manage the case in which job finished
        // check only if job can be reconfigured (malleable | flexible)
        if (jobType > 1){ 

            // get job monitor
            jobMonitor = job->jobMonitor;
            size_t step = jobMonitor->step; // current step

            // circular array
            step = jobMonitor->step % jobMonitor->steps;

            // collect information monitored by the monitoring thread // TODO: probably it is mandatory to add locks
            for(size_t jobGPU = 0; jobGPU < job->jobControl->jobResources->nGPUs; jobGPU++){

                // store results for computing the mean

                // store results of the current time step
                jobMonitor->gpuUsage[jobGPU][step] = schInfo->gpuUtilization[job->jobControl->jobResources->idGPUs[jobGPU]][0];
                jobMonitor->gpuTemperature[jobGPU][step] = schInfo->gpuTemperature[job->jobControl->jobResources->idGPUs[jobGPU]][0];
                jobMonitor->gpuEnergyConsumption[jobGPU][step] = schInfo->gpuPower[job->jobControl->jobResources->idGPUs[jobGPU]][0];
                jobMonitor->gpuPCIeThroughput[jobGPU][step] = schInfo->gpuPCIeThroughput[job->jobControl->jobResources->idGPUs[jobGPU]][0];
            }
            
            // update step
            jobMonitor->step++;


            // reconfiguration decisions are only took if a threshold is surpassed
            if(jobMonitor->step >= jobMonitor->steps){ // wait 5 seconds before taking decisions about what to do


                int jobReconfigured = 0;

                // get weighted mean GPU utilization
                double meanUsage = 0.0; // mean usage in the last steps time steps
                double w = 1.0; // initial weight

                for(size_t stp = 0; stp<jobMonitor->steps; stp++){
                    
                    size_t idx;

                    if(stp > step)
                        idx = jobMonitor->steps - stp - step;
                    else
                        idx = (step - stp) % jobMonitor->steps;

                    
                    for(size_t jobGPU = 0; jobGPU < job->jobControl->jobResources->nGPUs; jobGPU++){

                        
                        double us = (double)(jobMonitor->gpuUsage[jobGPU][idx]);
                        meanUsage += w * us; 
                    }

                    // update w
                    w -= 1.0 / (double)jobMonitor->steps;
                }  
                meanUsage /= ((double)jobMonitor->steps * (double)job->jobControl->jobResources->nGPUs); // divide the utilization by the number of GPUs * steps


                // if the mean usage is less than of %70, shrink
                if(jobResources->nGPUs > 1 && (meanUsage < 70.0)){
                                    
                    // new job resources
                    jobResources_t *reconfJobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));

                    // divide the number of GPUs by 2 for the new number of GPUs
                    reconfJobResources->nGPUs = jobResources->nGPUs / 2;
                    reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));

                    // maintain the GPUs // TODO: NON TOPOLOGY AWARE
                    for(size_t i = 0; i<reconfJobResources->nGPUs; i++){

                        reconfJobResources->idGPUs[i] = jobResources->idGPUs[i];
                    }

                    // allocate new resources and deallocate WHEN reconfiguration is FINISHED
                    // do not call allocate, since the resources are the same as the previous ones
                    //allocateResources(schInfo, reconfJobResources);

                    // schedule the reconfiguration
                    scheduleReconfiguration(job, iJob, reconfJobResources);
                
                    // job reconfigured
                    jobReconfigured = 1;

                    printf(" -- [RMS] Mean GPU usage of job %zu is %lf, so it will be shrinked (from %zu to %zu)\n", 
                        job->jobId, meanUsage, jobResources->nGPUs, reconfJobResources->nGPUs);
                }
                // if the mean usage is more than 90 and there are 
                else if(jobResources->nGPUs < schInfo->nGPUs && schInfo->nAvGPUs >= (2 * jobResources->nGPUs) && meanUsage > 90.0){
                   
                    // the previously allocated resources are maintained, but it is necessary to allocate the new ones
                    jobResources_t *reconfJobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t)); // final job resources
                    jobResources_t *reconfJobNewResources = (jobResources_t*)calloc(1, sizeof(jobResources_t)); // new job resources

                    // divide the number of GPUs by 2 for the new number of GPUs
                    reconfJobResources->nGPUs = jobResources->nGPUs * 2;
                    reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));


                    // new resources
                    reconfJobNewResources->nGPUs = reconfJobResources->nGPUs - jobResources->nGPUs;
                    reconfJobNewResources->idGPUs = (size_t*)calloc(reconfJobNewResources->nGPUs, sizeof(size_t));

                    // find the new resources
                    selectFirstAvailableGPUs(reconfJobNewResources->idGPUs, reconfJobNewResources->nGPUs, schInfo);

                    // allocate only new resources
                    allocateResources(schInfo, reconfJobNewResources);


                    // merge new and old resources
                    size_t idx1 = 0;
                    size_t idx2 = 0;

                    size_t n = 0;
                    // loop until both sets are merged in an ordered way
                    while(n < reconfJobResources->nGPUs){

                        if(idx1 < jobResources->nGPUs && jobResources->idGPUs[idx1] < reconfJobNewResources->idGPUs[idx2]){

                            reconfJobResources->idGPUs[n] = jobResources->idGPUs[idx1];
                            idx1++;
                        }
                        else if (idx2 < reconfJobNewResources->nGPUs){

                            reconfJobResources->idGPUs[n] = reconfJobNewResources->idGPUs[idx2];
                            idx2++;
                        }

                        n++;
                    }
                                        
                    // schedule the reconfiguration
                    scheduleReconfiguration(job, iJob, reconfJobResources);
                
                    // job reconfigured
                    jobReconfigured = 1;

                    printf(" -- [RMS] Mean GPU usage of job %zu is %lf, so it will be shrinked (from %zu to %zu)\n", 
                        job->jobId, meanUsage, jobResources->nGPUs, reconfJobResources->nGPUs);
                }


                // if job has been reconfigured, reinitialize step
                if(jobReconfigured){
                    
                    // reinitialize monitor (steps and GPU usage) : [TODO]: refactorize to a function?
                    jobMonitor->step = 0; 
                    
                    /*for(size_t jobGPU = 0; jobGPU < job->jobControl->jobResources->nGPUs; jobGPU++){

                        jobMonitor->gpuUsage[jobGPU] = 0;
                    }*/
                }
            }
        }
    }
}

void reconf(schInfo_t *schInfo){

    schInfo->rconf(schInfo);
}
