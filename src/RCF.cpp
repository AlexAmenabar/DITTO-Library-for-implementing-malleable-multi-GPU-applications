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


typedef struct monitorData_t {

    double meanUsage;

} monitorData_t;



// load data of the time step in the job monitor (from the global monitor)
size_t loadMonitorData(schInfo_t *schInfo, job_t *job){

    /*pthread_mutex_lock(&(printLock));
    printf(" -- Analysing malleability decisions: \n");
    pthread_mutex_unlock(&(printLock));*/

    // get job monitor
    jobMonitoring_t *jobMonitor = job->jobMonitor;
    size_t step = jobMonitor->step % jobMonitor->steps; // current step

    // collect information monitored by the monitoring thread // TODO: probably it is mandatory to add locks
    for(size_t jobGPU = 0; jobGPU < job->jobControl->jobResources->nGPUs; jobGPU++){

        size_t gpuIndex = job->jobControl->jobResources->idGPUs[jobGPU];
        int monitorIndex = (int)schInfo->monitorIndex - 1;
        if(monitorIndex < 0) 
            monitorIndex = JOB_MONITOR_STEPS-1;

        // store results for computing the mean

        // store results of the current time step
        jobMonitor->gpuUsage[jobGPU][step] = schInfo->gpuUtilization[gpuIndex][monitorIndex];
        jobMonitor->gpuTemperature[jobGPU][step] = schInfo->gpuTemperature[gpuIndex][monitorIndex];
        jobMonitor->gpuEnergyConsumption[jobGPU][step] = schInfo->gpuPower[gpuIndex][monitorIndex];
        jobMonitor->gpuPCIeThroughput[jobGPU][step] = schInfo->gpuPCIeThroughput[gpuIndex][monitorIndex];
    }
    
    // update step
    jobMonitor->step++;

    return step;
}


monitorData_t computeMeanUsage(job_t *job, jobMonitoring_t *jobMonitor, size_t step){

    monitorData_t monitorData;

    // get weighted mean GPU utilization
    double meanUsage = 0.0; // mean usage in the last steps time steps
    double w = 1.0; // initial weight
    double accW = 0.0;

    for(size_t stp = 0; stp<jobMonitor->steps; stp++){
        
        size_t idx;

        if(stp > step)
            idx = jobMonitor->steps - (stp - step);
        else
            idx = step - stp;

        for(size_t jobGPU = 0; jobGPU < job->jobControl->jobResources->nGPUs; jobGPU++){

            
            double us = (double)(jobMonitor->gpuUsage[jobGPU][idx]);
            meanUsage += w * us; 
        }

        // update w
        accW += w;
        w -= 1.0 / (double)jobMonitor->steps;
    }

    meanUsage /= ((double)accW * (double)job->jobControl->jobResources->nGPUs);
    //meanUsage /= ((double)jobMonitor->steps * (double)job->jobControl->jobResources->nGPUs); // divide the utilization by the number of GPUs * steps

    /*pthread_mutex_lock(&(printLock));
    printf("\n -- Mean usage = %lf, nGPUs = %zu, av. resources = %zu\n", meanUsage, job->jobControl->jobResources->nGPUs, schInfo->nAvGPUs);
    fflush(stdout);
    pthread_mutex_unlock(&(printLock));*/

    monitorData.meanUsage = meanUsage;
    return monitorData;
}


size_t computeGPUTopologyScore(schInfo_t *schInfo, size_t nGPUs, size_t *idGPUs, size_t gpuIndex){

    size_t score = 0;
    for(size_t i = 0; i<nGPUs; i++){

        size_t topoGPUIndex1 = idGPUs[gpuIndex];
        size_t topoGPUIndex2 = idGPUs[i];
        score += schInfo->gpuTopology[topoGPUIndex1 * schInfo->nGPUs + topoGPUIndex2];
    }
    return score;
}

size_t computeTopologyScore(schInfo_t *schInfo, size_t nGPUs, size_t *idGPUs){

    size_t score = 0;
    for(size_t i = 0; i<nGPUs; i++){

        score += computeGPUTopologyScore(schInfo, nGPUs, idGPUs, i);        
    }
    return score;
}


/*int improveN2NTopology(schInfo_t *schInfo, jobResources_t *jobResources, jobResources_t *reconfJobResources){

    // flag indicating if finished
    int finish = 0;
    int jobReconfigured = 0;

    // compute job score for current configuration
    size_t currentScore;
    size_t prevGPUScore = 99999;

    // loop and improve topology, until no improvements can be done
    while(!finish){
        
        // update current topoology score
        currentScore = computeTopologyScore(schInfo, reconfJobResources->nGPUs, reconfJobResources->idGPUs);

        // flag indicating if a improvement is achieved
        int improved = 0;

        // initial worst GPU data
        size_t gpuScore = 0;
        size_t gpuIndex = 0;
        size_t gpuId = schInfo->nGPUs;

        // loop over GPUs and find the worst one with worst score
        for(size_t i = 0; i<jobResources->nGPUs; i++){

            // compute the GPU score
            size_t tmpScore = computeGPUTopologyScore(schInfo, reconfJobResources->nGPUs, reconfJobResources->idGPUs, i);
            
            // if the GPU score is worse (higher) than higher found, update 
            if(tmpScore > gpuScore){
                
                gpuScore = tmpScore;
                gpuIndex = i; // index of the worse GPU
                gpuId = reconfJobResources->idGPUs[i]; // store old GPU
            }
        }


        // check whether it can be substituted by a better GPU
        size_t bestReconfGPU = schInfo->nGPUs; // no GPU selected 
        size_t bestScore = currentScore; // current score is the best score obtained until now

        // loop over system GPUs and check if they improve the topology score
        for(size_t i = 0; i<schInfo->nGPUs; i++){

            // if GPU is available, compute score
            if(schInfo->avGPUs[i]){

                // update GPU in reconf job resources
                reconfJobResources->idGPUs[gpuIndex] = i;

                // compute new topology score
                size_t tmpScore = computeTopologyScore(schInfo, reconfJobResources->nGPUs, reconfJobResources->idGPUs); 
                
                // check if this configuration is better
                if(tmpScore < bestScore){

                    bestReconfGPU = i;
                    bestScore = tmpScore;
                }
            }
        }

        // if the best score is better than the current score, update
        if(bestScore < currentScore){

            reconfJobResources->idGPUs[gpuIndex] = bestReconfGPU;
            improved = 1;

            // indicate allocation
            schInfo->avGPUs[gpuId] = 1; // now this is available
            schInfo->avGPUs[bestReconfGPU] = 0; // the selected new GPU is no longer available

            // there is a pending reconfiguration
            jobReconfigured = 1;
            prevGPUScore = gpuScore;
        }
        else{

            // set the previous GPU
            reconfJobResources->idGPUs[gpuIndex] = gpuId;
            prevGPUScore = gpuScore;
        }
        
        if(!improved){
            finish = 1;
        }
    }
    return jobReconfigured;
}*/

int improveN2NTopology(schInfo_t *schInfo, jobResources_t *jobResources, jobResources_t *reconfJobResources){

    int avGroup;
    size_t nGPUs = jobResources->nGPUs;

    // loop over groups
    if(nGPUs == 2 || nGPUs == 4){

        for(size_t group = 0; group < schInfo->nGPUs / nGPUs; group++){

            avGroup = 1;
            for(size_t gpu = group * nGPUs; gpu < (group+1) * nGPUs; gpu++){

                if(!schInfo->avGPUs[gpu]){
                
                    avGroup = 0;
                }
            }

            if(avGroup){

                for(size_t i = 0; i < nGPUs; i++){
                
                    reconfJobResources->idGPUs[i] = group * nGPUs + i;
                }

                return 1;
            }
        }
    }

    return 0;
}


// Reconfiguration policy: reconfigure any job that does not meet the established utilization conditions
// Shrink selection policy: keep GPUs (ordered)
// Expand allocation policy: keep possible and select first ones (ordered)
void utilization(schInfo_t *schInfo){

    // helper variables
    job_t *job;
    jobResources_t *jobResources;
    jobMonitoring_t *jobMonitor;
    size_t jobType, iJob, nRunningJobs;

    // get queue of running jobs
    jobQueue_t *runningQueue = &(schInfo->runningJobs);

    // loop over running jobs and make reconfiguration decisions
    nRunningJobs = getNumberOfJobsInQueue(&(schInfo->runningJobs));
    for(iJob = 0; iJob<nRunningJobs; iJob++){

        // get job information
        job = getJobFromQueue(runningQueue, iJob);
        jobResources = job->jobControl->jobResources;
        jobType = job->jobLauncher->jobType;
        jobMonitor = job->jobMonitor;

        // REVISE MONITOR
        // TODO: manage the case in which job finished
        // [Code here]

        // check only if job can be reconfigured (malleable | flexible)
        if (jobType > 1){ 

            // check whether the job finished
            int jobFinished = checkJobFinished(job->jobControl);

            // check if job finished
            if(jobFinished){
            
                // finish job
                finishJob(schInfo, job, iJob);

                // update values
                nRunningJobs --;
                iJob--;
            }
            else{

                // load relevant monitored data
                size_t step = loadMonitorData(schInfo, job); 

                if(checkReconfigurationDone(job->jobControl) == 0){

                    // compute mean utilization in the last JOB_MONITOR_STEPS time steps if we have enough mesaurements
                    if(jobMonitor->step >= jobMonitor->steps){ // wait 5 seconds before taking decisions about what to do

                        int jobReconfigured = 0;

                        // get mean utilization of GPUs in the last time steps
                        monitorData_t monitorData = computeMeanUsage(job, jobMonitor, step);
                        double meanUsage = monitorData.meanUsage;

                        // [Decision making]
                        // if the mean usage is less than of %70, shrink
                        if(jobResources->nGPUs > 1 && (meanUsage < 80.0)){
                                            
                            // new job resources
                            jobResources_t *reconfJobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));

                            // divide the number of GPUs by 2 for the new number of GPUs
                            reconfJobResources->nGPUs = jobResources->nGPUs / 2;
                            reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));

                            // maintain the GPUs // TODO: NON TOPOLOGY AWARE
                            for(size_t i = 0; i<reconfJobResources->nGPUs; i++){

                                reconfJobResources->idGPUs[i] = jobResources->idGPUs[i];
                            }
                            
                            // schedule reconfiguration and allocate resources
                            scheduleReconfiguration(schInfo, job, iJob, reconfJobResources);
                            schInfo->nShrinks ++;

                            // job reconfigured
                            jobReconfigured = 1;

                            /*pthread_mutex_lock(&(printLock));
                            //printf(" -- [RMS] Mean GPU usage of job %zu is %lf, shrink scheduled (from %zu to %zu)\n", job->jobId, meanUsage, jobResources->nGPUs, reconfJobResources->nGPUs);                    
                            printf(" Shrink scheduled for job %zu\n", job->jobId);
                            fflush(stdout);
                            pthread_mutex_unlock(&(printLock));*/
                        }
                        // if the mean usage is more than 90 and there are 
                        if(jobResources->nGPUs < schInfo->nGPUs && schInfo->nAvGPUs >= jobResources->nGPUs && meanUsage >= 90.0){
                            
                            // the previously allocated resources are maintained, but it is necessary to allocate the new ones
                            jobResources_t *reconfJobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t)); // final job resources
                            jobResources_t *reconfJobNewResources = (jobResources_t*)calloc(1, sizeof(jobResources_t)); // new job resources

                            // divide the number of GPUs by 2 for the new number of GPUs
                            reconfJobResources->nGPUs = jobResources->nGPUs * 2;
                            reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));

                            // new resources
                            reconfJobNewResources->nGPUs = reconfJobResources->nGPUs - jobResources->nGPUs;
                            reconfJobNewResources->idGPUs = (size_t*)calloc(reconfJobNewResources->nGPUs, sizeof(size_t));

                            // find the available GPUs for expanding
                            selectFirstAvailableGPUs(reconfJobNewResources->idGPUs, reconfJobNewResources->nGPUs, schInfo);

                            // merge new and old resources
                            size_t idx1 = 0;
                            size_t idx2 = 0;

                            size_t n = 0;

                            // loop until both sets are merged in an ordered way
                            while(n < reconfJobResources->nGPUs){

                                if((idx1 < jobResources->nGPUs && idx2 == reconfJobNewResources->nGPUs) || (idx1 < jobResources->nGPUs && jobResources->idGPUs[idx1] < reconfJobNewResources->idGPUs[idx2])){

                                    reconfJobResources->idGPUs[n] = jobResources->idGPUs[idx1];
                                    idx1++;
                                }
                                else if (idx2 < reconfJobNewResources->nGPUs){

                                    reconfJobResources->idGPUs[n] = reconfJobNewResources->idGPUs[idx2];
                                    idx2++;
                                }

                                n++;
                            }
                        
                            // deallocate temporal structure
                            deallocateJobResourcesStruct(&(reconfJobNewResources));

                            // schedule reconfiguration
                            scheduleReconfiguration(schInfo, job, iJob, reconfJobResources);
                            schInfo->nExpands ++;
                            
                            // job reconfigured
                            jobReconfigured = 1;

                            /*pthread_mutex_lock(&(printLock));
                            //printf(" -- [RMS] Mean GPU usage of job %zu is %lf, so it will be expanded (from %zu to %zu)\n", 
                            //    job->jobId, meanUsage, jobResources->nGPUs, reconfJobResources->nGPUs);
                            printf(" Expansion scheduled for job %zu\n", job->jobId);
                            fflush(stdout);
                            pthread_mutex_unlock(&(printLock));*/
                        }


                        // if job has been reconfigured, reinitialize step
                        if(jobReconfigured){
                            
                            // something to do here?
                            // reinitialize monitor (steps and GPU usage)
                            jobMonitor->step = 0; 
                            schInfo->nReconfigurations ++;
                        }
                    }
                }
            }
        }
    }
}


// Reconfiguration policy: reconfigure any job that does not meet the established utilization conditions
// Shrink selection policy: keep GPUs (ordered)
// Expand allocation policy: keep possible and select first ones (ordered)
void topology(schInfo_t *schInfo){

    // helper variables
    job_t *job;
    jobResources_t *jobResources;
    jobMonitoring_t *jobMonitor;
    size_t jobType, iJob, nRunningJobs;

    // get queue of running jobs
    jobQueue_t *runningQueue = &(schInfo->runningJobs);

    // loop over running jobs and make reconfiguration decisions
    nRunningJobs = getNumberOfJobsInQueue(&(schInfo->runningJobs));
    for(iJob = 0; iJob<nRunningJobs; iJob++){

        // get job information
        job = getJobFromQueue(runningQueue, iJob);
        jobResources = job->jobControl->jobResources;
        jobType = job->jobLauncher->jobType;
        jobMonitor = job->jobMonitor;

        // REVISE MONITOR
        // TODO: manage the case in which job finished
        // [Code here]

        // check only if job can be reconfigured (malleable | flexible)
        if (jobType > 1){ 

            // check whether the job finished
            int jobFinished = checkJobFinished(job->jobControl);

            // check if job finished
            if(jobFinished){
            
                // finish job
                finishJob(schInfo, job, iJob);

                // update values
                nRunningJobs --;
                iJob--;
            }
            else{

                // load relevant monitored data
                size_t step = loadMonitorData(schInfo, job); 

                // schedule a reconfiguration only if job already finished a previous reconfiguration
                if(checkReconfigurationDone(job->jobControl) == 0){

                    // compute mean utilization in the last JOB_MONITOR_STEPS time steps if we have enough mesaurements
                    if(jobMonitor->step >= jobMonitor->steps){ // wait 5 seconds before taking decisions about what to do

                        // [Decision making]
                        // if the mean usage is less than of %70, shrink
                        int jobReconfigured = 0;

                        // get mean utilization of GPUs in the last time steps
                        monitorData_t monitorData = computeMeanUsage(job, jobMonitor, step);
                        double meanUsage = monitorData.meanUsage;

                        // [Decision making]
                        // if the mean usage is less than of %70, shrink
                        if(jobResources->nGPUs > 1 && (meanUsage < 80.0)){
                                            
                            // new job resources
                            jobResources_t *reconfJobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));

                            // divide the number of GPUs by 2 for the new number of GPUs
                            reconfJobResources->nGPUs = jobResources->nGPUs / 2;
                            reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));

                            // maintain the GPUs // TODO: NON TOPOLOGY AWARE
                            for(size_t i = 0; i<reconfJobResources->nGPUs; i++){

                                reconfJobResources->idGPUs[i] = jobResources->idGPUs[i];
                            }
                            

                            // check if topologically a better reconfiguration can be done
                            jobResources_t *topoJobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));

                            // divide the number of GPUs by 2 for the new number of GPUs
                            topoJobResources->nGPUs = reconfJobResources->nGPUs;
                            topoJobResources->idGPUs = (size_t*)calloc(topoJobResources->nGPUs, sizeof(size_t));

                            int found = improveN2NTopology(schInfo, reconfJobResources, topoJobResources);

                            // schedule reconfiguration and allocate resources
                            if(!found){
                                scheduleReconfiguration(schInfo, job, iJob, reconfJobResources);
                            }
                            else{
                                scheduleReconfiguration(schInfo, job, iJob, topoJobResources);
                            }

                            schInfo->nShrinks ++;

                            // job reconfigured
                            jobReconfigured = 1;

                            /*pthread_mutex_lock(&(printLock));
                            //printf(" -- [RMS] Mean GPU usage of job %zu is %lf, shrink scheduled (from %zu to %zu)\n", job->jobId, meanUsage, jobResources->nGPUs, reconfJobResources->nGPUs);                    
                            printf(" Shrink scheduled for job %zu\n", job->jobId);
                            fflush(stdout);
                            pthread_mutex_unlock(&(printLock));*/
                        }
                        // if the mean usage is more than 90 and there are 
                        if(jobResources->nGPUs < schInfo->nGPUs && schInfo->nAvGPUs >= jobResources->nGPUs && meanUsage >= 90.0){
                            
                            // the previously allocated resources are maintained, but it is necessary to allocate the new ones
                            jobResources_t *reconfJobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t)); // final job resources
                            jobResources_t *reconfJobNewResources = (jobResources_t*)calloc(1, sizeof(jobResources_t)); // new job resources

                            // divide the number of GPUs by 2 for the new number of GPUs
                            reconfJobResources->nGPUs = jobResources->nGPUs * 2;
                            reconfJobResources->idGPUs = (size_t*)calloc(reconfJobResources->nGPUs, sizeof(size_t));

                            // new resources
                            reconfJobNewResources->nGPUs = reconfJobResources->nGPUs - jobResources->nGPUs;
                            reconfJobNewResources->idGPUs = (size_t*)calloc(reconfJobNewResources->nGPUs, sizeof(size_t));

                            // find the available GPUs for expanding
                            selectFirstAvailableGPUs(reconfJobNewResources->idGPUs, reconfJobNewResources->nGPUs, schInfo);

                            // merge new and old resources
                            size_t idx1 = 0;
                            size_t idx2 = 0;

                            size_t n = 0;

                            // loop until both sets are merged in an ordered way
                            while(n < reconfJobResources->nGPUs){

                                if((idx1 < jobResources->nGPUs && idx2 == reconfJobNewResources->nGPUs) || (idx1 < jobResources->nGPUs && jobResources->idGPUs[idx1] < reconfJobNewResources->idGPUs[idx2])){

                                    reconfJobResources->idGPUs[n] = jobResources->idGPUs[idx1];
                                    idx1++;
                                }
                                else if (idx2 < reconfJobNewResources->nGPUs){

                                    reconfJobResources->idGPUs[n] = reconfJobNewResources->idGPUs[idx2];
                                    idx2++;
                                }

                                n++;
                            }
                        
                            // deallocate temporal structure
                            deallocateJobResourcesStruct(&(reconfJobNewResources));


                            // check if topologically a better reconfiguration can be done
                            jobResources_t *topoJobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t));

                            // divide the number of GPUs by 2 for the new number of GPUs
                            topoJobResources->nGPUs = reconfJobResources->nGPUs;
                            topoJobResources->idGPUs = (size_t*)calloc(topoJobResources->nGPUs, sizeof(size_t));

                            int found = improveN2NTopology(schInfo, reconfJobResources, topoJobResources);

                            // schedule reconfiguration and allocate resources
                            if(!found){
                                scheduleReconfiguration(schInfo, job, iJob, reconfJobResources);
                            }
                            else{
                                scheduleReconfiguration(schInfo, job, iJob, topoJobResources);
                            }
                            schInfo->nExpands ++;
                            
                            // job reconfigured
                            jobReconfigured = 1;

                            /*pthread_mutex_lock(&(printLock));
                            //printf(" -- [RMS] Mean GPU usage of job %zu is %lf, so it will be expanded (from %zu to %zu)\n", 
                            //    job->jobId, meanUsage, jobResources->nGPUs, reconfJobResources->nGPUs);
                            printf(" Expansion scheduled for job %zu\n", job->jobId);
                            fflush(stdout);
                            pthread_mutex_unlock(&(printLock));*/
                        }
                        else{

                            // the previously allocated resources are maintained, but it is necessary to allocate the new ones
                            jobResources_t *topoJobResources = (jobResources_t*)calloc(1, sizeof(jobResources_t)); // final job resources

                            // divide the number of GPUs by 2 for the new number of GPUs
                            topoJobResources->nGPUs = jobResources->nGPUs;
                            topoJobResources->idGPUs = (size_t*)calloc(topoJobResources->nGPUs, sizeof(size_t));

                            int found = improveN2NTopology(schInfo, jobResources, topoJobResources);

                            // schedule reconfiguration and allocate resources
                            if(found){
                                
                                scheduleReconfiguration(schInfo, job, iJob, topoJobResources);

                                // job reconfigured
                                schInfo->nKeeps ++;
                                jobReconfigured = 1;
                            }
                        }


                        // if job has been reconfigured, reinitialize step
                        if(jobReconfigured){
                            
                            // something to do here?
                            // reinitialize monitor (steps and GPU usage)
                            jobMonitor->step = 0; 
                            schInfo->nReconfigurations ++;
                        }
                    }
                }
            }
        }
    }
}

void reconf(schInfo_t *schInfo){

    schInfo->rconf(schInfo);
}
