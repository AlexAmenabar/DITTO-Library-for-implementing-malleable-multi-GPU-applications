#include <cstddef>
#include <omp.h>
#include <unistd.h>
#include <stdio.h>      
#include <stdlib.h>     
#include <time.h>

#include <iostream>
#include <random>


size_t computeJobArrival(size_t timeStep, size_t tDiffMin, size_t tDiffMax){

    std::mt19937_64 rngTDiff(std::random_device{}());
    std::uniform_int_distribution<size_t> distTDiff(tDiffMin, tDiffMax);
    timeStep += distTDiff(rngTDiff);
    //std::cout << timeStep << std::endl;

    return timeStep;
}

size_t computeJobSize(size_t minN, size_t maxN){

    std::mt19937_64 rngN(std::random_device{}());
    std::uniform_int_distribution<size_t> distN(minN, maxN);
    return distN(rngN);
    //std::cout << N << std::endl;
}

void computeRequestedNumberGPUs(size_t* gpuRequest, size_t* mingpuRequest, size_t jobType, size_t request){

    size_t req = 0, minreq = 0;
    
    std::mt19937_64 rngGPUs(std::random_device{}());
    double mean = static_cast<double>(request);
    double stddev = 1.25;
    std::normal_distribution<double> normalGPUs(mean, stddev);

    // rigid
    if(jobType == 0){

        // rejection sampling to keep [1, 8]
        do {
            req = static_cast<int>(std::round(normalGPUs(rngGPUs)));
        } while ((req < 1 || req > 8) || (req & (req-1)) != 0);
        //std::cout << req << std::endl;

        // return request
        (*gpuRequest) = req;
        (*mingpuRequest) = req; 
    }
    else{

        // rejection sampling to keep [1, 8]
        do {
            req = static_cast<int>(std::round(normalGPUs(rngGPUs)));
        } while ((req < 2 || req > 8) || (req & (req-1)) != 0);
        //std::cout << req << std::endl;

        do {
            minreq = static_cast<int>(std::round(normalGPUs(rngGPUs)));
        } while ((minreq < 1 || minreq > req) || (minreq & (minreq-1)) != 0);


        if(req == minreq){

            if(req == 8){

                minreq /= 2;
            }
            else{
                minreq /= 2;

                if(minreq <= 0){
                    minreq = 1;
                }
            }
        }

        // return request
        (*gpuRequest) = req;
        (*mingpuRequest) = minreq; 
    }
}


int main(int argc, char* argv[]){


    // read input paramters
    int generateAll = strtoul(argv[1], NULL, 10);
    int nJobs = strtoul(argv[2], NULL, 10); // total number of jobs
    float pMalleableJobs = strtof(argv[3], NULL); // proportion of malleable jobs
    float pMoldableJobs = strtof(argv[4], NULL); // proportion of moldable jobs
    float pFlexibleJobs = strtof(argv[5], NULL); // proportion of flexible or adaptive jobs
    float pCorrectRequests = strtof(argv[6], NULL); // probability for a job to have an 'incorrect' resource request

    float pPhasesJobs = strtof(argv[7], NULL); // 
    float pCommunicationJobs = strtof(argv[8], NULL); // communication apps are only iterative currently
    
    int nGPUs = strtoul(argv[9], NULL, 10); // number of GPUs in the system
    size_t tDiffMax = (size_t)strtoul(argv[10], NULL, 10); // time step in which the job reaches to the system
    size_t tDiffMin = 1; // two jobs can not reach to the system exactly at the same time step

    const char *fileName, *fileName2, *fileName3, *fileName4;
    fileName = argv[11]; // file to store the workload

    if(generateAll){
        fileName2 = argv[12]; // file to store the workload
        fileName3 = argv[13]; // file to store the workload
        fileName4 = argv[14]; // file to store the workload
    }
    

    // open output file in write mode
    FILE *f = fopen(fileName, "w");
    fprintf(f, "%d\n\n", nJobs); // store the number of jobs in the workload


    FILE *f2, *f3, *f4;
    if(generateAll){

        f2 = fopen(fileName2, "w");
        f3 = fopen(fileName3, "w");
        f4 = fopen(fileName4, "w"); 
        fprintf(f2, "%d\n\n", nJobs); // store the number of jobs in the workload
        fprintf(f3, "%d\n\n", nJobs); // store the number of jobs in the workload
        fprintf(f4, "%d\n\n", nJobs); // store the number of jobs in the workload
    }



    // compute number of each type jobs
    int nMalleableJobs = (int)((float)nJobs * pMalleableJobs); // number of malleable jobs
    int nMoldableJobs = (int)((float)nJobs * pMoldableJobs); // number of moldable jobs
    int nFlexibleJobs = (int)((float)nJobs * pFlexibleJobs); // number of adaptive or flexible jobs
    int nRigidJobs = nJobs - nMalleableJobs - nMoldableJobs - nFlexibleJobs; // compute the number of rigid jobs

    int nCommunicationJobs = (int)((float)nJobs * pCommunicationJobs);
    int nPhasesJobs = (int)((float)nJobs * pPhasesJobs);
    int nIterativeJobs = nJobs - nCommunicationJobs - nPhasesJobs;

    // check
    printf( " -- Number of each type jobs: %d, %d, %d, %d\n", nRigidJobs, nMoldableJobs, nMalleableJobs, nFlexibleJobs);
    printf( " -- Number of each type apps: %d, %d, %d\n", nIterativeJobs, nPhasesJobs, nCommunicationJobs);


    // set bounds for parameters (TEMPORAL, IMPROVE WHEN POSSIBLE)
    size_t minN[3] = {10000, 250000, 1000000}; // SMALL; MEDIAN; LARGE JOBS
    size_t maxN[3] = {100000, 500000, 2000000}; // SMALL; MEDIAN; LARGE JOBS
    size_t T[3] = {500000, 1000000, 2000000}; // SMALL; MEDIAN; LARGE JOBS
    size_t K[3] = {500, 2000, 8000}; // constant for all jobs types
    size_t goodGPUs[3] = {2, 4, 8}; // SMALL; MEDIAN; LARGE JOBS (a different request is 'bad', underutilization or overutilization)


    // job arrival time step
    size_t arrival = 0;
        
    // helper variables
    size_t gpuRequest, mingpuRequest;

    size_t argcAppType[3] = {8, 12, 9}; // FVF   (F: fixed; V: variable) 

    // randomize
    srand(time(0));


    // loop and loop controller

    int nJobsPerType[4];
    nJobsPerType[0] = nRigidJobs;
    nJobsPerType[1] = nMoldableJobs;
    nJobsPerType[2] = nMalleableJobs;
    nJobsPerType[3] = nFlexibleJobs;
    

    int nJobsToGen = nJobs;
    while(nJobsToGen>0){

        // job type
        size_t jobType = rand() % 4;

        // priority
        size_t priority = 1;

        // job size (this affects GPU usage)
        int jobSize = rand() % 3; // 0: small; 1: medium; 2: big
        // global job duration
        int jobDuration = rand() % 3;
        // kernel execution duration
        int kernelDuration = rand() % 3;


        // resources (number of requested GPUs and minimum requested GPUs)
        computeRequestedNumberGPUs(&gpuRequest, &mingpuRequest, jobType, goodGPUs[jobSize]);

        // arrival time step
        arrival = computeJobArrival(arrival, tDiffMin, tDiffMax);

        // estimated duration (this depends on the size of the job, currently I am not able to compute it)
        size_t estimatedDuration = 0;

        size_t appType = 0; // temp, all apps are iterative //rand() % 3;


        // if a separate workload for each job type is necessary
        if(generateAll){
            
            // job type, priority, resources, min resources, arrival time step, estimated duration, app type, argc, argv 
            fprintf(f,  "0\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n", priority, gpuRequest, mingpuRequest, arrival, estimatedDuration, appType); 
            fprintf(f2, "1\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n", priority, gpuRequest, mingpuRequest, arrival, estimatedDuration, appType); 
            fprintf(f3, "2\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n", priority, gpuRequest, mingpuRequest, arrival, estimatedDuration, appType); 
            fprintf(f4, "3\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n", priority, gpuRequest, mingpuRequest, arrival, estimatedDuration, appType);
        }
        else{
            
            // job type, priority, resources, min resources, arrival time step, estimated duration, app type, argc, argv 
            fprintf(f, "%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n", jobType, priority, gpuRequest, mingpuRequest, arrival, estimatedDuration, appType);
        }


        // compute exact job size (N)
        size_t N = computeJobSize(minN[jobSize], maxN[jobSize]);

        // compute the argc of the job
        size_t jargc = argcAppType[appType];

        // if the job has phases, 
        /*if(jobType == 1){

        }*/

        // communication variables (TEMP)
        size_t pinned = 0; // non-pinned
        size_t async = 0; // sync
        size_t commSteps = 0; // one step
        size_t cores = 0; // single-core


        // write argc and argv            
        //for(size_t arg = 0; arg<jargc; arg++){

        // argc, argv
        fprintf(f, "%zu\n%zu\n%zu\n%zu\n0\n%zu\n%zu\n%zu\n%zu\n\n", jargc, N, T[jobSize], K[kernelDuration], pinned, async, commSteps, cores);

        if(generateAll){

            fprintf(f2, "%zu\n%zu\n%zu\n%zu\n0\n%zu\n%zu\n%zu\n%zu\n\n", jargc, N, T[jobSize], K[kernelDuration], pinned, async, commSteps, cores);
            fprintf(f3, "%zu\n%zu\n%zu\n%zu\n0\n%zu\n%zu\n%zu\n%zu\n\n", jargc, N, T[jobSize], K[kernelDuration], pinned, async, commSteps, cores);
            fprintf(f4, "%zu\n%zu\n%zu\n%zu\n0\n%zu\n%zu\n%zu\n%zu\n\n", jargc, N, T[jobSize], K[kernelDuration], pinned, async, commSteps, cores);
        }

        nJobsToGen --;
        nJobsPerType[jobType]--;
        //}


        /*// if a separate workload for each job type is necessary
        if(generateAll){
            
            // rigid, priority, resources, (min resources), arrival time step, estimated duration, app type, argc, argv 
            fprintf(f,  "0\n%zu\n%zu\n%zu\n0\n5\n%zu\n%zu\n%zu\n\n", priority, gpuRequest, arrival, appType, argcAppType[appType], N, T[jobSize], K[kernelDuration]);
            fprintf(f2, "1\n%zu\n%zu\n%zu\n0\n5\n%zu\n%zu\n%zu\n\n", priority, gpuRequest, arrival, appType, N, T[jobSize], K[kernelDuration]);
            fprintf(f3, "2\n%zu\n%zu\n%zu\n0\n5\n%zu\n%zu\n%zu\n\n", priority, gpuRequest, arrival, appType, N, T[jobSize], K[kernelDuration]);
            fprintf(f4, "3\n%zu\n%zu\n%zu\n0\n5\n%zu\n%zu\n%zu\n\n", priority, gpuRequest, arrival, appType, N, T[jobSize], K[kernelDuration]);

            nJobsToGen --;
        }
        else if(jobType == 0 && nRigidJobs > 0){

            // rigid, priority, resources, arrival time step, estimated duration, app type, argc, argv
            fprintf(f, "0\n1\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n\n", gpuRequest, arrival, estimatedDuration, appType, argcAppType[appType], N, T[jobSize], K[kernelDuration]);


            // argv


            nRigidJobs --;
            nJobsToGen --;
        }
        else if(jobType == 1 && nMoldableJobs > 0){

            // malleable, priority, resources, time step, app type            
            fprintf(f, "1\n1\n%d\n%zu\n%zu\n0\n5\n%zu\n%zu\n%zu\n\n", gpuRequest, mingpuRequest, arrival, N, T[jobDuration], K[kernelDuration]);

            nMoldableJobs--;
            nJobsToGen --;
        }
        else if(jobType == 2 && nMalleableJobs > 0){

            // malleable, priority, resources, time step, app type            
            fprintf(f, "2\n1\n%zu\n%zu\n%zu\n0\n5\n%zu\n%zu\n%zu\n\n", gpuRequest, mingpuRequest, arrival, N, T[jobDuration], K[kernelDuration]);

            nMalleableJobs--;
            nJobsToGen --;
        }
        else if(jobType == 3 && nFlexibleJobs > 0){

            // malleable, priority, resources, time step, app type            
            fprintf(f, "3\n1\n%zu\n%zu\n%zu\n0\n5\n%zu\n%zu\n%zu\n\n", gpuRequest, mingpuRequest, arrival, N, T[jobDuration], K[kernelDuration]);

            nFlexibleJobs--;
            nJobsToGen --;
        }*/
    }
}