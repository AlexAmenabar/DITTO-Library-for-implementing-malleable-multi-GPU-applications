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

    if(timeStep == 0)
        timeStep = 1;

    return timeStep;
}

size_t computeJobSize(size_t minN, size_t maxN){

    std::mt19937_64 rngN(std::random_device{}());
    std::uniform_int_distribution<size_t> distN(minN, maxN);
    //return distN(rngN);
    
    size_t diff = maxN / minN;
    size_t mult = rand() % diff;

    if(mult == 0) mult = 1;

    return minN * mult;
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
    int nJobs = strtoul(argv[2], NULL, 10); // total number of jobs to generate
    float pMoldableJobs = strtof(argv[3], NULL); // proportion of malleable jobs
    float pMalleableJobs = strtof(argv[4], NULL); // proportion of moldable jobs
    float pFlexibleJobs = strtof(argv[5], NULL); // proportion of flexible or adaptive jobs
    float pCorrectRequests = strtof(argv[6], NULL); // probability for a job to have an 'incorrect' resource request

    float pPhasesJobs = strtof(argv[7], NULL); // proportion of jobs that rely on phases
    float pCommunicationJobs = strtof(argv[8], NULL); // proportion of jobs that have communication
    
    int nGPUs = strtoul(argv[9], NULL, 10); // number of GPUs in the system
    size_t tDiffMax = (size_t)strtoul(argv[10], NULL, 10); // maximum time difference between two ssubsequent jobs
    size_t tDiffMin = 1; // minimum time difference between the arrivals of two subsequent jobs

    // files for storing workloads
    const char *fileName, *fileName1, *fileName2, *fileName3, *fileName4;
    fileName = argv[11]; // file to store the workload

    if(generateAll){
        fileName1 = argv[12]; // file to store the workload
        fileName2 = argv[13]; // file to store the workload
        fileName3 = argv[14]; // file to store the workload
        fileName4 = argv[15]; // file to store the workload
    }
    

    // open output file in write mode
    FILE *f = fopen(fileName, "w");
    fprintf(f, "%d\n\n", nJobs); // store the number of jobs in the workload


    FILE *f1, *f2, *f3, *f4;
    if(generateAll){

        f1 = fopen(fileName1, "w");
        f2 = fopen(fileName2, "w");
        f3 = fopen(fileName3, "w");
        f4 = fopen(fileName4, "w"); 
        fprintf(f1, "%d\n\n", nJobs); // store the number of jobs in the workload
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
    size_t minN[3] = {(size_t)1048576 * (size_t)256 / (size_t)4, (size_t)1048576 * (size_t)512 / (size_t)4, (size_t)1048576 * (size_t)1024 / (size_t)4}; // SMALL; MEDIAN; LARGE JOBS
    size_t maxN[3] = {(size_t)1048576 * (size_t)512 / (size_t)4, (size_t)1048576 * (size_t)1024 / (size_t)4, (size_t)1048576 * (size_t)2024 / (size_t)4}; // SMALL; MEDIAN; LARGE JOBS
    size_t T[4] = {500, 1000, 2000, 3000}; // SMALL; MEDIAN; LARGE JOBS
    size_t K[5] = {100, 150, 250, 500, 800}; // constant for all jobs types
    size_t goodGPUs[4] = {1, 2, 4, 8}; // SMALL; MEDIAN; LARGE JOBS (a different request is 'bad', underutilization or overutilization)
    size_t cpuK[4] = {1, 2, 4, 6}; // constant for all jobs types
    size_t P[2] = {1, 2};

    size_t kT[4] = {50, 100, 250, 500};

    //size_t minN[3] = {80, 160, 320}; // SMALL; MEDIAN; LARGE JOBS
    //size_t maxN[3] = {160, 320, 640}; // SMALL; MEDIAN; LARGE JOBS
    //size_t T[5] = {1, 2, 3, 4, 5}; // SMALL; MEDIAN; LARGE JOBS
    //size_t K[6] = {1, 2, 3, 4, 5, 6}; // constant for all jobs types
    //size_t goodGPUs[4] = {1, 2, 4, 8}; // SMALL; MEDIAN; LARGE JOBS (a different request is 'bad', underutilization or overutilization)
    //size_t cpuK[6] = {1, 2, 4, 8, 16, 32}; // constant for all jobs types
    //size_t P[4] = {1, 2, 4, 8};


    // job arrival time step
    size_t arrival = 0;
        
    // helper variables
    size_t gpuRequest, mingpuRequest;

    // iterative, phases, communication
    size_t argcAppType[3] = {9, 11, 10}; // FVF   (F: fixed; V: variable) 

    // randomize
    srand(time(0));


    // loop and loop controller
    int nJobsPerType[4];
    nJobsPerType[0] = nRigidJobs;
    nJobsPerType[1] = nMoldableJobs;
    nJobsPerType[2] = nMalleableJobs;
    nJobsPerType[3] = nFlexibleJobs;

    int nAppsPerType[3];
    nAppsPerType[0] = nIterativeJobs;
    nAppsPerType[1] = nPhasesJobs;
    nAppsPerType[2] = nCommunicationJobs;
    

    // generate workload
    int nJobsToGen = nJobs;
    while(nJobsToGen>0){

        // job type
        size_t jobType = rand() % 4;
        while(nJobsPerType[jobType] == 0)
            jobType = rand() % 4;

        nJobsPerType[jobType]--;


        size_t appType = rand() % 3; // temp, all apps are iterative //rand() % 3;
        while(nAppsPerType[appType] == 0)
            appType = rand() % 3;

        nAppsPerType[appType]--;

        // priority
        size_t priority = 1;

        // job size (this affects GPU usage)
        int jobSize = rand() % 3; // 0: small; 1: medium; 2: large

        // global job duration
        int jobDuration = rand() % 4;
        
        if(jobSize == 2 && jobDuration == 3)
            jobDuration = 2;
        
        // kernel execution duration
        int kernelDuration = rand() % 5;
        if(jobSize == 2 && kernelDuration == 4)
            kernelDuration = 3;


        // resources (number of requested GPUs and minimum requested GPUs)
        size_t iGPURequest = rand() % 4;
        gpuRequest = goodGPUs[iGPURequest];

        iGPURequest = rand() % 4;
        mingpuRequest = goodGPUs[iGPURequest];

        if(mingpuRequest > gpuRequest && gpuRequest > 1)
            mingpuRequest = gpuRequest / 2;
        else if(mingpuRequest > gpuRequest && gpuRequest == 1)
            mingpuRequest = 1;
        
            
        if(gpuRequest > nGPUs) gpuRequest = nGPUs;
        if(mingpuRequest > nGPUs) mingpuRequest = nGPUs;

        // arrival time step
        arrival = computeJobArrival(arrival, tDiffMin, tDiffMax);

        // estimated duration (this depends on the size of the job, currently I am not able to compute it)
        size_t estimatedDuration = 0;

        // if a separate workload for each job type is necessary
        if(generateAll){
            
            // job type, priority, resources, min resources, arrival time step, estimated duration, app type, argc, argv 
            fprintf(f1,  "0\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n", priority, gpuRequest, mingpuRequest, arrival, estimatedDuration, appType); 
            fprintf(f2, "1\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n", priority, gpuRequest, mingpuRequest, arrival, estimatedDuration, appType); 
            fprintf(f3, "2\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n", priority, gpuRequest, mingpuRequest, arrival, estimatedDuration, appType); 
            fprintf(f4, "3\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n", priority, gpuRequest, mingpuRequest, arrival, estimatedDuration, appType);
        }            
        // job type, priority, resources, min resources, arrival time step, estimated duration, app type, argc, argv 
        fprintf(f, "%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n", jobType, priority, gpuRequest, mingpuRequest, arrival, estimatedDuration, appType);


        // compute exact job size (N)
        size_t N = computeJobSize(minN[jobSize], maxN[jobSize]);
        //N = 80;

        size_t vT = T[jobDuration];
        //vT = 1;

        size_t vK = K[kernelDuration];
        //vK = 1;

        size_t s = 1; // temporal

        // compute the argc of the job
        size_t jargc = argcAppType[appType];


        // communication variables (TEMP)
        size_t pinned = 0; // non-pinned
        size_t async = 1; // sync
        size_t commSteps = 0; // one step
        size_t cores = 0; // single-core

        // argc, argv
        if(appType == 0){

            fprintf(f, "%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n1\n\n", jargc, N, vT, vK, s, pinned, async, commSteps, cores);

            if(generateAll){

                fprintf(f1, "%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n1\n\n", jargc, N, vT, vK, s, pinned, async, commSteps, cores);
                fprintf(f2, "%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n1\n\n", jargc, N, vT, vK, s, pinned, async, commSteps, cores);
                fprintf(f3, "%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n1\n\n", jargc, N, vT, vK, s, pinned, async, commSteps, cores);
                fprintf(f4, "%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n1\n\n", jargc, N, vT, vK, s, pinned, async, commSteps, cores);
            }
        }
        else if (appType == 1){
            
            size_t cpuDuration = rand() % 100;
            size_t phases = rand() % 2;
            jargc += phases;

            fprintf(f, "%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n1\n", jargc, N, vT, vK, cpuDuration, phases, s, pinned, async, commSteps, cores);
            
            if(generateAll){

                fprintf(f1, "%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n1\n", jargc, N, vT, vK, cpuDuration, phases, s, pinned, async, commSteps, cores);
                fprintf(f2, "%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n1\n", jargc, N, vT, vK, cpuDuration, phases, s, pinned, async, commSteps, cores);
                fprintf(f3, "%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n1\n", jargc, N, vT, vK, cpuDuration, phases, s, pinned, async, commSteps, cores);
                fprintf(f4, "%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n1\n", jargc, N, vT, vK, cpuDuration, phases, s, pinned, async, commSteps, cores);
            }

            for(size_t p = 0; p<phases; p++){
                
                size_t val = p % 2;
                fprintf(f, "%zu\n", val);

                if(generateAll){

                    fprintf(f1, "%zu\n", val);
                    fprintf(f2, "%zu\n", val);
                    fprintf(f3, "%zu\n", val);
                    fprintf(f4, "%zu\n", val);
                }
            }
        }
        else if(appType == 2){

            size_t it[5] = {1, 5, 10, 100, 1000};
            size_t nIterationsPerComm = it[rand() % 5];

            if(nIterationsPerComm > vT){
                nIterationsPerComm = vT - 1;
            }
            if(nIterationsPerComm == 0)
                nIterationsPerComm = 1;

            fprintf(f, "%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n1\n\n", jargc, N, vT, vK, nIterationsPerComm, s, pinned, async, commSteps, cores);

            if(generateAll){

                fprintf(f1, "%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n1\n\n", jargc, N, vT, vK, nIterationsPerComm, s, pinned, async, commSteps, cores);
                fprintf(f2, "%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n1\n\n", jargc, N, vT, vK, nIterationsPerComm, s, pinned, async, commSteps, cores);
                fprintf(f3, "%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n1\n\n", jargc, N, vT, vK, nIterationsPerComm, s, pinned, async, commSteps, cores);
                fprintf(f4, "%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n1\n\n", jargc, N, vT, vK, nIterationsPerComm, s, pinned, async, commSteps, cores);
            }
        }

        nJobsToGen --;
    }
}