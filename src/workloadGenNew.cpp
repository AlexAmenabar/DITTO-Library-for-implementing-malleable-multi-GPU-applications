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

    float pNBody = strtof(argv[7], NULL); // proportion of jobs that rely on phases
    float pFFT = strtof(argv[8], NULL); // proportion of jobs that have communication
    
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

    int nFFT = (int)((float)nJobs * pFFT);
    int nNBody = (int)((float)nJobs * pNBody);
    int nJacobi = nJobs - nFFT - nNBody;

    // check
    printf( " -- Number of each type jobs: %d, %d, %d, %d\n", nRigidJobs, nMoldableJobs, nMalleableJobs, nFlexibleJobs);
    printf( " -- Number of each type apps: %d, %d, %d\n", nJacobi, nNBody, nFFT);


    // job arrival time step
    size_t arrival = 0;
        
    // jacobi, nbody, FFT
    size_t argcAppType[3] = {9, 8, 9}; //

    // randomize
    srand(time(0));

    // loop and loop controller
    int nJobsPerType[4];
    nJobsPerType[0] = nRigidJobs;
    nJobsPerType[1] = nMoldableJobs;
    nJobsPerType[2] = nMalleableJobs;
    nJobsPerType[3] = nFlexibleJobs;

    int nAppsPerType[3];
    nAppsPerType[0] = nJacobi;
    nAppsPerType[1] = nNBody;
    nAppsPerType[2] = nFFT;

    // helper variables
    size_t gpuRequest, mingpuRequest;

    size_t probGPUs[4] = {65, 85, 95, 100};

    // generate workload
    int nJobsToGen = nJobs;
    while(nJobsToGen>0){
    
        // job type (fixed, moldable...)
        size_t jobType = rand() % 4;
        while(nJobsPerType[jobType] == 0)
            jobType = rand() % 4;

        nJobsPerType[jobType]--;

        // jacobi, nbody, fft
        size_t appType = rand() % 3; // temp, all apps are iterative //rand() % 3;
        while(nAppsPerType[appType] == 0)
            appType = rand() % 3;

        nAppsPerType[appType]--;

        // priority
        size_t priority = 1;

        // job size
        size_t nN, nN2, nT;
        size_t N, N2, T;

        if(appType == 0){

            nN = 3;
            nN2 = 4;
            nT = 3;
            size_t arrN[3] = {(size_t)512, (size_t)1024, (size_t)2048};
            size_t arrN2[4] = {(size_t)65536, (size_t)131072, (size_t)262144, (size_t)524288};
            size_t arrT[3] = {(size_t)10000, (size_t)50000, (size_t)100000};

            // select N, N2 and T
            size_t nIndex = rand() % nN;
            size_t tIndex = rand() % nT;
            size_t n2Index = rand() % nN2;

            N = arrN[nIndex];
            N2 = arrN2[n2Index];
            T = arrT[tIndex];
        }
        else if(appType == 1){

            nN = 4;
            nN2 = 0;
            nT = 3;
            size_t arrN[4] = {(size_t)16384, (size_t)65536, (size_t)131072, (size_t)262144};
            size_t arrT[3] = {(size_t)1000, (size_t)2000, (size_t)3000};

            // select N, N2 and T
            size_t nIndex = rand() % nN;
            size_t tIndex = rand() % nT;

            N = arrN[nIndex];
            T = arrT[tIndex];
        }
        else if(appType == 2){

            nN = 4;
            nN2 = 4;
            nT = 4;
            size_t arrN[4] = {(size_t)1024, (size_t)2048, (size_t)4096, (size_t)8192};
            size_t arrN2[4] = {(size_t)1024, (size_t)2048, (size_t)4096, (size_t)8192};
            size_t arrT[4] = {(size_t)10000, (size_t)15000, (size_t)20000, (size_t)25000};

            // select N, N2 and T
            size_t nIndex = rand() % nN;
            size_t tIndex = rand() % nT;

            N = arrN[nIndex];
            N2 = arrN2[nIndex];
            T = arrT[tIndex];            
        }

        size_t s = 1; 
        size_t jargc = argcAppType[appType];
        size_t pinned = 0; // non-pinned
        size_t async = 1; // sync
        size_t commSteps = 0; // one step
        size_t cores = 0; // single-core

        // select min and max number of GPUs
        size_t probValue = rand() % 100;
        if(probValue <= probGPUs[0]){

            gpuRequest = 1;
            mingpuRequest = 1;
        }
        else if(probValue > probGPUs[0] && probValue <= probGPUs[1]){

            gpuRequest = 2;
            mingpuRequest = rand() % 2 + 1; // 1 or 2
        }
        else if(probValue > probGPUs[1] && probValue <= probGPUs[2]){

            gpuRequest = 4;

            size_t val = rand() % 100;
            if(val <= 25){
                mingpuRequest = 4;
            }
            else if(val <= 70){
                mingpuRequest = 2;
            }
            else{
                mingpuRequest = 1;
            }
        }
        else{

            gpuRequest = 8;

            size_t val = rand() % 100;
            
            if(val <= 15){
                mingpuRequest = 8;
            }
            if(val <= 60){
                mingpuRequest = 4;
            }
            else if(val<=85){
                mingpuRequest = 2;
            }
            else 
                mingpuRequest = 1;
        }

        // arrival time step
        arrival = computeJobArrival(arrival, tDiffMin, tDiffMax);

        // estimated duration (this depends on the size of the job, currently I am not able to compute it)
        size_t estimatedDuration = 0;


        // job type, priority, resources, min resources, arrival time step, estimated duration, app type, argc, argv 
        fprintf(f, "%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n", jobType, priority, gpuRequest, mingpuRequest, arrival, estimatedDuration, appType);
        
        if(generateAll){
            
            // job type, priority, resources, min resources, arrival time step, estimated duration, app type, argc, argv 
            fprintf(f1,  "0\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n", priority, gpuRequest, mingpuRequest, arrival, estimatedDuration, appType); 
            fprintf(f2, "1\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n", priority, gpuRequest, mingpuRequest, arrival, estimatedDuration, appType); 
            fprintf(f3, "2\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n", priority, gpuRequest, mingpuRequest, arrival, estimatedDuration, appType); 
            fprintf(f4, "3\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n", priority, gpuRequest, mingpuRequest, arrival, estimatedDuration, appType);
        }            

        // app type dependant
        if(appType == 0 || appType == 2){

            fprintf(f, "%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n1\n\n", jargc, N, N2, T, s, pinned, async, commSteps, cores);

            if(generateAll){

                fprintf(f1, "%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n1\n\n", jargc, N, N2, T, s, pinned, async, commSteps, cores);
                fprintf(f2, "%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n1\n\n", jargc, N, N2, T, s, pinned, async, commSteps, cores);
                fprintf(f3, "%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n1\n\n", jargc, N, N2, T, s, pinned, async, commSteps, cores);
                fprintf(f4, "%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n1\n\n", jargc, N, N2, T, s, pinned, async, commSteps, cores);
            }
        }
        else if (appType == 1){
            
            fprintf(f, "%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n1\n\n", jargc, N, T, s, pinned, async, commSteps, cores);

            if(generateAll){

                fprintf(f1, "%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n1\n\n", jargc, N, T, s, pinned, async, commSteps, cores);
                fprintf(f2, "%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n1\n\n", jargc, N, T, s, pinned, async, commSteps, cores);
                fprintf(f3, "%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n1\n\n", jargc, N, T, s, pinned, async, commSteps, cores);
                fprintf(f4, "%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n%zu\n1\n\n", jargc, N, T, s, pinned, async, commSteps, cores);
            }
        }

        nJobsToGen --;
    }
}