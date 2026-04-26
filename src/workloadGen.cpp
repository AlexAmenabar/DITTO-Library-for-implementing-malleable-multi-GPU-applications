#include <cstddef>
#include <omp.h>
#include <unistd.h>
#include <stdio.h>      
#include <stdlib.h>     
#include <time.h>

#include <iostream>
#include <random>

int main(int argc, char* argv[]){

    // read input paramters
    const char *fileName = argv[1];
    int nJobs = strtoul(argv[2], NULL, 10);
    float pMalleableJobs = strtof(argv[3], NULL);
    float pMoldableJobs = strtof(argv[4], NULL);
    float pFlexibleJobs = strtof(argv[5], NULL);

    float pCorrectRequests = strtof(argv[6], NULL);

    int nGPUs = strtoul(argv[7], NULL, 10);
    size_t tDiffMax = (size_t)strtoul(argv[8], NULL, 10);
    size_t tDiffMin = 1;



    // open file in write mode
    FILE *f = fopen(fileName, "w");
    fprintf(f, "%d\n\n", nJobs);

    // compute number of malleable and non malleable jobs
    int nMalleableJobs = (int)((float)nJobs * pMalleableJobs);
    int nMoldableJobs = (int)((float)nJobs * pMoldableJobs);
    int nFlexibleJobs = (int)((float)nJobs * pFlexibleJobs);

    int nRigidJobs = nJobs - nMalleableJobs - nMoldableJobs - nFlexibleJobs;

    printf( " -- Number of each type jobs (%d, %d, %d, %d)\n", nRigidJobs, nMoldableJobs, nMalleableJobs, nFlexibleJobs);
    
    // set bounds for parameters
    size_t minN[3] = {10000, 250000, 1000000};
    size_t maxN[3] = {100000, 500000, 2000000};

    size_t T[3] = {500000, 1000000, 2000000};

    size_t K = 2000;


    size_t goodGPUs[3] = {2, 4, 8};


    // number of available GPUs
    size_t avGPUs = (size_t)nGPUs;

    // time step controller
    size_t timeStep = 0;
    
    
    int gpuRequest, mingpuRequest;


    // randomize
    srand(time(0));


    //printf(" Number of jobs = %d\n Percentage of malleable jobs = %f\n Number of malleable jobs = %d\n", nJobs, pMalleableJobs, nMalleableJobs);

    int nJobsToGen = nJobs;

    while(nJobsToGen>0){

        int jobType = rand() % 4;

        //decide job size
        int jobSize = rand() % 3; // 0: small; 1: medium; 2: big
        int jobDuration = rand() % 3;

        std::mt19937_64 rngTDiff(std::random_device{}());
        std::uniform_int_distribution<size_t> distTDiff(tDiffMin, tDiffMax);
        timeStep += distTDiff(rngTDiff);
        std::cout << timeStep << std::endl;

        std::mt19937_64 rngN(std::random_device{}());
        std::uniform_int_distribution<size_t> distN(minN[jobSize], maxN[jobSize]);
        size_t N = distN(rngN);
        std::cout << N << std::endl;


        /* Get number of GPUs */
        std::mt19937_64 rngGPUs(std::random_device{}());
        // pick one of your job sizes (example)
        double mean = static_cast<double>(goodGPUs[jobSize]);
        double stddev = 1.25;
        std::normal_distribution<double> normalGPUs(mean, stddev);

        // rejection sampling to keep [1, 8]
        do {
            gpuRequest = static_cast<int>(std::round(normalGPUs(rngGPUs)));
        } while ((gpuRequest < 2 || gpuRequest > 8) || (gpuRequest & (gpuRequest-1)) != 0);
        std::cout << gpuRequest << std::endl;


        //if(jobType > 0){
            
            // rejection sampling to keep [1, 8]
            do {
                mingpuRequest = static_cast<int>(std::round(normalGPUs(rngGPUs)));
            } while ((mingpuRequest < 1 || mingpuRequest > gpuRequest) || (mingpuRequest & (mingpuRequest-1)) != 0);


            if(gpuRequest == mingpuRequest){

                if(gpuRequest == 8){

                    mingpuRequest /= 2;
                }
                /*else if(mingpuRequest == 1){
                    
                    gpuRequest *= 2;
                }*/
                else{
                    mingpuRequest /= 2;

                    if(mingpuRequest <= 0){
                        mingpuRequest = 1;
                    }
                }
            }
        //}


        if(jobType == 0 && nRigidJobs > 0 || generateAll){

            // rigid, priority, resources, time step, app type
            fprintf(f, "0\n1\n%d\n%zu\n0\n3\n%zu\n%zu\n%zu\n\n", gpuRequest, timeStep, N, T[jobSize], K);

            nRigidJobs --;
            nJobsToGen --;
        }
        else if(jobType == 1 && nMoldableJobs > 0 || generateAll){

            // malleable, priority, resources, time step, app type            
            fprintf(f, "1\n1\n%d\n%d\n%zu\n0\n3\n%zu\n%zu\n%zu\n\n", gpuRequest, mingpuRequest, timeStep, N, T[jobDuration], K);

            nMoldableJobs--;
            nJobsToGen --;
        }
        else if(jobType == 2 && nMalleableJobs > 0 ||generateAll){

            // malleable, priority, resources, time step, app type            
            fprintf(f, "2\n1\n%d\n%d\n%zu\n0\n3\n%zu\n%zu\n%zu\n\n", gpuRequest, mingpuRequest, timeStep, N, T[jobDuration], K);

            nMalleableJobs--;
            nJobsToGen --;
        }
        else if(jobType == 3 && nFlexibleJobs > 0 || generateAll){

            // malleable, priority, resources, time step, app type            
            fprintf(f, "3\n1\n%d\n%d\n%zu\n0\n3\n%zu\n%zu\n%zu\n\n", gpuRequest, mingpuRequest, timeStep, N, T[jobDuration], K);

            nFlexibleJobs--;
            nJobsToGen --;
        }
    }
}