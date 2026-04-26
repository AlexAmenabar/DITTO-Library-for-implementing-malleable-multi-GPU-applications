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
    int nJobs = strtoul(argv[1], NULL, 10);

    float pCorrectRequests = strtof(argv[2], NULL);

    int nGPUs = strtoul(argv[3], NULL, 10);
    size_t tDiffMax = (size_t)strtoul(argv[4], NULL, 10);
    size_t tDiffMin = 1;

    const char *fileNameRigid = argv[5];
    const char *fileNameMoldable = argv[6];
    const char *fileNameMalleable = argv[7];
    const char *fileNameFlexible = argv[8];



    // open file in write mode
    FILE *frigid = fopen(fileNameRigid, "w");
    FILE *fmoldable = fopen(fileNameMoldable, "w");
    FILE *fmalleable = fopen(fileNameMalleable, "w");
    FILE *fflexible = fopen(fileNameFlexible, "w");

    
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


    fprintf(frigid, "%zu\n", nJobs);
    fprintf(fmoldable, "%zu\n", nJobs);
    fprintf(fmalleable, "%zu\n", nJobs);
    fprintf(fflexible, "%zu\n", nJobs);


    // randomize
    srand(time(0));


    //printf(" Number of jobs = %d\n Percentage of malleable jobs = %f\n Number of malleable jobs = %d\n", nJobs, pMalleableJobs, nMalleableJobs);

    int nJobsToGen = nJobs;

    while(nJobsToGen>0){

        int jobType = rand() % 4;

        //decide job size
        int jobSize = rand() % 3; // 0: small; 1: medium; 2: big
        int jobDuration = rand() % 3;

        int badRequest = rand() % 2; // if 1, bad request

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
        double stddev = badRequest ? 4: 1.25;
        std::normal_distribution<double> normalGPUs(mean, stddev);

        // rejection sampling to keep [1, 8]
        do {
            gpuRequest = static_cast<int>(std::round(normalGPUs(rngGPUs)));
        } while ((gpuRequest < 2 || gpuRequest > 8) || (gpuRequest & (gpuRequest-1)) != 0);
        std::cout << gpuRequest << std::endl;

        if(badRequest == 0) gpuRequest = goodGPUs[jobSize];




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

        printf(" -- %zu is a bad request (%d)? Good =  %zu, requested = %zu, min requested = %zu\n", nJobs - nJobsToGen, badRequest, goodGPUs[jobSize], gpuRequest, mingpuRequest);
        fflush(stdout);


        //if(jobType == 0 && nRigidJobs > 0 || generateAll){

            // rigid, priority, resources, time step, app type
            fprintf(frigid, "0\n1\n%d\n%zu\n0\n3\n%zu\n%zu\n%zu\n\n", gpuRequest, timeStep, N, T[jobSize], K);

         //}
            // malleable, priority, resources, time step, app type            
            fprintf(fmoldable, "1\n1\n%d\n%d\n%zu\n0\n3\n%zu\n%zu\n%zu\n\n", gpuRequest, mingpuRequest, timeStep, N, T[jobDuration], K);

        
            // malleable, priority, resources, time step, app type            
            fprintf(fmalleable, "2\n1\n%d\n%d\n%zu\n0\n3\n%zu\n%zu\n%zu\n\n", gpuRequest, mingpuRequest, timeStep, N, T[jobDuration], K);

        
            // malleable, priority, resources, time step, app type            
            fprintf(fflexible, "3\n1\n%d\n%d\n%zu\n0\n3\n%zu\n%zu\n%zu\n\n", gpuRequest, mingpuRequest, timeStep, N, T[jobDuration], K);

            nJobsToGen --;
    }
}