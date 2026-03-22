#include <cstddef>
#include <omp.h>
#include <unistd.h>
#include <stdio.h>      
#include <stdlib.h>     
#include <pthread.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "mockSch.hpp"

// include matrix summation 
#include "../testApps/toy_app_malleable.hpp"


typedef struct jobLauncher_t {

    int argc;
    void** argv;

    void (*launchFunc)(int, void* []);

} jobLauncher_t;


schInfo_t *schInfo;


void* launchJob(void *jobLauncherVoid){

    jobLauncher_t *jobLauncher = (jobLauncher_t*)jobLauncherVoid;
    jobLauncher->launchFunc(jobLauncher->argc, jobLauncher->argv);
    return NULL;
}


int main(int argc, char* argv[]){

    
    // allocate memory to store cuda data
    int nGPUs;
    cudaGetDeviceCount(&nGPUs); // get number of available devices
    
    // initialize the "scheduler"
    schInfo_t schInfo;
    schInfo.nGPUs = (size_t)nGPUs;
    schInfo.avGPUs = (char*)calloc(schInfo.nGPUs, sizeof(char));

    schInfo.nJobs = 0;
    schInfo.nMaxJobs = 1000;
    schInfo.jobControl = (jobControl_t*)calloc(schInfo.nMaxJobs, sizeof(jobControl_t));
    
    // launch process 1
    size_t a = 5;
    size_t b = 10;
    void* args[3];
    args[0] = &a;
    args[1] = &b;
    args[2] = &(schInfo.jobControl[0]);

    jobLauncher_t jobLauncher1;
    jobLauncher1.argc = 3;
    jobLauncher1.argv = args;
    jobLauncher1.launchFunc = &launch;

    pthread_t thr1;
    pthread_create(&thr1, NULL, launchJob, (void*)(&jobLauncher1));


    sleep(4);
    /*printf(" Notifying reconfiguration\n");
    fflush(stdout);
    notifyReconfiguration(4, NULL, NULL, NULL);
    printf(" Reconfiguration notified\n");*/


    // create process 2


    // create process 3

    
}