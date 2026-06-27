#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <unistd.h>
#include <cstddef>
#include <stdio.h>
#include <cstring>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include "toy_app_malleable.hpp"
#include "DITO_API.hpp"


#include "RMS.hpp"


void printArr(appStruct_t *appData){

    size_t i;

    for(i = 0; i<appData->N; i++){

        printf("%f ", appData->arr[i]);
    }
}

void runCPU(float *arr, size_t N, size_t K){

    for(size_t i = 0; i<N; i++){

        float val = arr[i];
        
        for (size_t k = 0; k < K; k++) {
        
            val = val * 1.000001f + 0.000001f; //(val * 1.01) * 0.999;// * 0.99;// / 1.5;//1.000001f + 0.000001f;
        }

        val = val * 0.5;
        arr[i] = val;
    }
}

void simulatePhases(appStruct_t *data){

    size_t T = data->T;
    size_t P = data->P;
    state_t *state = getState();

    for(size_t t = 0; t<T; t++){

        // reconfiguration point
        if(data->malleable == 1 && checkIfReconfiguration(getJobControl())){

            reconfigure(GPU2GPU);
        }

        // loop over app phases
        for(size_t p = 0; p<P; p++){

            // CPU
            if(data->phases[p] == 0){

                // no GPUs needed in this pahse, so move data to the CPU and send signal to the RMS
                if(data->malleable == 1){
                    
                    notifySigGPUs(getJobControl());

                    // wait the answer to the request
                    while(checkIfReconfiguration(getJobControl()) == 0){
                        
                        sleep(0.1);
                    }

                    reconfigure(GPU2GPU);
                }

                // move data from the GPU to the CPU
                runCPU((float*)(getDTIByIndex(0)->cpuData), getDTIByIndex(0)->N, data->cpuK);
            }
            // GPU
            else{

                // since this phase requires GPUs, check if it has, and, if not, request
                if(getNumberOfGPUs() == 0){
                    
                    if(data->malleable == 1){
                        
                        notifyReqGPUs(getJobControl());
                        
                        // wait the answer to the request
                        while(checkIfReconfiguration(getJobControl()) == 0){
                            
                            sleep(0.1);
                        }

                        reconfigure(GPU2GPU);
                    }
                }

                public_APP_Data_t *localData = appData;
                DTI_t **localArrDTI = arrDTI;
                size_t localnDTI = nDTI;
                size_t localmaxDTI = maxDTI;
                
                // firstprivates should be removed in the future
                #pragma omp parallel for num_threads (getNumberOfGPUs())
                for(size_t j = 0; j<(size_t)(getNumberOfGPUs()); j++){

                    appData = localData;
                    arrDTI = localArrDTI;
                    nDTI = localnDTI;
                    maxDTI = localmaxDTI;

                    // set device
                    cudaSetDevice(getGPUIds()[j]);

                    // run kernel
                    runKernel((float*)(getDTIByIndex(0)->gpuData[j]), getDTIByIndex(0)->nPerGPU[j], data->K);     

                    // sync devices
                    cudaDeviceSynchronize();
                }
            }
        }
    }
}


// TODO: REVISE WHAT HAPPENED HERE
// UPDATE PHASES APP: argc, argc +1 
void simulateIterative(appStruct_t *data){

    size_t T = data->T;
    state_t *state = getState();


    for(size_t t = 0; t<T; t++){

        // reconfiguration point
        if(checkIfReconfiguration(getJobControl())){

            reconfigure(GPU2GPU);
        }


        // app
        public_APP_Data_t *localData = appData;
        DTI_t **localArrDTI = arrDTI;
        size_t localnDTI = nDTI;
        size_t localmaxDTI = maxDTI;
            

        size_t nGPUs = getNumberOfGPUs();
        size_t *idGPUs = getGPUIds();

  
        // firstprivates should be removed in the future
        #pragma omp parallel for num_threads (getNumberOfGPUs())
        for(size_t j = 0; j<nGPUs; j++){ // get number of GPUs should receive a parameter such as appData or something

            appData = localData;
            arrDTI = localArrDTI;
            nDTI = localnDTI;
            maxDTI = localmaxDTI;

            // set device
            cudaSetDevice(idGPUs[j]);

            // run kernel
            runKernel((float*)(getDTIByIndex(0)->gpuData[j]), getDTIByIndex(0)->nPerGPU[j], data->K);     

            // sync devices
            cudaDeviceSynchronize();
        }
    }
}


void simulateIterativeCommunications(appStruct_t *data){
    
    // get number of time steps and state
    size_t T = data->T;
    state_t *state = getState();

    char startRunning = getJobControl()->startRunning; 
    while(!startRunning){
        
        sleep(1);
        pthread_mutex_lock(&(getJobControl()->lockStartRunning));
        startRunning = getJobControl()->startRunning; 
        pthread_mutex_unlock(&(getJobControl()->lockStartRunning));
    }

    printf(" -- Running\n");
    fflush(stdout);

    for(size_t t = 0; t<T; t++){

        // reconfiguration point, check if there is any pending reconfiguration
        if(checkIfReconfiguration(getJobControl())){

            reconfigure(GPU2GPU);
        }


        // copy data from global variables to local variables for enabling visibility to openMP threads (they are thread private)
        public_APP_Data_t *localData = appData;
        DTI_t **localArrDTI = arrDTI;
        size_t localnDTI = nDTI;
        size_t localmaxDTI = maxDTI;
            

        // get the information of the GPUs
        size_t nGPUs = getNumberOfGPUs();
        size_t *idGPUs = getGPUIds();

  
        #pragma omp parallel for num_threads (nGPUs)
        for(size_t j = 0; j<nGPUs; j++){ // get number of GPUs should receive a parameter such as appData or something

            appData = localData;
            arrDTI = localArrDTI;
            nDTI = localnDTI;
            maxDTI = localmaxDTI;

            // set device
            cudaSetDevice(idGPUs[j]);

            // run kernel
            runKernel((float*)(getDTIByIndex(0)->gpuData[j]), getDTIByIndex(0)->nPerGPU[j], data->K);     

            // sync devices
            cudaDeviceSynchronize();
        }

        struct timespec startTimer, endTimer;

        // check if data should be synchronized
        if(t % data->nIterationsForCommunications == 0){

            printf(" -- Starting communications...\n");
            fflush(stdout);

            // copy data to the GPU 0
            int dstDev = 0, srcDev;
            float **gpuArrs = (float**)calloc(nGPUs - 1, sizeof(float*));
            cudaError_t err;

            float **dData, *dSrcData, *dDstData;
            dData = (float**)getDTIByIndex(0)->gpuData;
            size_t *nPerGPU = getDTIByIndex(0)->nPerGPU;

            for(srcDev = 1; srcDev<nGPUs; srcDev++){ // get number of GPUs should receive a parameter such as appData or something
    
                // allocate memory for data in the dst device
                cudaSetDevice(dstDev); 
                err = cudaMalloc(&gpuArrs[srcDev-1], (size_t)nPerGPU[srcDev] * (size_t)sizeof(float)); 


                // copy data from source
                cudaSetDevice(srcDev);

                // try to enable peer access from src to dst
                int canAccess = 0;
                cudaDeviceCanAccessPeer(&canAccess, dstDev, srcDev);

                // if can access, move data directly, else, use the CPU
                if (canAccess) {

                    //printf(" -- P2P communications between GPUs from %d to %d ENABLED!\n", srcDev, dstDev);
                    
                    // start timer
                    clock_gettime(CLOCK_MONOTONIC, &(startTimer));

                    // enable peer access
                    cudaDeviceEnablePeerAccess(dstDev, 0);

                    // cpy data directly
                    cudaMemcpyPeer(gpuArrs[srcDev-1], dstDev, dData[srcDev], srcDev, (size_t)nPerGPU[srcDev] * (size_t)sizeof(float)); // copy P2P
                    cudaDeviceSynchronize();

                    // end timer
                    clock_gettime(CLOCK_MONOTONIC, &(endTimer));

                    //printf(" -- P2P communication finished\n");
                    //fflush(stdout);

                } else {

                    printf(" -- P2P communications between GPUs from %d to %d NOT ENABLED!\n", srcDev, dstDev);

                    // cpy data through the CPU
                    // 1. allocate memory in the CPU
                    float *tmpCPU = (float*)malloc(nPerGPU[srcDev] * sizeof(float));


                    // start timer
                    clock_gettime(CLOCK_MONOTONIC, &(startTimer));

                    // copy to the CPU
                    err = cudaMemcpy(tmpCPU, dData[srcDev], nPerGPU[srcDev], cudaMemcpyDeviceToHost); 

                    // copy to the GPU
                    err = cudaMemcpy(gpuArrs[srcDev-1], tmpCPU, nPerGPU[srcDev], cudaMemcpyHostToDevice); 

                    // end timer
                    clock_gettime(CLOCK_MONOTONIC, &(endTimer));

                    // dellocate temporal CPU memory
                    free(tmpCPU);

                    printf(" -- Communication through CPU finished\n");
                    fflush(stdout);
                }

                data->communicationTimeSrcDst += (endTimer.tv_sec - startTimer.tv_sec) + (endTimer.tv_nsec - startTimer.tv_nsec) / 1e9;


                // sync devices
                cudaDeviceSynchronize();
            }
            
            // do some computation?


            // store time

            // now the 0 is the src
            srcDev = 0;

            // move data again to the src GPUs
            for(dstDev = 1; dstDev<nGPUs; dstDev++){ // get number of GPUs should receive a parameter such as appData or something

                // set src device and check peer access to dst 
                cudaSetDevice(srcDev);

                // try to enable peer access
                int canAccess = 0;
                cudaDeviceCanAccessPeer(&canAccess, dstDev, srcDev);

                // if can access, move data directly, else, use the CPU
                if (canAccess) {

                    //printf(" -- P2P communications between GPUs from %d to %d ENABLED!\n", srcDev, dstDev);
                    
                    // set current device
                    cudaSetDevice(srcDev); // 

                    // start timer
                    clock_gettime(CLOCK_MONOTONIC, &(startTimer));

                    cudaDeviceEnablePeerAccess(dstDev, 0); // enable peer access to destination device

                    // cpy data directly
                    cudaMemcpyPeer(dData[dstDev], dstDev, gpuArrs[dstDev-1], srcDev, nPerGPU[dstDev] * sizeof(float)); // copy P2P
                    cudaDeviceSynchronize();

                    // end timer
                    clock_gettime(CLOCK_MONOTONIC, &(endTimer));

                    //printf(" -- P2P communication finished\n");
                    //fflush(stdout);

                } else {

                    printf(" -- P2P communications between GPUs from %d to %d NOT ENABLED!\n", srcDev, dstDev);

                    // cpy data through the CPU
                    // 1. allocate memory in the CPU
                    float* tmpCPU = (float*)malloc(nPerGPU[dstDev] * sizeof(float));

                    // start timer
                    clock_gettime(CLOCK_MONOTONIC, &(startTimer));

                    // copy to the CPU
                    err = cudaMemcpy(tmpCPU, gpuArrs[dstDev-1], nPerGPU[dstDev], cudaMemcpyDeviceToHost); 

                    // copy to the GPU
                    err = cudaMemcpy(dData[dstDev], tmpCPU, nPerGPU[dstDev], cudaMemcpyHostToDevice); 

                    // end timer
                    clock_gettime(CLOCK_MONOTONIC, &(endTimer));

                    // dellocate temporal CPU memory
                    free(tmpCPU);

                    printf(" -- Communication through CPU finished\n");
                    fflush(stdout);
                }

                data->communicationTimeDstSrc += (endTimer.tv_sec - startTimer.tv_sec) + (endTimer.tv_nsec - startTimer.tv_nsec) / 1e9;


                // sync devices
                cudaDeviceSynchronize();

                // deallocate gpuArrs
                cudaFree(gpuArrs[dstDev-1]);
            }
            
            // deallocate 
            free(gpuArrs);
        }
    }

    data->communicationTimeSrcDst = (double)((double)data->communicationTimeSrcDst / (double)T);
    data->communicationTimeDstSrc = (double)((double)data->communicationTimeDstSrc / (double)T);
}



void launch_iterative_app(int argc, void* argv[]){

    size_t i;

    // initialize DITTO
    initDITTO(argv[argc+1]);

    // APP: initialize data structure used by the application
    appStruct_t *appData = (appStruct_t*)calloc(1, sizeof(appStruct_t));
    appData->N = *(size_t*)argv[0];
    appData->T = *(size_t*)argv[1];
    appData->K = *(size_t*)argv[2];
    
    appData->s = *(size_t*)argv[3];
    //appData->async = *(size_t*)argv[4];

    size_t pinned = *(size_t*)argv[4];
    size_t async = *(size_t*)argv[5];
    size_t steps = *(size_t*)argv[6];
    size_t cores = *(size_t*)argv[7];

    appData->malleable = *(size_t*)argv[argc]; // 3 == argc


    // set communication type
    communicationType_t commType;
    if(pinned)
        commType.cudaMemoryType = pinnedComm;
    else 
        commType.cudaMemoryType = nonPinnedComm;

    if(async)
        commType.transmissionType = asyncComm;
    else
        commType.transmissionType = syncComm;
    
    if(steps == 0)
        commType.transferSteps = oneStepComm;
    else if(steps == 1)
        commType.transferSteps = twoStepsComm;
    else
        commType.transferSteps = stridedComm;

    if(cores)
        commType.transferCores = multiCoreComm;
    else
        commType.transferCores = singleCoreComm;


    // allocate memory for App Data
    if(appData->async > 1) // pinned memory
        cudaMallocHost(&appData->arr, appData->N * sizeof(float));
    else
        appData->arr = (float*)calloc(appData->N, sizeof(float));

    // initialize data
    for(i = 0; i<appData->N; i++)
        appData->arr[i] = (float)rand()/(float)(RAND_MAX);


    // Initialize DTIs for automatically managing CPU-GPU communications
    DTI_t *appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->N, sizeof(float), "appData", initializeDTIDescription(simple, ordered, commType));

    // configure all DTIs for automatic data transference // TODO: configure in initialization
    configureDTIs(getJobControl()->jobResources, NULL, CPU2GPU); // number of GPUs, old number of GPUs

    // send data to the GPU
    transferDataCPU2GPU();

    simulateIterative(appData);

    // transfer data fron the GPUs to the CPU
    transferDataGPU2CPU();



    if(appData->async > 1)
        cudaFreeHost(appDataDTI->cpuData);
    else
        free(appDataDTI->cpuData);


    // destroy streams
    freeDITTO();


    // signal that job finished
    jobFinished(getJobControl());
    
    pthread_exit(NULL);
}


void launch_phases_app(int argc, void* argv[]){

    size_t i;

    // initialize DITTO environment
    // temporal: simulate information received from the scheduler: number of GPUs and identifiers of the GPUs (pass argv to the initDITTO function?)
    initDITTO(argv[argc+1]);

    // APP: initialize data structure used by the application
    appStruct_t *appData = (appStruct_t*)calloc(1, sizeof(appStruct_t));
    appData->N = *(size_t*)argv[0];
    appData->T = *(size_t*)argv[1];
    appData->K = *(size_t*)argv[2];
    appData->cpuK = *(size_t*)argv[3];
    appData->P = *(size_t*)argv[4];

    appData->s = *(size_t*)argv[5];
    
    //appData->async = *(size_t*)argv[6];
    // communication info
    size_t pinned = *(size_t*)argv[6];
    size_t async = *(size_t*)argv[7];
    size_t steps = *(size_t*)argv[8];
    size_t cores = *(size_t*)argv[9];


    appData->phases = (size_t*)calloc(appData->P, sizeof(size_t));
    for(i = 0; i<appData->P; i++)
        appData->phases[i] = *(size_t*)argv[i + 10];

    appData->malleable = *(size_t*)argv[argc];


    // set communication type
    communicationType_t commType;
    if(pinned)
        commType.cudaMemoryType = pinnedComm;
    else 
        commType.cudaMemoryType = nonPinnedComm;

    if(async)
        commType.transmissionType = asyncComm;
    else
        commType.transmissionType = syncComm;
    
    if(steps == 0)
        commType.transferSteps = oneStepComm;
    else if(steps == 1)
        commType.transferSteps = twoStepsComm;
    else
        commType.transferSteps = stridedComm;

    if(cores)
        commType.transferCores = multiCoreComm;
    else
        commType.transferCores = singleCoreComm;


    // allocate memory for App Data
    if(async || pinned) // pinned memory
        cudaMallocHost(&appData->arr, appData->N * sizeof(float));
    else
        appData->arr = (float*)calloc(appData->N, sizeof(float));

    // initialize data
    for(i = 0; i<appData->N; i++)
        appData->arr[i] = (float)rand()/(float)(RAND_MAX);


    // Initialize DTIs for automatically managing CPU-GPU communications
    DTI_t *appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->N, sizeof(float), "appData", initializeDTIDescription(simple, ordered, commType));


    // configure all DTIs for automatic data transference // TODO: configure in initialization
    configureDTIs(getJobControl()->jobResources, NULL, CPU2GPU); // number of GPUs, old number of GPUs

    // send data to the GPU
    transferDataCPU2GPU();

    // program code (simulations)
    //printArr(appData);
    simulatePhases(appData);

    // transfer data fron the GPUs to the CPU
    transferDataGPU2CPU();
    


    if(async || pinned)
        cudaFreeHost(appDataDTI->cpuData);
    else
        free(appDataDTI->cpuData);


    // destroy streams
    freeDITTO();

    // signal that job finished
    jobFinished(getJobControl());

    pthread_exit(NULL);
}


void launch_communications_app(int argc, void* argv[]){

    size_t i;

    // initialize DITTO environment
    // temporal: simulate information received from the scheduler: number of GPUs and identifiers of the GPUs (pass argv to the initDITTO function?)
    initDITTO(argv[argc+1]);

    appStruct_t *appData = (appStruct_t*)calloc(1, sizeof(appStruct_t));
    appData->N = *(size_t*)argv[0];
    appData->T = *(size_t*)argv[1];
    appData->K = *(size_t*)argv[2];
    appData->nIterationsForCommunications = *(size_t*)argv[3];
    appData->s = *(size_t*)argv[4];
    //appData->async = *(size_t*)argv[5];

    // communication info
    size_t pinned = *(size_t*)argv[5];
    size_t async = *(size_t*)argv[6];
    size_t steps = *(size_t*)argv[7];
    size_t cores = *(size_t*)argv[8];
    
    appData->malleable = *(size_t*)argv[argc];


    // set communication type
    communicationType_t commType;
    if(pinned)
        commType.cudaMemoryType = pinnedComm;
    else 
        commType.cudaMemoryType = nonPinnedComm;

    if(async)
        commType.transmissionType = asyncComm;
    else
        commType.transmissionType = syncComm;
    
    if(steps == 0)
        commType.transferSteps = oneStepComm;
    else if(steps == 1)
        commType.transferSteps = twoStepsComm;
    else
        commType.transferSteps = stridedComm;

    if(cores)
        commType.transferCores = multiCoreComm;
    else
        commType.transferCores = singleCoreComm;


    // allocate memory for App Data
    if(async || pinned) // pinned memory
        cudaMallocHost(&appData->arr, appData->N * sizeof(float));
    else
        appData->arr = (float*)calloc(appData->N, sizeof(float));
    
    // initialize data
    for(i = 0; i<appData->N; i++)
        appData->arr[i] = (float)rand()/(float)(RAND_MAX);


    // Initialize DTIs for automatically managing CPU-GPU communications
    DTI_t *appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->N, sizeof(float), "appData", initializeDTIDescription(entire, nonerme, commType));


    // configure all DTIs for automatic data transference // TODO: configure in initialization
    configureDTIs(getJobControl()->jobResources, NULL, CPU2GPU); // number of GPUs, old number of GPUs

    // send data to the GPU
    transferDataCPU2GPU();

    // wait for signal before starting
    printf(" -- Simulating...\n");
    fflush(stdout);
    simulateIterativeCommunications(appData);

    // transfer data fron the GPUs to the CPU
    transferDataGPU2CPU();

    printf(" [RES]: %zu bytes; from %zu to %zu; %lf %lf\n", 
            appData->N * sizeof(float), getGPUIds()[0], getGPUIds()[1], appData->communicationTimeSrcDst, appData->communicationTimeDstSrc);


    if(async || pinned)
        cudaFreeHost(appDataDTI->cpuData);
    else
        free(appDataDTI->cpuData);


    // destroy streams
    freeDITTO();

    // signal that job finished
    jobFinished(getJobControl());

    pthread_exit(NULL);
}

void launch_reconf_test_app(int argc, void* argv[]){

    size_t i;

    // initialize DITTO environment
    printf(" Initializing DITTO\n");
    fflush(stdout);

    initDITTO(argv[argc+1]);
    printf(" DITTO initialized\n");
    fflush(stdout);

    // APP: initialize data structure used by the application
    appStruct_t *appData = (appStruct_t*)calloc(1, sizeof(appStruct_t));
    appData->N = *(size_t*)argv[0];
    appData->s = *(size_t*)argv[1];
    //appData->async = *(size_t*)argv[2];

    // communication info
    size_t pinned = *(size_t*)argv[2];
    size_t async = *(size_t*)argv[3];
    size_t steps = *(size_t*)argv[4];
    size_t cores = *(size_t*)argv[5];

    appData->malleable = *(size_t*)argv[argc];

    // allocate memory for App Data
    if(pinned) // pinned memory
        cudaMallocHost(&appData->arr, appData->N * sizeof(float));
    else
        appData->arr = (float*)calloc(appData->N, sizeof(float));
    
    // initialize data
    for(i = 0; i<appData->N; i++)
        appData->arr[i] = (float)i;//(float)rand()/(float)(RAND_MAX);

    
    // set communication type
    communicationType_t commType;
    if(pinned)
        commType.cudaMemoryType = pinnedComm;
    else 
        commType.cudaMemoryType = nonPinnedComm;

    if(async)
        commType.transmissionType = asyncComm;
    else
        commType.transmissionType = syncComm;
    
    if(steps == 0)
        commType.transferSteps = oneStepComm;
    else if(steps == 1)
        commType.transferSteps = twoStepsComm;
    else
        commType.transferSteps = stridedComm;

    if(cores)
        commType.transferCores = multiCoreComm;
    else
        commType.transferCores = singleCoreComm;


    // Initialize DTIs for automatically managing CPU-GPU communications
    DTI_t *appDataDTI;
    if(appData->s > 0)
        appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->N, sizeof(float), "appData", initializeComplexDTIDescription(complex, ordered, commType, appData->s));
    else
        appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->N, sizeof(float), "appData", initializeDTIDescription(simple, ordered, commType));

    for(size_t i = 0; i<appData->N; i++){

        printf("%f ", appData->arr[i]);
    }
    printf("\n");
    fflush(stdout);


    // configure all DTIs for automatic data transference // TODO: configure in initialization
    configureDTIs(getState()->jobResources, NULL, CPU2GPU); // number of GPUs, old number of GPUs

    // send data to the GPU
    transferDataCPU2GPU();

    struct timespec start, end;
    double etime = 0.0;

    // wait reconfiguration
    while(checkIfReconfiguration(getJobControl()) == 0){
        
        /*printf(" -- Checking reconf\n");
        fflush(stdout);*/ 
        sleep(1);
    }

    // print GPUs
    printf(" -- Printing current GPU configuration (%zu): ", getJobControl()->jobResources->nGPUs);
    for(size_t i = 0; i<getJobControl()->jobResources->nGPUs; i++){
        printf("%zu ", getJobControl()->jobResources->idGPUs[i]);
    }
    printf("\n");
    printf(" -- Printing GPU configuration for reconfiguration (%zu): ", getJobControl()->reconfJobResources->nGPUs);
    for(size_t i = 0; i<getJobControl()->reconfJobResources->nGPUs; i++){
        printf("%zu ", getJobControl()->reconfJobResources->idGPUs[i]);
    }
    printf("\n");
    fflush(stdout);


    clock_gettime(CLOCK_MONOTONIC, &start);
    reconfigure(GPU2GPU);
    clock_gettime(CLOCK_MONOTONIC, &end);
    etime += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("Reconfiguration finished in: %lf", etime);
    fflush(stdout);


    // transfer data fron the GPUs to the CPU
    for(i = 0; i<appData->N; i++){
        ((float*)(appDataDTI->cpuData))[i] = 0;
    }

    transferDataGPU2CPU();
    

    for(size_t i = 0; i<appData->N; i++){

        printf("%f ", appData->arr[i]);
    }
    printf("\n");
    fflush(stdout);

    int correct = 1;
    for(i = 0; i<appData->N; i++){
        if( ((float*)(appDataDTI->cpuData))[i] != (float)i){
            correct = 0;
            break;        
        }
    }
    
    if(!correct)
        printf(" -- [APP]: Incorrect result!!!");
    else
        printf(" -- [APP]: Correct results!!!!");

    if(pinned)
        cudaFreeHost(appDataDTI->cpuData);
    else
        free(appDataDTI->cpuData);


    // destroy streams
    freeDITTO();

    // deallocate streams

    // deallocate DITTO environment memory

    // signal that job finished
    jobFinished(getJobControl());

    pthread_exit(NULL);
}




void launch_reconfs_test_app(int argc, void* argv[]){

    size_t i;

    // initialize DITTO environment
    printf(" Initializing DITTO\n");
    fflush(stdout);

    initDITTO(argv[argc+1]);
    printf(" DITTO initialized\n");
    fflush(stdout);

    // APP: initialize data structure used by the application
    appStruct_t *appData = (appStruct_t*)calloc(1, sizeof(appStruct_t));
    appData->N = *(size_t*)argv[0];
    appData->s = *(size_t*)argv[1];
    //appData->async = *(size_t*)argv[2];

    // communication info
    size_t pinned = *(size_t*)argv[2];
    size_t async = *(size_t*)argv[3];
    size_t steps = *(size_t*)argv[4];
    size_t cores = *(size_t*)argv[5];

    appData->malleable = *(size_t*)argv[argc];

    // allocate memory for App Data
    if(pinned) // pinned memory
        cudaMallocHost(&appData->arr, appData->N * sizeof(float));
    else
        appData->arr = (float*)calloc(appData->N, sizeof(float));
    
    // initialize data
    for(i = 0; i<appData->N; i++)
        appData->arr[i] = (float)i;//(float)rand()/(float)(RAND_MAX);

    
    // set communication type
    communicationType_t commType;
    if(pinned)
        commType.cudaMemoryType = pinnedComm;
    else 
        commType.cudaMemoryType = nonPinnedComm;

    if(async)
        commType.transmissionType = asyncComm;
    else
        commType.transmissionType = syncComm;
    
    if(steps == 0)
        commType.transferSteps = oneStepComm;
    else if(steps == 1)
        commType.transferSteps = twoStepsComm;
    else
        commType.transferSteps = stridedComm;

    if(cores)
        commType.transferCores = multiCoreComm;
    else
        commType.transferCores = singleCoreComm;


    // Initialize DTIs for automatically managing CPU-GPU communications
    DTI_t *appDataDTI;
    if(appData->s > 0)
        appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->N, sizeof(float), "appData", initializeComplexDTIDescription(complex, ordered, commType, appData->s));
    else
        appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->N, sizeof(float), "appData", initializeDTIDescription(simple, ordered, commType));

    for(size_t i = 0; i<appData->N; i++){

        printf("%f ", appData->arr[i]);
    }
    printf("\n");
    fflush(stdout);


    // configure all DTIs for automatic data transference // TODO: configure in initialization
    configureDTIs(getState()->jobResources, NULL, CPU2GPU); // number of GPUs, old number of GPUs

    // send data to the GPU
    transferDataCPU2GPU();

    struct timespec start, end;
    double etime = 0.0;

    // wait reconfiguration
    while(checkIfReconfiguration(getJobControl()) == 0){
        
        /*printf(" -- Checking reconf\n");
        fflush(stdout);*/ 
        sleep(1);
    }

    // print GPUs
    printf(" -- Printing current GPU configuration (%zu): ", getJobControl()->jobResources->nGPUs);
    for(size_t i = 0; i<getJobControl()->jobResources->nGPUs; i++){
        printf("%zu ", getJobControl()->jobResources->idGPUs[i]);
    }
    printf("\n");
    printf(" -- Printing GPU configuration for reconfiguration (%zu): ", getJobControl()->reconfJobResources->nGPUs);
    for(size_t i = 0; i<getJobControl()->reconfJobResources->nGPUs; i++){
        printf("%zu ", getJobControl()->reconfJobResources->idGPUs[i]);
    }
    printf("\n");
    fflush(stdout);


    clock_gettime(CLOCK_MONOTONIC, &start);
    reconfigure(GPU2GPU);
    clock_gettime(CLOCK_MONOTONIC, &end);
    etime += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("Reconfiguration finished in: %lf", etime);
    fflush(stdout);


    // wait reconfiguration
    while(checkIfReconfiguration(getJobControl()) == 0){
        
        /*printf(" -- Checking reconf\n");
        fflush(stdout);*/ 
        sleep(1);
    }

    // print GPUs
    printf(" -- Printing current GPU configuration (%zu): ", getJobControl()->jobResources->nGPUs);
    for(size_t i = 0; i<getJobControl()->jobResources->nGPUs; i++){
        printf("%zu ", getJobControl()->jobResources->idGPUs[i]);
    }
    printf("\n");
    printf(" -- Printing GPU configuration for reconfiguration (%zu): ", getJobControl()->reconfJobResources->nGPUs);
    for(size_t i = 0; i<getJobControl()->reconfJobResources->nGPUs; i++){
        printf("%zu ", getJobControl()->reconfJobResources->idGPUs[i]);
    }
    printf("\n");
    fflush(stdout);


    clock_gettime(CLOCK_MONOTONIC, &start);
    reconfigure(GPU2GPU);
    clock_gettime(CLOCK_MONOTONIC, &end);
    etime += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("Reconfiguration finished in: %lf", etime);
    fflush(stdout);


    // transfer data fron the GPUs to the CPU
    for(i = 0; i<appData->N; i++){
        ((float*)(appDataDTI->cpuData))[i] = 0;
    }

    transferDataGPU2CPU();
    

    for(size_t i = 0; i<appData->N; i++){

        printf("%f ", appData->arr[i]);
    }
    printf("\n");
    fflush(stdout);

    int correct = 1;
    for(i = 0; i<appData->N; i++){
        if( ((float*)(appDataDTI->cpuData))[i] != (float)i){
            correct = 0;
            break;        
        }
    }
    
    if(!correct)
        printf(" -- [APP]: Incorrect result!!!");
    else
        printf(" -- [APP]: Correct results!!!!");

    if(pinned)
        cudaFreeHost(appDataDTI->cpuData);
    else
        free(appDataDTI->cpuData);


    // destroy streams
    freeDITTO();

    // deallocate streams

    // deallocate DITTO environment memory

    // signal that job finished
    jobFinished(getJobControl());

    pthread_exit(NULL);
}


void launch_reconfs_test_app_new(int argc, void* argv[]){


    size_t i;

    // initialize DITTO environment
    initDITTO(argv[argc+1]);

    // APP: initialize data structure used by the application
    appStruct_t *appData = (appStruct_t*)calloc(1, sizeof(appStruct_t));
    appData->N = *(size_t*)argv[0];
    appData->s = *(size_t*)argv[1];
    //appData->async = *(size_t*)argv[2];

    // communication info
    size_t pinned = *(size_t*)argv[2];
    size_t async = *(size_t*)argv[3];
    size_t steps = *(size_t*)argv[4];
    size_t cores = *(size_t*)argv[5];

    int enumValue = *(int*)argv[6];
    reconfDirEnum reconfDir = (reconfDirEnum)enumValue; 

    appData->malleable = *(size_t*)argv[argc];

    // allocate memory for App Data
    if(pinned) // pinned memory
        cudaMallocHost(&appData->arr, appData->N * sizeof(float));
    else
        appData->arr = (float*)calloc(appData->N, sizeof(float));
    
    // initialize data
    for(i = 0; i<appData->N; i++)
        appData->arr[i] = (float)i;//(float)rand()/(float)(RAND_MAX);

    
    // set communication type
    communicationType_t commType;
    if(pinned)
        commType.cudaMemoryType = pinnedComm;
    else 
        commType.cudaMemoryType = nonPinnedComm;

    if(async)
        commType.transmissionType = asyncComm;
    else
        commType.transmissionType = syncComm;
    
    if(steps == 0)
        commType.transferSteps = oneStepComm;
    else if(steps == 1)
        commType.transferSteps = twoStepsComm;
    else
        commType.transferSteps = stridedComm;

    if(cores)
        commType.transferCores = multiCoreComm;
    else
        commType.transferCores = singleCoreComm;


    // Initialize DTIs for automatically managing CPU-GPU communications
    DTI_t *appDataDTI;
    if(appData->s > 0)
        appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->N, sizeof(float), "appData", initializeComplexDTIDescription(complex, ordered, commType, appData->s));
    else
        appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->N, sizeof(float), "appData", initializeDTIDescription(simple, ordered, commType));


    // configure all DTIs for automatic data transference // TODO: configure in initialization
    configureDTIs(getState()->jobResources, NULL, CPU2GPU); // number of GPUs, old number of GPUs

    // send data to the GPU
    transferDataCPU2GPU();

    struct timespec start, end;
    double etime = 0.0;

    // wait reconfiguration
    while(checkIfReconfiguration(getJobControl()) == 0){
        
        /*printf(" -- Checking reconf\n");
        fflush(stdout);*/ 
        sleep(1);
    }


    clock_gettime(CLOCK_MONOTONIC, &start);
    reconfigure(reconfDir);
    clock_gettime(CLOCK_MONOTONIC, &end);
    etime = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("%lf\n", etime);

    // transfer data fron the GPUs to the CPU
    for(i = 0; i<appData->N; i++){
        ((float*)(appDataDTI->cpuData))[i] = 0;
    }

    transferDataGPU2CPU();

    int correct = 1;
    for(i = 0; i<appData->N; i++){
        if( ((float*)(appDataDTI->cpuData))[i] != (float)i){
            correct = 0;
            break;        
        }
    }    
    if(!correct)
        printf(" -- [APP]: Incorrect result!!!");


    if(pinned)
        cudaFreeHost(appDataDTI->cpuData);
    else
        free(appDataDTI->cpuData);

    // destroy streams
    freeDITTO();

    // signal that job finished
    jobFinished(getJobControl());

    pthread_exit(NULL);
}


void launch_malloc_test_app(int argc, void* argv[]){

    size_t i;

    // initialize DITTO environment
    // temporal: simulate information received from the scheduler: number of GPUs and identifiers of the GPUs (pass argv to the initDITTO function?)
    initDITTO(argv[argc+1]);

    // APP: initialize data structure used by the application
    appStruct_t *appData = (appStruct_t*)calloc(1, sizeof(appStruct_t));
    appData->N = *(size_t*)argv[0];
    appData->s = *(size_t*)argv[1];
    //appData->async = *(size_t*)argv[2];

    // communication info
    size_t pinned = *(size_t*)argv[2];
    size_t async = *(size_t*)argv[3];
    size_t steps = *(size_t*)argv[4];
    size_t cores = *(size_t*)argv[5];
    appData->malleable = *(size_t*)argv[argc];



    struct timespec startMalloc, endMalloc, startFree, endFree, startFirstMalloc, endFirstMalloc, startFirstFree, endFirstFree;
    double etimeMalloc = 0.0, etime = 0.0, etimeFree = 0.0, etimeFirstMalloc = 0.0, etimeFirstFree = 0.0;

    clock_gettime(CLOCK_MONOTONIC, &startFirstMalloc);
    // allocate memory for App Data
    if(async || pinned){ // pinned memory
        cudaMallocHost(&appData->arr, appData->N * sizeof(float));
    }
    else {
        appData->arr = (float*)calloc(appData->N, sizeof(float));
    }

    clock_gettime(CLOCK_MONOTONIC, &endFirstMalloc);
    etimeFirstMalloc += (endFirstMalloc.tv_sec - startFirstMalloc.tv_sec) + (endFirstMalloc.tv_nsec - startFirstMalloc.tv_nsec) / 1e9;

    // initialize data
    for(i = 0; i<appData->N; i++)
        appData->arr[i] = (float)rand()/(float)(RAND_MAX);



    // set communication type
    communicationType_t commType;
    if(pinned)
        commType.cudaMemoryType = pinnedComm;
    else 
        commType.cudaMemoryType = nonPinnedComm;

    if(async)
        commType.transmissionType = asyncComm;
    else
        commType.transmissionType = syncComm;
    
    if(steps == 0)
        commType.transferSteps = oneStepComm;
    else if(steps == 1)
        commType.transferSteps = twoStepsComm;
    else
        commType.transferSteps = stridedComm;

    if(cores)
        commType.transferCores = multiCoreComm;
    else
        commType.transferCores = singleCoreComm;


    // Initialize DTIs for automatically managing CPU-GPU communications
    DTI_t *DTI;
    size_t size = sizeof(float);
    if(appData->s > 0)
        DTI = createAutomaticDTI((void*)(appData->arr), appData->N, sizeof(float), "appData", initializeComplexDTIDescription(complex, ordered, commType, appData->s));
    else
        DTI = createAutomaticDTI((void*)(appData->arr), appData->N, sizeof(float), "appData", initializeDTIDescription(simple, ordered, commType));


    // configure all DTIs for automatic data transference // TODO: configure in initialization
    configureDTIs(getJobControl()->jobResources, NULL, CPU2GPU); // number of GPUs, old number of GPUs

    // send data to the GPU
    transferDataCPU2GPU();


    // intermediate array to recive data from the GPUs contiguously and then redistribute
    void **cData = (void**)calloc(getNumberOfGPUs(), sizeof(void*));

    // reconfiugre (manually for testing malloc and communication time)
    for(i = 0; i<getNumberOfGPUs(); i++){

        // set device
        //setGPUDevice(i);
        cudaSetDevice(getGPUIds()[i]);

        // allocate memory for cData 
        clock_gettime(CLOCK_MONOTONIC, &startMalloc);
        if(commType.cudaMemoryType == nonPinnedComm && commType.transmissionType == syncComm) { // non-pinned memory
            cData[i] = (void*)calloc(DTI->nPerGPU[i], size);
        }
        else { // pinned memory or async
            cudaMallocHost(&cData[i], DTI->nPerGPU[i] * size);
        }
        clock_gettime(CLOCK_MONOTONIC, &endMalloc);
        etimeMalloc += (endMalloc.tv_sec - startMalloc.tv_sec) + (endMalloc.tv_nsec - startMalloc.tv_nsec) / 1e9;

        
        // deallocate cData memory
        clock_gettime(CLOCK_MONOTONIC, &startFree);
        if(commType.cudaMemoryType == nonPinnedComm){
            free(cData[i]);
        }
        else{  // pinned
            cudaFreeHost(cData[i]); 
        }
        clock_gettime(CLOCK_MONOTONIC, &endFree);
        etimeFree += (endFree.tv_sec - startFree.tv_sec) + (endFree.tv_nsec - startFree.tv_nsec) / 1e9;
    }

    free(cData);

    // transfer data fron the GPUs to the CPU
    transferDataGPU2CPU();
    
    
    clock_gettime(CLOCK_MONOTONIC, &startFirstFree);
    if(pinned){
        cudaFreeHost(DTI->cpuData);
    }
    else{
        free(DTI->cpuData);
    }
    clock_gettime(CLOCK_MONOTONIC, &endFirstFree);
    etimeFirstFree += (endFirstFree.tv_sec - startFirstFree.tv_sec) + (endFirstFree.tv_nsec - startFirstFree.tv_nsec) / 1e9;


    printf("%lf %lf %lf %lf", etimeFirstMalloc, etimeMalloc, etimeFirstFree, etimeFree);

    // destroy streams
    freeDITTO();

    // signal that job finished
    jobFinished(getJobControl());

    pthread_exit(NULL);
}