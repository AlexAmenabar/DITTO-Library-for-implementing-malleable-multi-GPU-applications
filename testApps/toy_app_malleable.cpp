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

#include "toy_app_malleable.hpp"
#include "DITO_API.hpp"


#include "mockSch.hpp"


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

        /*if(t%100 == 0){
            
            printf(" In t = %zu\n", t);
            fflush(stdout);
        }*/

        // reconfiguration point
        if(data->malleable == 1 && checkIfReconfiguration(getJobControl())){
            //printf(" > Reconfiguring application\n"); 
            reconfigure();
        }

        // loop over app phases
        for(size_t p = 0; p<P; p++){

            // CPU
            if(data->phases[p] == 0){

                // no GPUs needed in this pahse, so move data to the CPU and send signal to the RMS
                /*printf(" -- Notifying no GPUs needed\n");
                fflush(stdout);*/
                if(data->malleable == 1){
                    
                    notifySigGPUs(getJobControl());

                    // wait the answer to the request
                    while(checkIfReconfiguration(getJobControl()) == 0){
                        
                        /*printf(" -- Checking reconf\n");
                        fflush(stdout);*/ 
                        sleep(0.1);
                    }

                    reconfigure();
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

                        reconfigure();
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

        /*if(t%100 == 0){
            
            printf(" In t = %zu\n", t);
            fflush(stdout);
        }*/

        // reconfiguration point
        if(checkIfReconfiguration(getJobControl())){
            //printf(" > Reconfiguring application\n"); 
            reconfigure();
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

    for(size_t t = 0; t<T; t++){

        // reconfiguration point, check if there is any pending reconfiguration
        if(checkIfReconfiguration(getJobControl())){

            reconfigure();
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
                err = cudaMalloc(&gpuArrs[srcDev-1], nPerGPU[srcDev] * sizeof(float)); 


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
                    cudaMemcpyPeer(gpuArrs[srcDev-1], dstDev, dData[srcDev], srcDev, nPerGPU[srcDev] * sizeof(float)); // copy P2P
                    cudaDeviceSynchronize();

                    // end timer
                    clock_gettime(CLOCK_MONOTONIC, &(endTimer));

                    //printf(" -- P2P communication finished\n");
                    //fflush(stdout);

                } else {

                    printf(" -- P2P communications between GPUs from %d to %d NOT ENABLED!\n", srcDev, dstDev);

                    // cpy data through the CPU
                    // 1. allocate memory in the CPU
                    float * tmpCPU = (float*)malloc(nPerGPU[srcDev] * sizeof(float));


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

    // initialize DITTO environment
    // temporal: simulate information received from the scheduler: number of GPUs and identifiers of the GPUs (pass argv to the initDITTO function?)
    
    //printf(" -- Initializing DITTO\n");
    //fflush(stdout);
    initDITTO(argv[argc+1]);
    //printf(" -- DITTO initialized\n");
    //fflush(stdout);

    // APP: initialize data structure used by the application
    appStruct_t *appData = (appStruct_t*)calloc(1, sizeof(appStruct_t));
    appData->N = *(size_t*)argv[0];
    appData->T = *(size_t*)argv[1];
    appData->K = *(size_t*)argv[2];
    appData->malleable = *(size_t*)argv[3]; // 3 == argc


    //printf(" -- N = %zu, T = %zu, K = %zu\n", appData->N, appData->T, appData->K);
    //fflush(stdout);


    appData->arr = (float*)calloc(appData->N, sizeof(float));
    for(i = 0; i<appData->N; i++)
        appData->arr[i] = (float)rand()/(float)(RAND_MAX);

    // Initialize DTIs for automatically managing CPU-GPU communications
    DTI_t *appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->N, sizeof(float), "appData", initializeDTIDescription(simple, ordered));

    // configure all DTIs for automatic data transference // TODO: configure in initialization
    configureDTIs(getNumberOfGPUs(), 0); // number of GPUs, old number of GPUs

    // send data to the GPU
    transferDataCPU2GPU();

    // program code (simulations)
    //printArr(appData);

    simulateIterative(appData);

    // transfer data fron the GPUs to the CPU
    transferDataGPU2CPU();


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

    appData->phases = (size_t*)calloc(appData->P, sizeof(size_t));
    for(i = 0; i<appData->P; i++){

        appData->phases[i] = *(size_t*)argv[i + 5];
    }

    appData->malleable = *(size_t*)argv[argc];


    // allocate memory for App Data
    appData->arr = (float*)calloc(appData->N, sizeof(float));
    for(i = 0; i<appData->N; i++)
        appData->arr[i] = (float)rand()/(float)(RAND_MAX);

    // Initialize DTIs for automatically managing CPU-GPU communications
    DTI_t *appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->N, sizeof(float), "appData", initializeDTIDescription(simple, ordered));


    // configure all DTIs for automatic data transference // TODO: configure in initialization
    configureDTIs(getNumberOfGPUs(), 0); // number of GPUs, old number of GPUs

    // send data to the GPU
    transferDataCPU2GPU();

    // program code (simulations)
    //printArr(appData);
    simulatePhases(appData);

    // transfer data fron the GPUs to the CPU
    transferDataGPU2CPU();
    
    // signal that job finished
    jobFinished(getJobControl());

    pthread_exit(NULL);
}


void launch_communications_app(int argc, void* argv[]){

    size_t i;

    // initialize DITTO environment
    // temporal: simulate information received from the scheduler: number of GPUs and identifiers of the GPUs (pass argv to the initDITTO function?)
    initDITTO(argv[argc+1]);

    // APP: initialize data structure used by the application
    printf(" -- [APP]: Loading argv\n");
    fflush(stdout);

    appStruct_t *appData = (appStruct_t*)calloc(1, sizeof(appStruct_t));
    appData->N = *(size_t*)argv[0];
    appData->T = *(size_t*)argv[1];
    appData->K = *(size_t*)argv[2];
    appData->nIterationsForCommunications = *(size_t*)argv[3];

    appData->malleable = *(size_t*)argv[argc];


    // allocate memory for App Data
    appData->arr = (float*)calloc(appData->N, sizeof(float));
    for(i = 0; i<appData->N; i++)
        appData->arr[i] = (float)rand()/(float)(RAND_MAX);

    // Initialize DTIs for automatically managing CPU-GPU communications
    DTI_t *appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->N, sizeof(float), "appData", initializeDTIDescription(entire, nonerme));


    //printf(" [APP]: %zu B (%zu GB) (N = %zu, size = %zu)\n", appData->N * sizeof(float), appData->N * sizeof(float) / 1000 / 1000 / 1000, appData->N, sizeof(float));

    // configure all DTIs for automatic data transference // TODO: configure in initialization
    configureDTIs(getNumberOfGPUs(), 0); // number of GPUs, old number of GPUs

    // send data to the GPU
    transferDataCPU2GPU();

    // program code (simulations)
    //printArr(appData);

    simulateIterativeCommunications(appData);

    // transfer data fron the GPUs to the CPU
    transferDataGPU2CPU();

    
    //printf(" Src-Dst = %lf, Dst-Src = %lf\n", appData->communicationTimeSrcDst, appData->communicationTimeDstSrc);

    printf(" [RES]: %zu bytes; from %zu to %zu; %lf %lf\n", 
            appData->N * sizeof(float), getGPUIds()[0], getGPUIds()[1], appData->communicationTimeSrcDst, appData->communicationTimeDstSrc);

    // signal that job finished
    jobFinished(getJobControl());

    pthread_exit(NULL);
}

void launch_reconf_test_app(int argc, void* argv[]){

    size_t i;

    // initialize DITTO environment
    // temporal: simulate information received from the scheduler: number of GPUs and identifiers of the GPUs (pass argv to the initDITTO function?)
    initDITTO(argv[argc-1]);

    // APP: initialize data structure used by the application
    appStruct_t *appData = (appStruct_t*)calloc(1, sizeof(appStruct_t));
    appData->N = *(int*)argv[0];
    appData->malleable = *(int*)argv[argc-2];

    // allocate memory for App Data
    appData->charArr = (char*)calloc(appData->N, sizeof(char));
    for(i = 0; i<appData->N; i++)
        appData->charArr[i] = (char)rand()/(char)(RAND_MAX);

    // Initialize DTIs for automatically managing CPU-GPU communications
    DTI_t *appDataDTI = createAutomaticDTI((void*)(appData->charArr), appData->N, sizeof(char), "appData", initializeDTIDescription(simple, ordered));


    // configure all DTIs for automatic data transference // TODO: configure in initialization
    configureDTIs(getNumberOfGPUs(), 0); // number of GPUs, old number of GPUs

    // send data to the GPU
    transferDataCPU2GPU();

    // wait reconfiguration
    while(checkIfReconfiguration(getJobControl()) == 0){
        
        /*printf(" -- Checking reconf\n");
        fflush(stdout);*/ 
        sleep(0);
    }

    struct timespec start, end;
    double etime;

    clock_gettime(CLOCK_MONOTONIC, &start);
    reconfigure();
    clock_gettime(CLOCK_MONOTONIC, &end);
    etime = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf(" %lf\n", etime);

    // transfer data fron the GPUs to the CPU
    transferDataGPU2CPU();
    
    free(appDataDTI->cpuData);

    // signal that job finished
    jobFinished(getJobControl());

    pthread_exit(NULL);
}