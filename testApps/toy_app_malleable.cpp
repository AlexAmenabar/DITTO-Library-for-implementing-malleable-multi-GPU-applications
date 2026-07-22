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



// TODO: REVISE WHAT HAPPENED HERE
// UPDATE PHASES APP: argc, argc +1 
void simulateIterative(appStruct_t *data){

    size_t T = data->T;

    for(size_t t = 0; t<T; t++){

        // reconfiguration point
        // check whether there is any pending reconfiguration
        if(checkIfReconfiguration(getJobControl())){

            reconfigure(CPU);
        }

        // copies for thread_local variables
        public_APP_Data_t *localData = appData;
        DTI_t **localArrDTI = arrDTI;
        size_t localnDTI = nDTI;
        size_t localmaxDTI = maxDTI;
            
        // get GPUs information
        size_t nGPUs = getNumberOfGPUs();
        size_t *idGPUs = getGPUIds();

        // firstprivates should be removed in the future
        #pragma omp parallel for num_threads (nGPUs)
        for(size_t j = 0; j<nGPUs; j++){ // get number of GPUs should receive a parameter such as appData or something

            appData = localData;
            arrDTI = localArrDTI;
            nDTI = localnDTI;
            maxDTI = localmaxDTI;

            // set device
            cudaSetDevice(idGPUs[j]);

            // run kernel
            runKernel((float*)(getDTIByIndex(0)->gpuData[j]), getDTIByIndex(0)->nPerGPU[j], data->K, nGPUs);     

            // synchronize devices
            cudaDeviceSynchronize();
        }
    }
}

void simulatePhases(appStruct_t *data){

    size_t T = data->T;
    size_t P = data->P;

    for(size_t t = 0; t<T; t++){

        // reconfiguration point
        if(checkIfReconfiguration(getJobControl())){

            reconfigure(CPU);
        }

        // loop over app phases
        for(size_t p = 0; p<P; p++){

            // CPU
            if(data->phases[p] == 0){

                // forget phases hints for now
                // no GPUs needed in this pahse, so move data to the CPU and send signal to the RMS
                /*if(data->malleable == 1){
                    
                    // notify that GPUs are not required in this phase
                    notifySigGPUs(getJobControl());

                    // wait the answer to the request
                    while(checkIfReconfiguration(getJobControl()) == 0){
                        
                        sleep(0.1);
                    }

                    // reconfigure
                    reconfigure(CPU);
                }*/

                // move data from the GPU to the CPU
                //runCPU((float*)(getDTIByIndex(0)->cpuData), getDTIByIndex(0)->N, data->cpuK);
                //sleep(data->cpuK);

                size_t t = rand() % 1000;
                if(t == 0)
                    t = 1;

                size_t time = data->cpuK * t;
                usleep(time);
            }
            // GPU
            else{

                // since this phase requires GPUs, check if it has, and, if not, request
                /*if(getNumberOfGPUs() == 0){
                    
                    if(data->malleable == 1){
                        
                        notifyReqGPUs(getJobControl());
                        
                        // wait the answer to the request
                        while(checkIfReconfiguration(getJobControl()) == 0){
                            
                            sleep(0.1);
                        }

                        reconfigure(GPU2GPU);
                    }
                }*/

                public_APP_Data_t *localData = appData;
                DTI_t **localArrDTI = arrDTI;
                size_t localnDTI = nDTI;
                size_t localmaxDTI = maxDTI;
                
                // get GPUs information
                size_t nGPUs = getNumberOfGPUs();

                // firstprivates should be removed in the future
                #pragma omp parallel for num_threads (nGPUs)
                for(size_t j = 0; j<nGPUs; j++){

                    appData = localData;
                    arrDTI = localArrDTI;
                    nDTI = localnDTI;
                    maxDTI = localmaxDTI;

                    // set device
                    cudaSetDevice(getGPUIds()[j]);

                    // run kernel
                    runKernel((float*)(getDTIByIndex(0)->gpuData[j]), getDTIByIndex(0)->nPerGPU[j], data->K, nGPUs);     

                    // sync devices
                    cudaDeviceSynchronize();
                }
            }
        }
    }
}



void simulateIterativeNCCL(appStruct_t *data){
    
    // get number of time steps and state
    size_t i;
    size_t T = data->T;

    double etComputation = 0.0, etCommunication = 0.0;
    struct timespec startComputation, endComputation;
    struct timespec startCommunication, endCommunication;

    // simulate iterations
    for(size_t t = 0; t<T; t++){

        // reconfiguration point, check if there is any pending reconfiguration
        if(checkIfReconfiguration(getJobControl())){

            // reconfigure and manage NCCL communicators
            freeNCCLComm(getState()->jobResources);
            reconfigure(CPU);
            initNCCLComm(getState()->jobResources);
        }

        // copy data from global variables to local variables for enabling visibility to openMP threads (they are thread private)
        public_APP_Data_t *localData = appData;
        DTI_t **localArrDTI = arrDTI;
        size_t localnDTI = nDTI;
        size_t localmaxDTI = maxDTI;
  

        // get the information of the GPUs
        size_t nGPUs = getNumberOfGPUs();
        size_t *idGPUs = getGPUIds();


        // [computation]
        clock_gettime(CLOCK_MONOTONIC, &startComputation);
        #pragma omp parallel for num_threads (nGPUs)
        for(i = 0; i<nGPUs; i++){ // get number of GPUs should receive a parameter such as appData or something

            appData = localData;
            arrDTI = localArrDTI;
            nDTI = localnDTI;
            maxDTI = localmaxDTI;

            // set device
            cudaSetDevice(idGPUs[i]);

            // run kernel
            runKernel((float*)(getDTIByIndex(0)->gpuData[i]), getDTIByIndex(0)->nPerGPU[i], data->K, nGPUs);     

            // sync devices
            cudaDeviceSynchronize();
        }
        clock_gettime(CLOCK_MONOTONIC, &endComputation);
        etComputation += (endComputation.tv_sec - startComputation.tv_sec) + (endComputation.tv_nsec - startComputation.tv_nsec) / 1e9;

        // [communication]
        clock_gettime(CLOCK_MONOTONIC, &startCommunication);
        // check if data should be synchronized
        if((t+1) % data->nIterationsForCommunications == 0 && nGPUs > 1){

            // copy data to the GPU 0
            cudaError_t err;

            float **dData;
            dData = (float**)getDTIByIndex(0)->gpuData;
            size_t *nPerGPU = getDTIByIndex(0)->nPerGPU;
            ncclComm_t *comms = getNCCLComms();
            cudaStream_t *streams = getCudaStreams();

            // start NCCL group
            ncclGroupStart();

            //#pragma omp parallel for num_threads (nGPUs)
            for (size_t i = 0; i < nGPUs; i++) {
                
                // set device
                cudaSetDevice(idGPUs[i]);
                ncclAllReduce(dData[i], dData[i], nPerGPU[i], ncclFloat, ncclSum, comms[i], streams[idGPUs[i]]);
            }
            fflush(stdout);
             
            ncclGroupEnd(); // Aquí es donde NCCL ejecuta la comunicación en paralelo

            // division (we are computing the mean)
            /*for (int i = 0; i < numGpus; i++) {
                cudaSetDevice(i); // Selecciona el GPU actual

                int threadsPerBlock = 256;
                int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;

                // Lanzamos el kernel en el GPU 'i' para que divida su propio array entre 'numGpus'
                computeMeanKernel<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(gpuValues[i], count, numGpus);
            }*/

            // sync streams
            for (size_t i = 0; i < nGPUs; i++) {

                cudaSetDevice(idGPUs[i]);
                err = cudaStreamSynchronize(streams[idGPUs[i]]);

                if(err != cudaSuccess)
                    printf("Job: Allreduce failed:  %s\n", cudaGetErrorString(err));
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &endCommunication);
        etCommunication += (endCommunication.tv_sec - startCommunication.tv_sec) + (endCommunication.tv_nsec - startCommunication.tv_nsec) / 1e9;
    }

    double comp = etComputation / (double)T;
    double comm = etCommunication / (double)T;


    pthread_mutex_lock(&(printLock));
    printf("%lf %lf %zu\n", comp, comm, T);
    fflush(stdout);
    pthread_mutex_unlock(&(printLock));

}


void simulateIterativeUnifiedMemory(appStruct_t *data){
    
    // get number of time steps and state
    size_t i;
    size_t T = data->T;

    double etComputation = 0.0;
    struct timespec startComputation, endComputation;

    clock_gettime(CLOCK_MONOTONIC, &startComputation);

    // simulate iterations
    for(size_t t = 0; t<T; t++){

        /*if(t % 10 == 0){
            printf(" T = %zu / %zu\n", t, T);
            fflush(stdout);
        }*/

        // reconfiguration point, check if there is any pending reconfiguration
        /*if(checkIfReconfiguration(getJobControl())){

            reconfigure(GPU2GPU);
        }*/

        // copy data from global variables to local variables for enabling visibility to openMP threads (they are thread private)
        public_APP_Data_t *localData = appData;
        DTI_t **localArrDTI = arrDTI;
        size_t localnDTI = nDTI;
        size_t localmaxDTI = maxDTI;
  

        // get the information of the GPUs
        size_t nGPUs = getNumberOfGPUs();
        size_t *idGPUs = getGPUIds();


        // [computation]
        #pragma omp parallel for num_threads (nGPUs)
        for(i = 0; i<nGPUs; i++){ // get number of GPUs should receive a parameter such as appData or something

            appData = localData;
            arrDTI = localArrDTI;
            nDTI = localnDTI;
            maxDTI = localmaxDTI;

            // set device
            cudaSetDevice(idGPUs[i]);

            // run kernel
            runGraphKernel((float*)(data->gValues), data->N / nGPUs, data->nPerNode, data->offPerNode, data->tmpAcc, data->indices, data->K, i);     

            // sync devices
            cudaDeviceSynchronize();
        }

        // [computation]
        #pragma omp parallel for num_threads (nGPUs)
        for(i = 0; i<nGPUs; i++){ // get number of GPUs should receive a parameter such as appData or something

            appData = localData;
            arrDTI = localArrDTI;
            nDTI = localnDTI;
            maxDTI = localmaxDTI;

            // set device
            cudaSetDevice(idGPUs[i]);

            // run kernel
            runUpdateNodesKernel((float*)(data->gValues), data->N / nGPUs, data->nPerNode, data->tmpAcc, i);     

            // sync devices
            cudaDeviceSynchronize();
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &endComputation);
    etComputation += (endComputation.tv_sec - startComputation.tv_sec) + (endComputation.tv_nsec - startComputation.tv_nsec) / 1e9;

    pthread_mutex_lock(&(printLock));
    printf("%lf\n", etComputation);    
    fflush(stdout);
    pthread_mutex_unlock(&(printLock));

}

/* Application MAIN functions */


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

    // communication info
    size_t pinned = *(size_t*)argv[4];
    size_t async = *(size_t*)argv[5];
    size_t steps = *(size_t*)argv[6];
    size_t cores = *(size_t*)argv[7];

    //int enumValue = *(int*)argv[8];
    //reconfDirEnum reconfDir = (reconfDirEnum)enumValue; 

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
        //appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->N, sizeof(float), "appData", initializeDTIDescription(entire, ordered, commType));
        appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->N, sizeof(float), "appData", initializeComplexDTIDescription(complex, ordered, commType, appData->s));
    else
        appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->N, sizeof(float), "appData", initializeDTIDescription(simple, ordered, commType));


    // configure all DTIs for automatic data transference // TODO: configure in initialization
    configureDTIs(getState()->jobResources, NULL, CPU2GPU); // number of GPUs, old number of GPUs

    // send data to the GPU
    transferDataCPU2GPU();

    // run application
    simulateIterative(appData);

    // move data to the CPU again
    transferDataGPU2CPU();


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

    //int enumValue = *(int*)argv[10];
    //reconfDirEnum reconfDir = (reconfDirEnum)enumValue; 

    appData->phases = (size_t*)calloc(appData->P, sizeof(size_t));
    for(i = 0; i<appData->P; i++)
        appData->phases[i] = *(size_t*)argv[i + 11];

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
        //appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->N, sizeof(float), "appData", initializeDTIDescription(entire, ordered, commType));
        appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->N, sizeof(float), "appData", initializeComplexDTIDescription(complex, ordered, commType, appData->s));
    else
        appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->N, sizeof(float), "appData", initializeDTIDescription(simple, ordered, commType));


    // configure all DTIs for automatic data transference // TODO: configure in initialization
    configureDTIs(getState()->jobResources, NULL, CPU2GPU); // number of GPUs, old number of GPUs

    // send data to the GPU
    transferDataCPU2GPU();

    // program code (simulations)
    simulatePhases(appData);

    // transfer data fron the GPUs to the CPU
    transferDataGPU2CPU();
    
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

// pending for being tested
void launch_NCCL_communications_app(int argc, void* argv[]){

    size_t i;

    // initialize DITTO environment
    // temporal: simulate information received from the scheduler: number of GPUs and identifiers of the GPUs (pass argv to the initDITTO function?)
    initDITTO(argv[argc+1]);
    initNCCLComm(getState()->jobResources);

    appStruct_t *appData = (appStruct_t*)calloc(1, sizeof(appStruct_t));
    appData->N = *(size_t*)argv[0];
    appData->T = *(size_t*)argv[1];
    appData->K = *(size_t*)argv[2];
    appData->nIterationsForCommunications = *(size_t*)argv[3];
    appData->s = *(size_t*)argv[4];

    // communication info
    size_t pinned = *(size_t*)argv[5];
    size_t async = *(size_t*)argv[6];
    size_t steps = *(size_t*)argv[7];
    size_t cores = *(size_t*)argv[8];
    
    //int enumValue = *(int*)argv[9];
    //reconfDirEnum reconfDir = (reconfDirEnum)enumValue; 

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
        appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->N, sizeof(float), "appData", initializeDTIDescription(entire, ordered, commType));
        //appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->N, sizeof(float), "appData", initializeComplexDTIDescription(complex, ordered, commType, appData->s));
    else
        appDataDTI = createAutomaticDTI((void*)(appData->arr), appData->N, sizeof(float), "appData", initializeDTIDescription(simple, ordered, commType));


    // configure all DTIs for automatic data transference // TODO: configure in initialization
    configureDTIs(getState()->jobResources, NULL, CPU2GPU); // number of GPUs, old number of GPUs

    // send data to the GPU
    transferDataCPU2GPU();


    // When we want to test network congestion, we want to make the job wait until
    // the rest of jobs start
    /*char startRunning = getJobControl()->startRunning; 
    while(!startRunning){
        
        sleep(1);
        pthread_mutex_lock(&(getJobControl()->lockStartRunning));
        startRunning = getJobControl()->startRunning; 
        pthread_mutex_unlock(&(getJobControl()->lockStartRunning));
    }*/



    // wait for signal before starting
    simulateIterativeNCCL(appData);

    // transfer data fron the GPUs to the CPU
    transferDataGPU2CPU();

    //printf(" [RES]: %zu bytes; from %zu to %zu; %lf %lf\n", 
    //        appData->N * sizeof(float), getGPUIds()[0], getGPUIds()[1], appData->communicationTimeSrcDst, appData->communicationTimeDstSrc);


    if(pinned)
        cudaFreeHost(appDataDTI->cpuData);
    else
        free(appDataDTI->cpuData);

    // destroy streams
    freeNCCLComm(getState()->jobResources);
    freeDITTO();


    // signal that job finished
    jobFinished(getJobControl());

    pthread_exit(NULL);
}


// pending for being tested
void launch_unified_memory_app(int argc, void* argv[]){


    size_t i;

    // initialize DITTO environment
    // temporal: simulate information received from the scheduler: number of GPUs and identifiers of the GPUs (pass argv to the initDITTO function?)
    initDITTO(argv[argc+1]);


    appStruct_t *appData = (appStruct_t*)calloc(1, sizeof(appStruct_t));
    appData->N = *(size_t*)argv[0];
    appData->T = *(size_t*)argv[1];
    appData->K = *(size_t*)argv[2];

    // communication info
    //size_t pinned = *(size_t*)argv[3];
    //size_t async = *(size_t*)argv[4];
    //size_t steps = *(size_t*)argv[5];
    //size_t cores = *(size_t*)argv[6];
    
    //int enumValue = *(int*)argv[7];
    //reconfDirEnum reconfDir = (reconfDirEnum)enumValue; 

    int pGroup = *(int*)argv[8];
    int pNGroup = *(int*)argv[9];

    appData->malleable = *(size_t*)argv[argc];


    // 
    size_t N = appData->N;
    size_t nGPUs = getState()->jobResources->nGPUs;


    // allocate unifed memory
    cudaError_t err = cudaMallocManaged((void**)&(appData->gValues), N * sizeof(float));
    
    if(err != cudaSuccess){
        printf("err = %s\n", cudaGetErrorString(err));
        fflush(stdout);
    }

    // initialize
    for(size_t i = 0; i<N; i++){
        appData->gValues[i] = (float)(rand() % 100);
    }

    // create array of indices for each element
    size_t capacity = (size_t)1024 * (size_t)1024;
    size_t next = 0;

    size_t *indices = (size_t*)malloc(capacity * sizeof(size_t));
    size_t *n = (size_t*)calloc(N, sizeof(size_t));
    size_t tN = 0;

    for(i = 0; i<getState()->jobResources->nGPUs; i++){

        // manage the region owned by this GPU
        for(size_t j = i * N / nGPUs; j < (i+1) * (N / nGPUs); j++){

            for(size_t l = 0; l<N; l++){

                int connected = 0;
                int val = rand() % 100;

                if(l >= i * (N / nGPUs) && l < (i+1) * (N / nGPUs)){

                    if(val < pGroup){
                        connected = 1;
                    }
                }
                else{

                    if(val < pNGroup){
                        connected = 1;
                    }
                }

                if (connected) {

                    if (next == capacity) {
                        capacity *= 2;
                        indices = (size_t*)realloc(indices,capacity * sizeof(size_t));
                    }

                    indices[next] = l;
                    n[j]++;
                    tN ++;
                    next++;
                }
            }
        }
    }

    cudaMallocManaged((void**)&(appData->indices), tN * sizeof(size_t));
    cudaMallocManaged((void**)&(appData->nPerNode), N * sizeof(size_t));
    cudaMallocManaged((void**)&(appData->offPerNode), N * sizeof(size_t));
    cudaMallocManaged((void**)&(appData->tmpAcc), N * sizeof(size_t));

    // copy indexes to unified memory
    next = 0;
    for(size_t i = 0; i<N; i++){

        appData->nPerNode[i] = n[i];
        appData->offPerNode[i] = next;

        for(size_t j = 0; j<n[i]; j++){

            appData->indices[next] = indices[next];
            next++;
        }
    }

    free(n);
    free(indices);


    // each GPU will write from [i * N] to [(i + 1) * N], being i the GPU index

    // set communication type
    /*communicationType_t commType;
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
        commType.transferCores = singleCoreComm;*/


    // wait for signal before starting
    simulateIterativeUnifiedMemory(appData);

    // transfer data fron the GPUs to the CPU
    //transferDataGPU2CPU();

    //printf(" [RES]: %zu bytes; from %zu to %zu; %lf %lf\n", 
    //        appData->N * sizeof(float), getGPUIds()[0], getGPUIds()[1], appData->communicationTimeSrcDst, appData->communicationTimeDstSrc);


	cudaFree(appData->indices);
	cudaFree(appData->nPerNode);
	cudaFree(appData->offPerNode);
	cudaFree(appData->gValues);
    cudaFree(appData->tmpAcc);

    // destroy streams
    freeDITTO();

    // signal that job finished
    jobFinished(getJobControl());

    pthread_exit(NULL);
}

// [SPECIFIC APPLICATIONS FOR EVALUATING MORE CONCRETE THINGS]

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
    
    
    pthread_mutex_lock(&(printLock));
    printf("%lf\n", etime);
    fflush(stdout);
    pthread_mutex_unlock(&(printLock));
    

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
    double etimeMalloc = 0.0, etimeFree = 0.0, etimeFirstMalloc = 0.0, etimeFirstFree = 0.0;

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
    configureDTIs(getState()->jobResources, NULL, CPU2GPU); // number of GPUs, old number of GPUs

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


    pthread_mutex_lock(&(printLock));
    printf("%lf %lf %lf %lf", etimeFirstMalloc, etimeMalloc, etimeFirstFree, etimeFree);
    fflush(stdout);
    pthread_mutex_unlock(&(printLock));

    // destroy streams
    freeDITTO();

    // signal that job finished
    jobFinished(getJobControl());

    pthread_exit(NULL);
}