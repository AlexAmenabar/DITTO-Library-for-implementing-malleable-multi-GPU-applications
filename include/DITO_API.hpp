#ifndef DITO_APP_H
#define DITO_APP_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <nccl.h>



// forwarded declarations
typedef void* (*GenericFunction)(void*);
typedef struct jobControl_t jobControl_t; // RMS communication
typedef struct jobResources_t jobResources_t; // RMS communication



/* Structures */
// Structure with the state information
typedef struct state_t {

    // Current job resources and job resources for reconfigurations copied from the RMS
    jobResources_t *jobResources;
    jobResources_t *reconfJobResources;

    // function to update application state 
    GenericFunction updateState;
    GenericFunction stateDeallocator;
    
    /// [TODO] Define here the variables to define the state of your application

} state_t;


/// @brief Enumeration that indicates the pattern for distributing data along GPUs
enum transmissionPatternsEnum {
    
    nonettp,
    entire,
    simple,
    p2pSimple,
    complex,
    p2pComplex,
    custom
};

/// @brief Enum indicating what to do with the remaining elements (nElements % nGPUs)
enum remainingElementsEnum {
    
    nonerme,
    first,
    ordered,
    last
};

/// @brief Enumeration that indicates the memory allocation type
enum cudaMemoryTypeEnum {

    pinnedComm,
    nonPinnedComm
};

/// @brief Enumeration that indicates if data distributions are synchronous or asynchronous
enum transmissionTypeEnum {

    syncComm,
    asyncComm
};

/// @brief Enumeration that indicates the communication type
enum transferStepsEnum {

    oneStepComm,
    twoStepsComm,
    stridedComm
};

/// @brief Enumeration that indicates the number of cores used for the data distribution
enum transferCoresEnum {

    singleCoreComm,
    multiCoreComm
};

/// @brief Structure that encapsulates all communication related information
typedef struct communicationType_t {

    cudaMemoryTypeEnum cudaMemoryType;
    transmissionTypeEnum transmissionType;
    transferStepsEnum transferSteps;
    transferCoresEnum transferCores;
} communicationType_t;

/// @brief Enumeration that indicates the data movement directions
enum reconfDirEnum {

    CPU2GPU,
    GPU2GPU,
    GPU2CPU,
    CPU
};

enum reconfTypeEnum {

    expand,
    shrink,
    keep
};


/// @brief Structure for describing the pattern for distributing data
typedef struct DTIDesctiption_t{

    // transmission data
    transmissionPatternsEnum tpttEnum;
    remainingElementsEnum rmEnum;
    communicationType_t commType;

    // the number of partitions TODO: change the variable name
    size_t s;

} DTIDesctiption_t;


/// @brief Structure with all the information for redistributing a data structure
typedef struct DTI_t {

    // 0: automatically managed arrays, 1: manually managed
    int type; // automatic (0) | manual (1)
    int dataLocation; // CPU (0) | GPU (1)

    void **gpuData; // [n_GPUS] data structure on each GPU
    void **prevGpuData; // [n_GPUS] data structure on each GPU for the previous configuration
    void *cpuData; // data structure in the CPU
    size_t N; // number of elements in the CPU array
    size_t size; // data type size
    const char *name; // name of the DTI
        
    // user-provided reconfiguration functions
    GenericFunction moveCPU2GPU; // CPU2GPU
    GenericFunction moveGPU2CPU; // GPU2CPU
    GenericFunction moveGPU2GPU; // GPU2GPU

    // user-provided description of the DTI 
    DTIDesctiption_t *description;

    // information of how data is organized among several GPUs
    size_t *nPerGPU; // [nGPUs] number of elements on each GPU (the sum of the elements on all partitions)
    size_t *nPartitionsPerGPU; // [nGPUs] number of partitions on each GPU
    size_t **nPerPartition; // [nGPUs x nPartitions] number of elements on each partition per GPU
    size_t **offsetPerPartition; // [nGPUs x nPartitions] index of the first element on each partition

    // data corresponding to the previous configuration
    size_t *prev_nPerGPU; // [nGPUs] number of elements on each GPU (the sum of the elements on all partitions)
    size_t *prev_nPartitionsPerGPU; // [nGPUs] number of partitions on each GPU
    size_t **prev_nPerPartition; // [nGPUs x nPartitions] number of elements on each partition per GPU
    size_t **prev_offsetPerPartition; // [nGPUs x nPartitions] index of the first element on each partition

} DTI_t;

/// @brief Structure that constains reconfiguration data
typedef struct reconfData_t {

    size_t **gpusToSplit; // [nGPUs]: the GPUs to move data for each GPU
    size_t *virtualTopology; // order in which GPUs have data
    // the number in the position of the GPUe indicates what position of the original array contains the GPU

} reconf_data_t;


/// @brief Structure that constains data of the application
typedef struct public_APP_data_t {

    // application state
    state_t *state;
    
    // job control (shared variable with the RMS)
    jobControl_t *jobControl; // communication with the scheduler

    // CUDA streams for asynchronous data movements // TODO: this is not the ideal place for this
    cudaStream_t *cudaStreams;
    ncclComm_t *ncclComms;

} public_APP_Data_t;


// DITTO management data, private for each thread
extern thread_local public_APP_data_t *appData; // app data related to DITTO (state and jobControl)
extern thread_local reconfData_t *reconfData; // reconfiguration data
extern thread_local size_t *virtualTopology;
extern thread_local DTI_t** arrDTI; // array of DTIs
extern thread_local size_t nDTI; // number of DTIs
extern thread_local size_t maxDTI; // maximum number of DTIs

/// initialize DITTO environment: state, array of DTIs...
void initDITTO(void *jobControl);

/// free DITTO environment TODO
void freeDITTO();

/// get Job control from the appData variable // TODO: I don't think this should be accesible
jobControl_t* getJobControl();

/* [STATE] */
/// initialize app state: number of GPUs and GPU identifiers
state_t* initState(jobResources_t *jobResources); 

/// get state variable from the DITTO application data
state_t* getState();

/// update the application state: the number of GPUs and the GPU identifiers using the information in the jobControl structure
state_t* updateState(state_t *state, jobControl_t *jobControl);

/// store the application state
state_t* storeState(state_t *state);

/// deallocate state memory
void freeState(state_t *state);

/// update application resources after reconfiguration
void updateApplicationResources(state_t *state);

/// Copy resources dedicated to the job from the jobControl (shared variable)
void cpyJobResourcesToState(state_t *state, jobResources_t *rmsJobResources);

/// Copy resources dedicated to the job from the jobControl (shared variable)
void cpyReconfResourcesToState(state_t *state, jobResources_t *reconfJobResources);

/// @brief Copy job resources to a new structure and return it
jobResources_t* cpyJobResources(jobResources_t *jobResources);

/// @brief Deallocate job resources
void freeJobResources(jobResources_t *jobResources);

/// get the number of GPUs available for the job
size_t getNumberOfGPUs();

/// get the GPU identifiers available for the job
size_t* getGPUIds();

cudaStream_t* getCudaStreams();

ncclComm_t* getNCCLComms();


/* [RECONFIGURATIONS] */

/// Reconfigure application: move data from the GPU to the CPU, update state, reconfigure DTIs for the new available resources, and move data again to the GPU. If necessary, reconfigure kernels too
void reconfigure(reconfDirEnum reconfDir);

/// Reconfigure DTIs for the new resources available
void reconfigureDTIs(jobResources_t *reconfJobResources, jobResources_t *jobResources);

/// Reconfigure kernels to the new number of GPUs
void* reconfigureKernels(GenericFunction f, void* params);

/// Move DTIs data from the CPU to the GPUs
void transferDataCPU2GPU();

/// Move DTIs data from the GPUs to the CPU
void transferDataGPU2CPU();

void transferDataGPU2GPU(reconfTypeEnum reconfEnum);



/* [DTI MANAGEMENT] */

DTI_t* createAutomaticDTI(void* cpuData, size_t N, size_t size, const char* name, DTIDesctiption_t *description);
DTI_t* createManualDTI(void* cpuData, void** gpuData, size_t N, size_t size, const char *name, GenericFunction cpu2gpu, GenericFunction gpu2cpu);
DTI_t* initializeDTI(size_t type, size_t dataLocation, void* cpuData, void** gpuData, size_t N, size_t size, const char* name, GenericFunction cpu2gpu, GenericFunction gpu2cpu, DTIDesctiption_t *description);
void freeDTI(DTI_t *DTI, size_t nGPUs);

void addDTI(DTI_t *DTI);
void configureDTIs(jobResources_t *jobResources, jobResources_t *reconfJobResources, reconfDirEnum reconfDir);
DTI_t* getDTI(const char *dtiName);
DTI_t* getDTIByIndex(int i);

void printDTI(DTI_t *DTI);

void setfDTICPU2GPU(DTI_t *DTI, GenericFunction f);
void setfDTIGPU2CPU(DTI_t *DTI, GenericFunction f);
void setDTIDescription(DTI_t *DTI, DTIDesctiption_t *description);
DTIDesctiption_t* getDTIDescription(DTI_t *DTI);
void setCPUData(DTI_t *DTI, void* cpuData);
void* getCPUData(DTI_t *DTI);
void setMultiGPUData(DTI_t *DTI, void** gpuData);
void** getMultiGPUData(DTI_t *DTI);
void setGPUData(DTI_t *DTI, void *gpuData, int i);
void* getGPUData(DTI_t *DTI, int i);


DTIDesctiption_t* initializeDTIDescription(transmissionPatternsEnum tpttEnum, remainingElementsEnum rmEnum, communicationType_t commType);
DTIDesctiption_t* initializeComplexDTIDescription(transmissionPatternsEnum tpttEnum, remainingElementsEnum rmEnum, communicationType_t commType, size_t nPartitions);
void freeDescription(DTIDesctiption_t *description, size_t nGPUs);

#endif
