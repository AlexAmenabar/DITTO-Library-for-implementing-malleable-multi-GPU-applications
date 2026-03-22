#ifndef DITO_APP_H
#define DITO_APP_H

// declare a type to represent userFunctions for data transmission
typedef void* (*GenericFunction)(void*);
typedef struct jobControl_t jobControl_t;

/* Structures */

// Structure with the state information
typedef struct state_t {

    size_t nGPUs; // number of GPUs in the current state
    size_t *idGPUs; // GPU identifiers

    // function to update application state 
    GenericFunction updateState;
    GenericFunction stateDeallocator;
    
    /// [TODO] Define here the variables to define the state of your application
} state_t;


// Structure with reconfiguration information
/*typedef struct reconfData_t {

    size_t nGPUs;
    size_t *idGPUs;

} reconfData_t;*/

/// @brief Enumeration that indicates the pattern for distributing data along GPUs
enum transmissionPatternsEnum {
    
    nonettp,
    all,
    simple,
    complex,
    custom
};

/// @brief Enum indicating what to do with the remaining elements (nElements % nGPUs)
enum remainingElementsEnum {
    
    nonerme,
    first,
    ordered,
    last
};

// Structure for describing data array redistributions for simple patterns
typedef struct DTIDesctiption_t{

    // simple description: used to create the complex description
    size_t *n;
    size_t *j;
    size_t *off;
    size_t *tn; // total number of elements

    // complex description: describes how data is organized in the GPU
    size_t *nPartitionsPerGPU; // [nGPUs] number of partitions on each GPU
    size_t *nElementsPerGPU; // [nGPUs] number of elements on each GPU (the sum of the elements on all partitions)
    size_t **nElementsPerPartition; // [nGPUs x ---] number of elements on each partition per GPU
    size_t **firstElementPerPartition; // [nGPUs x ---] index of the first element on each partition

    // enumerations providing more details about the DTI
    transmissionPatternsEnum tpttEnum;
    remainingElementsEnum rmEnum;

} DTIDesctiption_t;


/// @brief Structure to store information of how data is distributed acoss GPUs. Data (usually arrays) are divided in partitions, where each GPU receives
/// a set of partitions
typedef struct DTI_t {

    // 0: automatically managed arrays, 1: manually managed
    int type;
    int dataLocation; // 0: cpu; 1: gpu

    // pointers to CPU and GPU data
    void **gpuData; // [n_GPUS] pointers to device arrays (one per dev)
    void *cpuData; // array on the CPU
    size_t size; // data type size
    const char *name; // name of the DTI
    
    // number of elements in the CPU: number of elements in array, or struct, or whatever
    size_t N; // number of elements in the CPU array (if it is an array)
    
    // function to perform the data user-defined data redistributions
    GenericFunction moveCPU2GPU;
    GenericFunction moveGPU2CPU;

    // Helper information for managing communications in array-based 
    DTIDesctiption_t *description;

} DTI_t;

// app data to communicate the scheduler and the
typedef struct public_APP_data_t {

    // app state
    state_t *state;

    // reconfiguration management
    int pendingReconf = 0; // whether there is any pending reconfiguration
    pthread_mutex_t lockPendingReconf; // lock for synchronization
    //reconfData_t *reconfData;
    jobControl_t *jobControl; // communication with the scheduler

} public_APP_Data_t;

/* [DITTO initialization] */

/**
*
* Initialize environment for communicating the scheduler and the application 
* 
* The function initializes the necessary global variables for communicating both the scheduler and the application,
* which uses a lock variable to avoid race conditions when writing global variables. 
*
* @param[in] nGPUs Number of GPUs available for the execution at the begining
* @param[in] idGPUs Identifiers of the GPUs
*/
void initDITTO(void *jobControl);
void freeDITTO();

// get the state structure
state_t* getState();
void updateState();
state_t* storeState(state_t *state);


/* [RECONFIGURATIONS] */

//reconfData_t* getReconfigurationData();

// Scheduler notifies that there is a pending reconfiguration
void notifyReconfiguration(size_t nGPUs, size_t *idGPUs, size_t *src, size_t *dst);

// application checks whether there is any pending reconfiguration
int checIfkReconfiguration();

// application notifies that reconfiguration finished
void notifyReconfigurationDone();

// reconfigure application: state, data distribution and kernels
void reconfigure();

// reconfigure DTIs to the new number of GPUs
void reconfigureDTIs(size_t nGPUs, size_t nOldGPUs);

// reconfigure kernels to the new number of GPUs
void* reconfigureKernels(GenericFunction f, void* params);


/* [DTI MANAGEMENT] */

DTI_t* createAutomaticDTI(void* cpuData, size_t N, size_t size, const char* name, DTIDesctiption_t *description);
DTI_t* createManualDTI(void* cpuData, void** gpuData, size_t N, size_t size, const char *name, GenericFunction cpu2gpu, GenericFunction gpu2cpu);
void addDTI(DTI_t *DTI);
DTI_t* getDTI(const char *dtiName);
DTI_t* getDTIByIndex(int i);

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

DTIDesctiption_t* initializeDTIDescription(transmissionPatternsEnum tpttEnum, remainingElementsEnum rmEnum);
//DTIDesctiption_t* initializeComplexDTIDescription(transmissionPatternsEnum tpttEnum, remainingElementsEnum rmeEnum, size_t *n, size_t *j, size_t *off);
//DTIDesctiption_t* initializeCustomDTIDescription(transmissionPatternsEnum tpttEnum, remainingElementsEnum rmeEnum, size_t *n, size_t *j, size_t *off);

void printDTI(DTI_t *DTI);


/* [DATA TRANSMISSION - CALL DATA TRANSMISSION MODULE] */
void transferDataCPU2GPU();
void transferDataGPU2CPU();

#endif