#ifndef DITO_APP_H
#define DITO_APP_H

// declare a type to represent userFunctions for data transmission
typedef void* (*GenericFunction)(void*);
typedef struct jobControl_t jobControl_t;

/* Structures */

// Structure with the state information
typedef struct state_t {

    size_t nGPUs; // number of GPUs available for the application
    size_t *idGPUs; // available GPU identifiers

    // function to update application state 
    GenericFunction updateState;
    GenericFunction stateDeallocator;
    
    /// [TODO] Define here the variables to define the state of your application

} state_t;


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

    /* developer provided description */
    // enumerations providing more details about the DTI
    transmissionPatternsEnum tpttEnum;
    remainingElementsEnum rmEnum;
    size_t nPartitions; // the total number of elements N is diveded in nPartitions

} DTIDesctiption_t;


/// @brief Structure to store information of how data is distributed acoss GPUs. Data (usually arrays) are divided in partitions, where each GPU receives
/// a set of partitions
typedef struct DTI_t {

    // 0: automatically managed arrays, 1: manually managed
    int type;
    int dataLocation; // 0: cpu; 1: gpu

    // pointers to CPU and GPU data
    void **gpuData; // [n_GPUS] pointers to device arrays (one per dev)
    void *cpuData; // data array on the CPU
    size_t N; // number of elements in the CPU array (if it is an array)
    size_t size; // data type size
    const char *name; // name of the DTI
        
    // user-provided functions for managing the data redistributions
    GenericFunction moveCPU2GPU;
    GenericFunction moveGPU2CPU;

    // user-provided description of the DTI 
    DTIDesctiption_t *description;

    // information of how data is organized among several GPUs
    size_t *nPerGPU; // [nGPUs] number of elements on each GPU (the sum of the elements on all partitions)
    size_t **nPerPartition; // [nGPUs x nPartitions] number of elements on each partition per GPU
    size_t **offsetPerPartition; // [nGPUs x nPartitions] index of the first element on each partition

} DTI_t;

// app data to communicate the scheduler and the
typedef struct public_APP_data_t {

    // app state
    state_t *state;

    //reconfData_t *reconfData;
    jobControl_t *jobControl; // communication with the scheduler

} public_APP_Data_t;

/* [INITIALIZATION] */

/**
*
* Initialize environment for communicating the scheduler and the application 
* 
* The function initializes the necessary global variables for communicating both the scheduler and the application,
* which uses a lock variable to avoid race conditions when writing global variables. 
*
* @param[in] jobControl Structure that contains data about 
*/
void initDITTO(void *jobControl);
void freeDITTO();

/* [STATE] */
state_t* initState(size_t nGPUs, size_t *idGPUs); 
void freeState(state_t *state);
state_t* getState();
void updateState();
state_t* storeState(state_t *state);
size_t getNumberOfGPUs();
size_t* getGPUIds();

/* [NOTIFICATIONS] */

// application checks whether there is any pending reconfiguration
int checIfkReconfiguration();

// application notifies that reconfiguration finished
void notifyReconfigurationDone();
void notifySigGPUs();
void notifyReqGPUs();


/* [RECONFIGURATIONS] */

// reconfigure application: state, data distribution and kernels
void reconfigure();

// reconfigure DTIs to the new number of GPUs
void reconfigureDTIs(size_t nGPUs, size_t nOldGPUs);

// reconfigure kernels to the new number of GPUs
void* reconfigureKernels(GenericFunction f, void* params);
void transferDataCPU2GPU();
void transferDataGPU2CPU();


/* [DTI MANAGEMENT] */

DTI_t* createAutomaticDTI(void* cpuData, size_t N, size_t size, const char* name, DTIDesctiption_t *description);
DTI_t* createManualDTI(void* cpuData, void** gpuData, size_t N, size_t size, const char *name, GenericFunction cpu2gpu, GenericFunction gpu2cpu);
DTI_t* initializeDTI(size_t type, size_t dataLocation, void* cpuData, void** gpuData, size_t N, size_t size, const char* name, GenericFunction cpu2gpu, GenericFunction gpu2cpu, DTIDesctiption_t *description);
void freeDTI(DTI_t *DTI, size_t nGPUs);

void addDTI(DTI_t *DTI);
void configureDTIs(size_t nGPUs, size_t nOldGPUs);
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


DTIDesctiption_t* initializeDTIDescription(transmissionPatternsEnum tpttEnum, remainingElementsEnum rmEnum);
DTIDesctiption_t* initializeComplexDTIDescription(transmissionPatternsEnum tpttEnum, remainingElementsEnum rmEnum, size_t nPartitions);
void freeDescription(DTIDesctiption_t *description, size_t nGPUs);

#endif