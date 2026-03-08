#ifndef DITO_APP_H
#define DITO_APP_H

#include <omp.h>
#include <pthread.h>

#include "DDM.hpp"

typedef struct state_t {

    size_t nGPUs; // number of GPUs in the current state
    size_t *idGPUs; // GPU identifiers

    /// [TODO] Define here the variables to define the state of your application
} state_t;


typedef struct reconfData_t {

    size_t nGPUs;
    size_t *idGPUs;

} reconf_data_t;

// declare a type to represent userFunctions for data transmission
typedef void* (*UserFunction)(void*);


extern reconfData_t *reconfData;
extern int pendingReconf;
extern pthread_mutex_t lockPendingReconf;

extern state_t *state;
extern reconfData_t *appReconfData;

// sch
extern pthread_t thrSch;

// DTIs
extern DTI_t **arrDTI;
extern size_t nDTI;
extern size_t maxDTI;

// data transference functions
extern UserFunction *userCPU2GPUFunction;
extern UserFunction *userGPU2CPUFunction;
extern size_t nFuncs;
extern size_t maxFuncs;


/* DITTO initialization */

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
void initDITTO(size_t nGPUs, size_t *idGPUs);
void initState(size_t nGPUs, size_t *idGPUs);
state_t* getState();
void initReconfigurationData();
reconfData_t* getReconfigurationData();
void *runMockScheduler(void *arg);

// automatize in the future
void setCommunicationFunctions(UserFunction funcCPU2GPU, UserFunction funcGPU2CPU);


DTI_t* createDTI(void* cpuData, size_t N, size_t size, transmissionPatternsEnum tpttEnum, remainingElementsEnum rmEnum);
void addDTI(DTI_t *DTI);

/* [Reconfigurations] */
void notifyReconfiguration(size_t nGPUs, size_t *idGPUs, size_t *src, size_t *dst);
int checIfkReconfiguration();
void notifyReconfigurationDone();

void reconfigure();
void reconfigureKernels();
state_t* storeState(state_t *state);

/* [Data transmission] */
void* transferDataCPU2GPU(UserFunction userFunction, void *ret, void* args);
void* transferDataGPU2CPU(UserFunction userFunction, void *ret, void* args);

#endif