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
typedef void* (*GenericFunction)(void*);


// global variables
extern reconfData_t *reconfData;
extern int pendingReconf;
extern pthread_mutex_t lockPendingReconf;
// sch
extern pthread_t thrSch;

// Application data: state and reconfiguration data
extern state_t *state;
extern reconfData_t *appReconfData;

// DTIs: manage data transmissions
extern DTI_t **arrDTI;
extern size_t nDTI;
extern size_t maxDTI;



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

// initialize application structure
void initState(size_t nGPUs, size_t *idGPUs);

// get the state structure
state_t* getState();

// init reconfiguration data
void initReconfigurationData();

// Get reconfiguration data: return the appReconfData structure
reconfData_t* getReconfigurationData();

// Run mock scheduler to launch reconfigurations
void *runMockScheduler(void *arg);

// create DTI for automatically handling data transmissions
DTI_t* createAutomaticDTI(void* cpuData, size_t N, size_t size, const char* name, transmissionPatternsEnum tpttEnum, remainingElementsEnum rmEnum);
DTI_t* createManualDTI(void* cpuData, size_t N, size_t size, const char *name, GenericFunction cpu2gpu, GenericFunction gpu2cpu);
void addDTI(DTI_t *DTI);
DTI_t* getDTI(const char *dtiName);

/* [Reconfigurations] */

// Scheduler notifies that there is a pending reconfiguration
void notifyReconfiguration(size_t nGPUs, size_t *idGPUs, size_t *src, size_t *dst);

// application checks whether there is any pending reconfiguration
int checIfkReconfiguration();

// application notifies that reconfiguration finished
void notifyReconfigurationDone();

void reconfigure();
void reconfigureDTIs(size_t nGPUs, size_t nOldGPUs);
void* reconfigureKernels(GenericFunction f, void* params);

state_t* storeState(state_t *state);

/* [Data transmission] */
void* transferDataCPU2GPU(GenericFunction userFunction, void *ret, void* args);
void* transferDataGPU2CPU(GenericFunction userFunction, void *ret, void* args);

#endif