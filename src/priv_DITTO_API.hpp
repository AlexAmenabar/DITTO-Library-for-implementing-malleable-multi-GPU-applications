#ifndef PRIV_DITTO_API_H
#define PRIV_DITTO_API_H


// Forwarded declarations
typedef struct jobControl_t jobControl_t;

typedef struct state_t state_t;
//typedef struct reconfData_t reconfData_t;
typedef struct DTI_t DTI_t;
typedef struct DTIDesctiption_t DTIDesctiption_t;
typedef void* (*GenericFunction)(void*);


state_t* initState(size_t nGPUs, size_t *idGPUs); 
void freeState(state_t *state);

//reconfData_t* initReconfigurationData();
//void freeReconfigurationData(reconfData_t *reconfData);

DTI_t* initializeDTI(size_t type, size_t dataLocation, void* cpuData, void** gpuData, size_t N, size_t size, const char* name, GenericFunction cpu2gpu, GenericFunction gpu2cpu, DTIDesctiption_t *description);
void freeDTI(DTI_t *DTI, size_t nGPUs);

void freeDescription(DTIDesctiption_t *description, size_t nGPUs);


/* [MOCK SCHEDULER] */
//void* runMockScheduler(void *arg); 

#endif