#ifndef DDM_HPP
#define DDM_HPP

//TODO revise
#include "DITO_API.hpp"

typedef struct DTI_t DTI_t;
typedef struct jobResources_t jobResources_t;

/* [AUTOMATIC REDISTRIBUTION CONFIGURATIONS] */

// configure DTI for the next reconfiguraiton
void configureDTI(DTI_t *DTI, jobResources_t *jobResources, jobResources_t *reconfJobResources, reconfDirEnum reconfDir);
void configureEntireTransmission(DTI_t *DTI, jobResources_t *jobResources);
void configureSimpleTransmission(DTI_t *DTI, jobResources_t *jobResources);
void configureComplexTransmission(DTI_t *DTI, jobResources_t *jobResources);

/// @brief Function to decide the reconfiguration destination GPUs for each source GPU
void configureExpansion(jobResources_t *reconfJobResources, jobResources_t *jobResouces);
void configureShrink(jobResources_t *reconfJobResources, jobResources_t *jobResouces);
void configureN2N(jobResources_t *reconfJobResources, jobResources_t *jobResouces);

/// @brief 
void configureSimpleExpandTransmission(DTI_t *DTI, jobResources_t *reconfJobResources, jobResources_t *jobResouces);
void configureSimpleShrinkTransmission(DTI_t *DTI, jobResources_t *reconfJobResources, jobResources_t *jobResouces);
void configureSimpleN2NTransmission(DTI_t *DTI, jobResources_t *reconfJobResources, jobResources_t *jobResources);


#endif