#ifndef DDM_HPP
#define DDM_HPP

typedef struct DTI_t DTI_t;

/* [AUTOMATIC REDISTRIBUTION CONFIGURATIONS] */

// configure DTI for the next reconfiguraiton
void redistributeDTI(DTI_t *DTI, size_t nGPUs, size_t nOldGPUs);
void configureEntireTransmission(DTI_t *DTI, size_t nGPUs);
void configureSimpleTransmission(DTI_t *DTI, size_t nGPUs);

#endif