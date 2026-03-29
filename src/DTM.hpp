#ifndef CUDATM_HPP
#define CUDATM_HPP

// forwarded declarations
typedef struct DTI_t DTI_t;

void setGPUDevice();
void cpyDataCPU2GPU(DTI_t *DTI);
void cpyDataGPU2CPU(DTI_t *DTI);
void resetGPUs();

#endif