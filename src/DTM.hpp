#ifndef CUDATM_HPP
#define CUDATM_HPP

// forwarded declarations
typedef struct DTI_t DTI_t;

// general GPU functions
void setGPUDevice(size_t i);
void resetGPUs();

void initializeStreams();
void destroyStreams();

// Host-GPU communcations
void cpyDataCPU2GPU(DTI_t *DTI);
void cpyDataGPU2CPU(DTI_t *DTI);

// Reconfigure application by copying data directly from the source GPUs to the destination GPUs
void cpyDataP2P(DTI_t *DTI);

#endif