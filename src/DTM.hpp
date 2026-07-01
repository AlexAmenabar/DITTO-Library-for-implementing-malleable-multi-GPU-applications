#ifndef CUDATM_HPP
#define CUDATM_HPP

// forwarded declarations
typedef struct DTI_t DTI_t;
typedef struct jobResources_t jobResources_t;

// general GPU functions
void setGPUDevice(size_t i);

// Streams management
/// @brief Initialize CUDA streams
void initializeStreams(jobResources_t *jobResources);

/// @brief Destroy CUDA streams
void destroyStreams(jobResources_t *jobResources);

/// @brief Create NCCL communicators
void initializeNCCLComm(jobResources_t *jobResources);

/// @brief Destroy NCCL communicators
void destroyNCCLComm(jobResources_t *jobResources);

/// @brief TODO
void resetGPUs();

// Host-GPU communcations
void cpyDataCPU2GPU(DTI_t *DTI);
void cpyDataGPU2CPU(DTI_t *DTI);

// Reconfigure application by copying data directly from the source GPUs to the destination GPUs
void cpyDataGPU2GPU(DTI_t *DTI);

void reconfExpand(DTI_t *DTI);
void reconfShrink(DTI_t *DTI);
void reconfN2N(DTI_t *DTI);

#endif