#ifndef DDM_HPP
#define DDM_HPP

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include <type_traits>

/// @brief Enumeration that indicates the pattern for distributing data along GPUs
enum transmissionPatternsEnum {
    all,
    simple,
    complex,
    custom
};

/// @brief Enum indicating what to do with the remaining elements (nElements % nGPUs)
enum remainingElementsEnum {
    
    first,
    ordered,
    last
};


typedef struct infoComplexDTI_t{

    // simple description
    size_t *n;
    size_t *j;
    size_t *off;
    size_t *tn; // total number of elements

} infoComplexDTI_t;

typedef struct infoCustomDTI_t{

    // implementation description
    size_t *nPartitionsPerGPU; // [nGPUs] number of partitions on each GPU
    size_t *nElementsPerGPU; // [nGPUs] number of elements on each GPU (the sum of the elements on all partitions)
    size_t **nElementsPerPartition; // [nGPUs x ---] number of elements on each partition per GPU
    size_t **firstElementPerPartition; // [nGPUs x ---] index of the first element on each partition
    
} infoCustomDTI_t;


/// @brief Structure to store information of how data is distributed acoss GPUs. Data (usually arrays) are divided in partitions, where each GPU receives
/// a set of partitions
typedef struct DTI_t {

    void **gpuData; // [n_GPUS] pointers to device arrays
    void *cpuData; // array on the CPU
    size_t N; // number of elements in the CPU array
    size_t size; // size of the data type

    size_t nGPUs;

    infoComplexDTI_t *infoComplex;
    infoCustomDTI_t *infoCustom;

    // enumerations providing more details about the DTI
    transmissionPatternsEnum tpttEnum;
    remainingElementsEnum rmEnum;

} DTI_t;


/**
*
* Function to create the DTI structure for managing the communications between CPU and GPU
*/

DTI_t* initializeDTI(void* cpuData, size_t N, size_t size, transmissionPatternsEnum tpttEnum, remainingElementsEnum rmEnum);

void configureDTI(DTI_t *DTI, size_t nGPUs, size_t nOldGPUs, infoComplexDTI_t *infoComplex, infoCustomDTI_t *infoCustom);

void configureEntireTransmission(DTI_t *DTI);

void configureSimpleTransmission(DTI_t *DTI);

void configureComplexTransmission(DTI_t *DTI);

void configureCustomTransmission(DTI_t *DTI);

void printDTI(DTI_t *DTI);

#endif