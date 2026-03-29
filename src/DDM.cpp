#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstring>

#include "DDM.hpp"
#include "DTM.hpp"
#include "DITO_API.hpp"
#include "priv_DITTO_API.hpp"


/**
==============================
|| DATA DISTRIBUTION MODULE ||
==============================
*/

/* [CONFIGURE & RECONFIGURE] */

void configureDTI(DTI_t *DTI, size_t nGPUs, size_t nOldGPUs){

    size_t i;
    DTIDesctiption_t *description = DTI->description;

    // configure DTI only if DTI is automatic, else, functions are responsible of correctlu
    // deciding how to move data depending on the number of GPUs
    if(DTI->type == 0){

        ////////////////////////////////////////////////////
        // deallocate information related to old configuration (nOldGPUs)
        for(i = 0; i<nOldGPUs; i++){

            free(DTI->nPerPartition[i]);
            free(DTI->offsetPerPartition[i]);
        }
        if(DTI->nPerGPU)
            free(DTI->nPerGPU);
        if(DTI->nPerPartition)
            free(DTI->nPerPartition);
        if(DTI->offsetPerPartition)
            free(DTI->offsetPerPartition);

        // allocate for new configuration (nGPUs)
        DTI->nPerGPU = (size_t*)calloc(nGPUs, sizeof(size_t));
        DTI->nPerPartition = (size_t**)calloc(nGPUs, sizeof(size_t*));
        DTI->offsetPerPartition = (size_t**)calloc(nGPUs, sizeof(size_t*));

        for(i = 0; i<nGPUs; i++){

            DTI->nPerPartition[i] = (size_t*)calloc(description->nPartitions, sizeof(size_t));
            DTI->offsetPerPartition[i] = (size_t*)calloc(description->nPartitions, sizeof(size_t));
        }
        ////////////////////////////////////////////////////



        ////////////////////////////////////////////////////
        // deallocate old GPU pointers and allocate new ones
        if(DTI->gpuData) 
            free(DTI->gpuData);

        // allocate new pointers
        DTI->gpuData = (void**)calloc(nGPUs, sizeof(void*));
        ////////////////////////////////////////////////////



        ////////////////////////////////////////////////////
        // configure data following the DTI description
        switch (DTI->description->tpttEnum){

            case all:
                configureEntireTransmission(DTI, nGPUs);
                break;

            case simple:
                configureSimpleTransmission(DTI, nGPUs);
                break;

            // TODO
            case complex:
                configureComplexTransmission(DTI, nGPUs);
                break;
        }
        ////////////////////////////////////////////////////
    }
}

/* [AUTOMATIC REDISTRIBUTION CONFIGURATIONS] */

// All GPUs receive the entire array of N elements
void configureEntireTransmission(DTI_t *DTI, size_t nGPUs){

    size_t i;
    size_t N = DTI->N;
    DTIDesctiption_t *description = DTI->description;

    // configure information for all GPUs
    for(i = 0; i<nGPUs; i++){
        
        DTI->nPerGPU[i] = N; // the entire array is copied to all GPUs
        DTI->nPerPartition[i][0] = N; // the entire array is copied to all GPUs
        DTI->offsetPerPartition[i][0] = 0; // all start in 0 since the entire array is copied
    }
}

// N elements are distributed accross nGPUs devices
void configureSimpleTransmission(DTI_t *DTI, size_t nGPUs){

    size_t i, n, tmpOffset;
    size_t N = DTI->N;
    size_t nElements, rElements;
    DTIDesctiption_t *description = DTI->description;

    // compute the number of elements per GPU
    nElements = N / nGPUs;
    rElements = N % nGPUs;

    // configure the data distribution depending on the strategy for managing remaining elements
    // all remaining elements processed by the first GPU
    tmpOffset = 0;
    for(i = 0; i<nGPUs; i++){

        if(description->rmEnum == first && i == 0)
            n = nElements + rElements;  

        else if(description->rmEnum == ordered && i < rElements)
            n = nElements + 1;

        else if(description->rmEnum == last && i == nGPUs-1)
            n = nElements + rElements;
        
        else
            n = nElements;
            
        // there is only one partition divided between all GPUs
        DTI->nPerGPU[i] = n;
        DTI->nPerPartition[i][0] = n; // there is only one partition 
        DTI->offsetPerPartition[i][0] = tmpOffset; // all start in 0 since the entire array is copied

        tmpOffset += n;
    }
}

void configureComplexTransmission(DTI_t *DTI, size_t nGPUs){

    size_t i, n, tmpOffset;
    size_t N = DTI->N;
    size_t nElements, rElements, nElementsPerPartition, rElementsPerPartition, nPartitions;
    DTIDesctiption_t *description = DTI->description;
    nPartitions = description->nPartitions;

    // compute the number of elements per GPU
    nElements = N / nGPUs;
    nElementsPerPartition = N / (nGPUs * nPartitions);
    nElementsPerPartition = N % (nGPUs * nPartitions);
    rElements = N % nGPUs;

    // configure the data distribution depending on the strategy for managing remaining elements
    // all remaining elements processed by the first GPU
    tmpOffset = 0;
    for(i = 0; i<nGPUs; i++){

        if(description->rmEnum == first && i == 0)
            n = nElements + rElements;  

        else if(description->rmEnum == ordered && i < rElements)
            n = nElements + 1;

        else if(description->rmEnum == last && i == nGPUs-1)
            n = nElements + rElements;
        
        else
            n = nElements;
            
        // there is only one partition divided between all GPUs
        DTI->nPerGPU[i] = n;
        DTI->nPerPartition[i][0] = n; // there is only one partition 
        DTI->offsetPerPartition[i][0] = tmpOffset; // all start in 0 since the entire array is copied

        tmpOffset += n;
    }
}