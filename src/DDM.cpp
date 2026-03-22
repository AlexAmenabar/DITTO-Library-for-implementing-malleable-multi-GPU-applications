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

void redistributeDTI(DTI_t *DTI, size_t nGPUs, size_t nOldGPUs){

    // configure DTI only if DTI is automatic, else, functions are responsible of correctlu
    // deciding how to move data depending on the number of GPUs
    if(DTI->type == 0){

        // if the new number of GPUs is different from the previous one, deallocate memory // TODO: this should be revised for other configuration methods,
        // since deallocating description is not always correct. Sometimes description information will be neccessary
        if(nGPUs != nOldGPUs){

            freeDescription(DTI->description, nOldGPUs);
        }

        // deallocate old GPU pointers and allocate new ones
        if(DTI->gpuData) 
            free(DTI->gpuData);
        DTI->gpuData = (void**)calloc(nGPUs, sizeof(void*));


        // decide how to redistribute the data
        switch (DTI->description->tpttEnum){
            case all:
                configureEntireTransmission(DTI, nGPUs);
                break;

            case simple:
                configureSimpleTransmission(DTI, nGPUs);
                break;

            // TODO
            /*case complex:
                DTI->infoComplex = infoComplex;
                configureComplexTransmission(DTI);
                break;

            case custom:
                DTI->infoCustom = infoCustom;
                configureCustomTransmission(DTI);
                break;*/
        }
    }
}

/* [AUTOMATIC REDISTRIBUTION CONFIGURATIONS] */

void configureEntireTransmission(DTI_t *DTI, size_t nGPUs){

    size_t i;
    size_t N = DTI->N;
    DTIDesctiption_t *description = DTI->description;

    // allcoate memory for new configuration
    description->nElementsPerGPU = (size_t*)calloc(nGPUs, sizeof(size_t));
    description->nPartitionsPerGPU = (size_t*)calloc(nGPUs, sizeof(size_t));

    description->nElementsPerPartition = (size_t**)calloc(nGPUs, sizeof(size_t*));
    description->nElementsPerPartition[0] = (size_t*)calloc(1, sizeof(size_t));

    description->firstElementPerPartition = (size_t**)calloc(nGPUs, sizeof(size_t*));
    description->firstElementPerPartition[0] = (size_t*)calloc(1, sizeof(size_t));

    // configure
    for(i = 0; i<nGPUs; i++){
        
        description->nElementsPerGPU[i] = N;
        description->nPartitionsPerGPU[i] = 1; // the entire data array is a the unique partition per GPU
        description->nElementsPerPartition[i][0] = N; // there is only one partition 
        description->firstElementPerPartition[i][0] = 0; // all start in 0 since the entire array is copied
    }
}

void configureSimpleTransmission(DTI_t *DTI, size_t nGPUs){

    size_t i;
    size_t N = DTI->N;
    size_t n, tmpElement;
    DTIDesctiption_t *description = DTI->description;

    // allcate memory for new configuration (1 partition per GPU)
    description->nElementsPerGPU = (size_t*)calloc(nGPUs, sizeof(size_t));
    description->nPartitionsPerGPU = (size_t*)calloc(nGPUs, sizeof(size_t));

    description->nElementsPerPartition = (size_t**)calloc(nGPUs, sizeof(size_t*));
    description->firstElementPerPartition = (size_t**)calloc(nGPUs, sizeof(size_t*));

    for(i = 0; i<nGPUs; i++){

        description->nElementsPerPartition[i] = (size_t*)calloc(1, sizeof(size_t));
        description->firstElementPerPartition[i] = (size_t*)calloc(1, sizeof(size_t));
    }
    // compute number of elements per GPU
    size_t nElements = N / nGPUs;
    size_t rElements = N % nGPUs;

    // configure the data distribution depending on the strategy for managing remaining elements

    tmpElement = 0;

    // all remaining elements processed by the first GPU
    for(i = 0; i<nGPUs; i++){

        if(description->rmEnum == first && i == 0)
            n = nElements + rElements;  

        else if(description->rmEnum == ordered && i < rElements)
            n = nElements + 1;

        else if(description->rmEnum == last && i == nGPUs-1)
            n = nElements + rElements;
        
        else
            n = nElements;
            
        description->nElementsPerGPU[i] = n;
        description->nPartitionsPerGPU[i] = 1; // the entire data array is a the unique partition per GPU
        description->nElementsPerPartition[i][0] = n; // there is only one partition 
        description->firstElementPerPartition[i][0] = tmpElement; // all start in 0 since the entire array is copied

        tmpElement += n;
    }
}