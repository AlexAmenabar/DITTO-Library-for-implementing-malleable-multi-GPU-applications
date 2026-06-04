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
        //printf(" -- First phase\n");
        //fflush(stdout);
        // deallocate information related to old configuration (nOldGPUs)
        for(i = 0; i<nOldGPUs; i++){

            free(DTI->nPerPartition[i]);
            free(DTI->offsetPerPartition[i]);
        }
        if(DTI->nPerGPU)
            free(DTI->nPerGPU);
        if(DTI->nPartitionsPerGPU)
            free(DTI->nPartitionsPerGPU);
        if(DTI->nPerPartition)
            free(DTI->nPerPartition);
        if(DTI->offsetPerPartition)
            free(DTI->offsetPerPartition);

        // allocate for new configuration (nGPUs)
        //printf(" -- Second phase\n");
        //fflush(stdout);
        DTI->nPerGPU = (size_t*)calloc(nGPUs, sizeof(size_t));
        DTI->nPartitionsPerGPU = (size_t*)calloc(nGPUs, sizeof(size_t));
        DTI->nPerPartition = (size_t**)calloc(nGPUs, sizeof(size_t*));
        DTI->offsetPerPartition = (size_t**)calloc(nGPUs, sizeof(size_t*));
        ////////////////////////////////////////////////////



        ////////////////////////////////////////////////////
        //printf(" -- Third phase\n");
        //fflush(stdout);
        // deallocate old GPU pointers and allocate new ones
        if(DTI->gpuData) 
            free(DTI->gpuData);

        // allocate new pointers
        DTI->gpuData = (void**)calloc(nGPUs, sizeof(void*));
        ////////////////////////////////////////////////////



        ////////////////////////////////////////////////////
        //printf(" -- 4. phase\n");
        //fflush(stdout);
        // configure data following the DTI description
        switch (DTI->description->tpttEnum){

            case entire:
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
        
        // number of partitions per each GPU
        DTI->nPartitionsPerGPU[i] = 1; // one partition on each GPU

        // allocate memory for partition information
        DTI->nPerPartition[i] = (size_t*)calloc(DTI->nPartitionsPerGPU[i], sizeof(size_t));
        DTI->offsetPerPartition[i] = (size_t*)calloc(DTI->nPartitionsPerGPU[i], sizeof(size_t));

        // initialize information for guiding the information transference
        DTI->nPerGPU[i] = N; // the entire array is copied to all GPUs
        DTI->nPerPartition[i][0] = N; // the entire array is copied to all GPUs, so there is only one partition with all elements
        DTI->offsetPerPartition[i][0] = 0; // all start in 0 since the entire array is copied
    }
}

// N elements are distributed accross nGPUs devices
void configureSimpleTransmission(DTI_t *DTI, size_t nGPUs){

    size_t i, n, tmpOffset;
    size_t N = DTI->N;
    size_t nElements, rElements;
    DTIDesctiption_t *description = DTI->description;

    //printf(" -- %zu, %zu\n", N, nGPUs);
    //fflush(stdout);

    // compute the number of elements per GPU
    nElements = N / nGPUs;
    rElements = N % nGPUs;

    // configure the data distribution depending on the strategy for managing remaining elements
    // all remaining elements processed by the first GPU
    tmpOffset = 0;
    for(i = 0; i<nGPUs; i++){

        DTI->nPartitionsPerGPU[i] = 1; // one partition per each GPU

        // allocate memory for partitions information
        DTI->nPerPartition[i] = (size_t*)calloc(DTI->nPartitionsPerGPU[i], sizeof(size_t));
        DTI->offsetPerPartition[i] = (size_t*)calloc(DTI->nPartitionsPerGPU[i], sizeof(size_t));


        // manage remaining elements
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

// simplified version where only remaining elements are not handled: revise
void configureComplexTransmission(DTI_t *DTI, size_t nGPUs){

    size_t i, n, tmpOffset;
    size_t N = DTI->N;
    size_t nElementsPerGPU, rElementsPerGPU, nElementsPerPartition, rElementsPerPartition, nPartitionsPerGPU;
    DTIDesctiption_t *description = DTI->description;
    size_t s = description->s;

    // WARNING: We are assuming here that N is divisible by [s * nGPUs]
    // N: total number of elements in the DTI array
    // s: number of elements on each partition
    // nGPUs: number of GPUs for the reconfiguration

    // compute the number of partitions to be stored on each GPU (each partition of s elements)
    nPartitionsPerGPU =  N / (s * nGPUs);
    
    // initial number of elements and remaining elements dividing N by the number of GPUs
    nElementsPerGPU = s * nPartitionsPerGPU;
    rElementsPerGPU = 0;


    // initialize and allocate partitions information in the DTI
    for(i = 0; i<nGPUs; i++){

        // compute the number of elements per partition. It should be s, but it is necessary to manage remaining elemnts
        DTI->nPartitionsPerGPU[i] = nPartitionsPerGPU; // all GPUs receive the same number of partitions

        // allocate memory for partitions information
        DTI->nPerPartition[i] = (size_t*)calloc(DTI->nPartitionsPerGPU[i], sizeof(size_t));
        DTI->offsetPerPartition[i] = (size_t*)calloc(DTI->nPartitionsPerGPU[i], sizeof(size_t));
    }

    tmpOffset = 0;
    for(size_t par = 0; par < nPartitionsPerGPU; par++){
        
        for(i = 0; i<nGPUs; i++){
    
            // there is only one partition divided between all GPUs
            DTI->nPerGPU[i] += s;
            DTI->nPerPartition[i][par] = s; // there is only one partition 
            DTI->offsetPerPartition[i][par] = tmpOffset; // all start in 0 since the entire array is copied

            tmpOffset += s;
        }
    }

    /*nElementsPerGPU = N / nGPUs;
    rElementsPerGPU = N % nGPUs;

    // initial number of elements per partition
    nElementsPerPartition = s;

    // initialize partitions information
    for(i = 0; i<nGPUs; i++){
        
        // compute the number of elements per partition. It should be s, but it is necessary to manage remaining elemnts
        DTI->nPartitionsPerGPU[i] = nPartitionsPerGPU; // all GPUs receive the same number of partitions

        // allocate memory for partitions information
        DTI->nPerPartition[i] = (size_t*)calloc(DTI->nPartitionsPerGPU[i], sizeof(size_t));
        DTI->offsetPerPartition[i] = (size_t*)calloc(DTI->nPartitionsPerGPU[i], sizeof(size_t));
    }


    printf(" -- N = %zu, s = %zu, nPartitions = %zu, nElements per GPU = %zu, r elements per GPU = %zu, nElementsPerPartition = %zu\n",
            N, s, nPartitionsPerGPU, nElementsPerGPU, rElementsPerGPU, nElementsPerPartition);

    fflush(stdout);

    // configure the data distribution depending on the strategy for managing remaining elements
    // all remaining elements processed by the first GPU
    tmpOffset = 0;
    for(size_t par = 0; par < nPartitionsPerGPU; par++){
        
        for(i = 0; i<nGPUs; i++){

            // manage remaining elements (how much elements will be on each partition per GPU)
            if(description->rmEnum == first && i == 0)
                n = nElementsPerGPU + rElementsPerGPU;  

            else if(description->rmEnum == ordered && i < rElementsPerGPU)
                n = nElementsPerGPU + 1;

            else if(description->rmEnum == last && i == nGPUs-1)
                n = nElementsPerGPU + rElementsPerGPU;
            
            else
                n = nElementsPerGPU;
                
            // there is only one partition divided between all GPUs
            DTI->nPerGPU[i] += n;
            DTI->nPerPartition[i][par] = n; // there is only one partition 
            DTI->offsetPerPartition[i][par] = tmpOffset; // all start in 0 since the entire array is copied

            tmpOffset += n;
        }
    }*/
}