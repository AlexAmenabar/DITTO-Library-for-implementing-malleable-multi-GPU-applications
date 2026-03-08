#include "DDM.hpp"

/**
    [DATA DISTRIBUTION MODULE]
*/

DTI_t* initializeDTI(void* cpuData, size_t N, size_t size, transmissionPatternsEnum tpttEnum, remainingElementsEnum rmEnum){

    DTI_t *DTI = (DTI_t*)calloc(1, sizeof(DTI_t));
    infoCustomDTI_t *infoCustomDTI = DTI->infoCustom;

    // store DTI data
    DTI->cpuData = cpuData; // point to CPU data array
    DTI->N = N;
    DTI->size = size;

    DTI->infoComplex = (infoComplexDTI_t*)calloc(1, sizeof(infoComplexDTI_t));
    DTI->infoCustom = (infoCustomDTI_t*)calloc(1, sizeof(infoCustomDTI_t));

    // store enums
    DTI->tpttEnum = tpttEnum;
    DTI->rmEnum = rmEnum;

    return DTI;
}

void configureDTI(DTI_t *DTI, size_t nGPUs, size_t nOldGPUs, infoComplexDTI_t *infoComplex, infoCustomDTI_t *infoCustom){

    DTI->nGPUs = nGPUs;

    infoComplexDTI_t *infoComplexDTI = DTI->infoComplex;
    infoCustomDTI_t *infoCustomDTI = DTI->infoCustom;

    // allocate memory for GPU pointers
    if(DTI->gpuData) free(DTI->gpuData);
    DTI->gpuData = (void**)calloc(nGPUs, sizeof(void*));

    // deallocate memory allocated in previous configuration
    if(infoComplexDTI && infoComplexDTI->n) free(infoComplexDTI->n);
    if(infoComplexDTI && infoComplexDTI->j) free(infoComplexDTI->j);
    if(infoComplexDTI && infoComplexDTI->off) free(infoComplexDTI->off);
    if(infoComplexDTI && infoComplexDTI->tn) free(infoComplexDTI->tn);


    if(infoCustomDTI && infoCustomDTI->nElementsPerGPU) free(infoCustomDTI->nElementsPerGPU);
    if(infoCustomDTI && infoCustomDTI->nPartitionsPerGPU) free(infoCustomDTI->nPartitionsPerGPU);
    if(infoCustomDTI && infoCustomDTI->nElementsPerPartition) {

        for(size_t i = 0; i<nOldGPUs; i++){

            if(infoCustomDTI->nElementsPerPartition[i]) free(infoCustomDTI->nElementsPerPartition[i]);
        }
        free(infoCustomDTI->nElementsPerPartition);   
    }

    if(infoCustomDTI && infoCustomDTI->firstElementPerPartition) {

        for(size_t i = 0; i<nOldGPUs; i++){

            if(infoCustomDTI->firstElementPerPartition[i]) free(infoCustomDTI->firstElementPerPartition[i]);
        }  
        free(infoCustomDTI->firstElementPerPartition); 
    }


    // decide how to redistribute the data
    switch (DTI->tpttEnum){
        case all:
            configureEntireTransmission(DTI);
            break;

        case simple:
            configureSimpleTransmission(DTI);
            break;

        case complex:
            DTI->infoComplex = infoComplex;
            configureComplexTransmission(DTI);
            break;

        case custom:
            DTI->infoCustom = infoCustom;
            configureCustomTransmission(DTI);
            break;
    }
}

/// DTI management


void configureEntireTransmission(DTI_t *DTI){

    size_t i;

    size_t nGPUs = DTI->nGPUs;
    size_t N = DTI->N;

    infoCustomDTI_t *infoCustomDTI = DTI->infoCustom;

    printf(" -- Configuring entire transmission %zu, %zu\n", nGPUs, N);
    fflush(stdout);

    // allcoate memory for new configuration
    infoCustomDTI->nElementsPerGPU = (size_t*)calloc(nGPUs, sizeof(size_t));
    infoCustomDTI->nPartitionsPerGPU = (size_t*)calloc(nGPUs, sizeof(size_t));

    infoCustomDTI->nElementsPerPartition = (size_t**)calloc(nGPUs, sizeof(size_t*));
    infoCustomDTI->nElementsPerPartition[0] = (size_t*)calloc(1, sizeof(size_t));

    infoCustomDTI->firstElementPerPartition = (size_t**)calloc(nGPUs, sizeof(size_t*));
    infoCustomDTI->firstElementPerPartition[0] = (size_t*)calloc(1, sizeof(size_t));

    // configure
    for(i = 0; i<nGPUs; i++){
        
        infoCustomDTI->nElementsPerGPU[i] = N;
        infoCustomDTI->nPartitionsPerGPU[i] = 1; // the entire data array is a the unique partition per GPU
        infoCustomDTI->nElementsPerPartition[i][0] = N; // there is only one partition 
        infoCustomDTI->firstElementPerPartition[i][0] = 0; // all start in 0 since the entire array is copied
    }
}

void configureSimpleTransmission(DTI_t *DTI){

    size_t i;

    size_t nGPUs = DTI->nGPUs;
    size_t N = DTI->N;
    size_t n, tmpElement;

    infoCustomDTI_t *infoCustomDTI = DTI->infoCustom;

    // allcate memory for new configuration (1 partition per GPU)
    infoCustomDTI->nElementsPerGPU = (size_t*)calloc(nGPUs, sizeof(size_t));
    infoCustomDTI->nPartitionsPerGPU = (size_t*)calloc(nGPUs, sizeof(size_t));

    infoCustomDTI->nElementsPerPartition = (size_t**)calloc(nGPUs, sizeof(size_t*));
    infoCustomDTI->firstElementPerPartition = (size_t**)calloc(nGPUs, sizeof(size_t*));

    for(i = 0; i<nGPUs; i++){

        infoCustomDTI->nElementsPerPartition[i] = (size_t*)calloc(1, sizeof(size_t));
        infoCustomDTI->firstElementPerPartition[i] = (size_t*)calloc(1, sizeof(size_t));
    }
    // compute number of elements per GPU
    size_t nElements = N / nGPUs;
    size_t rElements = N % nGPUs;

    // configure the data distribution depending on the strategy for managing remaining elements

    tmpElement = 0;

    // all remaining elements processed by the first GPU
    for(i = 0; i<nGPUs; i++){

        if(DTI->rmEnum == first && i == 0)
            n = nElements + rElements;  

        else if(DTI->rmEnum == ordered && i < rElements)
            n = nElements + 1;

        else if(DTI->rmEnum == last && i == nGPUs-1)
            n = nElements + rElements;
        
        else
            n = nElements;
            
        infoCustomDTI->nElementsPerGPU[i] = n;
        infoCustomDTI->nPartitionsPerGPU[i] = 1; // the entire data array is a the unique partition per GPU
        infoCustomDTI->nElementsPerPartition[i][0] = n; // there is only one partition 
        infoCustomDTI->firstElementPerPartition[i][0] = tmpElement; // all start in 0 since the entire array is copied

        tmpElement += n;
    }
}

void configureComplexTransmission(DTI_t *DTI){

    printf(" Complex transmission configuration not implemented yet\n");

}

void configureCustomTransmission(DTI_t *DTI){

    printf(" Custom transmission configuration not implemented yet\n");
}


void printDTI(DTI_t *DTI){

    size_t i, j, nGPUs = DTI->nGPUs;
    infoCustomDTI_t *infoCustom = DTI->infoCustom;

    for(i = 0; i<nGPUs; i++){

        printf(" -- Printing GPU %zu data --\n", i);

        printf("     Number of partitions = %zu\n", infoCustom->nPartitionsPerGPU[i]);
        printf("     Number of elements   = %zu\n", infoCustom->nElementsPerGPU[i]);

        for(j = 0; j<infoCustom->nPartitionsPerGPU[i]; j++){
            printf("     Number of elements in partition %zu = %zu\n", j, infoCustom->nElementsPerPartition[i][j]);
            printf("     First element of partition %zu = %zu\n", j, infoCustom->firstElementPerPartition[i][j]);
        }
        printf("\n");
    }
    fflush(stdout);
}