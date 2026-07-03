#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstring>

#include "DDM.hpp"
#include "DTM.hpp"
#include "DITO_API.hpp"
#include "priv_DITTO_API.hpp"

#include "RMS.hpp"


/**
==============================
|| DATA DISTRIBUTION MODULE ||
==============================
*/

/* [CONFIGURE & RECONFIGURE] */

void configureDTI(DTI_t *DTI, jobResources_t *jobResources, jobResources_t *reconfJobResources, reconfDirEnum reconfDir){

    size_t i, j;
    DTIDesctiption_t *description = DTI->description;
    size_t nGPUs = jobResources->nGPUs; 
    size_t nReconfGPUs = 0; 
    

    if(DTI->type == 0){

        // if reconfiguration goes from GPUs to GPUs

        // if there is a reconfiguration to perform, then save DTI data in the previous one
        // in CPU2GPU and GPU2CPU reconfigurations, previous is not necessary, 
        if(reconfJobResources){

            // get the number of GPUs for the reconfiguration
            nReconfGPUs = reconfJobResources->nGPUs;

            //deallocate old DTI data and store current one as previous

            /*if(DTI->prev_nPartitionsPerGPU){

                for(i = 0; i<nGPUs; i++){
                    if(DTI->prev_nPerPartition[i]) 
                        free(DTI->prev_nPerPartition[i]);
                    if(DTI->prev_offsetPerPartition[i]) 
                        free(DTI->prev_offsetPerPartition[i]);
                }
            }*/

            //if(DTI->prev_nPerGPU) free(DTI->prev_nPerGPU);
            DTI->prev_nPerGPU = DTI->nPerGPU;

            //if(DTI->prev_nPartitionsPerGPU) free(DTI->prev_nPartitionsPerGPU);
            DTI->prev_nPartitionsPerGPU = DTI->nPartitionsPerGPU;

            //if(DTI->prev_nPerPartition) free(DTI->prev_nPerPartition);
            DTI->prev_nPerPartition = DTI->nPerPartition;

            //if(DTI->prev_offsetPerPartition) free(DTI->prev_offsetPerPartition);
            DTI->prev_offsetPerPartition = DTI->offsetPerPartition;

            // store GPU pointers in the prev pointers
            if(DTI->gpuData) 
                DTI->prevGpuData = DTI->gpuData;
        }
        else{

            nReconfGPUs = nGPUs;
            reconfJobResources = jobResources;
        }

        // if it is a reconfiguration, it will be used the number of GPUs for it. Else, the number
        // of GPUs in the job resources
        DTI->nPerGPU = (size_t*)calloc(nReconfGPUs, sizeof(size_t));
        DTI->nPartitionsPerGPU = (size_t*)calloc(nReconfGPUs, sizeof(size_t));
        DTI->nPerPartition = (size_t**)calloc(nReconfGPUs, sizeof(size_t*));
        DTI->offsetPerPartition = (size_t**)calloc(nReconfGPUs, sizeof(size_t*));

        // allocate new pointers
        DTI->gpuData = (void**)calloc(nReconfGPUs, sizeof(void*));

        // configure
        switch (DTI->description->tpttEnum){

            case entire:
                configureEntireTransmission(DTI, reconfJobResources);
                break;
            case simple:
                if(reconfDir == CPU2GPU || reconfDir == GPU2CPU || reconfDir == CPU)
                    configureSimpleTransmission(DTI, reconfJobResources); // 
                else if(nReconfGPUs > nGPUs)
                    configureSimpleExpandTransmission(DTI, reconfJobResources, jobResources);
                else if(nGPUs > nReconfGPUs)
                    configureSimpleShrinkTransmission(DTI, reconfJobResources, jobResources);
                else if(nGPUs == nReconfGPUs)
                    configureSimpleN2NTransmission(DTI, reconfJobResources, jobResources);
                break;
            case complex:
                if(reconfDir == CPU2GPU || reconfDir == GPU2CPU || reconfDir == CPU)
                    configureComplexTransmission(DTI, reconfJobResources); // 
                else if(nReconfGPUs > nGPUs)
                    configureSimpleExpandTransmission(DTI, reconfJobResources, jobResources);
                else if(nGPUs > nReconfGPUs)
                    configureSimpleShrinkTransmission(DTI, reconfJobResources, jobResources);
                else if(nGPUs == nReconfGPUs)
                    configureSimpleN2NTransmission(DTI, reconfJobResources, jobResources);
                break;
        }

        if(reconfJobResources == jobResources)
            reconfJobResources = NULL;
    }
}


/* [AUTOMATIC REDISTRIBUTION CONFIGURATIONS] */


void configureExpansion(jobResources_t *reconfJobResources, jobResources_t *jobResources){

    size_t i, j, f;

    // get current and reconfiguration resources
    size_t *idGPUs = jobResources->idGPUs;
    size_t *idReconfGPUs = reconfJobResources->idGPUs;

    size_t nGPUs = jobResources->nGPUs;
    size_t nReconfGPUs = reconfJobResources->nGPUs;

    // relation in the number of reconf and current GPUs
    size_t D = nReconfGPUs / nGPUs;

    // GPUs to reconfigure from each GPU
    size_t **gpusToSplit = (size_t**)calloc(nGPUs, sizeof(size_t*)); // identifiers of the GPUs to distribute data from the GPU
    size_t *nSelectedGPUs = (size_t*)calloc(nGPUs, sizeof(size_t)); // number of GPUs selected already to distribute data from the GPU
    size_t nSelectedTotal = 0;

    for(i = 0; i<nGPUs; i++){
        gpusToSplit[i] = (size_t*)calloc(D, sizeof(size_t)); // initialize
    }

    // GPUs already selected (0 | 1)
    size_t *selectedGPUs = (size_t*)calloc(nReconfGPUs, sizeof(size_t)); // nReconfGPUs > nGPUs

    // select the GPUs to redistribute data from each GPU
    for(i = 0; i<nGPUs; i++){

        // id of the source GPU
        size_t srcGPU = idGPUs[i];

        // keep data if possible (look if the srcGPU appears too in the reconf GPUs array)
        f = 0; j = 0;
        while(j<nReconfGPUs && !f){

            if(srcGPU == idReconfGPUs[j]){
                
                f = 1;
                gpusToSplit[i][0] = srcGPU; // keep a part of the data in the GPU
                nSelectedGPUs[i] ++; // one more GPU selected
                selectedGPUs[j] = 1; // the GPU has been selected
                nSelectedTotal ++;
            }
            j++;
        }
    }

#ifdef TOPOLOGYAWARE

    // [Topology aware]
    i = 0;

    // loop over GPUs until all GPUs are selected in a topologically 
    while(nSelectedTotal < (D * nGPUs)){

        if(nSelectedGPUs[i] < D){

            size_t srcDev = idGPUs[i];
            size_t bIndex;

            // find the best GPU topologically
            size_t aff = 9999999;

            // loop over GPUs
            for(size_t j = 0; j<nReconfGPUs; j++){

                size_t dstDev = idReconfGPUs[j];

                if(srcDev != dstDev && gpuTopology[srcDev * gNGPUs + dstDev] < aff && !selectedGPUs[j]){

                    aff = gpuTopology[srcDev * gNGPUs + dstDev];
                    bIndex = j;
                }
            }

            // select GPU
            gpusToSplit[i][nSelectedGPUs[i]] = idReconfGPUs[bIndex];
            nSelectedGPUs[i] ++;
            nSelectedTotal ++;
            selectedGPUs[bIndex] = 1;
        }

        i = (i+1) % nGPUs;
    }

#else

    // select the rest GPUs to distribute data (non-topology aware)
    for(i = 0; i<nGPUs; i++){

        // id of the source GPU
        size_t srcGPU = idGPUs[i];

        size_t nextIndex = 0;
        while(nSelectedGPUs[i] < D){

            // get GPU id in the reconfiguration GPUs array
            size_t idGPU = idReconfGPUs[nextIndex];
            
            // check validation: 
            //  (i) not selected yet
            //  (ii) not in current GPUs array
            int valid = 1;

            // i
            if(selectedGPUs[nextIndex]){
            
                valid = 0;
            }
            
            // ii
            if(valid){
            
                for(j = 0; j<nSelectedGPUs[i]; j++){

                    if(gpusToSplit[i][j] == idGPU){
                        valid = 0; // not valid
                    }
                }
            }
            if(valid){
            
                for(j = 0; j<nGPUs; j++){

                    if(idGPUs[j] == idGPU){
                        valid = 0; // not valid
                    }
                }
            }

            // if valid, add the GPU to the array of selected GPUs
            if(valid){

                gpusToSplit[i][nSelectedGPUs[i]] = idGPU;
                nSelectedGPUs[i] ++;
                nSelectedTotal ++;
                selectedGPUs[nextIndex] = 1;
            }

            // update loop index
            nextIndex ++;
        }
    }

#endif

    // store the GPUs to split
    reconfData->gpusToSplit = gpusToSplit;

    /*printf(" [APP]: Splitting information:\n");
    for(i = 0; i<nGPUs; i++){

        printf(" -- GPU %zu (%zu): ", idGPUs[i], i);

        for(j = 0; j<D; j++){

            printf("%zu ", gpusToSplit[i][j]);
        }
        printf("\n");
    }
    printf("\n");*/
}

void configureShrink(jobResources_t *reconfJobResources, jobResources_t *jobResources){
    
    size_t i, j, f;

    // get current and reconfiguration resources
    size_t *idGPUs = jobResources->idGPUs;
    size_t *idReconfGPUs = reconfJobResources->idGPUs;

    size_t nGPUs = jobResources->nGPUs;
    size_t nReconfGPUs = reconfJobResources->nGPUs;

    // relation in the number of reconf and current GPUs
    size_t D = nGPUs / nReconfGPUs;

    // GPUs to reconfigure from each GPU
    size_t nSelectedTotal = 0;
    size_t *segmentSelected = (size_t*)calloc(nReconfGPUs, sizeof(size_t)); // 
    size_t *nSelectedGPUs = (size_t*)calloc(nReconfGPUs, sizeof(size_t)); // number of source GPUs selected for each reconf GPU
    size_t **gpusToSplit = (size_t**)calloc(nGPUs, sizeof(size_t*)); // GPU identifiers to move from each source GPU
    for(i = 0; i<nGPUs; i++){
        gpusToSplit[i] = (size_t*)calloc(1, sizeof(size_t)); // initialize
    }

    // GPU selected (0 | 1)
    size_t *selectedGPUs = (size_t*)calloc(nGPUs, sizeof(size_t));

    //printf(" Virtual topology: ");
    //for(i = 0; i<nGPUs; i++)
    //    printf(" %zu", virtualTopology[i]);
    //printf("\n");
    //fflush(stdout);

    // for each reconf GPU, find the GPU in the virtual topology
    for(i = 0; i<nReconfGPUs; i++){

        // id of the dst GPU
        size_t dstGPU = idReconfGPUs[i];

        // find the index of the GPU in the virtual topology
        size_t inVrt = 0;

        j = 0;
        while(j < nGPUs && !inVrt){

            if(dstGPU == virtualTopology[j]){

                inVrt = 1; // found
            }
            else{
                j++;
            }
        }
    
        // it is possible to keep data, rebuild the entire segment in the i.th GPU
        if(inVrt){

            // check if the data segment of the GPU is already selected
            size_t s = j / D; // segment
            
            // if valid, then select all the GPUs that contain the segment for sending their data to the current dst GPU
            if(!segmentSelected[s]){

                // we know that the current GPU is in both configurations, so loop over the GPUs that currently contain
                // the data segment and indicate to send it to the indicated destination GPU
                for(size_t l = s * D; l<s*D + D; l++){

                    // get the GPU id
                    size_t srcGPU = virtualTopology[l];

                    // find the GPU in the array of GPUs (we know that it is)
                    size_t indexGPU = 0;
                    while(srcGPU != idGPUs[indexGPU]){
                        indexGPU++;
                    }

                    // add GPU information for later distribution
                        
                    // we are shrinking, move to only one GPU
                    gpusToSplit[indexGPU][0] = dstGPU; // move data to dstGPU
                    nSelectedGPUs[i] ++; // one more source GPU selected for i.th
                    selectedGPUs[indexGPU] = 1; // GPU selected
                    
                    nSelectedTotal ++;
                }

                // segment selected
                segmentSelected[s] = 1;
            }
        }
    }

#ifdef TOPOLOGYAWARE
    
    // GPUaware
    i = 0;
    
    // find the best partition for each GPU
    while(nSelectedTotal < (D * nReconfGPUs)){

        if(nSelectedGPUs[i] < D){

            // id of the dst GPU
            size_t dstGPU = idReconfGPUs[i];

            size_t aff = 9999999;
            size_t localaff = 0;

            size_t segmentIndex;

            // loop over partitions, compute affinities and selected the one with lower value
            for(size_t s = 0; s < nReconfGPUs; s++){

                localaff = 0;

                if(!segmentSelected[s]){

                    for(size_t l = 0; l<D; l++){

                        // get GPU id from virtual topology
                        size_t srcGPU = virtualTopology[s * D + l];

                        // get affinity value
                        localaff += gpuTopology[srcGPU * gNGPUs + dstGPU];
                    }

                    // store
                    if(localaff < aff){
                    
                        aff = localaff;
                        segmentIndex = s;
                    }
                }
            }
            
            // select segment
            for(size_t l = segmentIndex * D; l < segmentIndex * D + D; l++){

                // get the GPU id
                size_t srcGPU = virtualTopology[l];

                // find the GPU in the array of GPUs (we know that it is)
                size_t indexGPU = 0;
                while(srcGPU != idGPUs[indexGPU]){
                    indexGPU++;
                }
                    
                // we are shrinking, move to only one GPU
                gpusToSplit[indexGPU][0] = dstGPU; // move data to dstGPU
                nSelectedGPUs[i] ++; // one more source GPU selected for i.th
                selectedGPUs[indexGPU] = 1; // GPU selected

                nSelectedTotal ++;
            }

            // segment selected
            segmentSelected[segmentIndex] = 1;
        }

        // move to the next GPU
        i = (i+1) % nReconfGPUs;
    }

#else

    // all possible data is kept in GPUs, now, manage the rest
    // find the available GPUs and set
    for(i = 0; i<nReconfGPUs; i++){

        size_t dstGPU = idReconfGPUs[i];

        size_t nextIndex = 0;
        while(nSelectedGPUs[i] < D){

            if(!selectedGPUs[nextIndex]){

                gpusToSplit[nextIndex][0] = dstGPU;
                selectedGPUs[nextIndex] = 1;

                nSelectedGPUs[i] ++;
            }
            nextIndex++;
        }
    }

#endif

    // store the GPUs to split
    reconfData->gpusToSplit = gpusToSplit;

    //printf(" [APP]: Splitting information:\n");
    //for(i = 0; i<nGPUs; i++){

    //    printf(" -- GPU %zu (%zu): %zu\n", idGPUs[i], i, gpusToSplit[i][0]);
    //}
    //printf("\n");
}

void configureN2N(jobResources_t *reconfJobResources, jobResources_t *jobResources){

    size_t i, j, f;

    // get current and reconfiguration resources
    size_t *idGPUs = jobResources->idGPUs;
    size_t *idReconfGPUs = reconfJobResources->idGPUs;
    size_t nGPUs = jobResources->nGPUs;

    // GPUs to reconfigure from each GPU
    size_t **gpusToSplit = (size_t**)calloc(nGPUs, sizeof(size_t*)); // GPU identifiers to move from each source GPU
    for(i = 0; i<nGPUs; i++){
        gpusToSplit[i] = (size_t*)calloc(1, sizeof(size_t)); // initialize
    }

    // GPU selected (0 | 1)
    size_t *selectedGPUs = (size_t*)calloc(nGPUs, sizeof(size_t)); // post GPU
    size_t *selectedReconfGPUs = (size_t*)calloc(nGPUs, sizeof(size_t)); // post GPU

    // find if the GPUs in idGPUs appear in idReconfGPUs
    for(i = 0; i<nGPUs; i++){

        // get GPU id
        size_t srcGPU = idGPUs[i];

        // try to find the GPU
        j = 0;
        while(j < nGPUs && srcGPU != idReconfGPUs[j])
            j++;

        // if found
        if(j < nGPUs){

            gpusToSplit[i][0] = srcGPU;
            selectedGPUs[i] = 1;
            selectedReconfGPUs[j] = 1;
        }
    }

#ifdef TOPOLOGYAWARE

    // topology aware
    for(i = 0; i<nGPUs; i++){

        // get GPU id
        size_t srcGPU = idGPUs[i];

        if(!selectedGPUs[i]){

            size_t aff = 9999999;
            size_t bIndex = 0;

            // loop over reconfiguration GPUs and find a not selected one
            for(j = 0; j<nGPUs; j++){

                size_t dstGPU = idReconfGPUs[j];

                if(!selectedReconfGPUs[j] && gpuTopology[srcGPU * gNGPUs + dstGPU] < aff && srcGPU != dstGPU){

                    aff = gpuTopology[srcGPU * gNGPUs + dstGPU];
                    bIndex = j;
                }
            }
            
            gpusToSplit[i][0] = idReconfGPUs[bIndex];
            selectedGPUs[i] = 1;
            selectedReconfGPUs[bIndex] = 1;
        }
    }

#else

    // find GPUs for the remaining GPUs
    for(i = 0; i<nGPUs; i++){

        // get GPU id
        size_t srcGPU = idGPUs[i];

        if(!selectedGPUs[i]){

            // loop over reconfiguration GPUs and find a not selected one
            j = 0;
            while(j<nGPUs && selectedReconfGPUs[j]){
                j++;
            }

            if(j<nGPUs){
        
                gpusToSplit[i][0] = idReconfGPUs[j];
                selectedGPUs[i] = 1;
                selectedReconfGPUs[j] = 1;
            }
        }
    }

#endif

    reconfData->gpusToSplit = gpusToSplit;

    //printf(" [APP]: Splitting information:\n");
    //for(i = 0; i<nGPUs; i++){
//
    //    printf(" -- GPU %zu (%zu): %zu\n", idGPUs[i], i, gpusToSplit[i][0]);
    //}
    //printf("\n");

}



/* Metadata generation for reconfiugrations */

// [GPU - CPU - GPU]

// All GPUs receive the entire array of N elements
void configureEntireTransmission(DTI_t *DTI, jobResources_t *jobResources){

    size_t i, nGPUs, *idGPUs;
    size_t N = DTI->N;
    DTIDesctiption_t *description = DTI->description;

    idGPUs = jobResources->idGPUs;
    nGPUs = jobResources->nGPUs;

    // allocate new virtual topology
    if(virtualTopology) free(virtualTopology);
    virtualTopology = (size_t*)calloc(nGPUs, sizeof(size_t));


    // configure information for all GPUs
    for(i = 0; i<nGPUs; i++){
        
        // add GPU to virtual topology
        virtualTopology[i] = idGPUs[i];

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
void configureSimpleTransmission(DTI_t *DTI, jobResources_t *jobResources){

    size_t i, n, tmpOffset, nGPUs, *idGPUs;
    size_t N = DTI->N;
    size_t nElements, rElements;
    DTIDesctiption_t *description = DTI->description;

    idGPUs = jobResources->idGPUs;
    nGPUs = jobResources->nGPUs;

    // allocate new virtual topology
    if(virtualTopology) 
        free(virtualTopology);
    virtualTopology = (size_t*)calloc(nGPUs, sizeof(size_t));

    // compute the number of elements per GPU
    nElements = N / nGPUs;
    rElements = N % nGPUs;

    // configure the data distribution depending on the strategy for managing remaining elements
    // all remaining elements processed by the first GPU
    tmpOffset = 0;
    for(i = 0; i<nGPUs; i++){

        // add GPU to virtual topology
        virtualTopology[i] = idGPUs[i];

        // manage partitions
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
void configureComplexTransmission(DTI_t *DTI, jobResources_t *jobResources){

    size_t i, n, tmpOffset, nGPUs, *idGPUs;
    size_t N = DTI->N;
    size_t nElementsPerGPU, rElementsPerGPU, nElementsPerPartition, rElementsPerPartition, nPartitionsPerGPU;
    DTIDesctiption_t *description = DTI->description;
    size_t s = description->s;

    idGPUs = jobResources->idGPUs;
    nGPUs = jobResources->nGPUs;

    // allocate new virtual topology
    if(virtualTopology) free(virtualTopology);
    virtualTopology = (size_t*)calloc(nGPUs, sizeof(size_t));


    // WARNING: We are assuming here that N is divisible by [s * nGPUs]
    // N: total number of elements in the DTI array
    // s: number of elements on each partition
    // nGPUs: number of GPUs for the reconfiguration

    // compute the number of partitions to be stored on each GPU (each partition of s elements)
    //nPartitionsPerGPU =  N / (s * nGPUs);
    
    nPartitionsPerGPU = s;
    size_t partitionSize = N / (nPartitionsPerGPU * nGPUs); // the partition size depends on the number of partitions per GPU and the number of GPUs

    // initial number of elements and remaining elements dividing N by the number of GPUs
    nElementsPerGPU = partitionSize * nPartitionsPerGPU;
    rElementsPerGPU = 0;

    // initialize and allocate partitions information in the DTI
    for(i = 0; i<nGPUs; i++){

        // add GPU to virtual topology
        virtualTopology[i] = idGPUs[i];

        // compute the number of elements per partition. It should be s, but it is necessary to manage remaining elemnts
        DTI->nPartitionsPerGPU[i] = nPartitionsPerGPU; // all GPUs receive the same number of partitions

        // allocate memory for partitions information
        DTI->nPerPartition[i] = (size_t*)calloc(DTI->nPartitionsPerGPU[i], sizeof(size_t));
        DTI->offsetPerPartition[i] = (size_t*)calloc(DTI->nPartitionsPerGPU[i], sizeof(size_t));
    }


    tmpOffset = 0;
    for(size_t par = 0; par < nPartitionsPerGPU; par++){
        
        //printf(" Partition = %zu\n", par);

        for(i = 0; i<nGPUs; i++){
    
            //printf("  GPU = %zu: n %zu, nP %zu, oP %zu\n", i, partitionSize, partitionSize, tmpOffset);
            // there is only one partition divided between all GPUs
            DTI->nPerGPU[i] += partitionSize;
            DTI->nPerPartition[i][par] = partitionSize; // there is only one partition 
            DTI->offsetPerPartition[i][par] = tmpOffset; // all start in 0 since the entire array is copied

            tmpOffset += partitionSize;
        }
    }

    //fflush(stdout);
}



// [Expansions]

void configureSimpleExpandTransmission(DTI_t *DTI, jobResources_t *reconfJobResources, jobResources_t *jobResources){

    size_t i, j, n, tmpOffset, next = 0;
    size_t N = DTI->N;
    size_t nElements, rElements;
    DTIDesctiption_t *description = DTI->description;

    // get gpus to move from each GPU
    size_t **gpusToSplit = reconfData->gpusToSplit;

    // get current and reconfiguration GPUs information
    size_t *idGPUs = jobResources->idGPUs;
    size_t *idReconfGPUs = reconfJobResources->idGPUs;
    size_t nGPUs = jobResources->nGPUs;
    size_t nReconfGPUs = reconfJobResources->nGPUs;

    size_t D = nReconfGPUs / nGPUs;


    // each GPU receives only one partition
    //n = N / nReconfGPUs; 

    // number of data partitions
    size_t nPartitionsPerGPU = description->s;
    if(nPartitionsPerGPU == 0){
        nPartitionsPerGPU = 1;
    }

    // number of elements in the new partitions (after reconfigurations)
    n = N / (nPartitionsPerGPU * nReconfGPUs); // partition size

    // memory for the new virtual topology
    size_t *newVirtualTopology = (size_t*)calloc(nReconfGPUs, sizeof(size_t));

    // create metadata 
    // - number of elemnets per GPU
    // - offsets
    tmpOffset = 0;

    // create new virtual topology
    for(i = 0; i<nGPUs; i++){

        // get GPU i in virtual topology (that defines the order of the data structure)
        size_t srcGPU = virtualTopology[i];

        // find the GPU in the array of current GPU configuration
        size_t indexGPU = 0;
        while(idGPUs[indexGPU] != srcGPU){

            indexGPU++;
        }

        // store the GPUs related to this in the new virtual topology
        for(j = 0; j<D; j++){

            newVirtualTopology[i * D + j] = gpusToSplit[indexGPU][j];
        }
    }


    // create metadata
    // this data is used when we want to reconstruct the complete array in the CPU, but it is not
    // directly used for the peer to peer data redistribution
    for(i = 0; i<nReconfGPUs; i++){

        // get the i.th GPU in the data structure order
        size_t gpu = newVirtualTopology[i];
        size_t gpuIndex = 0;

        // get GPU index in the array of reconfiguration GPUs
        while(idReconfGPUs[gpuIndex] != gpu){

            gpuIndex++;
        }

        // build metadata

        // each GPU receives only one partition
        DTI->nPartitionsPerGPU[gpuIndex] = nPartitionsPerGPU; 

        // allocate memory for partitions information
        DTI->nPerPartition[gpuIndex] = (size_t*)calloc(DTI->nPartitionsPerGPU[gpuIndex], sizeof(size_t));
        DTI->offsetPerPartition[gpuIndex] = (size_t*)calloc(DTI->nPartitionsPerGPU[gpuIndex], sizeof(size_t));

        // total number of element to the GPU
        DTI->nPerGPU[gpuIndex] = n * DTI->nPartitionsPerGPU[gpuIndex]; // each GPU receives a entire partition

        // loop over partitions and compute offsets and number of elements
        for(j = 0; j<DTI->nPartitionsPerGPU[gpuIndex]; j++){

            // there is only one partition divided between all GPUs
            DTI->nPerPartition[gpuIndex][j] = n; // there is only one partition 
            DTI->offsetPerPartition[gpuIndex][j] = tmpOffset + j * n * nReconfGPUs; // all start in 0 since the entire array is copied
        
            //printf(" -- GPU %zu (%zu): N per GPU = %zu, Number of partitions = %zu, N per partition = %zu, offset per partition = %zu\n", 
            //        gpu, gpuIndex, DTI->nPerGPU[gpuIndex], DTI->nPartitionsPerGPU[gpuIndex], DTI->nPerPartition[gpuIndex][j], DTI->offsetPerPartition[gpuIndex][j]);
            //fflush(stdout);
        }

        tmpOffset += n;
    }

    // update virtual topology
    if(virtualTopology) free(virtualTopology);
    virtualTopology = newVirtualTopology;
}



// [Shrinks]

void configureSimpleShrinkTransmission(DTI_t *DTI, jobResources_t *reconfJobResources, jobResources_t *jobResources){

    size_t i, j, n, tmpOffset = 0, next = 0;
    size_t N = DTI->N;
    size_t nElements, rElements;
    DTIDesctiption_t *description = DTI->description;

    // get gpus to move from each GPU
    size_t **gpusToSplit = reconfData->gpusToSplit;

    // get current and reconfiguration GPUs information
    size_t *idGPUs = jobResources->idGPUs;
    size_t *idReconfGPUs = reconfJobResources->idGPUs;
    size_t nGPUs = jobResources->nGPUs;
    size_t nReconfGPUs = reconfJobResources->nGPUs;

    // each GPU receives only one partition
    //n = N / nReconfGPUs; 

    // number of data partitions
    size_t nPartitionsPerGPU = description->s;
    if(nPartitionsPerGPU == 0){
        nPartitionsPerGPU = 1;
    }
    n = N / (nPartitionsPerGPU * nReconfGPUs); // partition size


    size_t D = nGPUs / nReconfGPUs;


    // allocate memory for the new virtual topology
    size_t *newVirtualTopology = (size_t*)calloc(nReconfGPUs, sizeof(size_t));
    for(i = 0; i<nReconfGPUs; i++){
        newVirtualTopology[i] = 99999;
    }


    next = 0;

    // loop over old virtual topology and create new one
    for(i = 0; i<nGPUs; i++){

        // get i.th GPU in the virtual topology (that defines the order of the data structure)
        size_t srcGPU = virtualTopology[i];

        // find the GPU in the array of current GPU configuration
        size_t indexGPU = 0;
        while(idGPUs[indexGPU] != srcGPU){

            indexGPU++;
        }

        // destination gpu
        size_t dstGPU = gpusToSplit[indexGPU][0];

        // check if the destination GPU is already in the newvirtual topology
        j = 0;
        while(j < nReconfGPUs && newVirtualTopology[j] != dstGPU){
            j++;
        }

        // if not in newVirtualTopology, add
        if(j == nReconfGPUs){

            newVirtualTopology[next] = dstGPU;
            next++;
        }
    }

    // create metadata
    // this metadata is used only for rebuilding the entire array on the CPU, not for shrinking
    // from GPUs to GPUs
    for(i = 0; i<nReconfGPUs; i++){

        size_t gpu = newVirtualTopology[i];
        size_t gpuIndex = 0;

        // get GPU index
        while(idReconfGPUs[gpuIndex] != gpu){

            gpuIndex++;
        }

        // each GPU receives only one partition
        DTI->nPartitionsPerGPU[gpuIndex] = nPartitionsPerGPU; 

        // allocate memory for partitions information
        DTI->nPerPartition[gpuIndex] = (size_t*)calloc(DTI->nPartitionsPerGPU[gpuIndex], sizeof(size_t));
        DTI->offsetPerPartition[gpuIndex] = (size_t*)calloc(DTI->nPartitionsPerGPU[gpuIndex], sizeof(size_t));

        // there is only one partition divided between all GPUs
        DTI->nPerGPU[gpuIndex] = n * nPartitionsPerGPU; // each GPU receives a entire partition

        for(j = 0; j<nPartitionsPerGPU; j++){
        
            DTI->nPerPartition[gpuIndex][j] = n; // all partitions have the same number of elements
            
            // partitions are combined to create bigger partitions when shrinking. Therefore, tmpOffset
            // depends on the GPU we are moving, while [j * n * nReconfGPUs] is the internal offset
            // between elements in the partition
            DTI->offsetPerPartition[gpuIndex][j] = tmpOffset + j * n * nReconfGPUs; // all start in 0 since the entire array is copied
        
            //printf(" -- GPU %zu (%zu): N per GPU = %zu, Number of partitions = %zu, N per partition = %zu, offset per partition = %zu\n", 
            //        gpu, gpuIndex, DTI->nPerGPU[gpuIndex], DTI->nPartitionsPerGPU[gpuIndex], DTI->nPerPartition[gpuIndex][0], DTI->offsetPerPartition[gpuIndex][0]);
            //fflush(stdout);
        }
        tmpOffset += n;
    }

    // update virtual topology
    if(virtualTopology) 
        free(virtualTopology);
    virtualTopology = newVirtualTopology;
}



// [N2N]

void configureSimpleN2NTransmission(DTI_t *DTI, jobResources_t *reconfJobResources, jobResources_t *jobResources){

    size_t i, j, n, tmpOffset, next = 0;
    size_t N = DTI->N;
    size_t nElements, rElements;
    DTIDesctiption_t *description = DTI->description;

    // get gpus to move from each GPU
    size_t **gpusToSplit = reconfData->gpusToSplit;

    // get current and reconfiguration GPUs information
    size_t *idGPUs = jobResources->idGPUs;
    size_t *idReconfGPUs = reconfJobResources->idGPUs;
    size_t nGPUs = jobResources->nGPUs;


    // each GPU receives only one partition
    //n = N / nGPUs; 

    // number of data partitions
    size_t nPartitionsPerGPU = description->s;
    if(nPartitionsPerGPU == 0){
        nPartitionsPerGPU = 1;
    }
    n = N / (nPartitionsPerGPU * nGPUs); // partition size


    // deallocate current virtual topology and allocate memory for the new one
    size_t *newVirtualTopology = (size_t*)calloc(nGPUs, sizeof(size_t));
    next = 0;

    // loop over old virtual topology and create new one
    for(i = 0; i<nGPUs; i++){

        // get i.th GPU in the virtual topology (that defines the order of the data structure)
        size_t srcGPU = virtualTopology[i];

        // find the GPU in the array of current GPU configuration (we know the GPU is in the array)
        size_t indexGPU = 0;
        while(idGPUs[indexGPU] != srcGPU){

            indexGPU++;
        }

        // destination gpu
        size_t dstGPU = gpusToSplit[indexGPU][0];

        // if not in newVirtualTopology, add
        newVirtualTopology[next] = dstGPU;
        next++;
    }

    // create metadata
    tmpOffset = 0;
    for(i = 0; i<nGPUs; i++){

        size_t gpu = newVirtualTopology[i];
        size_t gpuIndex = 0;

        // get GPU index in the array of reconfiguration GPUs (we know the GPU is in the array)
        while(idReconfGPUs[gpuIndex] != gpu){

            gpuIndex++;
        }

        // each GPU receives only one partition
        DTI->nPartitionsPerGPU[gpuIndex] = nPartitionsPerGPU; 

        // allocate memory for partitions information
        DTI->nPerPartition[gpuIndex] = (size_t*)calloc(DTI->nPartitionsPerGPU[gpuIndex], sizeof(size_t));
        DTI->offsetPerPartition[gpuIndex] = (size_t*)calloc(DTI->nPartitionsPerGPU[gpuIndex], sizeof(size_t));

        // there is only one partition divided between all GPUs
        DTI->nPerGPU[gpuIndex] = n * nPartitionsPerGPU; // each GPU receives a entire partition

        for(j = 0; j<nPartitionsPerGPU; j++){

            DTI->nPerPartition[gpuIndex][j] = n; // there is only one partition 
            DTI->offsetPerPartition[gpuIndex][j] = tmpOffset + j * n * nGPUs; // all start in 0 since the entire array is copied
        
            //printf(" -- GPU %zu (%zu): N per GPU = %zu, Number of partitions = %zu, N per partition = %zu, offset per partition = %zu\n", 
            //        gpu, gpuIndex, DTI->nPerGPU[gpuIndex], DTI->nPartitionsPerGPU[gpuIndex], DTI->nPerPartition[gpuIndex][0], DTI->offsetPerPartition[gpuIndex][0]);
            //fflush(stdout);
        }

        tmpOffset += n;
    }

    free(virtualTopology);
    virtualTopology = newVirtualTopology;
}