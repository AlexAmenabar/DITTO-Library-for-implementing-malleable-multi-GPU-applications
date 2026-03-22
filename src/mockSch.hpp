#ifndef MOCKSCH_HPP
#define MOCKSCH_HPP

typedef struct jobControl_t {

    // dedicated resources
    size_t nGPUs;
    size_t *idGPUs;

    // monitoring
    double *gpuUsage;

} jobControl_t;


typedef struct schInfo_t {

    size_t nGPUs;
    char *avGPUs; // 0 | 1

    size_t nJobs;
    size_t nMaxJobs;
    jobControl_t *jobControl;

} schInfo_t;

#endif