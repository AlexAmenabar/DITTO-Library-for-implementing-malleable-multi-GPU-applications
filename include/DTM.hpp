#ifndef CUDATM_HPP
#define CUDATM_HPP

#include "DDM.hpp"


int getNumberOfGPUs();
void setGPUDevice();
void cpyDataCPU2GPU(DTI_t *DTI);
void cpyDataGPU2CPU(DTI_t *DTI);

#endif