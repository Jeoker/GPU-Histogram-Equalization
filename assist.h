#ifndef ASSIST
#define ASSIST

#include <cuda.h>
double CLOCK();

cudaError_t checkCuda(cudaError_t result);
#endif
