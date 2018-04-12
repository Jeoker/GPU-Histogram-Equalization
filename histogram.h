#ifndef HISTOGRAM
#define HISTOGRAM

#include <cuda.h>

__global__ void kernel(unsigned char *input,
					   unsigned char *output);

void histogram_gpu(unsigned char *data, 
                   unsigned int height, 
                   unsigned int width);

void histogram_gpu_warmup(unsigned char *data, 
                   		  unsigned int height, 
                   		  unsigned int width);

#endif
