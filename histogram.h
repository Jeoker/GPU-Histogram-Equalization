#ifndef HISTOGRAM
#define HISTOGRAM

#include <cuda.h>

__global__ void kernel(unsigned char *input,
					   unsigned char *output);

void histogram_gpu(unsigned char *grey_value, 
				   unsigned int *pixel_count,
				   unsigned int compress_size,
                   unsigned int height, 
                   unsigned int width,
				   unsigned char *output_cpu);

void histogram_gpu_warmup(unsigned char *data, 
                   		  unsigned int height, 
                   		  unsigned int width);

#endif
