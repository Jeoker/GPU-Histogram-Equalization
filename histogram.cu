#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include "histogram.h"
#include "assist.h"

#define TIMER_CREATE(t)               \
  cudaEvent_t t##_start, t##_end;     \
  cudaEventCreate(&t##_start);        \
  cudaEventCreate(&t##_end);               
 
#define TIMER_START(t)                \
  cudaEventRecord(t##_start);         \
  cudaEventSynchronize(t##_start);    \
 
#define TIMER_END(t)                             \
  cudaEventRecord(t##_end);                      \
  cudaEventSynchronize(t##_end);                 \
  cudaEventElapsedTime(&t, t##_start, t##_end);  \
  cudaEventDestroy(t##_start);                   \
  cudaEventDestroy(t##_end);     

#define TILE_SIZE 512
#define INTENSITY_RANGE 256

/* Switch of time counting */
#define CUDA_TIMING
#define CPU_SWITCH

unsigned char *input_gpu;
unsigned char *output_gpu;

/* Warm up kernel */
__global__ void kernel(unsigned char *input, 
                       unsigned char *output) {
        
  	int location = blockIdx.x * TILE_SIZE + threadIdx.x;
	
	output[location] = location % 255;
}

/* Processing GPU kernel */
__global__ void count_intensity(unsigned int *input,
								unsigned int size,
							    unsigned int *intensity_num) {
	
    unsigned int location = blockIdx.x * TILE_SIZE + threadIdx.x;

	if (location < (size >> 2)) {
		atomicAdd(&intensity_num[(unsigned char)(input[location] & 0xFF000000)], 1);
		atomicAdd(&intensity_num[(unsigned char)(input[location] & 0x00FF0000)], 1);
		atomicAdd(&intensity_num[(unsigned char)(input[location] & 0x0000FF00)], 1);
		atomicAdd(&intensity_num[(unsigned char)(input[location] & 0x000000FF)], 1);
	}
}

__global__ void prefixSum(unsigned int *intensity_num,
						  unsigned char *min_index) {
	
	for (int i = 1; i < INTENSITY_RANGE; ++i) {
		intensity_num[i] += intensity_num[i - 1];
		if (intensity_num[i] < intensity_num[i - 1]) {
			*min_index = i;
		}
	}
}

__global__ void probability(unsigned int *intensity_num,
						    double *intensity_pro,
						    unsigned int size,
							unsigned char *min_index) {
	unsigned int index = threadIdx.x;
	if (index < INTENSITY_RANGE) {
		intensity_pro[index] = ((double) (intensity_num[index] - intensity_num[*min_index])) / (size - intensity_num[*min_index]);
	}
}

__global__ void histo_equalized(unsigned char* input,
							    unsigned int size,
							    double *intensity_pro,
							    unsigned char *output) {

  	unsigned int location = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (location < size) {
		output[location] = (unsigned char) ((INTENSITY_RANGE - 1) * 
                            intensity_pro[input[location]]);
	}
}

void histogram_gpu(unsigned char *data, 
                   unsigned int height, 
                   unsigned int width) {
                         
    /* Both are the same size (CPU/GPU). */
	int size = width * height;
	int gridSize = 1 + ((size - 1) / TILE_SIZE);
	
    unsigned int *intensity_num;
	double *intensity_pro;
	unsigned char *min_index;

	checkCuda(cudaMalloc((void**) &input_gpu, size * sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**) &output_gpu, size * sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**) &intensity_num, INTENSITY_RANGE * sizeof(unsigned int)));
	checkCuda(cudaMalloc((void**) &intensity_pro, INTENSITY_RANGE * sizeof(double)));
	checkCuda(cudaMalloc((void**) &min_index, 1 * sizeof(double)));
		
     /* Copy data to GPU */
    checkCuda(cudaMemcpy(input_gpu, 
			  data, 
			  size * sizeof(char), 
			  cudaMemcpyHostToDevice));
	checkCuda(cudaMemset(intensity_num, 0, INTENSITY_RANGE * sizeof(unsigned int)));
	checkCuda(cudaMemset(min_index, 0, 1 * sizeof(unsigned int)));
	checkCuda(cudaDeviceSynchronize());

        
     /* Execute algorithm */
	dim3 dimGrid(gridSize);
    dim3 dimBlock(TILE_SIZE);

     /* Kernel Call */
	#if defined(CUDA_TIMING)
		float Ktime;
		TIMER_CREATE(Ktime);
		TIMER_START(Ktime);
	#endif
    

	count_intensity<<<dimGrid, dimBlock>>>((unsigned int *)input_gpu,
										    size,
										    intensity_num);
	prefixSum<<<1, 1>>>(intensity_num, min_index);

	probability<<<1, INTENSITY_RANGE>>>(intensity_num, intensity_pro, size, min_index);

	histo_equalized<<<dimGrid, dimBlock>>>(input_gpu, size, intensity_pro, output_gpu);

	#if defined(CUDA_TIMING)
		TIMER_END(Ktime);
		printf("Kernel Execution Time: %f ms\n", Ktime);
	#endif

	 /* Retrieve results from the GPU */
	checkCuda(cudaMemcpy(data, 
						output_gpu, 
						size * sizeof(unsigned char), 
						cudaMemcpyDeviceToHost));
                        
     /* Free resources and end the program */
	checkCuda(cudaFree(output_gpu));
	checkCuda(cudaFree(input_gpu));
	checkCuda(cudaFree(intensity_num));
	checkCuda(cudaFree(intensity_pro));
	checkCuda(cudaFree(min_index));
}

void histogram_gpu_warmup(unsigned char *data, 
					      unsigned int height, 
                          unsigned int width) {
                         
    /* Both are the same size (CPU/GPU). */
	int size = height*width;
	
	int gridSize = 1 + (( size - 1) / TILE_SIZE);
	
	 /* Allocate arrays in GPU memory */
	checkCuda(cudaMalloc((void**) &input_gpu ,size*sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**) &output_gpu ,size*sizeof(unsigned char)));
	checkCuda(cudaMemset(output_gpu , 0 , size*sizeof(unsigned char)));
				
	 /* Copy data to GPU */
	checkCuda(cudaMemcpy(input_gpu, 
		   			 	data, 
			     	    size*sizeof(char), 
						cudaMemcpyHostToDevice));

	checkCuda(cudaDeviceSynchronize());
        
	 /* Execute algorithm */
	dim3 dimGrid(gridSize);
	dim3 dimBlock(TILE_SIZE);
        
	kernel<<<dimGrid, dimBlock>>>(input_gpu, 
								  output_gpu);
                                             
	checkCuda(cudaDeviceSynchronize());
        
	 /* Retrieve results from the GPU */
	checkCuda(cudaMemcpy(data, 
						 output_gpu, 
						 size*sizeof(unsigned char), 
						 cudaMemcpyDeviceToHost));
                        
    /* Free resources and end the program */
	checkCuda(cudaFree(output_gpu));
	checkCuda(cudaFree(input_gpu));
}

