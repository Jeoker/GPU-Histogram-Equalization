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

#define TILE_SIZE 16
#define INTENSITY_RANGE 256

/* Switch of time counting */
#define CUDA_TIMING

unsigned char *input_gpu;
unsigned char *output_gpu;
                

/* Warm up kernel */
__global__ void kernel(unsigned char *input, 
                       unsigned char *output) {
        
  	int x = blockIdx.x * TILE_SIZE + threadIdx.x;
	int y = blockIdx.y * TILE_SIZE+ threadIdx.y;
                
	int location = 	y * TILE_SIZE * gridDim.x + x;
	
	output[location] = x % 255;

}

/* Processing GPU kernel */
__global__ void count_intensity(unsigned char *input,
								unsigned int height,
								unsigned int width,
							    unsigned int *intensity_num) {
  	unsigned int x = blockIdx.x * TILE_SIZE + threadIdx.x;
	unsigned int y = blockIdx.y * TILE_SIZE + threadIdx.y;

	unsigned int location = y * TILE_SIZE * gridDim.x + x;

	if (x < width && y < height) {
		atomicAdd(&intensity_num[input[location]], 1);
	}
}

__global__ void prefixSum(unsigned int *intensity_num,
						  double *intensity_sum,
						  unsigned int height,
						  unsigned int width) {
	for (int i = 1; i < INTENSITY_RANGE; ++i) {
		intensity_num[i] += intensity_num[i - 1];
	}

	for (int i = 0; i < INTENSITY_RANGE; ++i) {
		intensity_sum[i] = intensity_num[i] / (height * width);	
	}
}

__global__ void histo_equalized (unsigned char *input,
					  unsigned char *output,
					  unsigned int height,
			      	  unsigned int width,
			   		  double *intensity_sum) {
  	unsigned int x = blockIdx.x * TILE_SIZE + threadIdx.x;
	unsigned int y = blockIdx.y * TILE_SIZE+ threadIdx.y;

	unsigned int location = y * TILE_SIZE * gridDim.x + x;

    if (x < width && y < height) {
		output[location] = (unsigned char) ((INTENSITY_RANGE - 1) * intensity_sum[location]);
	}

}

void histogram_gpu(unsigned char *data, 
                   unsigned int height, 
                   unsigned int width) {
                         
	int gridXSize = 1 + ((width - 1) / TILE_SIZE);
	int gridYSize = 1 + ((height - 1) / TILE_SIZE);
	
	int XSize = gridXSize * TILE_SIZE;
	int YSize = gridYSize * TILE_SIZE;
	
	 /* Both are the same size (CPU/GPU). */
	int size = XSize * YSize;
		
	 /* Allocate arrays in GPU memory */

	/* Maybe assigned type according to the input size */
	unsigned int *intensity_num;
	double *intensity_sum;

	checkCuda(cudaMalloc((void**) &input_gpu, size * sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**) &output_gpu, size * sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**) &intensity_num, INTENSITY_RANGE * sizeof(unsigned int)));
	checkCuda(cudaMalloc((void**) &intensity_sum, INTENSITY_RANGE * sizeof(double)));

	
     /* Copy data to GPU */
    checkCuda(cudaMemcpy(input_gpu, 
			  data, 
			  size * sizeof(char), 
			  cudaMemcpyHostToDevice));
	checkCuda(cudaMemset(intensity_num, 0, INTENSITY_RANGE * sizeof(unsigned int)));
	checkCuda(cudaDeviceSynchronize());
        
     /* Execute algorithm */
	dim3 dimGrid(gridXSize, gridYSize);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);

     /* Kernel Call */
	#if defined(CUDA_TIMING)
		float Ktime;
		TIMER_CREATE(Ktime);
		TIMER_START(Ktime);
	#endif

	count_intensity<<<dimGrid, dimBlock>>>(input_gpu,
										    height,
										    width,
										    intensity_num);
    checkCuda(cudaDeviceSynchronize());

	prefixSum<<<1, 1>>>(intensity_num, intensity_sum, height, width);

	histo_equalized<<<dimGrid, dimBlock>>>(input_gpu, output_gpu, height, width, intensity_sum);

    checkCuda(cudaDeviceSynchronize());

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
	checkCuda(cudaFree(intensity_sum));
}

void histogram_gpu_warmup(unsigned char *data, 
					      unsigned int height, 
                          unsigned int width) {
                         
	int gridXSize = 1 + (( width - 1) / TILE_SIZE);
	int gridYSize = 1 + ((height - 1) / TILE_SIZE);
	
	int XSize = gridXSize*TILE_SIZE;
	int YSize = gridYSize*TILE_SIZE;
	
	 /* Both are the same size (CPU/GPU). */
	int size = XSize*YSize;
	
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
	dim3 dimGrid(gridXSize, gridYSize);
	dim3 dimBlock(TILE_SIZE, TILE_SIZE);
        
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

