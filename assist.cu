/* Some helper functions */
/* including CUDA debug function */
/* and clocking function */
#include <time.h>
#include <stdio.h>
#include "assist.h"
#include <omp.h>
/* Switch of debuging */
#define DEBUG

double CLOCK() {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

cudaError_t checkCuda(cudaError_t result) {
	#if defined(DEBUG) || defined(_DEBUG)
		if (result != cudaSuccess) {
			fprintf(stderr, "CUDA Runtime Error: %s\n",
					cudaGetErrorString(result));
			exit(-1);
		}
	#endif
		return result;
}
