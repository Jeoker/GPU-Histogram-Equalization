#ifndef MAIN
#define MAIN
extern double CLOCK();

extern void histogram_gpu(unsigned char *data, 
                          unsigned int height, 
                          unsigned int width);
                          
extern void histogram_gpu_warmup(unsigned char *data, 
                          		 unsigned int height, 
                                 unsigned int width);
#endif
