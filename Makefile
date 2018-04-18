# environment
SM := 35

GCC := g++
NVCC := nvcc

# Remove function
RM = rm -f

# Specify opencv Installation
#opencvLocation = /usr/local/opencv
opencvLIB= -L/shared/apps/opencv-3.0.0-beta/INSTALL/lib
opencvINC= -I/shared/apps/opencv-3.0.0-beta/INSTALL/include

GENCODE_FLAGS := -gencode arch=compute_$(SM),code=sm_$(SM) -arch sm_$(SM)
LIB_FLAGS := -lcudadevrt -lcudart

NVCCFLAGS :=
GccFLAGS = -fopenmp -O3 

TARGET = histogram

all: build

build: $(TARGET)

assist.o: ./assist.cu
	$(NVCC) $(NVCCFLAGS) $^ -dc -o $@ $(GENCODE_FLAGS)

histogram.o: ./histogram.cu 
	$(NVCC) $(NVCCFLAGS) $^ -dc -o $@ $(GENCODE_FLAGS)

dlink.o: ./assist.o ./histogram.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(GENCODE_FLAGS) -dlink

main.o: ./main.cpp ./assist.h ./histogram.h ./main.h ./rlc_encode.h
	$(GCC) $(GccFLAGS) $(opencvLIB) $(opencvINC) -c $< -o $@ -std=c++11

rlc_encode.o: ./rlc_encode.cpp ./rlc_encode.h
	$(GCC) $(GccFLAGS) $(opencvLIB) $(opencvINC) -c $< -o $@ -std=c++11

$(TARGET): main.o assist.o histogram.o rlc_encode.o
		$(NVCC) $(NVCCFLAGS) $(opencvLIB) $(opencvINC) $^ -o $@ $(GENCODE_FLAGS) -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_photo -lopencv_video	

clean:
	$(RM) *.o
	$(RM) $(TARGET)
