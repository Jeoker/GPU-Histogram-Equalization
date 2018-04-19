#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <string>
#include <ctime>

#define TILE_SIZE 16

using namespace cv;
using namespace std;

cudaError_t eqHist(unsigned char *input, unsigned int width, unsigned int height, unsigned char *output);

// Kernel
// Add GPU kernel and functions
__global__ void kernel(unsigned char *input, unsigned char *output)
{

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    int location = y * TILE_SIZE * gridDim.x + x;

    output[location] = x % 255;
}

int main(int argc, const char **argv)
{
    // rudimentory timer
    clock_t start, stop;
    clock_t duration, durationCV, durationCUDA;

    // read in image
    if (argc != 2)
    {
        cout << "Usage: " << argv[0] << " IMAGE_LOCATION" << endl;
        return -1;
    }
    Mat src = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    if (src.empty())
    {
        cout << "�޷�����ͼƬ��" << endl;
        return -1;
    }

    // Mat -> vector
    int imgWidth = src.cols;
    int imgHeight = src.rows;
    int imgSize = imgWidth * imgHeight;
    unsigned char *img = new unsigned char[imgSize];
    memcpy(img, src.data, imgSize * sizeof(unsigned char));

    start = clock();
    // get histogram
    unsigned int Hist[256] = {0};
    for (int i = 0; i < imgSize; ++i)
        ++Hist[img[i]];
    int mincdf;
    for (int i = 0; i < 256; ++i)
    {
        if (Hist[i] > 0)
        {
            mincdf = i;
            break;
        }
    }
    // histogram addition
    unsigned int cdfHist[256] = {0};
    cdfHist[0] = Hist[0];
    for (int i = 1; i < 256; ++i)
    {
        cdfHist[i] = cdfHist[i - 1] + Hist[i];
    }

    // generate look-up table
    double lutTemp[256] = {0};
    for (int i = mincdf + 1; i < 256; ++i)
    {
        lutTemp[i] = 255.0 * (cdfHist[i] - cdfHist[mincdf]) / (imgSize - cdfHist[mincdf]);
    }
    unsigned char lut[256] = {0};
    for (int i = 0; i < 256; ++i)
    {
        lut[i] = round(lutTemp[i]);
    }
    // transform LUT to output
    unsigned char *img2 = new unsigned char[imgSize];
    for (int i = 0; i < imgSize; ++i)
    {
        img2[i] = lut[img[i]];
    }
    stop = clock();
    duration = stop - start;

    // using OpenCV
    Mat imgCV;
    start = clock();
    equalizeHist(src, imgCV);
    stop = clock();
    durationCV = stop - start;

    // vector -> Mat
    Mat output(imgHeight, imgWidth, CV_8U);
    memcpy(output.data, img2, imgSize * sizeof(unsigned char));

    // comparing OpenCV & CPU version
    bool testOK = true;
    for (int i = 0; i < imgSize; ++i)
    {
        int temp = abs(int(imgCV.data[i]) - int(img2[i]));
        if (temp != 0)
        {
            testOK = false;
            break;
        }
    }
    if (testOK)
        cout << "User defined function PASSED!" << endl;
    else
        cout << "User defined function FAILED!" << endl;

    imshow("CPU-Output", output);
    imshow("OpenCV-Output", imgCV);
    imshow("Origin", src);
    waitKey(0);

    // GPU calculation
    unsigned char *imgGPU = new unsigned char[imgSize];
    start = clock();
    cudaError_t cudaStatus = eqHist(img, imgWidth, imgHeight, imgGPU);
    stop = clock();
    durationCUDA = stop - start;
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "eqHist on CUDA failed!");
        return 1;
    }

    cout << "Calculation Time:\nUser: " << duration << "\tOpenCV: " << durationCV
         << "\tCalculation Time:\nCUDA: " << durationCUDA << endl;

    // comparing OpenCV & GPU
    bool testGPU = true;
    int count = 0;
    for (int i = 0; i < imgSize; ++i)
    {
        int temp = int(imgCV.data[i]) - int(imgGPU[i]);
        if (abs(temp) > 20)
            ++count;
    }
    //GPU Error shouldn't be more than 1%
    if (count > imgSize / 100)
        testGPU = false;

    if (testGPU)
        cout << "CUDA PASSED! DIFFERENCE COUNT: " << count << endl;
    else
        cout << "CUDA FAILED! DIFFERENCE COUNT: " << count << endl;

    // vector -> Mat
    Mat outputGPU(imgHeight, imgWidth, CV_8U);
    memcpy(outputGPU.data, imgGPU, imgSize * sizeof(unsigned char));

    imshow("CUDA�����ͼƬ", outputGPU);

    waitKey(0);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    system("pause");
    return 0;
}

//histogram equalization������
cudaError_t eqHist(unsigned char *input, unsigned int width, unsigned int height, unsigned char *output)
{
    cudaError_t cudaStatus;

    int gridXSize = 1 + ((width - 1) / TILE_SIZE);
    int gridYSize = 1 + ((height - 1) / TILE_SIZE);
    int XSize = gridXSize * TILE_SIZE;
    int YSize = gridYSize * TILE_SIZE;

    // Both are the same size (CPU/GPU).
    int size = XSize * YSize;

    unsigned char *input_gpu;
    unsigned char *output_gpu;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for image
    cudaStatus = cudaMalloc((void **)&input_gpu, size * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void **)&output_gpu, size * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Set tho output image to 0
    cudaStatus = cudaMemset(output_gpu, 0, 256 * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemset failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(input_gpu, input, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Execute algorithm
    dim3 dimGrid(gridXSize, gridYSize);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);

    // Kernel Call
    kernel<<<dimGrid, dimBlock>>>(input_gpu, output_gpu);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(output, output_gpu, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(input_gpu);
    cudaFree(output_gpu);

    return cudaStatus;
}
