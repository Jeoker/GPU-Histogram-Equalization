## GPU Final Project: Histogram-Equalization

    See https://www.math.uci.edu/icamp/courses/math77c/demos/hist_eq.pdf for theory details.

### GPU programm Implementation:

Step 0. Read image in CPU and send to GPU

Step 1. Calculate histogram

Step 2. Calculate look-up table

Step 3. Apply look-up table to original image and get new image

Step 4. Send image back to CPU and save it

### Optimizations
1. Use GPU to accelerate calculationmulti-thread model to calculate intensity possibility
    - 1.1 Calculate histogram value by spilit the picture into small parts. Each block responsible for serveral rows
    - 1.2 Use reduction tree to sum up
    - 1.3 Parallel division
Note: this part is actually abandoned because we used Run-length encoding instead

2. Send 4 pixel values into kernals at a time (bit operation during histogram sum up)
    - 2.1 GPU take at least 32 bit from global memory to L2 memory, if we read one pixel every kernal, a lot of bandwidth is wasted
    - 2.2 Cast unsigned char -> int, so we can send 4 pixels value into L2 memory at once
    - 3.3 Use bit operation to seperate each pixel in a kernal
    
 3. Run-length encoding optimization
    - 3.1 Compress the image in CPU before sending to gpu, so that the memory copy time will be optimized
    - 3.2 This approach only work well when image has big block of similar color
    - Taking too much time in encoding and decoding, thus abandoned
    
### Output result

Our GPU program accelerate this algorithm for around 400%.
For an image with a size of 8000 * 4000, the running time is:
 - Kernel Execution Time: 15.074208 ms
 - CPU execution time: 219.912 ms
 - GPU execution time: 34.5817 ms
 - Percentage difference: 0%

<p align="center">
  <img src="https://github.com/Jeoker/GPU-Histogram-Equalization/blob/master/input/Geotagged_articles_wikimap_RENDER_ca_huge.png" width="800px" height="400px"/>
</p>
<center>
  The input image
</center>

<p align="center">
  <img src="https://github.com/Jeoker/GPU-Histogram-Equalization/blob/master/output/output_gpu.jpg" width="800px" height="400px"/>
</p>
<center>
  The output image
</center>
