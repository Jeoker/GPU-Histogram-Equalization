# GPU-Histogram-Equalization
GPU Final Project

See https://www.math.uci.edu/icamp/courses/math77c/demos/hist_eq.pdf as theory details.

Road map:

Implementation plan:
1. Based on the size of the image. We will dynamically choose to use CPU or GPU.
2. If the input image is very large. We need to do image spilit so that to fit in memory.

GPU programm Implementation:
1. Use multi-thread model to calculate intensity possibility.
    1.1 Calculate histogram value by spilit the picture into small parts. Each block responsible for serveral rows
    1.2 Use reduction tree to sum up.
    1.3 Parallel division.
2. Calculate the histogram equalized image by CPU or GPU?? (TBD by experienment)
