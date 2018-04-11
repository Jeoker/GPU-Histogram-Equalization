#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <string>
#include <ctime>

using namespace cv;
using namespace std;

int main(int argc, const char** argv)
{
	// rudimentory timer
	clock_t start, stop;
	clock_t duration, durationCV, durationCUDA;

	// read in image
	if (argc != 2) {
		cout << "Usage: " << argv[0] << " IMAGE_LOCATION" << endl;
		return -1;
	}
	Mat src = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	if (src.empty()) {
		cout << "ÎÞ·¨ÔØÈëÍ¼Æ¬£¡" << endl;
		return -1;
	}

	// Mat -> vector
	int imgWidth = src.cols;
	int imgHeight = src.rows;
	int imgSize = imgWidth*imgHeight;
	unsigned char *img = new unsigned char[imgSize];
	memcpy(img, src.data, imgSize * sizeof(unsigned char));

	start = clock();
	// get histogram
	unsigned int Hist[256] = { 0 };
	for (int i = 0; i < imgSize; ++i) ++Hist[img[i]];
	int mincdf;
	for (int i = 0; i < 256; ++i) {
		if (Hist[i] > 0) {
			mincdf = i;
			break;
		}
	}
	// histogram addition
	unsigned int cdfHist[256] = { 0 };
	cdfHist[0] = Hist[0];
	for (int i = 1; i < 256; ++i) {
		cdfHist[i] = cdfHist[i - 1] + Hist[i];
	}

	// generate look-up table
	double lutTemp[256] = { 0 };
	for (int i = mincdf + 1; i < 256; ++i) {
		lutTemp[i] = 255.0 * (cdfHist[i] - cdfHist[mincdf]) / (imgSize - cdfHist[mincdf]);
	}
	unsigned char lut[256] = { 0 };
	for (int i = 0; i < 256; ++i) {
		lut[i] = round(lutTemp[i]);
	}
	// transform LUT to output
	unsigned char *img2 = new unsigned char[imgSize];
	for (int i = 0; i < imgSize; ++i) {
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
	for (int i = 0; i < imgSize; ++i) {
		int temp = abs(int(imgCV.data[i]) - int(img2[i]));
		if (temp != 0) {
			testOK = false;
			break;
		}
	}
	if (testOK) cout << "User defined function PASSED!" << endl;
	else cout << "User defined function FAILED!" << endl;

	imshow("CPU-Output", output);
	imshow("OpenCV-Output", imgCV);
	imshow("Origin", src);
	waitKey(0);

    return 0;
}
}
