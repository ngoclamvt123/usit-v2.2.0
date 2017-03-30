/*
 * ifpp.cpp
 * iterative fourier push pull extraction
 *
 * Iris extractor: extracts the iris texture out of an eye image and maps it into doubly dimensionless coordinates.
 *
 *  Created on: 13.05.2011
 *      Author: Peter Wild
 *      Author: Heinz Hofbauer
 */

#include "version.h"
#include <map>
#include <list>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo/photo.hpp>
#include <boost/regex.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;
using namespace cv;

/** no globbing in win32 mode **/
int _CRT_glob = 0;

/** Program modes **/
static const int MODE_ERROR = 0, MODE_MAIN = 1, MODE_HELP = 2;

/*
 * Print command line usage for this program
 */
void printUsage() {
    printVersion();
	cout << "+-----------------------------------------------------------------------------+" << endl;
    cout << "| ifpp - extracts the iris texture out of an eye image                        |" << endl;
    cout << "|                                                                             |" << endl;
    cout << "| MODES                                                                       |" << endl;
    cout << "|                                                                             |" << endl;
    cout << "| (# 1) iris texture extraction from eye images                               |" << endl;
    cout << "| (# 2) usage                                                                 |" << endl;
    cout << "|                                                                             |" << endl;
    cout << "| ARGUMENTS                                                                   |" << endl;
    cout << "|                                                                             |" << endl;
    cout << "+------+------------+---+---+-------------------------------------------------+" << endl;
    cout << "| Name | Parameters | # | ? | Description                                     |" << endl;
    cout << "+------+------------+---+---+-------------------------------------------------+" << endl;
    cout << "| -i   | infile     | 1 | N | source image (* = any)                          |" << endl;
    cout << "| -o   | outfile    | 1 | N | target image (?n = n-th * in infile)            |" << endl;
    cout << "| -s   | wdth hght  | 1 | Y | target width and height                         |" << endl;
    cout << "|      |            |   |   | default: same size as original                  |" << endl;
    cout << "| -vm  | visualfile | 1 | Y | visual mask image (?n = n-th * in infile, off)  |" << endl;
    cout << "| -vc  | visualfile | 1 | Y | visual center image (?n = n-th * in infile, off)|" << endl;
    cout << "| -vp  | visualfile | 1 | Y | visual polar image (?n = n-th * in infile, off) |" << endl;
    cout << "| -vb  | visualfile | 1 | Y | visual border image (?n = n-th * in infile, off)|" << endl;
    cout << "| -vs  | visualfile | 1 | Y | segmentation image (?n = n-th * in infile, off) |" << endl;
    cout << "| -bm  | binarymask | 1 | Y | iris mask image (?n = n-th * in infile, off)    |" << endl;
    cout << "| -q   |            | 1 | Y | quiet mode on (off)                             |" << endl;
    cout << "| -t   |            | 1 | Y | time progress on (off)                          |" << endl;
    cout << "| -h   |            | 2 | N | prints usage                                    |" << endl;
    cout << "+------+------------+---+---+-------------------------------------------------+" << endl;
    cout << "|                                                                             |" << endl;
    cout << "| AUTHOR                                                                      |" << endl;
    cout << "|                                                                             |" << endl;
    cout << "| Peter Wild (pwild@cosy.sbg.ac.at)                                           |" << endl;
    cout << "| Heinz Hofbauer (hhofbaue@cosy.sbg.ac.at)                                    |" << endl;
    cout << "|                                                                             |" << endl;
    cout << "| JPEG2000 Hack                                                               |" << endl;
    cout << "| Thomas Bergmueller (thomas.bergmueller@authenticvision.com)                 |" << endl;
    cout << "|                                                                             |" << endl;
    cout << "| COPYRIGHT                                                                   |" << endl;
    cout << "|                                                                             |" << endl;
    cout << "| (C) 2011 All rights reserved. Do not distribute without written permission. |" << endl;
    cout << "+-----------------------------------------------------------------------------+" << endl;
}
/** ------------------------------- Matrix helpers ------------------------------- **/

/**
 * Adds and multiplies a floating point matrix with a given value
 * src: source floating point matrix (CV_32FC1)
 * dst: target floating point matrix (CV_32FC1)
 * a: value to be added
 * m: value to multiply the sum
 */
void fmatAddMult(const Mat& src, Mat& dst, const float a, const float f){
	MatConstIterator_<float> s = src.begin<float>();
	MatIterator_<float> d = dst.begin<float>();
	MatConstIterator_<float> e = src.end<float>();
	for (; s!=e; s++, d++){
		*d = (*s + a) * f;
	}
}

/**
 * Gets average value
 * src: source floating point matrix (CV_32FC1)
 * avg: resulting average value
 */
void fmatAverage(const Mat& src, float& avg){
	avg = 0;
	MatConstIterator_<float> s = src.begin<float>();
	MatConstIterator_<float> e = src.end<float>();
	for (; s!=e; s++){
		avg += *s;
	}
	avg /= (src.cols * src.rows);
}

/**
 * Adds and multiplies a floating point matrix with a given value
 * src: source floating point matrix (CV_32FC1)
 * dst: target floating point matrix (CV_32FC1)
 * a: value to be added
 */
void fmatAdd(const Mat& src, Mat& dst, const float a){
	MatConstIterator_<float> s = src.begin<float>();
	MatIterator_<float> d = dst.begin<float>();
	MatConstIterator_<float> e = src.end<float>();
	for (; s!=e; s++, d++){
		*d = (*s + a);
	}
}

/**
 * Modifies contour, such that it has mean zero
 * contour: contour to be normalized (CV_32FC1)
 * mean: average value
 */
void meanZero(Mat& contour, float& mean){
	float * b = (float *)contour.data;
	int width = contour.cols;
	double sum = 0;
	for (int i=0; i<width; i++, b++){
		sum += *b;
	}
	mean = sum / width;
	b = (float *)contour.data;
	for (int i=0; i<width; i++, b++){
		*b -= mean;
	}
}

/** ------------------------------- Mask generation ------------------------------- **/

/**
 * Fast method to calculate a standard uniform-intervalled histogram
 *
 * src: input image
 * hist: MatND array
 * channel: number of channel to be evaluated (default: 0)
 * bins: number of bins in histogram (default: 256)
 * min: minimum pixel value (default: 0)
 * max: maximum pixel value (default: 256)
 */
void calcHist(const Mat& src, MatND& hist, const int channel = 0, const int bins = 256, const float min = 0, const float max = 256){
	int channels[] = {channel};
	int histSize[] = {bins}; // number of bins
	float s_ranges[] = { min, max }; //lower inclusive, upper exclusive
	const float * ranges[] = {s_ranges};
	calcHist(&src,1,channels,Mat(),hist,1,histSize,ranges);
}

/**
 * Computation of quantile values out of histograms, such that:
 * quantile * 100% are less than returned value, and
 * (1-quantile) * 100 % are greater or equal than returned value.
 * computation assumes uniform distribution of pixel values within interval
 *
 * hist:      histogram used for computation
 * count:     pixel count, i.e. number of pixels in histogram
 * quantile:  quantile between 0 and 1
 * bins:      number of bins in histogram (default: 256)
 * min:       minimum pixel value (default: 0)
 * max:       maximum pixel value (default: 256)
 *
 * returns float value between min and max
 */
float histQuantile(const MatND& hist, const int count, float quantile, const int bins = 256, const float min = 0, const float max = 256){
	CV_Assert(0 <= quantile && quantile <= 1);
	float left = cvRound(quantile * count); // that many pixels should be less than returned value assuming uniform distribution of real pixel values within histogram interval
	float sum = 0;
	int iv = 0;
	for (sum = hist.at<float>(iv); iv < 255 && sum <= left; sum += hist.at<float>(++iv));
	for (;hist.at<float>(iv) == 0; iv--);
	float ivSize = (max-min)/bins;
	return (min + (iv + 1 - (sum - left)/hist.at<float>(iv))*ivSize);
}

/*
 * Calculates magnitude and orientation images.
 *
 * srcX: CV_32FC1 input image, values in x-direction
 * srcY: CV_32FC1 input image, values in y-direction
 * mag:   CV_8UC1 (values normalized to 0..255) or CV_32FC1 (absolute numbers, floating point) destination image (inplace possible)
 * orient: CV_8UC1 (values normalized to 0..180) or CV_32FC1 (absolute angle in degrees 0..360, floating point) destination image (inplace possible)
 * scaleFactor: returns scaling Factor for magnitude conversion to CV_8UC1, if was necessary (default: 1)
 * logScale: if true, magnitude image is subjected logarithmic scaling
 */
void vec2polar(const Mat& srcX, const Mat& srcY, Mat& mag, Mat& orient, float& scaleFactor, const bool logScale = false)
{
	CV_Assert(srcX.type() == CV_32FC1);
	CV_Assert(srcY.type() == CV_32FC1);
	CV_Assert(mag.empty() || mag.type() == CV_8UC1 || mag.type() == CV_32FC1);
	CV_Assert(orient.empty() || orient.type() == CV_8UC1 || orient.type() == CV_32FC1);
	CV_Assert(srcY.size() == srcX.size());
	CV_Assert(mag.empty() || mag.size() == srcX.size());
	CV_Assert(orient.empty() || orient.size() == srcX.size());
	const unsigned int MAG32F = 2, ORIENT32F = 8, CALCMAG = 1, CALCORIENT = 4;
	const unsigned int CALCMAG8U = CALCMAG, CALCMAG32F = CALCMAG+MAG32F, CALCORIENT8U = CALCORIENT, CALCORIENT32U = CALCORIENT + ORIENT32F, CALCMAG8UORIENT8U = CALCMAG+CALCORIENT, CALCMAG32FORIENT8U = CALCMAG+MAG32F+CALCORIENT, CALCMAG8UORIENT32F = CALCMAG+CALCORIENT+ORIENT32F, CALCMAG32FORIENT32F = CALCMAG+MAG32F+CALCORIENT+ORIENT32F;
	Mat tmp; // magnitude is a bit more complicated than orientation due to scaling issues, we need a 32f helper image
	float max;
	float *px, * py, * pmag, * porient32f;
	uchar * porient8u;
	int height = srcX.rows;
	int width = srcX.cols;
	int magoffset, orientoffset, offset;
	scaleFactor = 1;
	unsigned int mode = 0;
	if (!mag.empty()){
		mode |= CALCMAG;
		if (mag.type() == CV_32FC1){
			mode |= MAG32F;
			tmp = mag;
		}
		else {
			tmp.create(height,width,CV_32FC1);
		}
	}
	if (!orient.empty()){
		mode |= CALCORIENT;
		if (orient.type() == CV_32FC1){
			mode |= ORIENT32F;
		}
	}
	px = (float *)(srcX.data);
	py = (float *)(srcY.data);
	offset = srcX.step/sizeof(float) - srcX.cols;
	if (mode == CALCMAG8U){
		pmag = (float *)(tmp.data);
		magoffset = tmp.step/sizeof(float) - tmp.cols;
		max = 0;
		if (logScale){
			for (int y=0; y < height; y++, pmag+= magoffset, px += offset, py += offset){
				for (int x=0; x < width; x++, pmag++, px++, py++){
					float fx = *px, fy = *py;
					*pmag = log(1 + sqrt(fx * fx + fy * fy));
					if (*pmag > max) max = *pmag;
				}
			}
		}
		else {
			for (int y=0; y < height; y++, pmag+= magoffset, px += offset, py += offset){
				for (int x=0; x < width; x++, pmag++, px++, py++){
					float fx = *px, fy = *py;
					*pmag = sqrt(fx * fx + fy * fy);
					if (*pmag > max) max = *pmag;
				}
			}
		}
		scaleFactor = 255.0/max;
		convertScaleAbs(tmp,mag,scaleFactor);
	}
	else if (mode == CALCMAG32F){
		pmag = (float *)(tmp.data);
		magoffset = tmp.step/sizeof(float) - tmp.cols;
		if (logScale){
			for (int y=0; y < height; y++, pmag+= magoffset, px += offset, py += offset){
				for (int x=0; x < width; x++, pmag++, px++, py++){
					float fx = *px, fy = *py;
					*pmag = log(1 + sqrt(fx * fx + fy * fy));
				}
			}
		}
		else {
			for (int y=0; y < height; y++, pmag+= magoffset, px += offset, py += offset){
				for (int x=0; x < width; x++, pmag++, px++, py++){
					float fx = *px, fy = *py;
					*pmag = sqrt(fx * fx + fy * fy);
				}
			}
		}
	}
	else if (mode == CALCORIENT8U){
		porient8u = (uchar *)(orient.data);
		orientoffset = orient.step - orient.cols;
		for (int y=0; y < height; y++, porient8u += orientoffset, px += offset, py += offset){
			for (int x=0; x < width; x++, porient8u++, px++, py++){
				float fx = *px, fy = *py;
				*porient8u = (fy == 0 && fx == 0) ? 0 : saturate_cast<uchar>(fastAtan2(fy,fx) / 2);
			}
		}
	}
	else if (mode == CALCORIENT32U){
		porient32f = (float *)(orient.data);
		orientoffset = orient.step/sizeof(float) - orient.cols;
		for (int y=0; y < height; y++, porient32f += orientoffset, px += offset, py += offset){
			for (int x=0; x < width; x++, porient32f++, px++, py++){
				float fx = *px, fy = *py;
				*porient32f = (fy == 0 && fx == 0) ? 0 : fastAtan2(fy,fx);
			}
		}
	}
	else if (mode == CALCMAG8UORIENT8U){
		pmag = (float *)(tmp.data);
		magoffset = tmp.step/sizeof(float) - tmp.cols;
		porient8u = (uchar *)(orient.data);
		orientoffset = orient.step - orient.cols;
		max = 0;
		if (logScale){
			for (int y=0; y < height; y++, pmag+= magoffset, porient8u += orientoffset, px += offset, py += offset){
				for (int x=0; x < width; x++, pmag++, porient8u++, px++, py++){
					float fx = *px, fy = *py;
					*pmag = log(1 + sqrt(fx * fx + fy * fy));
					*porient8u = (fy == 0 && fx == 0) ? 0 : saturate_cast<uchar>(fastAtan2(fy,fx) / 2); //(cvRound(atan2f(fy,fx)*90 / M_PI) + 180)% 180;
					if (*pmag > max) max = *pmag;
				}
			}
		}
		else {
			for (int y=0; y < height; y++, pmag+= magoffset, porient8u += orientoffset, px += offset, py += offset){
				for (int x=0; x < width; x++, pmag++, porient8u++, px++, py++){
					float fx = *px, fy = *py;
					*pmag = cvSqrt(fx * fx + fy * fy);
					*porient8u = (fy == 0 && fx == 0) ? 0 : saturate_cast<uchar>(fastAtan2(fy,fx) / 2); //(cvRound(atan2f(fy,fx)*90 / M_PI) + 180)% 180;
					if (*pmag > max) max = *pmag;
				}
			}
		}
		scaleFactor = 255.0/max;
		convertScaleAbs(tmp,mag,scaleFactor);
	}
	else if (mode == CALCMAG32FORIENT8U){
		pmag = (float *)(tmp.data);
		magoffset = tmp.step/sizeof(float) - tmp.cols;
		porient8u = (uchar *)(orient.data);
		orientoffset = orient.step - orient.cols;
		if (logScale){
			for (int y=0; y < height; y++, pmag+= magoffset, porient8u += orientoffset, px += offset, py += offset){
				for (int x=0; x < width; x++, pmag++, porient8u++, px++, py++){
					float fx = *px, fy = *py;
					*pmag = log(1 + sqrt(fx * fx + fy * fy));
					*porient8u = (fy == 0 && fx == 0) ? 0 : saturate_cast<uchar>(fastAtan2(fy,fx) / 2); //cvRound((M_PI + atan2f(fy,fx))*127.5 / M_PI);
				}
			}
		}
		else {
			for (int y=0; y < height; y++, pmag+= magoffset, porient8u += orientoffset, px += offset, py += offset){
				for (int x=0; x < width; x++, pmag++, porient8u++, px++, py++){
					float fx = *px, fy = *py;
					*pmag = sqrt(fx * fx + fy * fy);
					*porient8u = (fy == 0 && fx == 0) ? 0 : saturate_cast<uchar>(fastAtan2(fy,fx) / 2); //cvRound((M_PI + atan2f(fy,fx))*127.5 / M_PI);
				}
			}
		}
	}
	else if (mode == CALCMAG8UORIENT32F){
		pmag = (float *)(tmp.data);
		magoffset = tmp.step/sizeof(float) - tmp.cols;
		porient32f = (float *)(orient.data);
		orientoffset = orient.step/sizeof(float) - orient.cols;
		max = 0;
		if (logScale){
			for (int y=0; y < height; y++, pmag+= magoffset, porient32f += orientoffset, px += offset, py += offset){
				for (int x=0; x < width; x++, pmag++, porient32f++, px++, py++){
					float fx = *px, fy = *py;
					*pmag = log(1 + sqrt(fx * fx + fy * fy));
					*porient32f = (fy == 0 && fx == 0) ? 0 : fastAtan2(fy,fx);
					if (*pmag > max) max = *pmag;
				}
			}
		}
		else {
			for (int y=0; y < height; y++, pmag+= magoffset, porient32f += orientoffset, px += offset, py += offset){
				for (int x=0; x < width; x++, pmag++, porient32f++, px++, py++){
					float fx = *px, fy = *py;
					*pmag = sqrt(fx * fx + fy * fy);
					*porient32f = (fy == 0 && fx == 0) ? 0 : fastAtan2(fy,fx);
					if (*pmag > max) max = *pmag;
				}
			}
		}
		scaleFactor = 255.0/max;
		convertScaleAbs(tmp,mag,scaleFactor);
	}
	else if (mode == CALCMAG32FORIENT32F){
		pmag = (float *)(tmp.data);
		magoffset = tmp.step/sizeof(float) - tmp.cols;
		porient32f = (float *)(orient.data);
		orientoffset = orient.step/sizeof(float) - orient.cols;
		if (logScale){
			for (int y=0; y < height; y++, pmag+= magoffset, porient32f += orientoffset, px += offset, py += offset){
				for (int x=0; x < width; x++, pmag++, porient32f++, px++, py++){
					float fx = *px, fy = *py;
					*pmag = log(1 + sqrt(fx * fx + fy * fy));
					*porient32f = (fy == 0 && fx == 0) ? 0 : fastAtan2(fy,fx);
				}
			}
		}
		else {
			for (int y=0; y < height; y++, pmag+= magoffset, porient32f += orientoffset, px += offset, py += offset){
				for (int x=0; x < width; x++, pmag++, porient32f++, px++, py++){
					float fx = *px, fy = *py;
					*pmag = sqrt(fx * fx + fy * fy);
					*porient32f = (fy == 0 && fx == 0) ? 0 : fastAtan2(fy,fx);
				}
			}
		}
	}
}

/**
 * Generates destination regions map from source image
 *
 * src: 		single channel 8-bit input image
 * dst: 		single channel 32-bit regions map image
 * count: 		outputs number of regions
 */
void regionsmap(const Mat& src, Mat& dst, int& count){
	CV_Assert(src.type() == CV_8UC1);
	CV_Assert(dst.type() == CV_32SC1);
	int width = src.cols;
	int height = src.rows;
	int labelsCount = 0;
	int maxRegions = ((width / 2) + 1) * ((height/2)+1)+1;
	int * map = new int[maxRegions];
	for (int i=0; i< maxRegions; i++) map[i] = i; // identity mapping
	uchar * psrc = src.data;
	int * pdst = (int *)(dst.data);
	int srcoffset = src.step - src.cols;
	int srcline = src.step;
	int dstoffset = dst.step / sizeof(int) - dst.cols;
	int dstline = dst.step / sizeof(int);
	dst.setTo(0);
	// 1) processing first row
	if (*psrc != 0) *pdst = ++labelsCount;
	if (width > 1) {
		psrc++; pdst++;
	}
	for (int x=1; x < width; x++, psrc++, pdst++){
		if (*psrc != 0) {// if pixel is a region pixel, check left neightbor
			if (psrc[-1] != 0){ // label like left neighbor
				*pdst = pdst[-1];
			}
			else { // label new region
				 *pdst = ++labelsCount;
			}
		}
	}
	if (height > 1){
		psrc += srcoffset;
		pdst += dstoffset;
	}
	// 2) for all other rows
	for (int y=1; y < height; y++, psrc+= srcoffset, pdst += dstoffset){
		// first pixel in row only checks upper and upper-right pixels
		if (*psrc != 0){
			if (psrc[-srcline] != 0){ // check upper pixel
				*pdst = pdst[-dstline];
			}
			else if (1 < width && psrc[1-srcline] != 0){ // check upper right
				*pdst = pdst[1-dstline];
			}
			else {
				*pdst = ++labelsCount;
			}
		}
		if (width > 1){
			psrc++;
			pdst++;
		}
		// all other pixels in the row check for left and three upper pixels
		for (int x=1; x < width-1; x++, psrc++, pdst++){
			if (*psrc != 0){
				if (psrc[-1] != 0){// check left neighbor
					*pdst = pdst[-1];
				}
				else if (psrc[-1-srcline] != 0){// label like left upper
					*pdst = pdst[-1-dstline];
				}
				else if (psrc[-srcline] != 0){// check upper
					*pdst = pdst[-dstline];
				}
				if (psrc[1-srcline] != 0){
					if (*pdst == 0){ // label pixel as the above right
						*pdst = pdst[1-dstline];
					}
					else {
						int label1 = *pdst;
						int label2 = pdst[1-dstline];
						if ((label1 != label2) && (map[label1] != map[label2])){
							if (map[label1] == label1){ // map unmapped to already mapped
								map[label1] = map[label2];
							}
							else if (map[label2] == label2){ // map unmapped to already mapped
								map[label2] = map[label1];
							}
							else { // both values are already mapped
								map[map[label1]] = map[label2];
								map[label1] = map[label2];
							}
							// reindexing
							for (int i=1; i <= labelsCount; i++){
								if (map[i] != i){
									int j = map[i];
									while (j != map[j]){
										j = map[j];
									}
									map[i] = j;
								}
							}
						}
					}
				}
				if (*pdst == 0)
				{
					*pdst = ++labelsCount;
				}
			}
		}
		if (*psrc != 0){
			if (psrc[-1] != 0){// check left neighbor
				*pdst = pdst[-1];
			}
			else if (psrc[-1-srcline] != 0){// label like left upper
				*pdst = pdst[-1-dstline];
			}
			else if (psrc[-srcline] != 0){// check upper
				*pdst = pdst[-dstline];
			}
			else
			{
				*pdst = ++labelsCount;
			}
		}
		psrc++;
		pdst++;
	}
	int * remap = new int[maxRegions];
	count = 0;
	for (int i=1; i <= labelsCount; i++){
		if (map[i] == i) {
			remap[i] = ++count;
		}
	}
	remap[0] = 0;
	// complete remapping
	for (int i=1; i <= labelsCount; i++){
		if (map[i] != i) remap[i] = remap[map[i]];
	}
	pdst = (int *) (dst.data);
	for (int y=0; y < height; y++, pdst += dstoffset){
		for (int x=0; x < width; x++, pdst++){
			*pdst = remap[*pdst];
		}
	}
	delete[] remap;
	delete[] map;
}

/**
 * The regsize operator filters out too large or too small binary large objects (regions) in a region map
 *
 * regmap:  regions map (to calculate this, simply compute bm::regionsmap(mask,regmap,count), where count returns # of regions)
 * mask:    output mask with filtered regions
 * count:   number of connected components in regmap
 * minSize: only regions larger or equal than minSize are kept
 * maxSize: only regions smaller or equal than maxSize are kept
 */
void maskRegsize(const Mat& regmap, Mat& mask, const int count, const int minSize = INT_MIN, const int maxSize = INT_MAX){
	CV_Assert(regmap.type() == CV_32SC1);
	CV_Assert(mask.type() == CV_8UC1);
	CV_Assert(mask.size() == regmap.size());
	int * map = new int[count+1];
	// resetting map to now count region size
	for (int i=0; i<count+1; i++) map[i] = 0;
	int width = regmap.cols;
	int height = regmap.rows;
	int * pmap = (int *) (regmap.data);
	int mapoffset = regmap.step / sizeof(int) - regmap.cols;
	for (int y=0; y < height; y++, pmap += mapoffset){
		for (int x=0; x < width; x++, pmap++){
			if (*pmap > 0){
				map[*pmap]++;
			}
		}
	}
	// delete too large and too small regions
	pmap = (int *) (regmap.data);
	uchar * pmask = mask.data;
	int maskoffset = mask.step - mask.cols;
	for (int y=0; y < height; y++, pmap += mapoffset, pmask += maskoffset){
		for (int x=0; x < width; x++, pmap++, pmask++){
			if (*pmap > 0){
				int size = map[*pmap];
				if (size < minSize || size > maxSize) *pmask = 0; else *pmask = 255;
			} else *pmask = 0;
		}
	}
	delete[] map;
}

/**
 * Computes mask for reflections in image
 *
 * src:        input image
 * mask:       output image
 * roiPercent: parameter for the number of highest pixel intensities in percent
 * maxSize:    maximum size of reflection region
 * dilateSize: size of circular structuring element for dilate operation
 * dilateIterations: iterations of dilate operation
 */
void maskReflections(const Mat& src, Mat& mask, const float roiPercent = 20, const int dilateSize = 5, const int dilateIterations = 2, const int maxSize = 1000){
	CV_Assert(src.type() == CV_8UC1);
	CV_Assert(mask.type() == CV_8UC1);
	CV_Assert(mask.size() == src.size());
	MatND hist;
	calcHist(src,hist);
	float light = cvFloor(histQuantile(hist,src.cols*src.rows,(100-roiPercent)*0.01));
	threshold(src,mask,light-1,255,THRESH_BINARY);
	if (dilateSize > 0){
		Mat dilated;
		Mat kernel(dilateSize,dilateSize,CV_8UC1);
		kernel.setTo(Scalar(0));
		circle(kernel,Point(dilateSize/2,dilateSize/2),1,Scalar(255),CV_FILLED);
		dilate(mask,dilated,kernel,Point(-1,-1),dilateIterations);
		mask = dilated;
	}
	Mat regions(mask.rows,mask.cols,CV_32SC1);
	int count = 0;
	regionsmap(mask,regions,count);
	maskRegsize(regions,mask,count,0,maxSize);
}

/**
 * Main eye mask selecting pupillary and limbic boundary pixels
 *
 * src: source CV_8UC1 image
 * mask: destination CV_8UC1 mask
 * reflect: used reflection mask
 */
void maskEye(const Mat& src, Mat& mask, Mat& reflect){
	// initial assertions
	// initial assertions
	CV_Assert(src.type() == CV_8UC1);
	CV_Assert(mask.type() == CV_8UC1);
	CV_Assert(mask.size() == src.size());
	const float roiPercent = 20;
	//const float roiReflections = 10;
	const int apertureSize = 7;
	//const int maxReflectSize = 1000;
	//const int dilateSize = 7;
	//const int dilateIterations = 3;
	int width = mask.cols;
	int height = mask.rows;
	const float avgSigma = width/200;
	const int avgSize = cvCeil(avgSigma * 3) + (1+(cvCeil(avgSigma * 3) % 2)) % 2;
	int cellWidth = width/30;
	int cellHeight = height/30;
	int gridWidth = width / cellWidth + (width % cellWidth == 0 ? 0 : 1);
	int gridHeight = height / cellHeight + (height % cellHeight == 0 ? 0 : 1);
	//Mat msk(src.rows,src.cols,CV_8UC1);
	//maskReflections(src, msk, roiReflections, dilateSize, dilateIterations, maxReflectSize);
	Mat gradX(src.rows,src.cols,CV_32FC1);
	Mat gradY(src.rows,src.cols,CV_32FC1);
	Sobel(src,gradX,gradX.depth(),1,0,apertureSize);
	Sobel(src,gradY,gradY.depth(),0,1,apertureSize);
	GaussianBlur(gradX,gradX,Size(avgSize,avgSize),avgSigma);
	GaussianBlur(gradY,gradY,Size(avgSize,avgSize),avgSigma);
	Mat mag(mask.rows,mask.cols,CV_32FC1);
	Mat empty;
	float scaleFactor;
	vec2polar(gradX,gradY,mag,empty,scaleFactor);
	uchar * pmask = reflect.data;
	float * pmag = (float *)(mag.data);
	int magoffset = mag.step/sizeof(float) - width;
	int maskoffset = reflect.step - width;
	float max = 0;
	for (int y=0; y < height; y++, pmag+= magoffset, pmask += maskoffset){
		for (int x=0; x < width; x++, pmag++, pmask++){
			if (*pmask != 0){
				*pmag = 0;
			}
			if (*pmag > max) max = *pmag;
		}
	}
	MatND hist;
	calcHist(mag,hist,0,1000,0,max+.000001);
	float minval = histQuantile(hist,width*height,(100-roiPercent)*0.01,1000,0,max+.000001);
	pmag = (float *)(mag.data);
	for (int y=0; y < height; y++, pmag += magoffset){
		for (int x=0; x < width; x++, pmag++){
			if (*pmag < minval) *pmag = 0;
		}
	}
	for (int y = 0; y < gridHeight; y++) {
		for (int x = 0; x < gridWidth; x++) {
			int cX = x*cellWidth;
			int cY = y*cellHeight;
			int cWidth = min(cellWidth, width - x*cellWidth);
			int cHeight = min(cellHeight, height - y*cellHeight);
			float * pgradX = ((float *) (gradX.data)) + (gradX.step/sizeof(float))*cY + cX;
			float * pgradY = ((float *) (gradY.data)) + (gradY.step/sizeof(float))*cY + cX;
			pmag = ((float *) (mag.data)) + (mag.step/sizeof(float))*cY + cX;
			int gradXCellOffset = gradX.step/ sizeof(float) - cWidth;
			int gradYCellOffset = gradY.step/ sizeof(float) - cWidth;
			magoffset = mag.step/sizeof(float) - cWidth;
			double sumX = 0;
			double sumY = 0;
			double sumMag = 0;
			for (int b=0; b < cHeight; b++, pmag += magoffset, pgradX += gradXCellOffset, pgradY+= gradYCellOffset){
			  for (int a=0; a < cWidth; a++, pmag++, pgradX++, pgradY++){
				  if (*pmag > 0) {
					  sumX += *pgradX;
					  sumY += *pgradY;
					  sumMag += *pmag;
				  }
			  }
			}
			if (sumMag > 0) {
				sumX /= sumMag;
				sumY /= sumMag;
			}
			pmag = ((float *) (mag.data)) + (mag.step/sizeof(float))*cY + cX;
			bool is_significant = ((sumX * sumX + sumY * sumY) > 0.5);
			pmask = mask.data + (mask.step)*cY + cX;
			maskoffset = mask.step - cWidth;
			for (int b=0; b < cHeight; b++, pmag += magoffset, pmask += maskoffset){
			  for (int a=0; a < cWidth; a++, pmag++, pmask++){
				  *pmask = (*pmag > 0 && is_significant) ? 255 : 0;
			  }
			}
		}
	}
}

/** ------------------------------- Center detection ------------------------------- **/

/**
 * Type for a bi-directional ray ith originating point and direction
 * x: x-coordinate of origin
 * y: y-coordinate of origin
 * alpha: direction in degrees
 */
struct BidRay{
	float x;
	float y;
	float fx;
	float fy;
	float mag;
	BidRay(float _x, float _y, float _fx, float _fy, float _mag){
		x = _x;
		y = _y;
		fx = _fx;
		fy = _fy;
		mag = _mag;
	}
};

/**
 * calculates determinant of vectors (x1, y1) and (x2, y2)
 *
 * x1: first vector's x-coordinate
 * y1: first vector's x-coordinate
 * x2: second vector's y-coordinate
 * y2: second vector's y-coordinate
 */
inline float det(const float &x1, const float &y1, const float &x2, const float &y2) {
	return x1*y2 - y1*x2;
}

/**
 * intersects two lines (x1, y1) + s*(fx1, fy1) and (x2, y2) + t*(fx2, fy2)
 *
 * x1: x-coordinate of point on line 1
 * y1: y-coordinate of point on line 1
 * fx1: direction-vector x-coordinate of line 1
 * fy1: direction-vector y-coordinate of line 1
 * x2: x-coordinate of point on line 2
 * y2: y-coordinate of point on line 2
 * fx2: direction-vector x-coordinate of line 2
 * fy2: direction-vector y-coordinate of line 2
 * sx: intersection point x-coordinate
 * sy: intersection point y-coordinate
 */
int intersect(const float &x1, const float &y1, const float &fx1, const float &fy1, const float &x2, const float &y2, const float &fx2, const float &fy2, float &sx, float &sy){
	if (det(fx1,fy1,fx2,fy2) == 0){
		if (det(fx1,fy1,x2-x1,y2-y1) == 0){
			sx = x1;
			sy = y1;
			return -1; // equal
		}
		sx = NAN;
		sy = NAN;
		return 0; // parallel
	}
	float Ds = det(x2-x1,y2-y1,-fx2,-fy2);
	float D = det(fx1,fy1,-fx2,-fy2);
	float s = Ds / D;
	sx = x1 + s*fx1;
	sy = y1 + s*fy1;
	return 1;
}

/**
 * intersects a line (x1, y1) + s*(fx1, fy1) with an axis parallel to the x-axis
 *
 * x1: x-coordinate of point on line 1
 * y1: y-coordinate of point on line 1
 * fx1: direction-vector x-coordinate of line 1
 * fy1: direction-vector y-coordinate of line 1
 * y2: y-coordinate of point on axis parallel to x-axis
 * sx: intersection point x-coordinate (sy is always equal y2)
 */
int intersectX(const float &x1, const float &y1, const float &fx1, const float &fy1, const float &y2, float &sx){
	if (fy1 == 0){
		if (y2-y1 == 0){
			sx = x1;
			return -1; // equal
		}
		sx = NAN;
		return 0; // parallel
	}
	sx = x1 + ((y2-y1)*fx1 / fy1);
	return 1;
}

/**
 * intersects a line (x1, y1) + s*(fx1, fy1) with an axis parallel to the y-axis
 *
 * x1: x-coordinate of point on line 1
 * y1: y-coordinate of point on line 1
 * fx1: direction-vector x-coordinate of line 1
 * fy1: direction-vector y-coordinate of line 1
 * x2: x-coordinate of point on axis parallel to y-axis
 * sy: intersection point x-coordinate (sx is always equal x2)
 */
int intersectY(const float &x1, const float &y1, const float &fx1, const float &fy1, const float &x2, float &sy){
	if (fx1 == 0){
		if (x2-x1 == 0){
			sy = y1;
			return -1; // equal
		}
		sy = NAN;
		return 0; // parallel
	}
	sy = y1 + ((x2-x1)*fy1 / fx1);
	return 1;
}

/**
 * intersects a line (x, y) + s*(fx, fy) with an axis parallel rectangle
 *
 * x: x-coordinate of point on line
 * y: y-coordinate of point on line
 * fx: direction-vector x-coordinate of line
 * fy: direction-vector y-coordinate of line
 * left: left coordinate of rectangle
 * top: top coordinate of rectangle
 * width: width of rectangle
 * height: height of rectangle
 * px: first intersection point x-coordinate
 * py: first intersection point y-coordinate
 * qx: first intersection point x-coordinate
 * qy: first intersection point y-coordinate
 */
int intersectRect(const float &x, const float &y, const float &fx, const float &fy, const float &left, const float &top, const float &right, const float &bottom, float &px, float &py, float &qx, float &qy){
	float leftY, bottomX, rightY, topX;
	int lefti = intersectY(x,y,fx,fy,left,leftY);
	bool leftHit = (lefti != 0) && leftY >= top && leftY <= bottom;
	int topi = intersectX(x,y,fx,fy,top,topX);
	bool topHit = (topi != 0) && topX >= left && topX <= right;
	int righti = intersectY(x,y,fx,fy,right,rightY);
	bool rightHit = (righti != 0) && rightY >= top && rightY <= bottom;
	int bottomi = intersectX(x,y,fx,fy,bottom,bottomX);
	bool bottomHit = (bottomi != 0) && bottomX >= left && bottomX <= right;
	if (leftHit){
		if (bottomHit){
			if (rightHit){
				if (topHit){
					// left, bottom, right, top
					px = left;
					py = leftY;
					qx = right;
					qy = rightY;
					return -1;
				}
				else {
					// left, bottom, right
					px = left;
					py = leftY;
					qx = right;
					qy = rightY;
					return -1;
				}
			}
			else {
				if (topHit){
					// left, bottom, top
					px = topX;
					py = top;
					qx = bottomX;
					qy = bottom;
					return -1;
				}
				else {
					// left, bottom
					px = left;
					py = leftY;
					qx = bottomX;
					qy = bottom;
					return 1;
				}
			}
		}
		else {
			if (rightHit){
				if (topHit){
					// left, right, top
					px = left;
					py = leftY;
					qx = right;
					qy = rightY;
					return -1;
				}
				else {
					// left, right
					px = left;
					py = leftY;
					qx = right;
					qy = rightY;
					return 1;
				}
			}
			else {
				if (topHit){
					// left, top
					px = left;
					py = leftY;
					qx = topX;
					qy = top;
					return 1;
				}
			}
		}
	}
	else {
		if (bottomHit){
			if (rightHit){
				if (topHit){
					// bottom, right, top
					px = topX;
					py = top;
					qx = bottomX;
					qy = bottom;
					return -1;
				}
				else {
					// bottom, right
					px = bottomX;
					py = bottom;
					qx = right;
					qy = rightY;
					return 1;
				}
			}
			else {
				if (topHit){
					// bottom, top
					px = topX;
					py = top;
					qx = bottomX;
					qy = bottom;
					return 1;
				}
			}
		}
		else {
			if (rightHit){
				if (topHit){
					// right, top
					px = topX;
					py = top;
					qx = right;
					qy = rightY;
					return 1;
				}
			}
		}
	}
	px = NAN;
	py = NAN;
	qx = NAN;
	qy = NAN;
	return 0;
}

/**
 * Draws a line onto accumulator matrix using Bresenham's algorithm
 * Increases a rectangular accumulator by adding a given value to all points on a line
 * line:    line to be drawn
 * accu:    floating point canvas (accumulator)
 * border:  outer accu boundary rectangle in user space coordinates
 * value:   value to add to accu on points on the line
 *
 * returns true, if values are added to the accu
 */
bool drawLine(const BidRay& line, Mat_<float>& accu, const Rect_<float>& border){
	// intersect line with border
	float cellWidth = border.width/accu.cols;
	float cellHeight = border.height/accu.rows;
	float rx = border.x+cellWidth/2, ry = border.y+cellHeight/2;
	float rwidth = border.x+border.width-cellWidth, rheight = border.y+border.height-cellHeight;
	float px, py, qx, qy;
	float incValue = line.mag / 1000;
	int accuLine = (accu.step/sizeof(float));

	int res = intersectRect(line.x,line.y,line.fx,line.fy,rx,ry,rwidth,rheight,px,py,qx,qy);
	if (res != 0){
	  int x1 = min(max(cvRound((px-rx)/cellWidth+0.5),0),accu.cols-1);
	  int y1 = min(max(cvRound((py-ry)/cellHeight+0.5),0),accu.rows-1);
	  int x2 = min(max(cvRound((qx-rx)/cellWidth+0.5),0),accu.cols-1);
	  int y2 = min(max(cvRound((qy-ry)/cellHeight+0.5),0),accu.rows-1);
	  // line intersects with border, so draw line onto accu
	  float * p = (float *) (accu.data);
	  int t, dx, dy, incx, incy, pdx, pdy, ddx, ddy, es, el, err;
	  dx = x2 - x1;
	  dy = y2 - y1;
	  incx = (dx > 0) ? 1 : (dx < 0) ? -1 : 0;
	  incy = (dy > 0) ? accuLine : (dy < 0) ? -accuLine : 0;
	  if(dx<0) dx = -dx;
	  if(dy<0) dy = -dy;
	  if (dx>dy) {
		pdx=incx; // parallel step
		pdy=0;
		ddx=incx; // diagonal step
		ddy=incy;
		es=dy; // error step
		el=dx;
	  } else {
		pdx=0; // parallel step
		pdy=incy;
		ddx=incx; // diagonal step
		ddy=incy;
		es=dx; // error step
		el=dy;
	  }
	  p += x1 + y1*accuLine;
	  err = el/2;
	  // setPixel
	  *p += incValue;
	  // Calculate pixel
	  for(t=0; t<el; ++t) {// t counts Pixels, el is also count
		// update error
		err -= es;
		if(err<0) {
		  // make error term positive
		  err += el;
		  // step towards slower direction
		  p += ddx + ddy;
		}
		else  {
		  // step towards faster direction
		  p += pdx + pdy;
		}
		*p += incValue;

	  }
	  return true;

	}
	return false;
}

/*
 * Calculates circle center in source image.
 *
 * gradX:			CV_32FC1 image, gradient in x direction
 * gradY:			CV_32FC1 image, gradient in y direction
 * mask:			CV_8UC1 mask image to exclude wrong points for gradient extraction
 * center:		    center point of main circle in source image
 * accuPrecision:   stop condition for accuracy of center
 * accuSize:	    size of the accumulator array
 */
void eyeCenter(const Mat& gradX, const Mat& gradY, const Mat& mask, Point& center, const float accuPrecision = .5, const int accuSize = 10){
	// initial assertions
	CV_Assert(gradX.type() == CV_32FC1);
	CV_Assert(gradY.type() == CV_32FC1);
	CV_Assert(mask.type() == CV_8UC1);
	CV_Assert(gradX.size() == mask.size());
	CV_Assert(gradY.size() == mask.size());
	// initial declarations
	int width = mask.cols;
	int height = mask.rows;
	int accuScaledSize = (accuSize+1)/2;
	Rect_<float> accuRect(0,0,width,height);
	Mat_<float> accu(accuSize,accuSize);
	Mat_<float> accuScaled(accuScaledSize,accuScaledSize);
	// create candidates list
	std::list<BidRay> candidates;
	float * px = (float *)(gradX.data);
	float * py = (float *)(gradY.data);
	uchar * pmask = (uchar *)(mask.data);
	int xoffset = gradX.step/sizeof(float) - width;
	int yoffset = gradY.step/sizeof(float) - width;
	int maskoffset = mask.step - width;

	for (int y=0; y < height; y++, px += xoffset, py += yoffset,pmask += maskoffset){
		for (int x=0; x < width; x++, px++, py++, pmask++){
			if (*pmask > 0){
				float fx = *px, fy = *py;
				candidates.push_back(BidRay(x,y,fx,fy,sqrt(fx*fx+fy*fy)));
			}
		}
	}
	while (accuRect.width > accuPrecision || accuRect.height > accuPrecision){
		accu.setTo(Scalar(0));
		bool isIn = true;
		if (candidates.size() > 0){
			for (std::list<BidRay>::iterator it = candidates.begin(); it != candidates.end();(isIn) ? ++it : it = candidates.erase(it)){
				isIn = drawLine(*it,accu,accuRect);
			}
		}
		pyrDown(accu,accuScaled);
		float * p = (float *) (accuScaled.data);
		float maxCellValue = 0;
		int maxCellX = accuScaled.cols / 2;
		int maxCellY = accuScaled.rows / 2;
		int accuOffset = accuScaled.step / sizeof(float) - accuScaled.cols;
		for (int y=0; y < accuScaledSize; y++, p += accuOffset){
			for (int x=0; x < accuScaledSize; x++, p++){
				if (*p > maxCellValue){
					maxCellX = x;
					maxCellY = y;
					maxCellValue = *p;
				}
			}
		}
		accuRect.x += ((maxCellX * accuRect.width + 0.5) / accuScaledSize) - (accuRect.width * 0.25);
		accuRect.y += ((maxCellY * accuRect.height + 0.5) / accuScaledSize) - (accuRect.height * 0.25);
		accuRect.width /= 2;
		accuRect.height /= 2;
	}
	center.x = cvRound(accuRect.x + accuRect.width / 2);
	center.y = cvRound(accuRect.y + accuRect.height / 2);
}

/** ------------------------------- Rubbersheet transform ------------------------------- **/


/*
 * Calculates the mapped (polar) image of source using transformation center (polar origin) and radius.
 *
 * src:       (cartesian) source image (possibly multi-channel)
 * dst:		  (polar) destination image (possibly multi-channel)
 * centerX:   x-coordinate of polar origin in (floating point) pixels of the source image
 * centerY:   y-coordinate of polar origin in (floating point) pixels of the source image
 * radius:    radius in pixels of the source image (to map whole image, this should be the maximum of distances between origin and corners)
 * interpolation: interpolation mode
 */
void rubbersheet(const Mat& src, Mat& dst, const Mat& inner, const Mat& outer, const int interpolation = INTER_LINEAR) {
	CV_Assert(src.channels() == dst.channels());
	CV_Assert(src.type() == dst.type());
	CV_Assert(src.depth() == CV_8U);
	CV_Assert(2 * dst.cols == inner.cols);
	CV_Assert(outer.cols == inner.cols);
	CV_Assert(interpolation == INTER_NEAREST || interpolation == INTER_LINEAR);
	int nChannels = src.channels();
	// image processing helpers
	int dstheight = dst.rows;
	int dstwidth = dst.cols;
	int srcheight = src.rows;
	int srcwidth = src.cols;
	uchar * pdst = dst.data;
	int dstoffset = dst.step - dstwidth * nChannels;
	int srcstep = src.step;
	float roffset = 1.f / dstheight;
	float r = 0;
	if (interpolation == INTER_NEAREST){
		for (int y=0; y < dstheight; y++, pdst+= dstoffset, r+=roffset){
			float * pinner = (float *) inner.data;
			float * pouter = (float *) outer.data;
			for (int x=0; x < dstwidth; x++, pinner++, pouter++){
				float a = 0.5f + *pinner + r * (*pouter - *pinner);
				pinner++; pouter++;
				float b =  0.5f + *pinner + r * (*pouter - *pinner);
				int coordX = cvRound(a);
				int coordY = cvRound(b);
				if (coordX < 0 || coordY < 0 || coordX >= srcwidth || coordY >= srcheight){
					for (int i=0; i< nChannels; i++,pdst++){
						*pdst = 0;
					}
				}
				else {
					uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
					for (int i=0; i< nChannels; i++,pdst++,psrc++){
						*pdst = *psrc;
					}
				}
			}
		}
	}
	else {
		for (int y=0; y < dstheight; y++, pdst+= dstoffset, r+= roffset){
			float * pinner = (float *) inner.data;
			float * pouter = (float *) outer.data;
			for (int x=0; x < dstwidth; x++, pinner++, pouter++){
				float a = 0.5f + *pinner + r * (*pouter - *pinner);
				pinner++; pouter++;
				float b =  0.5f + *pinner + r * (*pouter - *pinner);
				int coordX = cvFloor(a);
				int coordY = cvFloor(b);
				if (coordX >= 0){
					if (coordY >= 0){
						if (coordX < srcwidth-1){
							if (coordY < srcheight-1){
								float dx = a-coordX;
								float dy = b-coordY;
								uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
								for (int i=0; i< nChannels; i++,pdst++,psrc++){
									*pdst = saturate_cast<uchar>((1-dy)*((1-dx)*((float)(*psrc)) + dx*(float)(psrc[nChannels]))+ dy*((1-dx)*((float)(psrc[srcstep])) + dx*(float)(psrc[nChannels+srcstep])));
								}
							}
							else if (coordY == srcheight-1){ // unten raus
								float dx = a-coordX;
								float dy = b-coordY;
								uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
								for (int i=0; i< nChannels; i++,pdst++,psrc++){
									*pdst = saturate_cast<uchar>((1-dy)*((1-dx)*((float)(*psrc)) + dx*(float)(psrc[nChannels])));
								}
							}
							else {
								for (int i=0; i< nChannels; i++,pdst++){
									*pdst = 0;
								}
							}
						}
						else if (coordX == srcwidth-1){
							if (coordY < srcheight-1){// rechts raus
								float dx = a-coordX;
								float dy = b-coordY;
								uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
								for (int i=0; i< nChannels; i++,pdst++,psrc++){
									*pdst = saturate_cast<uchar>((1-dx)*((1-dy)*((float)(*psrc))+ dy*((float)(psrc[srcstep]))));
								}
							}
							else if (coordY == srcheight-1){ // unten rechts raus
								float dx = a-coordX;
								float dy = b-coordY;
								uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
								for (int i=0; i< nChannels; i++,pdst++,psrc++){
									*pdst = saturate_cast<uchar>((1-dy)*(1-dx)*((float)(*psrc)));
								}
							}
							else {
								for (int i=0; i< nChannels; i++,pdst++){
									*pdst = 0;
								}
							}
						}
						else {
							for (int i=0; i< nChannels; i++,pdst++){
								*pdst = 0;
							}
						}
					}
					else if (coordY == -1){
						if (coordX < srcwidth-1){// oben raus
								float dx = a-coordX;
								float dy = b-coordY;
								uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
								for (int i=0; i< nChannels; i++,pdst++,psrc++){
									*pdst = saturate_cast<uchar>((dy)*((1-dx)*((float)(psrc[srcstep])) + dx*(float)(psrc[nChannels+srcstep])));
								}
						}
						else if (coordX == srcwidth-1){// oben rechts raus
								float dx = a-coordX;
								float dy = b-coordY;
								uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
								for (int i=0; i< nChannels; i++,pdst++,psrc++){
									*pdst = saturate_cast<uchar>((dy)*(1-dx)*((float)(psrc[srcstep])));
								}
						}
						else {
							for (int i=0; i< nChannels; i++,pdst++){
								*pdst = 0;
							}
						}
					}
					else {
						for (int i=0; i< nChannels; i++,pdst++){
							*pdst = 0;
						}
					}
				}
				else if (coordX == -1){
					if (coordY >= 0){
						if (coordY < srcheight-1){// links raus
							float dx = a-coordX;
							float dy = b-coordY;
							uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
							for (int i=0; i< nChannels; i++,pdst++,psrc++){
								*pdst = saturate_cast<uchar>(dx*((1-dy)*(float)(psrc[nChannels]) + dy*(float)(psrc[nChannels+srcstep])));
							}
						}
						else if (coordY == srcheight-1){ // links unten raus
							float dx = a-coordX;
							float dy = b-coordY;
							uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
							for (int i=0; i< nChannels; i++,pdst++,psrc++){
								*pdst = saturate_cast<uchar>((1-dy)*(dx)*((float)(psrc[nChannels])));
							}
						}
						else {
							for (int i=0; i< nChannels; i++,pdst++){
								*pdst = 0;
							}
						}
					}
					else if (coordY == -1){ // links oben raus
						float dx = a-coordX;
						float dy = b-coordY;
						uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
						for (int i=0; i< nChannels; i++,pdst++,psrc++){
							*pdst = saturate_cast<uchar>((dy)*(dx)*((float)(psrc[nChannels+srcstep])));
						}
					}
					else {
						for (int i=0; i< nChannels; i++,pdst++){
							*pdst = 0;
						}
					}
				}
				else {
					for (int i=0; i< nChannels; i++,pdst++){
						*pdst = 0;
					}
				}
			}
		}
	}
}

/** ------------------------------- Boundary detection ------------------------------- **/

/*
 * Calculates the mapped (polar) image of source using transformation center (polar origin) and radius.
 *
 * src:       (cartesian) source image (possibly multi-channel)
 * dst:		  (polar) destination image (possibly multi-channel)
 * centerX:   x-coordinate of polar origin in (floating point) pixels of the source image
 * centerY:   y-coordinate of polar origin in (floating point) pixels of the source image
 * radius:    radius in pixels of the source image (to map whole image, this should be the maximum of distances between origin and corners)
 * interpolation: interpolation mode
 */
void cart2polar(const Mat& src, Mat& dst, const float centerX = 0, const float centerY = 0, const float radius = -1, const int interpolation = INTER_LINEAR) {
	CV_Assert(src.type() == CV_8UC1);
	CV_Assert(src.type() == dst.type());
	CV_Assert(interpolation == INTER_NEAREST || interpolation == INTER_LINEAR);
	// image processing helpers
	int dstheight = dst.rows;
	int dstwidth = dst.cols;
	int srcheight = src.rows;
	int srcwidth = src.cols;
	float rad = radius;
	if (rad < 0){
		float dist1 = centerX*centerX;
		float dist2 = centerY*centerY;
		float dist3 = (srcwidth-centerX)*(srcwidth-centerX);
		float dist4 = (srcheight-centerY)*(srcheight-centerY);
		rad = max(max(sqrt(dist1+dist2),sqrt(dist1+dist4)),max(sqrt(dist3+dist2),sqrt(dist3+dist4)));
	}
	uchar * pdst = dst.data;
	uchar * psrc = src.data;
	int dstoffset = dst.step - dstwidth;
	int srcstep = src.step;
	float roffset = rad/dstheight;
	float r = roffset/2;
	float thetaoffset = 2.f * M_PI / dstwidth;
	float xoffset = centerX - 0.5f;
	float yoffset = centerY - 0.5f;
	if (interpolation == INTER_NEAREST){
		for (int y=0; y < dstheight; y++, pdst+= dstoffset, r+= roffset){
			float theta = thetaoffset/2;
			for (int x=0; x < dstwidth; x++, theta += thetaoffset,pdst++){
				float a = xoffset + r * cos(theta);//std::cos(theta);
				float b = yoffset + r * sin(theta);//std::sin(theta);
				int coordX = cvRound(a);
				int coordY = cvRound(b);
				if (coordX < 0 || coordY < 0 || coordX >= srcwidth || coordY >= srcheight){
					*pdst = 0;
				}
				else {
					*pdst = psrc[coordY*srcstep+coordX];
				}
			}
		}
	}
	else {
		for (int y=0; y < dstheight; y++, pdst+= dstoffset, r+= roffset){
			float theta = thetaoffset/2;
			for (int x=0; x < dstwidth; x++, theta += thetaoffset,pdst++){
				float a = xoffset + r * cos(theta);//std::cos(theta);
				float b = yoffset + r * sin(theta);//std::sin(theta);
				int coordX = cvFloor(a);
				int coordY = cvFloor(b);
				if (coordX >= 0){
					if (coordY >= 0){
						if (coordX < srcwidth-1){
							if (coordY < srcheight-1){
								float dx = a-coordX;
								float dy = b-coordY;
								int offset = coordY*srcstep+coordX;
								*pdst = saturate_cast<uchar>((1-dy)*((1-dx)*((float)(psrc[offset])) + dx*(float)(psrc[offset+1]))+ dy*((1-dx)*((float)(psrc[offset+srcstep])) + dx*(float)(psrc[offset+1+srcstep])));
							}
							else if (coordY == srcheight-1){ // unten raus
								float dx = a-coordX;
								float dy = b-coordY;
								int offset = coordY*srcstep+coordX;
								*pdst = saturate_cast<uchar>((1-dy)*((1-dx)*((float)(psrc[offset])) + dx*(float)(psrc[offset+1])));
							}
							else {
								*pdst = 0;
							}
						}
						else if (coordX == srcwidth-1){
							if (coordY < srcheight-1){// rechts raus
								float dx = a-coordX;
								float dy = b-coordY;
								int offset = coordY*srcstep+coordX;
								*pdst = saturate_cast<uchar>((1-dx)*((1-dy)*((float)(psrc[offset]))+ dy*((float)(psrc[offset+srcstep]))));
							}
							else if (coordY == srcheight-1){ // unten rechts raus
								float dx = a-coordX;
								float dy = b-coordY;
								int offset = coordY*srcstep+coordX;
								*pdst = saturate_cast<uchar>((1-dy)*(1-dx)*((float)(psrc[offset])));
							}
							else {
								*pdst = 0;
							}
						}
						else {
							*pdst = 0;
						}
					}
					else if (coordY == -1){
						if (coordX < srcwidth-1){// oben raus
							float dx = a-coordX;
							float dy = b-coordY;
							int offset = coordY*srcstep+coordX;
							*pdst = saturate_cast<uchar>((dy)*((1-dx)*((float)(psrc[offset+srcstep])) + dx*(float)(psrc[offset+1+srcstep])));

						}
						else if (coordX == srcwidth-1){// oben rechts raus
							float dx = a-coordX;
							float dy = b-coordY;
							int offset = coordY*srcstep+coordX;
							*pdst = saturate_cast<uchar>((dy)*(1-dx)*((float)(psrc[offset+srcstep])));
						}
						else {
							*pdst = 0;
						}
					}
					else {
						*pdst = 0;
					}
				}
				else if (coordX == -1){
					if (coordY >= 0){
						if (coordY < srcheight-1){// links raus
							float dx = a-coordX;
							float dy = b-coordY;
							int offset = coordY*srcstep+coordX;
							*pdst = saturate_cast<uchar>(dx*((1-dy)*(float)(psrc[offset+1]) + dy*(float)(psrc[offset+1+srcstep])));
						}
						else if (coordY == srcheight-1){ // links unten raus
							float dx = a-coordX;
							float dy = b-coordY;
							int offset = coordY*srcstep+coordX;
							*pdst = saturate_cast<uchar>((1-dy)*(dx)*((float)(psrc[offset+1])));
						}
						else {
							*pdst = 0;
						}
					}
					else if (coordY == -1){ // links oben raus
						float dx = a-coordX;
						float dy = b-coordY;
						int offset = coordY*srcstep+coordX;
						*pdst = saturate_cast<uchar>((dy)*(dx)*((float)(psrc[offset+1+srcstep])));
					}
					else {
						*pdst = 0;
					}
				}
				else {
					*pdst = 0;
				}
			}
		}
	}
}

/**
 * Returns a gabor 2D kernel
 * kernel: output CV_32FC1 image of specific size
 * lambda: wavelength of cosine factor
 * theta: orientation of normal to parallel stripes
 * psi: phase offset
 * sigma: gaussian sigma parameter
 * gamma: spatial aspect ratio
 */
void gaborKernel(Mat& kernel,float lambda,float theta,float psi,float sigma,float gamma){
	CV_Assert(kernel.type() == CV_32FC1);
	CV_Assert(kernel.cols%2==1 && kernel.rows%2==1);
	float * p = (float *)kernel.data;
	int width = kernel.cols;
	int height = kernel.rows;
	int offset = kernel.step/sizeof(float) - width;
	int rx = width/2;
	int ry = height/2;
	float sigma_x = sigma;
	float sigma_y = sigma/gamma;
	for (int y=-ry,i=0;i<height;y++,i++,p+=offset){
		for (int x=-rx,j=0;j<width;x++,j++,p++){
			float x_theta = x*std::cos(theta)+y*std::sin(theta);
			float y_theta =-x*std::sin(theta)+y*std::cos(theta);
			*p = 1 / (2*M_PI*sigma_x*sigma_y) * std::exp(-.5f*((x_theta*x_theta)/(sigma_x*sigma_x) + (y_theta*y_theta)/(sigma_y*sigma_y)))*std::cos(2*M_PI/lambda*x_theta+psi);
		}
	}
}

/**
 * Normalizes a kernel, such that the area under the kernel is 1
 * kernel: output CV_32FC1 image of specific size
 */
void kernelNormArea(Mat& kernel){
	double sum = 0;
	MatIterator_<float> it;
	for (it = kernel.begin<float>(); it < kernel.end<float>(); it++){
		sum += *it;
	}
	if (sum != 0){
		for (it = kernel.begin<float>(); it < kernel.end<float>(); it++){
			*it /= sum;
		}
	}
}

/**
 * Normalizes a kernel, such that the value of the absolute maximum (peak) of the kernel is 1
 * kernel: output CV_32FC1 image of specific size
 */
void kernelNormPeak(Mat& kernel){
	float peak = 0;
	MatIterator_<float> it;
	for (it = kernel.begin<float>(); it < kernel.end<float>(); it++){
		float val = std::abs(*it);
		if (val > peak) peak = val;
	}
	//std::cout << " normalized to " << peak;
	if (peak != 0){
		for (it = kernel.begin<float>(); it < kernel.end<float>(); it++){
			*it /= peak;
		}
	}
}



/*
 * Extends an image circular on right and left side and copies upper and lower sides
 * src: source image
 * dst: extended image 8extension in x-axis only
 * offsetX: shift-x offset
 * offsetY: shift-y offset
 */
void extendBorder(const Mat& src, Mat& dst, const int offsetX, const int offsetY){
	CV_Assert(dst.cols >= src.cols + offsetX); // extend on both sides
	CV_Assert(dst.rows >= src.rows + offsetY);
	const int width = src.cols;
	const int height = src.rows;
	const int twidth = dst.cols;
	const int theight = dst.rows;
	const int extendRight = twidth-(width+offsetX);
	const int extendBLine = offsetY+height-1;
	const int extendTLine = offsetY+1;
	Mat q1 (src,Rect(width-offsetX,0,offsetX,height));
	Mat q2 (src,Rect(0,0,extendRight,height));
	Mat d0 (dst,Rect(offsetX,offsetY,width,height));
	Mat d1 (dst,Rect(0,offsetY,offsetX,height));
	Mat d2 (dst,Rect(width+offsetX,offsetY,extendRight,height));
	q1.copyTo(d1);
	q2.copyTo(d2);
	src.copyTo(d0);
	for (int i=0; i<=offsetY; i++){ // this hack removes the first line
		Mat q (dst,Rect(0,extendTLine,twidth,1));
		Mat d (dst,Rect(0,i,twidth,1));
		q.copyTo(d);
	}
	for (int i=offsetY+height; i<theight; i++){
		Mat q (dst,Rect(0,extendBLine,twidth,1));
		Mat d (dst,Rect(0,i,twidth,1));
		q.copyTo(d);
	}
}

/**
 * Normalizes a function based on DFT by keeping the first numberCoeffs fourier coefficients only
 * Tip: f should be normalized (i.e. its average ideally should be zero)!
 * f:            original signal, should be Mat (1, size, CV_32FC1);
 * norm:         normalized signal, should be Mat (1, size, CV_32FC1);
 * numberCoeffs: number of fourrier coefficients to keep
 */
void fourierNormalize(const Mat& f, Mat& norm, float& energy, const int numberCoeffs = -1){
	int size = f.cols;
	int dftSize = getOptimalDFTSize(f.cols);
	if (dftSize != size){
		Mat g = Mat(1,dftSize,f.type());
		float * src = (float *) f.data;
		float * dst = (float *) g.data;
		for (int i=0; i<size; i++)
			dst[i] = src[i];
		for (int i=size; i<dftSize; i++)
			dst[i] = src[0];
		Mat gFourier = Mat(1,dftSize,f.type());
		dft(g,gFourier,CV_DXT_FORWARD);
		if (numberCoeffs >= 0){
			float * data = (float *) gFourier.data;
			for (int i = 1 + numberCoeffs*2; i < dftSize; i++){
				data[i] = 0;
			}
		}
		dft(gFourier,g,CV_DXT_INV_SCALE);
		src = (float *) g.data;
		dst = (float *) norm.data;
		for (int i=0; i<size; i++)
			dst[i] = src[i];
	}
	else {
		Mat fFourier = Mat(1,dftSize,f.type());
		dft(f,fFourier,CV_DXT_FORWARD);
		if (numberCoeffs >= 0){
			float * data = (float *) fFourier.data;
			energy = 0;
			int datastop = 1 + numberCoeffs*2;
			for (int i = 1; i < datastop; i++){
				energy += data[i] * data[i];
			}
			for (int i = datastop; i < size; i++){
				data[i] = 0;
			}
		}
		dft(fFourier,norm,CV_DXT_INV_SCALE);
	}
}

/**
 * Extracts a contour with spacing 1 from clustered candidate yvalues
 * contour: contour to be extracted with type Mat (1,width,CV_32FC1)
 * yval: y-values per angle with type Mat (3*width, 1, CV_32FC1)
 * labels: label per y-value with type Mat (3*width,1, CV_8UC1)
 * centers: central kmean y-values with type Mat (3,1, CV_32FC1);
 * index: label index
 */
void kmeans2Contour(Mat& contour, const Mat& yval, const Mat& yenergy, const Mat& labels, const float center, const int index){
	CV_Assert((yval.cols * yval.rows % 3) == 0);
	int pos = 0; float fact1 = 1, fact2 = 1;
	MatConstIterator_<uchar> l = labels.begin<uchar>();
	MatConstIterator_<uchar> lend = labels.end<uchar>();
	float * y = (float *) yval.data;
	int ystep = yval.step / sizeof (float);
	float * e = (float *) yenergy.data;
	int estep = yenergy.step / sizeof (float);
	MatIterator_<float> c = contour.begin<float>();
	float maxdist = 0; // maximum distance to center
	float maxenergy = 0;
	while (l != lend) {
		if (*l == index){
			float dist = abs(y[pos*ystep]-center);
			float energ = e[pos*estep];
			if (maxdist < dist) maxdist = dist;
			if (maxenergy < energ) maxenergy = energ;
		}
		l++;pos = 0;
	} // find next label point
	pos = 0;
	l = labels.begin<uchar>();
	while (l != lend && *l != index) {l++;pos++;} // find next label point
	if (l == lend) { contour.setTo(Scalar(center)); return;}
	float starty = y[pos*ystep];
	float maxweight = fact1 * (1 - (abs(starty-center) / maxdist)) + fact2*(e[pos*estep] / maxenergy);
	int startx = pos / 3;
	l++;pos++;
	while (pos % 3 != 0){
		if (*l == index){
			float yv = y[pos*ystep];
			float weight = fact1 * (1 - (abs(yv-center) / maxdist)) + fact2 * (e[pos*estep] / maxenergy);
			if (weight > maxweight) {
				starty = y[pos*ystep];
				maxweight = weight;
			}
		}
		l++;pos++;
	} // take best value
	float originy = starty;
	int originx = startx;
	c += startx;
	while (l != lend){
		// find next point
		float endy;
		int endx;
		while (l != lend && *l != index) {l++;pos++;} // find next label point
		if (l == lend) break;
		endy = y[pos*ystep];
		maxweight = fact1 * (1 - (abs(endy-center) / maxdist)) + fact2 * (e[pos*estep] / maxenergy);
		endx = pos / 3;
		l++;pos++;
		while (pos % 3 != 0){
			if (*l == index){
				float yv = y[pos*ystep];
				float weight = fact1 * (1 - (abs(yv-center) / maxdist)) + fact2 * (e[pos*estep] / maxenergy);
				if (weight > maxweight) {
					endy = y[pos*ystep];
					maxweight = weight;
				}
			}
			l++;pos++;
		} // take best value
		// now interpolate between startx and endx including startx
		int diffx = endx-startx;
		float delta = (endy - starty) / diffx;
		for (int i=0; i<diffx; i++, c++){
			*c = starty + i * delta;
		}
		startx = endx;
		starty = endy;
	}
	// now interpolate between startx and originx including startx
	int diffx1 = contour.cols - startx;
	int diffx = diffx1 + originx;
	float delta = (originy - starty) / diffx;
	for (int i=0; i<diffx1; i++, c++){
		*c = starty + i * delta;
	}
	c = contour.begin<float>();
	for (int i=diffx1; i<diffx; i++, c++){
		*c = starty + i * delta;
	}
}

/**
 * Retrieves the energy of the contour in the source image
 * (adds up source values at specified points)
 * src: original image (height, width, CV_32FC1)
 * contour: contour signal (1, width, CV_32FC1)
 * energy: resulting energy
 */
void getEnergy(const Mat& src,Mat& contour, float& energy, int offset = 0){
	//Mat tmp(src.rows,src.cols,CV_32FC1);
	//tmp.setTo(Scalar(0));
	CV_Assert(src.cols == contour.cols);
	float * s = (float *)contour.data;
	float * p = (float *)src.data;
	//float *tmpp = (float *)tmp.data;
	int stride = src.step / sizeof(float);
	//int tmpstride = tmp.step / sizeof(float);
	int width = src.cols;
	const int maxheight = src.rows - 1;
	energy = 0;
	if (offset == 0){
		for (int x=0; x<width; x++,s++){
			energy += p[x+max(0,min(cvRound(s[0]),maxheight))*stride];
		//tmpp[x+max(0,min(cvRound(s[0]),maxheight))*tmpstride] = 255;
		}
	}
	else {
		for (int x=0; x<width; x++,s++){
			energy += p[x+max(0,min(cvRound(s[0])+offset,maxheight))*stride];
			//tmpp[x+max(0,min(cvRound(s[0]),maxheight))*tmpstride] = 255;
		}
	}
	//kernelWrite(tmp,"temp.bmp");
}

/**
 * DEBUG: DELETE PLEASE
 * Visualizes a floating point kernel (using peak normalization)
 * kernel: output CV_32FC1 image of specific size
 * norm: CV_8UC1 normalized image
 */
bool kernelWrite(cv::Mat& kernel, std::string filename){
	CV_Assert(kernel.type() == CV_32FC1);
	cv::Mat norm(kernel.rows,kernel.cols,CV_8UC1);
	cv::MatIterator_<float> it;
	cv::MatIterator_<uchar> it2;
	float maxAbs = 0;
	for (it = kernel.begin<float>(); it < kernel.end<float>(); it++){
		if (std::abs(*it) > maxAbs) maxAbs = std::abs(*it);
	}
	//std::cout << "maxAbs: " << maxAbs << std::endl;
	for (it = kernel.begin<float>(), it2 = norm.begin<uchar>(); it < kernel.end<float>(); it++, it2++){
		*it2 = cv::saturate_cast<uchar>(cvRound(127 + (*it / maxAbs) * 127));
	}
	return cv::imwrite(filename,norm);
}

/**
 * DEBUG: DELETE PLEASE
 * Visualizes a floating point kernel (using peak normalization)
 * kernel: output CV_32FC1 image of specific size
 * norm: CV_8UC1 normalized image
 */
void float2ucharpic(cv::Mat& fpic, cv::Mat& upic){
	cv::MatIterator_<float> it;
	cv::MatIterator_<uchar> it2;
	float maxAbs = 0;
	for (it = fpic.begin<float>(); it < fpic.end<float>(); it++){
		if (std::abs(*it) > maxAbs) maxAbs = std::abs(*it);
	}
	//std::cout << "maxAbs: " << maxAbs << std::endl;
	for (it = fpic.begin<float>(), it2 = upic.begin<uchar>(); it < fpic.end<float>(); it++, it2++){
		*it2 = cv::saturate_cast<uchar>(cvRound(127 + (*it / maxAbs) * 127));
	}
}

/**
 * Optimizes a contour to fit the highest peak in the filtered image within a given range
 */
void fitContour(const Mat& contour, Mat& newContour, const Mat& filtered, int range){
	float * f = (float *)filtered.data;
	int stride = filtered.step / sizeof(float);
	int width = filtered.cols;
	int height = filtered.rows;
	MatConstIterator_<float> c = contour.begin<float>();
	MatIterator_<float> n = newContour.begin<float>();
	for (int x=0; x<width; x++, f++, c++, n++){
		int y0 = max(1,min(height-1,cvRound(*c)));
		int ystart = max(1,min(height-1,y0-range));
		int yend = max(1,min(height-1,y0+range));
		float *row = f + ystart * stride;
		int maxy = y0;
		float fily =0;
		for (int y=ystart; y < yend; y++, row += stride){
			if (*row > fily){
				fily = *row;
				maxy = y;
			}
		}
		//cout << "optimized from " << *c << " to " << maxy << endl;
		*n = maxy;
	}
}

void refineContour(const Mat& contour, Mat& newContour, const Mat& filtered, int range = 10){
	float mean;
	float cenergy;
	fmatAverage(contour,mean);
	fmatAdd(contour,newContour,-mean);
	fourierNormalize(newContour,newContour,cenergy,3);
	fmatAdd(newContour,newContour,mean);
	//fitContour(newContour,newContour,filtered,range);
	//meanZero(newContour,mean);
	//fourierNormalize(newContour,newContour,cenergy,3);
	//fmatAdd(newContour,newContour,mean);
}

/**
 * Estimates inner and outer contours in the polar image
 * src: source image
 * innerContour: pupilary boundary
 * outerContour: limbic boundary
 */
void eyeborderPolar(const Mat& src, Mat& inner, Mat& outer){
	CV_Assert(src.type() == CV_8UC1);
	// 1. convolve image with gabor kernels (long)
	Mat filtered(src.rows,src.cols,CV_32FC1);
	Mat filter(29,29,CV_32FC1);
	gaborKernel(filter,8*M_PI,-M_PI/2,M_PI/2,6,1);
	kernelNormPeak(filter);
	filter2D(src,filtered,filtered.depth(),filter,cvPoint(-1,-1),0,BORDER_REPLICATE);
	Mat debug1(src.rows,src.cols,CV_8UC1);
	float2ucharpic(filtered,debug1);
	Mat debug2(src.rows,src.cols,CV_8UC3);
	cv::cvtColor(debug1,debug2,CV_GRAY2BGR);


	// 2. find 3 best maximum vertical responses
	int height = src.rows;
	int width = src.cols;
	Mat yval(3*width,1,CV_32FC1); // candidate y-values
	Mat yenergy(3*width,1,CV_32FC1); // candidate energy
	Mat labels(3*width,1,CV_8UC1); // candidate best labels
	Mat centers(3,1,CV_32FC1); // candidate centers

	float * f = (float *)filtered.data;
	int stride = filtered.step / sizeof(float);
	MatIterator_<float> e = yenergy.begin<float>();
	MatIterator_<float> v = yval.begin<float>();
	for (int x=0; x<width; x++, f++){
		float *row = f + 1 * stride;
		int maxy = 0;
		float fily = 0;
		for (int y=1; y < height-1; y++, row += stride){
			if (*row > fily){
				fily = *row;
				maxy = y;
			}
		}
		*v = maxy;
		*e = fily;
		v++;
		e++;
		row = f + (maxy+1) * stride;
		int maxy1 = -1;
		float fily1 = 0;
		bool nonmax = true;
		for (int y=maxy+1; y < height-1; y++, row += stride){
				if (nonmax){
					float *p = row - stride;
					if (*row < *p) continue;
					else nonmax = false;
				}
				if (*row > fily1){
					fily1 = *row;
					maxy1 = y;
				}
		}
		row = f + (maxy-1) * stride;
		int maxy2 = -1;
		float fily2 = 0;
		nonmax = true;
		for (int y=maxy -1; y > 1; y--, row -= stride){
			if (nonmax){
				float *p = row + stride;
				if (*row < *p) continue;
				else nonmax = false;
			}
			if (*row > fily2){
				fily2 = *row;
				maxy2 = y;
			}
		}
		if (fily1 > fily2){
			if (maxy1 == -1){
				*v = maxy; v++;
				*e = fily; e++;
			}
			else {
				*v = maxy1; v++;
				*e = fily1; e++;
			}
			if (maxy2 == -1){
				*v = maxy; v++;
				*e = fily; e++;
			}
			else {
				*v = maxy2; v++;
				*e = fily2; e++;
			}
			/*debug2.at<Vec3b>(cvRound(*b),x)[0] = 255;
									debug2.at<Vec3b>(cvRound(*b),x)[1] = 0;
									debug2.at<Vec3b>(cvRound(*b),x)[2] = 0;*/
		}
		else{
			if (maxy2 == -1){
				*v = maxy; v++;
				*e = fily; e++;
			}
			else {
				*v = maxy2; v++;
				*e = fily2; e++;
			}
			if (maxy1 == -1){
				*v = maxy; v++;
				*e = fily; e++;
			}
			else {
				*v = maxy1; v++;
				*e = fily1; e++;
			}
		}
	}

	// 3. kmeans clustering of image points
	// use forward prediction model to include contour points
	kmeans(yval,3,labels,TermCriteria(CV_TERMCRIT_ITER,100,1e-100),3,KMEANS_PP_CENTERS,centers);
	MatIterator_<float> bit = yval.begin<float>();
	MatIterator_<uchar> lit = labels.begin<uchar>();
	for (int i=0, x=0; i<512; i++, x++){
		for (int i=0; i<3; i++, bit++, lit++){
			for (int j=0; j<3; j++){
				if (*lit == j) debug2.at<Vec3b>(cvRound(*bit),x)[j] = 255;
				else debug2.at<Vec3b>(cvRound(*bit),x)[j] = 0;
			}

		}
	}
	imwrite("filtered.png",debug2);
	// build up contour
	Mat cont1 (1,width,CV_32FC1);
	Mat cont2 (1,width,CV_32FC1);
	Mat cont3 (1,width,CV_32FC1);
	float center1 = centers.at<float>(0,0), center2 = centers.at<float>(1,0), center3 = centers.at<float>(2,0);
	Mat reliable3 (1,width,CV_8UC1);
	if (center1 < center2){
		if (center2 < center3){
			// 1 2 3
			kmeans2Contour(cont1,yval,yenergy,labels,center1,0); // upper
			kmeans2Contour(cont2,yval,yenergy,labels,center2,1); // middle
			kmeans2Contour(cont3,yval,yenergy,labels,center3,2); // lower
		}
		else if (center1 < center3){
			//1 3 2
			kmeans2Contour(cont1,yval,yenergy,labels,center1,0); // upper
			kmeans2Contour(cont2,yval,yenergy,labels,center3,2); // middle
			kmeans2Contour(cont3,yval,yenergy,labels,center2,1); // lower
		}
		else {
			// 3 1 2
			kmeans2Contour(cont1,yval,yenergy,labels,center3,2); // upper
			kmeans2Contour(cont2,yval,yenergy,labels,center1,0); // middle
			kmeans2Contour(cont3,yval,yenergy,labels,center2,1); // lower
		}
	}
	else {
		if (center1 < center3){
			// 2 1 3
			kmeans2Contour(cont1,yval,yenergy,labels,center2,1); // upper
			kmeans2Contour(cont2,yval,yenergy,labels,center1,0); // middle
			kmeans2Contour(cont3,yval,yenergy,labels,center3,2); // lower
		}
		else if (center2 < center3){
			// 2 3 1
			kmeans2Contour(cont1,yval,yenergy,labels,center2,1); // upper
			kmeans2Contour(cont2,yval,yenergy,labels,center3,2); // middle
			kmeans2Contour(cont3,yval,yenergy,labels,center1,0); // lower
		}
		else {
			// 3 2 1
			kmeans2Contour(cont1,yval,yenergy,labels,center3,2); // upper
			kmeans2Contour(cont2,yval,yenergy,labels,center2,1); // middle
			kmeans2Contour(cont3,yval,yenergy,labels,center1,0); // lower
		}
	}
	float mean1, mean2, mean3;
	float genergy1,genergy2,genergy3, energy1, energy2, energy3;
	float maxenergy;
	refineContour(cont1,cont1,filtered);
	refineContour(cont2,cont2,filtered);
	refineContour(cont3,cont3,filtered);
	getEnergy(filtered,cont1,genergy1);
	getEnergy(filtered,cont2,genergy2);
	getEnergy(filtered,cont3,genergy3);
	maxenergy = max(max(genergy1, genergy2),genergy3);
	genergy1 /= maxenergy;
	genergy2 /= maxenergy;
	genergy3 /= maxenergy;
	fmatAverage(cont1,mean1);
	fmatAverage(cont2,mean2);
	fmatAverage(cont3,mean3);
	float maxmean = max(max(mean1,mean2),mean3);
	energy1 = ((mean1 / maxmean) + 4*genergy1)/5;
	energy2 = ((mean2 / maxmean) + 4*genergy2)/5;
	energy3 = ((mean3 / maxmean) + 4*genergy3)/5;
	//cout << "energy 1: " << energy1 << " " << genergy1 << endl;
	//cout << "energy 2: " << energy2 << " " << genergy2 << endl;
	//cout << "energy 3: " << energy3 << " " << genergy3 << endl;

	/*
	float* t = (float *)cont3.data;
	uchar* r = reliable3.data;
	for (int i=0; i< width; i++,t++,r++){
		cout << "i: " << i << " data: " << *t << " reliable: " << (*r != 0) << endl;
	}*/

	// TEST
	float innerE, outerE;
	if (energy1 > energy2){
		if (energy2 > energy3){
			// 1 2 3
			inner = cont1; innerE = energy1;
			outer = cont2; outerE = energy2;
		}
		else if (energy1 > energy3){
			//1 3 2
			inner = cont1; innerE = energy1;
			outer = cont3; outerE = energy3;

		}
		else {
			// 3 1 2
			inner = cont1; innerE = energy1;
			outer = cont3; outerE = energy3;
		}
	}
	else {
		if (energy1 > energy3){
			// 2 1 3
			// eigentlich 1 2
			inner = cont1; innerE = energy1;
			outer = cont2; outerE = energy2;

		}
		else if (energy2 > energy3){
			// 2 3 1
			inner = cont2; innerE = energy2;
			outer = cont3; outerE = energy3;
		}
		else {
			// 3 2 1
			inner = cont2; innerE = energy2;
			outer = cont3; outerE = energy3;
		}
	}
	//cout << "innerE: " << innerE << endl;
	//cout << "outerE: " << outerE << endl;
	//fitContour(inner,inner,filtered,10);

	float outerAvg, innerAvg, genergy4;
	fmatAverage(outer, outerAvg);
	fmatAverage(inner, innerAvg);
	if (innerE > outerE){
		int from = cvRound((outerAvg-innerAvg) / 2);
		int to = cvRound(max(height-innerAvg,3 * (outerAvg-innerAvg) / 2));
		int maxi = 0;
		float menergy = outerE;
		for (int i=from; i<to; i++){
			getEnergy(filtered,inner,genergy4,i);
			if (genergy4 > menergy) {
				menergy = genergy4;
				maxi = i;
			}
		}
		fmatAdd(inner,outer,maxi);
		//fitContour(outer,outer,filtered,20);
	}
	else {
		int from = cvRound(innerAvg / 2 - outerAvg);
		int to = cvRound((innerAvg-outerAvg) / 2);
		int maxi = 0;
		float menergy = innerE;
		for (int i=from; i<to; i++){
			getEnergy(filtered,outer,genergy4,i);
			if (genergy4 > menergy) {
				menergy = genergy4;
				maxi = i;
			}
		}
		fmatAdd(outer,inner,maxi);
		//fitContour(inner,inner,filtered,20);
	}


	/*
	innerContour = contours[(energy1 > energy2) ? 0 : 1];
	if (energy1 > energy2) {
		if (energy 1 > energy3){

		}
		else {

		}
		outerContour = contours[(energy2 > energy3) ? 1 : 2];
	}
	else {
		outerContour = contours[2];
	}*/
	/*			for (int i=0; i<10; i++)
					cout << i << ": " << labels.at<int>(i,0) << endl;

				cout << "center: " << centers.at<float>(0,0) << " and " << centers.at<float>(1,0) << endl;
	*/
}

/** ------------------------------- Contour refinement ------------------------------- **/

/**
 * Calculates y-resolution for use with mappolar2cart
 * cartwidth: width of the cartesian source image (prior transformed with cart2polar)
 * cartheight: height of the cartesian source image (prior transformed with cart2polar)
 * polarheight: height of the destinatipon polar image (of the transformation cart2polar)
 * centerX: used center x-coordinate in the cart2polar transformation
 * centerY: used center y-coordinate in the cart2polar transformation
 * radius: used radius in the cart2polar transformation
 */
float getResY(const int cartwidth, const int cartheight, const int polarheight, const float centerX = 0, const float centerY = 0, const float radius = -1) {
	float dist1 = centerX*centerX;
	float dist2 = centerY*centerY;
	float dist3 = (cartwidth-centerX)*(cartwidth-centerX);
	float dist4 = (cartheight-centerY)*(cartheight-centerY);
	float rad = max(max(sqrt(dist1+dist2),sqrt(dist1+dist4)),max(sqrt(dist3+dist2),sqrt(dist3+dist4)));
	return rad / polarheight;
}

/*
 * Copies one floating point matrix to another
 * src: source matrix (CV_32FC1)
 * dst: destination matrix (CV_32FC1);
 */
void copy(const Mat& src, Mat& dst){
	CV_Assert(src.size == dst.size);
	MatConstIterator_<float> s = src.begin<float>();
	MatIterator_<float> d = dst.begin<float>();
	MatConstIterator_<float> e = src.end<float>();
	for (; s!=e; s++, d++){
		*d = *s;
	}
}

/**
 * Maps a polar contour back to cartesian coordinates
 * polar: array of radius values from center position
 * cart array of x- and y-positions from upper left corner
 * centerX: x-coordinate of center in cartesian coordinate system (upper-left corner)
 * centerY: y-coordinate of center in cartesian coordinate system (upper left corner)
 * resY: y-resolution (distance) of polar image (i.e. pixels per unit in polar coords, typically use method getResY to calculate resolution)
 * resX: x-resolution (angular) of polar image (i.e. 2 Pi divided by number of pixels used to represent the circle, typically 2.f * M_PI / polar.cols)
 */
void mappolar2cart(const Mat& polar, Mat& cart, const float centerX = 0, const float centerY = 0, const float resY = 1, const float resX = -1, const bool pixeloffset = true) {
	CV_Assert(polar.type() == CV_32FC1);
	CV_Assert(cart.type() == CV_32FC1);
	CV_Assert(2*polar.cols == cart.cols);
	float * s = (float *)polar.data;
	float * d = (float *)cart.data;
	int width = polar.cols;
	const float thetaoffset = (resX < 0) ? (2.f * M_PI / polar.cols) : resX;
	float theta = thetaoffset/2;
	if (pixeloffset){
		float xoffset = centerX - 0.5f;
		float yoffset = centerY - 0.5f;
		for (int x=0; x<width; x++,s++,d++, theta += thetaoffset){
			*d = xoffset + (*s + 0.5f) * resY * cos(theta); // x coordiante
			d++;
			*d = yoffset + (*s + 0.5f) * resY * sin(theta);; // y coordinate
		}
	}
	else if (centerX == 0 && centerY == 0 && resY == 1){ // fast version
		for (int x=0; x<width; x++,s++,d++, theta += thetaoffset){
			*d = *s * cos(theta); // x coordiante
			d++;
			*d = *s * sin(theta);; // y coordinate
		}
	}
	else {
		for (int x=0; x<width; x++,s++,d++, theta += thetaoffset){
			*d = centerX + *s * resY * cos(theta); // x coordiante
			d++;
			*d = centerY + *s * resY * sin(theta);; // y coordinate
		}
	}
}

/**
 * Resamples a closed contour from its new nucleus
 * contourOld: old polar contour, even spacing of 2 Pi (1,width,CV_32FC1), stretch-normalized (divide by resolution resY)
 * dx: center x-coordinate difference to old center (old_cx + dx = new_cx)
 * dy: center y-coordinate difference to old center (old_cy + dy = new_cy)
 * contourNew: new polar contour, even spacing of 2 Pi (1,width,CV_32FC1)
 */
void resampleContour(const Mat& contourOld, const float dx, const float dy, Mat& contourNew){
	float a, b, t1, x, y, t2, k, m, rad;
	int firstindex, lastindex;
	int width = contourOld.cols;
	const float thetaoffset = 2.f * M_PI / width;
	float thetastart = thetaoffset / 2;
	float theta = thetastart;
	float thetafactor = M_PI/180.f;
	float * s = (float *)contourOld.data;
	float * d = (float *)contourNew.data;
	a = (*s) * cos(theta) - dx;
	b = (*s) * sin(theta) - dy;
	//float a2 = a * a;
	//float b2 = b * b;
	//float r1 = (cv::sqrt(a2 + b2)) - 0.5f;
	t1 = (cv::fastAtan2(b, a)) * thetafactor;
	s++;
	theta += thetaoffset;
	firstindex = (int)(ceil((t1 - thetastart + 2* M_PI) / thetaoffset)) % width;
	for (int i=1; i<width; i++,s++, theta += thetaoffset){
		// calculate polar coordinates of the next point with respect to N
		x = (*s) * cos(theta) - dx;
		y = (*s) * sin(theta) - dy;
		//float x2 = x * x;
		//float y2 = y * y;
		//float r2 = (cv::sqrt(x2 + y2)) - 0.5f;
		t2 = (cv::fastAtan2(y, x)) * thetafactor;
		// von r1 bis r2 in Polarkoordinaten liegen Punkte auf der geraden
		// Geradengleichung aufstellen
		// y = k*x + m entspricht r(t) = m / (sin(t) + k*cos(t))
		lastindex = (int)(ceil((t2 - thetastart + 2*M_PI) / thetaoffset)) % width;
		if (lastindex != firstindex){ // at least one value is in between
			if (abs(x-a) < 0.00001){
				for (int j = firstindex; j != lastindex; j = ((j+1) % width)){
					rad = j*thetaoffset + thetastart;
					d[j] = a / cos(rad);
				}
			}
			else {
				k = (y-b)/(x-a);
				m = ((y-k*x) + (b-k*a))/2;
				for (int j = firstindex; j != lastindex; j = ((j+1) % width)){
					rad = j*thetaoffset + thetastart;
					d[j] = m / (sin(rad) - k * cos(rad));
				}
			}
		}
		a = x; b = y; t1 = t2;
		firstindex = lastindex;
		// r1, t1, r2, t2 are coordinates with respect to new coordinate system
		// s references contour point to be processed, it has *s distance to origin and has angle theta
		// now calculate: what would be the angle with respect to N ?
	}
	s = (float *)contourOld.data;
	theta = thetastart;
	x = (*s) * cos(theta) - dx;
	y = (*s) * sin(theta) - dy;
	t2 = (cv::fastAtan2(y, x)) * thetafactor;
	lastindex = (int)(ceil((t2 - thetastart + 2*M_PI) / thetaoffset)) % width;
	if (lastindex != firstindex){ // at least one value is in between
		if (abs(x-a) < 0.00001){
			for (int j = firstindex; j != lastindex; j = ((j+1) % width)){
				rad = j*thetaoffset + thetastart;
				d[j] = a / cos(rad);
			}
		}
		else {
			k = (y-b)/(x-a);
			m = ((y-k*x) + (b-k*a))/2;
			for (int j = firstindex; j != lastindex; j = ((j+1) % width)){
				rad = j*thetaoffset + thetastart;
				d[j] = m / (sin(rad) - k * cos(rad));
			}
		}
	}
}

void pullAndPush(const Mat& polarCont, Point2f& center){
	int width = polarCont.cols;
	Mat cart(1,2*width,CV_32FC1);
	Mat polar(1,width,CV_32FC1);
	copy(polarCont,polar);
	float fx = 0, fy = 0, dx = 0, dy = 0;
	int iter = 0;
	do {
		fx = 0; fy = 0;
		mappolar2cart(polar,cart,0,0,1,-1,false);
		for (MatIterator_<float> s = cart.begin<float>(), e = cart.end<float>(); s != e; s++){
			fx += (*s);
			s++;
			fy += (*s);
		}
		fx /= width;
		fy /= width;
		//cout << " moving fx " << fx << " fy " << fy << endl;
		dx += fx;
		dy += fy;
		resampleContour(polarCont,dx,dy,polar);
		iter++;
	}
	while ((abs(fx) > 0.1 || abs(fy) > 0.1) && iter < 1000);
	//cout << "iterations: " << iter << endl;
	center.x = dx;
	center.y = dy;
}

/**
 * Retrieves average contour value
 * contour: cartesian contour signal (1, 2*width, CV_32FC1)
 * energy: resulting average value
 */
void getCentroid(const Mat& contour, Point& center){
	float x = 0, y = 0;
	float * s = (float *)contour.data;
	int width = contour.cols / 2;
	for (int i=0; i<width; i++){
		x += *s;
		s++;
		y += *s;
		s++;
	}
	center.x = cvRound(x / width);
	center.y = cvRound(y / width);
}

/**
 * Estimates inner and outer contours in the polar image
 * src: source image (cartesian coordinates)
 * center: old center coordinates, overwritten with new refined center
 */
void refineEyeCenter(const Mat& src, Point& center){
	CV_Assert(src.type() == CV_8UC1);
	int width = src.cols;
	int height = src.rows;
	int polarwidth = width;
	int polarheight = height;
	Mat polar (polarheight,polarwidth,CV_8UC1);
	cart2polar(src,polar,center.x,center.y,-1,INTER_LINEAR);
	// 1. convolve image with gabor kernels (long) and take maximum response for slight variation
	Mat filtered1(polarheight,polarwidth,CV_32FC1);
	Mat filtered2(polarheight,polarwidth,CV_32FC1);
	Mat filtered3(polarheight,polarwidth,CV_32FC1);
	Mat filtered4(polarheight,polarwidth,CV_32FC1);
	Mat filter1(29,109,CV_32FC1);
	gaborKernel(filter1,8*M_PI,-M_PI/2,M_PI/2,2,0.1);
	kernelNormPeak(filter1);
	Mat filter2(29,109,CV_32FC1);
	gaborKernel(filter2,8*M_PI,-9*M_PI/16,M_PI/2,2,0.1);
	kernelNormPeak(filter2);
	Mat filter3(29,109,CV_32FC1);
	gaborKernel(filter3,8*M_PI,-7*M_PI/16,M_PI/2,2,0.1);
	kernelNormPeak(filter3);
	Mat filter4(29,29,CV_32FC1);
	gaborKernel(filter4,8*M_PI,-M_PI/2,M_PI/2,6,1);
	kernelNormPeak(filter4);
	//kernelWrite(filter4,"kernel.png");
	filter2D(polar,filtered1,filtered1.depth(),filter1,cvPoint(-1,-1),0,BORDER_REPLICATE);
	filter2D(polar,filtered2,filtered2.depth(),filter2,cvPoint(-1,-1),0,BORDER_REPLICATE);
	max(filtered1,filtered2,filtered3);
	filter2D(polar,filtered2,filtered2.depth(),filter3,cvPoint(-1,-1),0,BORDER_REPLICATE);
	max(filtered2,filtered3,filtered1);// in filtered1 steht jetzt das fertige
	filter2D(polar,filtered4,filtered4.depth(),filter4,cvPoint(-1,-1),0,BORDER_REPLICATE);
	// 2. find 3 best maximum vertical responses
	Mat yval(3*width,1,CV_32FC1); // candidate y-values
	Mat yenergy(3*width,1,CV_32FC1); // candidate energy
	Mat labels(3*width,1,CV_8UC1); // candidate best labels
	Mat centers(3,1,CV_32FC1); // candidate centers

	float * f = (float *)filtered1.data;
	int stride = filtered1.step / sizeof(float);
	MatIterator_<float> e = yenergy.begin<float>();
	MatIterator_<float> v = yval.begin<float>();
	for (int x=0; x<width; x++, f++){
		float *row = f + 1 * stride;
		int maxy = 0;
		float fily = 0;
		for (int y=1; y < height-1; y++, row += stride){
			if (*row > fily){
				fily = *row;
				maxy = y;
			}
		}
		*v = maxy;
		*e = fily;
		v++;
		e++;
		row = f + (maxy+1) * stride;
		int maxy1 = -1;
		float fily1 = 0;
		bool nonmax = true;
		for (int y=maxy+1; y < height-1; y++, row += stride){
				if (nonmax){
					float *p = row - stride;
					if (*row < *p) continue;
					else nonmax = false;
				}
				if (*row > fily1){
					fily1 = *row;
					maxy1 = y;
				}
		}
		row = f + (maxy-1) * stride;
		int maxy2 = -1;
		float fily2 = 0;
		nonmax = true;
		for (int y=maxy -1; y > 1; y--, row -= stride){
			if (nonmax){
				float *p = row + stride;
				if (*row < *p) continue;
				else nonmax = false;
			}
			if (*row > fily2){
				fily2 = *row;
				maxy2 = y;
			}
		}
		if (fily1 > fily2){
			if (maxy1 == -1){
				*v = maxy; v++;
				*e = fily; e++;
			}
			else {
				*v = maxy1; v++;
				*e = fily1; e++;
			}
			if (maxy2 == -1){
				*v = maxy; v++;
				*e = fily; e++;
			}
			else {
				*v = maxy2; v++;
				*e = fily2; e++;
			}
			/*debug2.at<Vec3b>(cvRound(*b),x)[0] = 255;
									debug2.at<Vec3b>(cvRound(*b),x)[1] = 0;
									debug2.at<Vec3b>(cvRound(*b),x)[2] = 0;*/
		}
		else{
			if (maxy2 == -1){
				*v = maxy; v++;
				*e = fily; e++;
			}
			else {
				*v = maxy2; v++;
				*e = fily2; e++;
			}
			if (maxy1 == -1){
				*v = maxy; v++;
				*e = fily; e++;
			}
			else {
				*v = maxy1; v++;
				*e = fily1; e++;
			}
		}
	}

	// 3. kmeans clustering of image points
	// use forward prediction model to include contour points
	kmeans(yval,3,labels,TermCriteria(CV_TERMCRIT_ITER,100,1e-100),3,KMEANS_PP_CENTERS,centers);
	// build up contour
	Mat cont1 (1,width,CV_32FC1);
	Mat cont2 (1,width,CV_32FC1);
	Mat cont3 (1,width,CV_32FC1);
	float center1 = centers.at<float>(0,0), center2 = centers.at<float>(1,0), center3 = centers.at<float>(2,0);
	Mat reliable3 (1,width,CV_8UC1);
	if (center1 < center2){
		if (center2 < center3){
			// 1 2 3
			kmeans2Contour(cont1,yval,yenergy,labels,center1,0); // upper
			kmeans2Contour(cont2,yval,yenergy,labels,center2,1); // middle
			kmeans2Contour(cont3,yval,yenergy,labels,center3,2); // lower
		}
		else if (center1 < center3){
			//1 3 2
			kmeans2Contour(cont1,yval,yenergy,labels,center1,0); // upper
			kmeans2Contour(cont2,yval,yenergy,labels,center3,2); // middle
			kmeans2Contour(cont3,yval,yenergy,labels,center2,1); // lower
		}
		else {
			// 3 1 2
			kmeans2Contour(cont1,yval,yenergy,labels,center3,2); // upper
			kmeans2Contour(cont2,yval,yenergy,labels,center1,0); // middle
			kmeans2Contour(cont3,yval,yenergy,labels,center2,1); // lower
		}
	}
	else {
		if (center1 < center3){
			// 2 1 3
			kmeans2Contour(cont1,yval,yenergy,labels,center2,1); // upper
			kmeans2Contour(cont2,yval,yenergy,labels,center1,0); // middle
			kmeans2Contour(cont3,yval,yenergy,labels,center3,2); // lower
		}
		else if (center2 < center3){
			// 2 3 1
			kmeans2Contour(cont1,yval,yenergy,labels,center2,1); // upper
			kmeans2Contour(cont2,yval,yenergy,labels,center3,2); // middle
			kmeans2Contour(cont3,yval,yenergy,labels,center1,0); // lower
		}
		else {
			// 3 2 1
			kmeans2Contour(cont1,yval,yenergy,labels,center3,2); // upper
			kmeans2Contour(cont2,yval,yenergy,labels,center2,1); // middle
			kmeans2Contour(cont3,yval,yenergy,labels,center1,0); // lower
		}
	}
	float mean1, mean2, mean3;
	float genergy1,genergy2,genergy3, cenergy1, cenergy2, cenergy3, energy1, energy2, energy3;
	float maxenergy;
	meanZero(cont1,mean1);
	meanZero(cont2,mean2);
	meanZero(cont3,mean3);
	fourierNormalize(cont1,cont1,cenergy1,3);
	fourierNormalize(cont2,cont2,cenergy2,3);
	fourierNormalize(cont3,cont3,cenergy3,3);
	maxenergy = max(max(cenergy1, cenergy2),cenergy3);
	cenergy1 = 1 - (cenergy1 / maxenergy);
	cenergy2 = 1 - (cenergy2 / maxenergy);
	cenergy3 = 1 - (cenergy3 / maxenergy);
	fmatAdd(cont1,cont1,mean1);
	fmatAdd(cont2,cont2,mean2);
	fmatAdd(cont3,cont3,mean3);
	getEnergy(filtered4,cont1,genergy1);
	getEnergy(filtered4,cont2,genergy2);
	getEnergy(filtered4,cont3,genergy3);
	maxenergy = max(max(genergy1, genergy2),genergy3);
	genergy1 /= maxenergy;
	genergy2 /= maxenergy;
	genergy3 /= maxenergy;
	energy1 = 0.5 * (cenergy1 + 4*genergy1)/5;
	energy2 = (cenergy2 + 4*genergy2)/5;
	energy3 = 2 * (cenergy3 + 4*genergy3)/5;
	Mat contNorm (1,polarwidth,CV_32FC1);
	//float resX = 2.f * M_PI / polarwidth;
	float resY = getResY(width, height, polarheight, center.x, center.y);
	if (energy1 > energy2){
		if (energy2 > energy3){
			// 1 2 3
			fmatAddMult(cont1,contNorm,0.5,resY);
		}
		else if (energy1 > energy3){
			//1 3 2
			fmatAddMult(cont1,contNorm,0.5,resY);
		}
		else {
			// 3 1 2
			fmatAddMult(cont3,contNorm,0.5,resY);
		}
	}
	else {
		if (energy1 > energy3){
			// 2 1 3
			fmatAddMult(cont1,contNorm,0.5,resY);
/*
			addAndMult(cont1,cont1,0.5,getResY(width,height,polarheight,center.x,center.y));

			f = cont1.begin<float>();
			for (int i=0;f!=e;i++, f++){
				cout << i << ": " << *f << endl;
			}
			mappolar2cart(cont1, cart, center.x,center.y, resY, resX);
			Mat newcont (1,cont1.cols,CV_32FC1);
			resampleContour(cont1,10.f,3.f,newcont);
			Mat visual;
			src.copyTo(visual);
			float * c = (float *) cart.data;
			for (int i=0; i < polarwidth-1;i++){
				line(visual,Point2f(c[2*i], c[2*i+1]),Point2f(c[2*i+2], c[2*i+3]),Scalar(0,0,0,0),3);
			}
			for (int i=0; i < polarwidth-1;i++){
				line(visual,Point2f(c[2*i], c[2*i+1]),Point2f(c[2*i+2], c[2*i+3]),Scalar(255,255,255,0));
			}
			imwrite("visual.png", visual);*/
		}
		else if (energy2 > energy3){
			// 2 3 1
			fmatAddMult(cont2,contNorm,0.5,resY);
		}
		else {
			// 3 2 1
			fmatAddMult(cont3,contNorm,0.5,resY);

		}
	}
	Point2f centernew (0,0);
	pullAndPush(contNorm,centernew);
	center.x = cvRound(0.5f + center.x + centernew.x);
	center.y = cvRound(0.5f + center.y + centernew.y);
}



/** ------------------------------- Helpers ------------------------------- **/

class CommandLine{
public:
	/*
	 * Default constructor for command line parameter parsing.
	 * This constructor should be called for parsing command lines for executables.
	 * Note, that all options require '-' as prefix and may contain an arbitrary
	 * number of optional arguments.
	 *
	 * argc: number of parameters
	 * argv: string array of argument values
	 */
	CommandLine(int argc, char *argv[]){
		for (int i=1; i< argc; i++){
			char * argument = argv[i];
			if (strlen(argument) > 1 && argument[0] == '-' && (argument[1] < '0' || argument[1] > '9')){
				std::vector<std::string> opt;
				char * argument2;
				while (i + 1 < argc && (strlen(argument2 = argv[i+1]) <= 1 || argument2[0] != '-'  || (argument2[1] >= '0' && argument2[1] <= '9'))){
					std::string arg2(argument2);
					opt.push_back(arg2);
					i++;
				}
				std::string arg(argument);
				content.insert(std::pair<std::string, std::vector<std::string> > (arg, opt));
				opt.clear();
			}
			else {
				CV_Error(CV_StsBadArg,"Invalid command line format");
			}
		}
	}

	/*
	 * Destructor
	 */
	~CommandLine(){
		content.clear();
	}

	/*
	 * Returns number of options
	 */
	unsigned int sizeOpts(void){
		return content.size();
	}

	/*
	 * Returns number of parameters in an option
	 *
	 * option: name of the option
	 */
	unsigned int sizePars(std::string option){
		std::map<std::string, std::vector<std::string> >::iterator it = content.find(option);
		return (it != content.end()) ? it->second.size() : 0;
	}

	/*
	 * Returns the list of parameters for a given option
	 *
	 * option: name of the option
	 */
	std::vector<std::string> * getOpt(std::string option){
		std::map<std::string, std::vector<std::string> >::iterator it = content.find(option);
		return (it != content.end()) ? &(it->second) : 0;
	}

	/*
	 * Returns a specific parameter type (int) given an option and parameter index
	 *
	 * option: name of option
	 * param: name of parameter
	 */
	int getParInt(std::string option, unsigned int param = 0){
		std::map<std::string, std::vector<std::string> >::iterator it = content.find(option);
		if (it != content.end()) {
			if (param < it->second.size()) {
				return atoi(it->second[param].c_str());
			}
		}
		return 0;
	}

	/*
	 * Returns a specific parameter type (long) given an option and parameter index
	 *
	 * option: name of option
	 * param: name of parameter
	 */
	long getParLong(std::string option, unsigned int param = 0){
		std::map<std::string, std::vector<std::string> >::iterator it = content.find(option);
		if (it != content.end()) {
			if (param < it->second.size()) {
				return atol(it->second[param].c_str());
			}
		}
		return 0;
	}

	/*
	 * Returns a specific parameter type (double) given an option and parameter index
	 *
	 * option: name of option
	 * param: name of parameter
	 */
	double getParDouble(std::string option, unsigned int param = 0){
		std::map<std::string, std::vector<std::string> >::iterator it = content.find(option);
		if (it != content.end()) {
			if (param < it->second.size()) {
				return atof(it->second[param].c_str());
			}
		}
		return 0;
	}

	/*
	 * Returns a specific parameter type (float) given an option and parameter index
	 *
	 * option: name of option
	 * param: name of parameter
	 */
	float getParFloat(std::string option, unsigned int param = 0){
		std::map<std::string, std::vector<std::string> >::iterator it = content.find(option);
		if (it != content.end()) {
			if (param < it->second.size()) {
				return atof(it->second[param].c_str());
			}
		}
		return 0;
	}

	/*
	 * Returns a specific parameter type (string) given an option and parameter index
	 *
	 * option: name of option
	 * param: name of parameter
	 */
	std::string getPar(std::string option, unsigned int param = 0){
		std::map<std::string, std::vector<std::string> >::iterator it = content.find(option);
		if (it != content.end()) {
			if (param < it->second.size()) {
				return it->second[param];
			}
		}
		return 0;
	}

	/*
	 * Checks, if each command line option is valid, i.e. exists in the options array
	 *
	 * validOptions: list of valid options separated by pipe (i.e. |) character
	 */
	void checkOpts(std::string validOptions){
		std::vector<std::string> tokens;
		const std::string delimiters = "|";
		// skip delimiters at beginning
		std::string::size_type lastPos = validOptions.find_first_not_of(delimiters,0);
		// find first non-delimiter
		std::string::size_type pos = validOptions.find_first_of(delimiters, lastPos);
		while (std::string::npos != pos || std::string::npos != lastPos){
			// add found token to vector
			tokens.push_back(validOptions.substr(lastPos,pos - lastPos));
			// skip delimiters
			lastPos = validOptions.find_first_not_of(delimiters,pos);
			// find next non-delimiter
			pos = validOptions.find_first_of(delimiters,lastPos);
		}
		sort(tokens.begin(), tokens.end());
		for (std::map<std::string, std::vector<std::string> >::iterator it = content.begin(); it != content.end(); it++){
			if (!binary_search(tokens.begin(),tokens.end(),it->first)){
				CV_Error(CV_StsBadArg,"Command line parameter '" + it->first + "' not allowed.");
				return;
			}
		}
	}

	/*
	 * Checks, if a specific required option exists in the command line
	 *
	 * option: option name
	 */
	void checkOptExists(std::string option){
		std::map<std::string, std::vector<std::string> >::iterator it = content.find(option);
		if (it == content.end()) CV_Error(CV_StsBadArg,"Command line parameter '" + option + "' is required, but does not exist.");
	}

	/*
	 * Checks, if a specific option has the appropriate number of parameters
	 *
	 * option: option name
	 * size: appropriate number of parameters for the option
	 */
	void checkOptSize(std::string option, unsigned int size = 1){
		std::map<std::string, std::vector<std::string> >::iterator it = content.find(option);
		if (it->second.size() != size) CV_Error(CV_StsBadArg,"Command line parameter '" + option + "' has unexpected size.");
	}


	/*
	 * Checks, if a specific option has the appropriate number of parameters
	 *
	 * option: option name
	 * min: minimum appropriate number of parameters for the option
	 * max: maximum appropriate number of parameters for the option
	 */
	void checkOptRange(std::string option, unsigned int min = 0, unsigned int max = 1){
		std::map<std::string, std::vector<std::string> >::iterator it = content.find(option);
		unsigned int size = it->second.size();
		if (size < min || size > max) CV_Error(CV_StsBadArg,"Command line parameter '" + option + "' is out of range.");
	}

	/*
	 * Returns a string representation of the command line
	 */
	std::string toString(void){
		std::ostringstream out;
		for (std::map<std::string, std::vector<std::string> >::iterator it = content.begin(); it != content.end(); it++){
			out << (*it).first << " ";
			std::vector<std::string> opt = it->second;
			for (std::vector<std::string>::iterator it2 = opt.begin(); it2 != opt.end(); it2++){
				out << *it2 << " ";
			}
		}
		return out.str();
	}

	/*
	 * Prints command line to STDOUT
	 */
	void print(void){
		std::cout << toString() << std::endl;
	}
private:
	std::map<std::string, std::vector<std::string> > content;
};

class Timing{
public:
	int progress;
	int total;

	/*
	 * Default constructor for timing.
	 * Automatically calls init()
	 */
	Timing(long seconds, bool eraseMode){
		updateInterval = seconds;
		progress = 1;
		total = 100;
		eraseCount=0;
		erase = eraseMode;
		init();
	}

	/*
	 * Destructor
	 */
	~Timing(){}

	/*
	 * Initializes timing variables
	 */
	void init(void){
		start = boost::posix_time::microsec_clock::universal_time();
		lastPrint = start - boost::posix_time::seconds(updateInterval);
	}

	/*
	 * Clears printing (for erase option only)
	 */
	void clear(void){
		for (unsigned int i=0; i<eraseCount;i++) std::cout << "\r";
		for (unsigned int i=0; i<eraseCount;i++) std::cout << " ";
		for (unsigned int i=0; i<eraseCount;i++) std::cout << "\r";
		eraseCount = 0;
	}

	/*
	 * Updates current time and returns true, if output should be printed
	 */
	bool update(void){
		current = boost::posix_time::microsec_clock::universal_time();
		return ((current - lastPrint > boost::posix_time::seconds(updateInterval)) || (progress == total));
	}

	/*
	 * Prints timing object to STDOUT
	 */
	void print(void){
		lastPrint = current;
		double percent = 100*((double)progress) / total;
		boost::posix_time::time_duration passedSec = (current - start);
		boost::posix_time::time_duration togoSec = boost::posix_time::seconds((long)((double)(passedSec.total_milliseconds()) * (100-percent) / percent / 1000));
		std::stringstream out;
		out << "Progress ... " << percent << "\% (" << (progress) << "/" << total << " Total " << passedSec;
		if (progress != total) out << " Remaining ca. "  << togoSec;
		out << ")";
		std::string output = out.str();
		if (erase){
			for (unsigned int i=0; i<eraseCount;i++) std::cout << "\r";
			std::cout << output;
			for (unsigned int i=output.length(); i<eraseCount;i++) std::cout << " ";
			for (unsigned int i=output.length(); i<eraseCount;i++) std::cout << "\r";
			eraseCount = output.length();
		}
		else {
			std::cout << output << std::endl;
		}
	}
private:
	long updateInterval;
	boost::posix_time::ptime start;
	boost::posix_time::ptime current;
	boost::posix_time::ptime lastPrint;
	unsigned int eraseCount;
	bool erase;
};

class FilePattern{
private:
	/*
	 * Formats a given string, such that it can be used as a regular expression (escapes sprcial characters and uses * and ? as wildcards)
	 */
	std::string formatPattern(std::size_t pos, std::size_t n){
		std::string pattern(filePattern.substr(pos,n));
		pattern = boost::algorithm::replace_all_copy(pattern, "\\", "\\\\");
		pattern = boost::algorithm::replace_all_copy(pattern, ".", "\\.");
		pattern = boost::algorithm::replace_all_copy(pattern, "+", "\\+");
		pattern = boost::algorithm::replace_all_copy(pattern, "[", "\\[");
		pattern = boost::algorithm::replace_all_copy(pattern, "{", "\\{");
		pattern = boost::algorithm::replace_all_copy(pattern, "|", "\\|");
		pattern = boost::algorithm::replace_all_copy(pattern, "(", "\\(");
		pattern = boost::algorithm::replace_all_copy(pattern, ")", "\\)");
		pattern = boost::algorithm::replace_all_copy(pattern, "^", "\\^");
		pattern = boost::algorithm::replace_all_copy(pattern, "$", "\\$");
		pattern = boost::algorithm::replace_all_copy(pattern, "}", "\\}");
		pattern = boost::algorithm::replace_all_copy(pattern, "]", "\\]");
		pattern = boost::algorithm::replace_all_copy(pattern, "*", "([^/\\\\]*)");
		pattern = boost::algorithm::replace_all_copy(pattern, "?", "([^/\\\\])");
		return pattern;
	}

	/*
	 * Appends the files in the specified path according to yet unresolved pattern by recursively calling this function.
	 *
	 * current: an index such that 0...current-1 of pattern are already considered/matched yielding path
	 * path:    the current directory (or empty)
	 * files:   the list to which new files can be applied
	 */
	void recursiveGetFiles(const unsigned int& current, const std::string& path, std::list<std::string>& files){
		std::size_t first_unknown = filePattern.find_first_of("*?",current); // find unknown * in filePattern
		if (first_unknown != std::string::npos){
			std::size_t last_dirpath = filePattern.find_last_of("/\\",first_unknown);
			std::size_t next_dirpath = filePattern.find_first_of("/\\",first_unknown);
			if (next_dirpath != std::string::npos){
				boost::regex expr((last_dirpath != std::string::npos && last_dirpath > current) ? formatPattern(last_dirpath+1,next_dirpath-last_dirpath-1) : formatPattern(current,next_dirpath-current));
				boost::filesystem::directory_iterator end_itr; // default construction yields past-the-end
				try {
					for ( boost::filesystem::directory_iterator itr( ((path.length() > 0) ? path + filePattern[current-1] : (last_dirpath != std::string::npos && last_dirpath > current) ? "" : "./") + ((last_dirpath != std::string::npos && last_dirpath > current) ? filePattern.substr(current,last_dirpath-current) : "")); itr != end_itr; ++itr )
					{
						if (boost::filesystem::is_directory(itr->path()) && boost::regex_match(itr->path().filename().string(), expr)){
							recursiveGetFiles((int)(next_dirpath+1),((path.length() > 0) ? path + filePattern[current-1] : "") + ((last_dirpath != std::string::npos && last_dirpath > current) ? filePattern.substr(current,last_dirpath-current) + filePattern[last_dirpath] : "") + itr->path().filename().string(),files);
						}
					}
				}
				catch (boost::filesystem::filesystem_error &e){}
			}
			else {
				boost::regex expr((last_dirpath != std::string::npos && last_dirpath > current) ? formatPattern(last_dirpath+1,filePattern.length()-last_dirpath-1) : formatPattern(current,filePattern.length()-current));
				boost::filesystem::directory_iterator end_itr; // default construction yields past-the-end
				try {
					for ( boost::filesystem::directory_iterator itr(((path.length() > 0) ? path +  filePattern[current-1] : (last_dirpath != std::string::npos && last_dirpath > current) ? "" : "./") + ((last_dirpath != std::string::npos && last_dirpath > current) ? filePattern.substr(current,last_dirpath-current) : "")); itr != end_itr; ++itr )
					{
						if (boost::regex_match(itr->path().filename().string(), expr)){
							files.push_back(((path.length() > 0) ? path + filePattern[current-1] : "") + ((last_dirpath != std::string::npos && last_dirpath > current) ? filePattern.substr(current,last_dirpath-current) + filePattern[last_dirpath] : "") + itr->path().filename().string());
						}
					}
				}
				catch (boost::filesystem::filesystem_error &e){
                    cerr << "error in recursiveGetFiles("<<current<<", "<<path<<", ...) in line "<< __LINE__ << endl;
                    cerr << e.what() << endl;
                }
			}
		}
		else { // no unknown symbols
			boost::filesystem::path file(((path.length() > 0) ? path + "/" : "") + filePattern.substr(current,filePattern.length()-current));
			if (boost::filesystem::exists(file)){
				files.push_back(file.string());
			}
		}
	}
	std::string filePattern;
public:
	/*
	 * Default constructor for file pattern processing.
	 *
	 * pattern: actual file pattern, e.g. *.jpg
	 */
	FilePattern(const std::string& pattern){
		filePattern = pattern;
	}

	/*
	 * Destructor
	 */
	FilePattern(){
		filePattern.clear();
	}

	/*
	 * Retrieves a list of files from the filesystem, which correspond to the fiven file pattern.
	 * Returns true, if the operation was successful (returned at least one file)
	 *
	 * files: empty list of filenames, where new files are appended
	 * returns: true, if at least one file is appended
	 */
	void getFiles(std::list<std::string>& files){
		recursiveGetFiles(0,"",files);
	}

	/*
	 * Renames a given filename corresponding to the actual file pattern using a renaming pattern.
	 * Wildcards can be referred to as ?1, ?2, ... in the order they appeared in the file pattern.
	 *
	 * infile: path of the file to be renamed
	 * outfile: path of the renamed file
	 * renamePattern: renaming pattern using ?1, ?2, ... as placeholders for wildcards
	 * returns: true, if renaming was successful
	 */
	void renameFile(const std::string& infile, std::string& outfile, const std::string& renamePattern, const std::string& par = "?"){
		std::size_t first_unknown = renamePattern.find_first_of(par,0); // find unknown ? in renamePattern
		if (first_unknown != std::string::npos){
			std::string formatOut = renamePattern;
			for (int i=49; i<58; i++){
				formatOut = boost::algorithm::replace_all_copy(formatOut, par + (char) i, std::string("$") + (char) i);
			}
			boost::regex patternOut(formatPattern(0,filePattern.length()));
			outfile = boost::regex_replace(infile,patternOut,formatOut,boost::match_default | boost::format_perl);
		} else {
			outfile = renamePattern;
		}
	}
};

/** ------------------------------- Program ------------------------------- **/

/*
 * Main program
 */
int main(int argc, char *argv[])
{
	int mode = MODE_HELP;
	try {
    	CommandLine cmd(argc,argv);
    	if (cmd.sizeOpts() == 0 || cmd.getOpt("-h") != 0) mode = MODE_HELP;
    	else mode = MODE_MAIN;
    	if (mode == MODE_MAIN){
			// validate command line
			cmd.checkOpts("-i|-o|-s|-vm|-vc|-vp|-vb|-vs|-bm|-q|-t");
			cmd.checkOptExists("-i");
			cmd.checkOptSize("-i",1);
			FilePattern infiles(cmd.getPar("-i"));
			cmd.checkOptExists("-o");
			cmd.checkOptSize("-o",1);
			string outfiles = cmd.getPar("-o");
			int outwidth = -1, outheight = -1;
			if (cmd.getOpt("-s") != 0){
				cmd.checkOptSize("-s",2);
				outwidth = cmd.getParInt("-s",0);
				outheight = cmd.getParInt("-s",1);
			}
			string vmaskfiles;
			if (cmd.getOpt("-vm") != 0){
				cmd.checkOptSize("-vm",1);
				vmaskfiles = cmd.getPar("-vm");
			}
			string vcenterfiles;
			if (cmd.getOpt("-vc") != 0){
				cmd.checkOptSize("-vc",1);
				vcenterfiles = cmd.getPar("-vc");
			}
			string vpolarfiles;
			if (cmd.getOpt("-vp") != 0){
				cmd.checkOptSize("-vp",1);
				vpolarfiles = cmd.getPar("-vp");
			}
			string vborderfiles;
			if (cmd.getOpt("-vb") != 0){
				cmd.checkOptSize("-vb",1);
				vborderfiles = cmd.getPar("-vb");
			}
			string vsegmentfiles;
			if (cmd.getOpt("-vs") != 0){
				cmd.checkOptSize("-vs",1);
				vsegmentfiles = cmd.getPar("-vs");
			}
			string binmaskfiles;
			if (cmd.getOpt("-bm") != 0){
				cmd.checkOptSize("-bm",1);
				binmaskfiles = cmd.getPar("-bm");
			}
			int apertureSize = 5;
			unsigned int accuSize = 100;
			float accuPrecision = .5;
			bool q = false;
			if (cmd.getOpt("-q") != 0){
				cmd.checkOptSize("-q",0);
				q = true;
			}
			bool t = false;
			if (cmd.getOpt("-t") != 0){
				cmd.checkOptSize("-t",0);
				t = true;
			}
			// starting routine
			Timing timing(1,q);
			std::list<std::string> files;
			infiles.getFiles(files);
			CV_Assert(files.size() > 0);
			timing.total = files.size();
			for (std::list<std::string>::iterator infile = files.begin(); infile != files.end(); ++infile, timing.progress++){
				if (!q) cout << "Loading image '" << *infile << "' ..."<< endl;
				// MODIFICATION TB, June 3rd, 2014
				// additional conversion step to enable direct JP2K processing (Bug in CV)
				// Loading of JP2k in color is supported, loading as grayscale, however, not
				Mat imgCol = imread(*infile, CV_LOAD_IMAGE_COLOR);			
				Mat img;			
				cvtColor(imgCol,img,CV_BGR2GRAY);
				CV_Assert(img.data != 0);
				CV_Assert(img.depth() == CV_8U);
				int width = img.cols;
				int height = img.rows;
				if (!q) cout << "done" << endl << "Removing reflections ..."<< endl;
				Mat mask(height,width,CV_8UC1);
				Mat img2(height,width,CV_8UC1);
				const float roiReflections = 15;//10
				const int maxReflectSize = 2000;//1000
				const int dilateSize = 10;//7
				const int dilateIterations = 4;//3
				maskReflections(img, mask, roiReflections, dilateSize, dilateIterations, maxReflectSize);
				inpaint(img,mask,img2,10,INPAINT_NS);
				if (!q) cout << "done" << endl << "Generating mask ..."<< endl;
				Mat mask2(height,width,CV_8UC1);
				maskEye(img2,mask2,mask);
				if (!q) cout << "done" << endl;
				if (!vmaskfiles.empty()){
					string vmaskfile;
					infiles.renameFile(*infile,vmaskfile,vmaskfiles);
					if (!q) cout << "Storing visual mask '" << vmaskfile << "' ..."<< endl;
					if (!imwrite(vmaskfile,mask)) CV_Error(CV_StsError,"Could not save image '" + vmaskfile + "'");
					if (!q) cout << "done" << endl;
				}
				if (!q) cout << "Finding circle center ..."<< endl;
				Point center;
				Mat gradX(height,width,CV_32FC1);
				Mat gradY(height,width,CV_32FC1);
				Sobel(img2,gradX,gradX.depth(),1,0,apertureSize);
				Sobel(img2,gradY,gradY.depth(),0,1,apertureSize);
				eyeCenter(gradX,gradY,mask2,center,accuPrecision,accuSize);
				cout << "Center before: (x=" << center.x << ";y=" << center.y << ")" << endl;
				refineEyeCenter(img2,center);
				cout << "Center after: (x=" << center.x << ";y=" << center.y << ")" << endl;
				if (!q) cout << "done" << endl;
				if (center.x < 0 || center.y < 0){
					if (!q) cout << "Center not found." << endl;
					// make an educated guess
					center.x = width/2;
					center.y = height/2;
				}
				else {
					if (!q) cout << "Center: (x=" << center.x << ";y=" << center.y << ")" << endl;
					if (!vcenterfiles.empty()){
						Mat visual;
						img.copyTo(visual);
						line(visual,Point(center.x,0),Point(center.x,img.rows),Scalar(0,0,0,0),3);
						line(visual,Point(0,center.y),Point(img.cols,center.y),Scalar(0,0,0,0),3);
						line(visual,Point(center.x,0),Point(center.x,img.rows),Scalar(255,255,255,0));
						line(visual,Point(0,center.y),Point(img.cols,center.y),Scalar(255,255,255,0));
						string vcenterfile;
						infiles.renameFile(*infile,vcenterfile,vcenterfiles);
						if (!q) cout << "Storing visual center '" << vcenterfile << "' ..."<< endl;
						if (!imwrite(vcenterfile,visual)) CV_Error(CV_StsError,"Could not save image '" + vcenterfile + "'");
						if (!q) cout << "done" << endl;
					}
				}
				if (!q) cout << "Applying Cartesian to Polar Transformation ..."<< endl;
				int polarwidth = (outwidth < 0) ? width : outwidth;
				int polarheight = (outwidth < 0) ? height : cvRound(outwidth * height / ((float)(width)));
				if (outwidth < 0){
					outwidth = width;
				}
				if (outheight < 0){
					outheight = height;
				}
				Mat polar (polarheight,polarwidth,CV_8UC1);
				cart2polar(img2,polar,center.x,center.y,-1,INTER_LINEAR);
				if (!q) cout << "done" << endl;
				if (!vpolarfiles.empty()){
					string vpolarfile;
					infiles.renameFile(*infile,vpolarfile,vpolarfiles);
					if (!q) cout << "Storing visual polar image '" << vpolarfile << "' ..."<< endl;
					if (!imwrite(vpolarfile,polar)) CV_Error(CV_StsError,"Could not save image '" + vpolarfile + "'");
					if (!q) cout << "done" << endl;
				}
				if (!q) cout << "Finding eye boundaries ..."<< endl;
				Mat inner (1,polarwidth,CV_32FC1);
				Mat outer (1,polarwidth,CV_32FC1);
				eyeborderPolar(polar, inner, outer);
				if (!q) cout << "done" << endl;
				if (!vborderfiles.empty()){
					Mat visual;
					polar.copyTo(visual);
					float * c = (float *) inner.data;
					for (int i=0; i < polarwidth-1;i++, c++){
						line(visual,Point2f(i, *c),Point2f(i+1, c[1]),Scalar(0,0,0,0),3);
					}
					c = (float *) inner.data;
					for (int i=0; i < polarwidth-1;i++, c++){
						line(visual,Point2f(i, *c),Point2f(i+1, c[1]),Scalar(255,255,255,0));
					}
					c = (float *) outer.data;
					for (int i=0; i < polarwidth-1;i++, c++){
						line(visual,Point2f(i, *c),Point2f(i+1, c[1]),Scalar(0,0,0,0),3);
					}
					c = (float *) outer.data;
					for (int i=0; i < polarwidth-1;i++, c++){
						line(visual,Point2f(i, *c),Point2f(i+1, c[1]),Scalar(255,255,255,0));
					}
					string vborderfile;
					infiles.renameFile(*infile,vborderfile,vborderfiles);
					if (!q) cout << "Storing image '" << vborderfile << "' ..."<< endl;
					if (!imwrite(vborderfile,visual)) CV_Error(CV_StsError,"Could not save image '" + vborderfile + "'");
					if (!q) cout << "done" << endl;
				}
				Mat innerCart (1,2*polarwidth,CV_32FC1);
				Mat outerCart (1,2*polarwidth,CV_32FC1);
				Mat out (outheight,outwidth,CV_8UC1);
				float resX = 2.f * M_PI / polarwidth;
				float resY = getResY(width, height, polarheight, center.x, center.y);
				mappolar2cart(inner, innerCart, center.x,center.y, resY, resX);
				mappolar2cart(outer, outerCart, center.x,center.y, resY, resX);
				if (!vsegmentfiles.empty()){
					Mat visual;
					img.copyTo(visual);
					float * c = (float *) innerCart.data;
					for (int i=0; i < polarwidth-1;i++){
						line(visual,Point2f(c[2*i], c[2*i+1]),Point2f(c[2*i+2], c[2*i+3]),Scalar(0,0,0,0),3);
					}
					for (int i=0; i < polarwidth-1;i++){
						line(visual,Point2f(c[2*i], c[2*i+1]),Point2f(c[2*i+2], c[2*i+3]),Scalar(255,255,255,0));
					}
					c = (float *) outerCart.data;
					for (int i=0; i < polarwidth-1;i++){
						line(visual,Point2f(c[2*i], c[2*i+1]),Point2f(c[2*i+2], c[2*i+3]),Scalar(0,0,0,0),3);
					}
					for (int i=0; i < polarwidth-1;i++){
						line(visual,Point2f(c[2*i], c[2*i+1]),Point2f(c[2*i+2], c[2*i+3]),Scalar(255,255,255,0));
					}
					string vsegmentfile;
					infiles.renameFile(*infile,vsegmentfile,vsegmentfiles);
					if (!q) cout << "Storing image '" << vsegmentfile << "' ..."<< endl;
					if (!imwrite(vsegmentfile,visual)) CV_Error(CV_StsError,"Could not save image '" + vsegmentfile + "'");
					if (!q) cout << "done" << endl;
				}
                if( !binmaskfiles.empty()){
					Mat bw(height,width, CV_8UC1, Scalar(0));
                    Point iris_points[1][polarwidth];
					float * it = (float *) outerCart.data;
					for (int i=0; i < polarwidth; i++){
						iris_points[0][i] = Point2i(cvRound(*it),cvRound(it[1]));
						it+=2;
					}
					const Point* irispoints[1] = { iris_points[0] };

                    Point pupil_points[1][polarwidth];
					it = (float *) innerCart.data;
					for (int i=0; i < polarwidth; i++){
						pupil_points[0][i] = Point2f(cvRound(*it),cvRound(it[1]));
						it+=2;
					}
					const Point* pupilpoints[1] = { pupil_points[0] };

                    fillPoly(bw,irispoints,&polarwidth,1,Scalar(255,255,255));
					fillPoly(bw,pupilpoints,&polarwidth,1,Scalar(0,0,0));

					string binmaskfile;
					infiles.renameFile(*infile,binmaskfile,binmaskfiles);
					if (!q) cout << "Storing binary mask image '" << binmaskfile << "' ..."<< endl;
					if (!imwrite(binmaskfile,bw)) CV_Error(CV_StsError,"Could not save image '" + binmaskfile + "'");
					if (!q) cout << "done" << endl;
                }
				if (!q) cout << "Applying Rubbersheet transform ..."<< endl;
				rubbersheet(img2, out, innerCart, outerCart, INTER_LINEAR);
				string outfile;
				infiles.renameFile(*infile,outfile,outfiles);
				if (!q) cout << "done" << endl << "Storing image '" << outfile << "' ..."<< endl;
				if (!imwrite(outfile,out)) CV_Error(CV_StsError,"Could not save image '" + outfile + "'");
				if (!q) cout << "done" << endl;
				if (t && timing.update()) timing.print();
			}
			if (t && q) timing.clear();
    	}
    	else if (mode == MODE_HELP){
			// validate command line
			cmd.checkOpts("-h");
			if (cmd.getOpt("-h") != 0) cmd.checkOptSize("-h",0);
			// starting routine
			printUsage();
    	}
    }
	catch (exception& e){
	   	cerr << "Exit with errors:" << e.what() << endl;
	   	exit(EXIT_FAILURE);
	}
    return EXIT_SUCCESS;
}
