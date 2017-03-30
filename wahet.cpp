/*
 * wahet.cpp
 *
 * Author: P. Wild (pwild@cosy.sbg.ac.at)
 *
 * Iris segmentation tool extracting the iris texture out of an eye image and mapping it into doubly dimensionless coordinates.
 * The software implements the following technique:
 *
 * Weighted Adaptive Hough and Ellipsopolar Transform
 *
 * see:
 *
 * A. Uhl and P. Wild. Weighted Adaptive Hough and Ellipsopolar Transforms for Real-time Iris Segmentation.
 * In Proceedings of the 5th International Conference on Biometrics (ICB'12), 8 pages, New Delhi, India,
 * March 29 - April 1, 2012.
 *
 */
#define _USE_MATH_DEFINES
#include <cmath>

#ifdef _WIN32 
#include <boost\math\special_functions\fpclassify.hpp>
#ifndef INFINITY
#define INFINITY (DBL_MAX+DBL_MAX)
#endif
#ifndef NAN
#define NAN (INFINITY-INFINITY)
#endif
#endif

#include "version.h"
#include <cstdio>
#include <map>
#include <vector>
#include <string>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo/photo.hpp>
#include <boost/regex.hpp>
#include <boost/filesystem.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;
using namespace cv;

/** no globbing in win32 mode **/
int _CRT_glob = 0;

/** Program modes **/
static const int MODE_MAIN = 1, MODE_HELP = 2;

/*
 * Print command line usage for this program
 */
void printUsage() {
    printVersion();
	printf("+-----------------------------------------------------------------------------+\n");
	printf("| wahet - Weighted Adaptive Hough and Ellipsopolar Transform                  |\n");
	printf("|                                                                             |\n");
	printf("| MODES                                                                       |\n");
	printf("|                                                                             |\n");
    printf("| (# 1) iris texture extraction from eye images                               |\n");
    printf("| (# 2) usage                                                                 |\n");
    printf("|                                                                             |\n");
    printf("| ARGUMENTS                                                                   |\n");
    printf("|                                                                             |\n");
    printf("+------+------------+---+---+-------------------------------------------------+\n");
    printf("| Name | Parameters | # | ? | Description                                     |\n");
    printf("+------+------------+---+---+-------------------------------------------------+\n");
    printf("| -i   | infile     | 1 | N | input eye image (use * as wildcard, all other   |\n");
    printf("|      |            |   |   | file parameters may refer to n-th * with ?n)    |\n");
    printf("| -o   | outfile    | 1 | N | output iris texture image                       |\n");
	printf("| -m   | maskfile   | 1 | N | output iris noise mask image                    |\n");
    printf("| -s   | wdth hght  | 1 | Y | size, i.e. width and height, of output (512x64) |\n");
    printf("| -e   |            | 1 | Y | enhance iris texture on (off)                   |\n");
    printf("| -q   |            | 1 | Y | quiet mode on (off)                             |\n");
    printf("| -t   |            | 1 | Y | time progress on (off)                          |\n");
    printf("| -rm  | rmaskfile  | 1 | Y | write source reflection mask (off)              |\n");
    printf("| -rr  | rremovfile | 1 | Y | write image with removed reflections (off)      |\n");
    printf("| -em  | emaskfile  | 1 | Y | write image masking boundary edges (off)        |\n");
    printf("| -gr  | gradfile   | 1 | Y | write gradient magnitude and phase image (off)  |\n");
    printf("| -ic  | icentfile  | 1 | Y | write result of initial center detection (off)  |\n");
    printf("| -po  | polarfile  | 1 | Y | write polar image (off)                         |\n");
    printf("| -fb  | fboundfile | 1 | Y | write first boundary in polar coords (off)      |\n");
    printf("| -ep  | ellpolfile | 1 | Y | write ellipsopolar image (off)                  |\n");
    printf("| -ib  | iboundfile | 1 | Y | write inner boundary candidate (off)            |\n");
    printf("| -ob  | oboundfile | 1 | Y | write outer boundary candidate (off)            |\n");
    printf("| -bm  | binmaskfile| 1 | Y | write binary segmentation mask (off)            |\n");
    printf("| -sr  | segresfile | 1 | Y | write segmentation result (off)                 |\n");
    printf("| -lt  | thickness  | 1 | Y | thickness for lines to be drawn (1)             |\n");
    printf("| -h   |            | 2 | N | prints usage                                    |\n");
    printf("| -so  | scale fac. | 1 | N | scale the area of the outer ellipse (def.=1.0)  |\n");
    printf("| -si  | scale fac. | 1 | N | scale the area of the inner ellipse (def.=1.0)  |\n");
    printf("| -tr  | translate  | 1 | N | horizontal translation of ellipses (def.=0.0)   |\n");
    printf("|      |            |   |   | the factor is by iris radius(x) (-1 would be a  |\n");
    printf("|      |            |   |   | translation by iris radius to the left)         |\n");
    printf("| -l   | logfile    | 1 | N | log parameters for unrolling to this file.      |\n");
    printf("+------+------------+---+---+-------------------------------------------------+\n");
    printf("|                                                                             |\n");
    printf("| EXAMPLE USAGE                                                               |\n");
    printf("|                                                                             |\n");
    printf("| -i *.tiff -o ?1_texture.png -s 512 32 -e -q -t                              |\n");
    printf("|                                                                             |\n");
    printf("| AUTHOR                                                                      |\n");
    printf("|                                                                             |\n");
    printf("| Peter Wild (pwild@cosy.sbg.ac.at)                                           |\n");
    printf("| Heinz Hofbauer (hhofbaue@cosy.sbg.ac.at)                                    |\n");
    printf("|                                                                             |\n");
    printf("| JPEG2000 Hack                                                               |\n");
    printf("| Thomas Bergmueller (thomas.bergmueller@authenticvision.com)                 |\n");
    printf("|                                                                             |\n");
    printf("|                                                                             |\n");
    printf("| COPYRIGHT                                                                   |\n");
    printf("|                                                                             |\n");
    printf("| (C) 2012 All rights reserved. Do not distribute without written permission. |\n");
    printf("+-----------------------------------------------------------------------------+\n");
}

/** ------------------------------- OpenCV helpers ------------------------------- **/

/**
 * Visualizes a CV_32FC1 image (using peak normalization)
 *
 * filename: name of the file to be stored
 * norm: CV_32FC1 image
 *
 * returning imwrite result code
 */
bool imwrite2f(const string filename, const Mat& src){
	Mat norm(src.rows,src.cols,CV_8UC1);
	MatConstIterator_<float> it;
	MatIterator_<uchar> it2;
	float maxAbs = 0;
	for (it = src.begin<float>(); it < src.end<float>(); it++){
		if (std::abs(*it) > maxAbs) maxAbs = std::abs(*it);
	}
	for (it = src.begin<float>(), it2 = norm.begin<uchar>(); it < src.end<float>(); it++, it2++){
		*it2 = saturate_cast<uchar>(cvRound(127 + (*it / maxAbs) * 127));
	}
	return imwrite(filename,norm);
}

/**
 * Calculate a standard uniform upper exclusive lower inclusive 256-bin histogram for range [0,256]
 *
 * src: CV_8UC1 image
 * histogram: CV_32SC1 1 x 256 histogram matrix
 */
void hist2u(const Mat& src, Mat& histogram){
	histogram.setTo(0);
	MatConstIterator_<uchar> s = src.begin<uchar>();
	MatConstIterator_<uchar> e = src.end<uchar>();
	int * p = (int *)histogram.data;
	for (; s!=e; s++){
		p[*s]++;
	}
}

/**
 * Calculate a standard uniform upper exclusive lower inclusive 256-bin histogram for range [0,256]
 *
 * src: CV_32FC1 image
 * histogram: CV_32SC1 1 x 256 histogram matrix
 * min: minimal considered value (inclusive)
 * max: maximum considered value (exclusive)
 */
void hist2f(const Mat& src, Mat& histogram, const float min = 0, const float max = 256){
	histogram.setTo(0);
	MatConstIterator_<float> s = src.begin<float>();
	MatConstIterator_<float> e = src.end<float>();
	int bins = histogram.rows * histogram.cols;
	float binsize = (max-min) / bins;
	int * p = (int *)histogram.data;
	for (; s!=e; s++){
		if (*s >= min) {
			if (*s < max){
				int idx = cvFloor((*s-min) / binsize);
				p[(idx < 0) ? 0 : ((idx > bins-1) ? bins-1 : idx)]++;
			}
		}
	}
}

/**
 * Computate upper exclusive lower inclusive uniform histogram quantile, i.e. quantile * 100% are
 * less than returned value, and c(1-quantile) * 100 % are greater or equal than returned value.
 *
 * histogram: CV_32SC1 1 x 256 histogram matrix
 * count: histogram member count
 * quantile:  quantile between 0 and 1
 *
 * returning quantile bin between 0 and 256
 */
int histquant2u(const Mat& histogram, const int count, const float quantile){
	int * s = (int *)histogram.data;
	int left = max(0,min(count,cvRound(quantile * count)));
	int sum = 0;
	int p = 0;
	for (; sum < left; p++) {
		sum += s[p];
	}
	if (p > 0 && (sum - left > left - sum + s[p-1])) p--;
	return p;
}

/** ------------------------------- Clahe ------------------------------- **/

/*
 * Retrieves the (bilinear) interpolated byte from 4 bytes
 *
 * x: distance to left byte
 * y: distance to right byte
 * r: distance to upper byte
 * s: distance to lower byte
 * b1: upper left byte
 * b2: upper right byte
 * b3: lower left byte
 * b4: lower right byte
 */
uchar interp(const double x, const double y, const double r, const double s, const uchar b1, const uchar b2, const uchar b3, const uchar b4) {
  double w1 = (x + y);
  double w2 = x / w1;
  w1 = y / w1;
  double w3 = (r + s);
  double w4 = r / w3;
  w3 = s / w3;
  return saturate_cast<uchar>(w3 * (w1 * b1 + w2 * b2) + w4 * (w1 * b3 + w2 * b4));
}

/*
 * Retrieves the bilinear interpolated byte from 2 bytes
 *
 * x:  distance to left byte
 * y:  distance to right byte
 * b1: left byte
 * b2: right byte
 */
uchar interp(const double x, const double y, const uchar b1, const uchar b2) {
  double w1 = (x + y);
  double w2 = x / w1;
  w1 = y / w1;
  return saturate_cast<uchar>(w1 * b1 + w2 * b2);
}

/*
 * Inplace histogram clipping according to Zuiderveld (counts excess and redistributes excess by adding the average increment)
 *
 * hist: CV_32SC1 1 x 256 histogram matrix
 * clipFactor: between 0 (maximum slope M/N, where M #pixel in window, N #bins) and 1 (maximum slope M)
 * pixelCount: number of pixels in window
 */
void clipHistogram(Mat& hist, const float clipFactor, const int pixelCount) {
	double minSlope = ((double) pixelCount) / 256;
	int clipLimit = std::min(pixelCount, std::max(1, cvCeil(minSlope + clipFactor * (pixelCount - minSlope))));
	int distributeCount = 0;
	MatIterator_<int> p = hist.begin<int>();
	MatIterator_<int> e = hist.end<int>();
	for (; p!=e; p++){
		int binsExcess = *p - clipLimit;
		if (binsExcess > 0) {
			distributeCount += binsExcess;
			*p = clipLimit;
		}
	}
	int avgInc = distributeCount / 256;
	int maxBins = clipLimit - avgInc;
	for (p = hist.begin<int>(); p!=e; p++){
		if (*p <= maxBins) {
			distributeCount -= avgInc;
			*p += avgInc;
		}
		else if (*p < clipLimit) {
			distributeCount -= (clipLimit - *p);
			*p = clipLimit;
		}
	}
	while (distributeCount > 0) {
		for (p = hist.begin<int>(); p!=e && distributeCount > 0; p++){
			if (*p < clipLimit) {
				(*p)++;
				distributeCount--;
			}
		}
	}
}

/*
 * Contrast-limited adaptive histogram equalization (supports in-place)
 *
 * src: CV_8UC1 image
 * dst: CV_8UC1 image (in-place operation is possible)
 * cellWidth: patch size in x direction (greater or equal to 2)
 * cellHeight: patch size in y direction (greater or equal to 2)
 * clipFactor: histogram clip factor between 0 and 1
 */
void clahe(const Mat& src, Mat& dst, const int cellWidth = 10, const int cellHeight = 10, const float clipFactor = 1.){
	Mat hist(1,256,CV_32SC1);
	Mat roi;
	uchar * sp, * dp;
	int height = src.rows;
	int width = src.cols;
	int gridWidth = width / cellWidth + (width % cellWidth == 0 ? 0 : 1);
	int gridHeight = height / cellHeight + (height % cellHeight == 0 ? 0 : 1);
	int bufSize = (gridWidth + 2)*256;
	int bufOffsetLeft = bufSize - 256;
	int bufOffsetTop = bufSize - gridWidth * 256;
	int bufOffsetTopLeft = bufSize - (gridWidth + 1) * 256;
	Mat buf(1, bufSize, CV_8UC1);
	MatIterator_<uchar> pbuf = buf.begin<uchar>(), ebuf = buf.end<uchar>();
	MatIterator_<int> phist, ehist = hist.end<int>();
	uchar * curr, * topleft, * top, * left;
	int pixelCount, cX, cY, cWidth, cHeight, cellOrigin, cellOffset;
	double sum;
	// process first row, first cell
	cX = 0;
	cY = 0;
	cWidth = min(cellWidth, width);
	cHeight = min(cellHeight, height);
	pixelCount = cWidth*cHeight;
	sum = 0;
	roi = Mat(src,Rect(cX,cY,cWidth,cHeight));
	hist2u(roi,hist);
	if (clipFactor < 1) clipHistogram(hist,clipFactor,pixelCount);
	// equalization
	for(phist = hist.begin<int>(); phist!=ehist; phist++, pbuf++){
		sum += *phist;
		*pbuf = saturate_cast<uchar>(sum * 255 / pixelCount);
	}
	// paint first corner cell
	cWidth = min(cellWidth / 2, cWidth);
	cHeight = min(cellHeight / 2, cHeight);
	cellOrigin = src.step * cY + cX;
	cellOffset = src.step - cWidth;
	sp = (uchar *)(src.data + cellOrigin);
	dp = (uchar *)(dst.data + cellOrigin);
	curr = buf.data;
	for (int b=0; b < cHeight; b++, sp+= cellOffset, dp += cellOffset){
	  for (int a=0; a < cWidth; a++, sp++, dp++){
		*dp = curr[*sp];
	  }
	}
	// process first row, other cells
	for (int x = 1; x < gridWidth; x++) {
		cX = x*cellWidth;
		cWidth = min(cellWidth, width - x*cellWidth);
		cHeight = min(cellHeight, height);
		pixelCount = cWidth*cHeight;
		sum = 0;
		roi.release();
		roi = Mat(src,Rect(cX,cY,cWidth,cHeight));
		hist2u(roi,hist);
		if (clipFactor < 1) clipHistogram(hist,clipFactor,pixelCount);
		// equalization
		for(phist = hist.begin<int>(); phist!=ehist; phist++, pbuf++){
			sum += *phist;
			*pbuf = saturate_cast<uchar>(sum * 255 / pixelCount);
		}
		// paint first row, other cells
		cX += cellWidth/2 - cellWidth;
		cWidth = min(cellWidth, width - x*cellWidth + cellWidth/2);
		cHeight = min(cellHeight / 2, height);
		cellOrigin = src.step * cY + cX;
		cellOffset = src.step - cWidth;
		sp = (uchar *)(src.data + cellOrigin);
		dp = (uchar *)(dst.data + cellOrigin);
		curr = buf.data + (curr - buf.data + 256) % bufSize;
		left = buf.data + (curr - buf.data + bufOffsetLeft) % bufSize;
		for (int b=0; b < cHeight; b++, sp+= cellOffset, dp += cellOffset){
		  for (int a=0; a < cWidth; a++, sp++, dp++){
			  *dp = interp(a,cWidth-a,left[*sp], curr[*sp]);
		  }
		}
	}
	// process (i.e. paint) first row, last cell (only if necessary)
	if (width % cellWidth > cellWidth / 2 || width % cellWidth == 0) {
		cWidth = (width - cellWidth / 2) % cellWidth;
		cHeight = min(cellHeight / 2, height);
		cX = width-cWidth;
		cellOrigin = src.step * cY + cX;
		cellOffset = src.step - cWidth;
		sp = (uchar *)(src.data + cellOrigin);
		dp = (uchar *)(dst.data + cellOrigin);
		for (int b=0; b < cHeight; b++, sp+= cellOffset, dp += cellOffset){
		  for (int a=0; a < cWidth; a++, sp++, dp++){
			*dp = curr[*sp];
		  }
		}
	}
	// process rest of rows
	for (int y = 1; y < gridHeight; y++) {
		// process other rows, first cell
		cX = 0;
		cY = y*cellHeight;
		cWidth = min(cellWidth, width);
		cHeight = min(cellHeight, height - y*cellHeight);
		pixelCount = cWidth*cHeight;
		sum = 0;
		roi.release();
		roi = Mat(src,Rect(cX,cY,cWidth,cHeight));
		hist2u(roi,hist);
		if (clipFactor < 1) clipHistogram(hist,clipFactor,pixelCount);
		// equalization
		if (pbuf == ebuf) pbuf = buf.begin<uchar>();
		for(phist = hist.begin<int>(); phist!=ehist; phist++, pbuf++){
			sum += *phist;
			*pbuf = saturate_cast<uchar>(sum * 255 / pixelCount);
		}
		// paint other rows, first cell
		cY += cellHeight/2 - cellHeight;
		cWidth = min(cellWidth / 2, width);
		cHeight = min(cellHeight, height - y*cellHeight + cellHeight/2);
		cellOrigin = src.step * cY + cX;
		cellOffset = src.step - cWidth;
		sp = (uchar *)(src.data + cellOrigin);
		dp = (uchar *)(dst.data + cellOrigin);
		curr = buf.data + (curr - buf.data + 256) % bufSize;
		top = buf.data + (curr - buf.data + bufOffsetTop) % bufSize;
		for (int b=0; b < cHeight; b++, sp+= cellOffset, dp += cellOffset){
		  for (int a=0; a < cWidth; a++, sp++, dp++){
			  *dp = interp(b,cHeight-b,top[*sp], curr[*sp]);
		  }
		}
		// process other rows, rest of cells
		for (int x = 1; x < gridWidth; x++) {
			cX = x*cellWidth;
			cY = y*cellHeight;
			cWidth = min(cellWidth, width - x*cellWidth);
			cHeight = min(cellHeight, height - y*cellHeight);
			pixelCount = cWidth*cHeight;
			sum = 0;
			roi.release();
			roi = Mat(src,Rect(cX,cY,cWidth,cHeight));
			hist2u(roi,hist);
			if (clipFactor < 1) clipHistogram(hist,clipFactor,pixelCount);
			// equalization
			if (pbuf == ebuf) pbuf = buf.begin<uchar>();
			for(phist = hist.begin<int>(); phist!=ehist; phist++, pbuf++){
				sum += *phist;
				*pbuf = saturate_cast<uchar>(sum * 255 / pixelCount);
			}
			// paint other rows, rest of cells
			cX += cellWidth/2 - cellWidth;
			cY += cellHeight/2 - cellHeight;
			cWidth = min(cellWidth, width - x*cellWidth + cellWidth/2);
			cHeight = min(cellHeight, height - y*cellHeight + cellHeight/2);
			cellOrigin = src.step * cY + cX;
			cellOffset = src.step - cWidth;
			sp = (uchar *)(src.data + cellOrigin);
			dp = (uchar *)(dst.data + cellOrigin);
			curr = buf.data + (curr - buf.data + 256) % bufSize;
			top = buf.data + (curr - buf.data + bufOffsetTop) % bufSize;
			topleft = buf.data + (curr - buf.data + bufOffsetTopLeft) % bufSize;
			left = buf.data + (curr - buf.data + bufOffsetLeft) % bufSize;
			for (int b=0; b < cHeight; b++, sp+= cellOffset, dp += cellOffset){
			  for (int a=0; a < cWidth; a++, sp++, dp++){
				  *dp = interp(a, cWidth-a,b,cHeight-b,topleft[*sp],top[*sp],left[*sp],curr[*sp]);
			  }
			}
		}
		// process (i.e. paint) other rows, last cell (only if necessary)
		if (width % cellWidth > cellWidth / 2 || width % cellWidth == 0) {
			cWidth = (width - cellWidth / 2) % cellWidth;
			cHeight = min(cellHeight, height - y*cellHeight + cellHeight/2);
			cX = width-cWidth;
			cellOrigin = src.step * cY + cX;
			cellOffset = src.step - cWidth;
			sp = (uchar *)(src.data + cellOrigin);
			dp = (uchar *)(dst.data + cellOrigin);
			top = buf.data + (curr - buf.data + bufOffsetTop) % bufSize;
			for (int b=0; b < cHeight; b++, sp+= cellOffset, dp += cellOffset){
			  for (int a=0; a < cWidth; a++, sp++, dp++){
				  *dp = interp(b,cHeight-b,top[*sp], curr[*sp]);
			  }
			}
		}
	}
	// process (i.e. paint) last row (only if necessary)
	if (height % cellHeight > cellHeight / 2 || height % cellHeight == 0) {
		// paint last row, first cell
		cWidth =  min(cellWidth / 2, width);
		cHeight = (height - cellHeight / 2) % cellHeight;
		cX = 0;
		cY = height-cHeight;
		cellOrigin = src.step * cY + cX;
		cellOffset = src.step - cWidth;
		sp = (uchar *)(src.data + cellOrigin);
		dp = (uchar *)(dst.data + cellOrigin);
		curr = buf.data + (curr - buf.data + bufOffsetTop + 256) % bufSize;
		for (int b=0; b < cHeight; b++, sp+= cellOffset, dp += cellOffset){
		  for (int a=0; a < cWidth; a++, sp++, dp++){
			*dp = curr[*sp];
		  }
		}
		// paint last row, other cells
		for (int x = 1; x < gridWidth; x++) {
			cX = (x-1)*cellWidth + cellWidth/2;
			cWidth = min(cellWidth, width - x*cellWidth + cellWidth/2);
			cHeight = (height - cellHeight / 2) % cellHeight;
			cellOrigin = src.step * cY + cX;
			cellOffset = src.step - cWidth;
			sp = (uchar *)(src.data + cellOrigin);
			dp = (uchar *)(dst.data + cellOrigin);
			left = curr;
			curr = buf.data + (curr - buf.data + 256) % bufSize;
			for (int b=0; b < cHeight; b++, sp+= cellOffset, dp += cellOffset){
			  for (int a=0; a < cWidth; a++, sp++, dp++){
				  *dp = interp(a,cWidth-a,left[*sp], curr[*sp]);
			  }
			}
		}
		// paint last row, last cell (only if necessary)
		if (width % cellWidth > cellWidth / 2 || width % cellWidth == 0) {
			cWidth = (width - cellWidth / 2) % cellWidth;
			cHeight = (height - cellHeight / 2) % cellHeight;
			cX = width-cWidth;
			cellOrigin = src.step * cY + cX;
			cellOffset = src.step - cWidth;
			sp = (uchar *)(src.data + cellOrigin);
			dp = (uchar *)(dst.data + cellOrigin);
			for (int b=0; b < cHeight; b++, sp+= cellOffset, dp += cellOffset){
			  for (int a=0; a < cWidth; a++, sp++, dp++){
				  *dp = curr[*sp];
			  }
			}
		}
	}
}

/** ------------------------------- Mask generation ------------------------------- **/

/*
 * Masks a region of interest within a floating point matrix
 *
 * src: CV_32FC1 matrix
 * dst: CV_32FC1 matrix
 * mask: CV_8UC1 region of interest matrix
 * onread: for any p: dst[p] := set if mask[p] = onread, otherwise dst[p] = src[p]
 * set: set value for onread in mask, see onread
 */
void maskValue(const Mat& src, Mat& dst, const Mat& mask, const uchar onread = 0, const uchar set = 0){

	MatConstIterator_<float> s = src.begin<float>();
	MatIterator_<float> d = dst.begin<float>();
	MatConstIterator_<float> e = src.end<float>();
	MatConstIterator_<uchar> r = mask.begin<uchar>();
	for (; s!=e; s++, d++, r++){
		*d = (*r == onread) ? set : *s;
	}
}

/**
 * Generates destination regions map from source image
 *
 * src: CV_8UC1 image
 * dst: CV_32SC1 regions map image (same size as src)
 * count: outputs number of regions
 */
void regionsmap(const Mat& src, Mat& dst, int& count){
	int width = src.cols;
	int height = src.rows;
	int labelsCount = 0;
	int maxRegions = ((width / 2) + 1) * ((height/2)+1)+1;
	Mat regsmap(1,maxRegions,CV_32SC1);
	int * map = (int *)regsmap.data;
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
	Mat regsremap(1,maxRegions,CV_32SC1);
	int * remap = (int *)regsremap.data;
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
}

/**
 * Filters out too large or too small binary large objects (regions) in a region map
 *
 * regmap:  CV_32SC1 regions map (use regionsmap() to calculate this object)
 * mask:    CV_8UC1 output mask with filtered regions (same size as regmap)
 * count:   number of connected components in regmap
 * minSize: only regions larger or equal than minSize are kept
 * maxSize: only regions smaller or equal than maxSize are kept
 */
void maskRegsize(const Mat& regmap, Mat& mask, const int count, const int minSize = INT_MIN, const int maxSize = INT_MAX){
	Mat regs(1,count+1,CV_32SC1);
	int * map = (int *)regs.data;
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
}

/**
 * Computes mask for reflections in image
 *
 * src: CV_8UC1 input image
 * mask: CV_8UC1 output image (same size as src)
 * roiPercent: parameter for the number of highest pixel intensities in percent
 * maxSize: maximum size of reflection region between 0 and 1
 * dilateSize: size of circular structuring element for dilate operation
 * dilateIterations: iterations of dilate operation
 */
void createReflectionMask(const Mat& src, Mat& mask, const float roiPercent = 20, const float maxSizePercent = 3, const int dilateSize = 11, const int dilateIterations = 1){
	CV_Assert(src.type() == CV_8UC1);
	CV_Assert(mask.type() == CV_8UC1);
	CV_Assert(mask.size() == src.size());
	Mat regions(mask.rows,mask.cols,CV_32SC1);
	//Mat src2(src.rows,src.cols,CV_8UC1);
	//blur(src,src2,Size(3,3));
	adaptiveThreshold(src,mask,255,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY,23,-60);
	int count = 0;
	regionsmap(mask,regions,count);
	maskRegsize(regions,mask,count,10,1000);
	Mat kernel(dilateSize,dilateSize,CV_8UC1);
	kernel.setTo(0);
	circle(kernel,Point(dilateSize/2,dilateSize/2),dilateSize/2,Scalar(255),CV_FILLED);
	dilate(mask,mask,kernel,Point(-1,-1),dilateIterations);
}

/**
 * Main eye mask selecting pupillary and limbic boundary pixels
 *
 * src: CV_8UC1 image
 * mask: CV_8UC1 mask (same size as src)
 * gradX: CV_32FC1 gradient image in x-direction
 * gradY: CV_32FC1 gradient image in y-direction
 * mag: CV_32FC1 gradient magnitude
 */
void createBoundaryMask(const Mat& src, Mat& mask, const Mat& gradX, const Mat& gradY, const Mat& mag){
	const float roiPercent = 20; // was 20
	const int histbins = 1000;
	int width = mask.cols;
	int height = mask.rows;
	int cellWidth = width/30;
	int cellHeight = height/30;
	int gridWidth = width / cellWidth + (width % cellWidth == 0 ? 0 : 1);
	int gridHeight = height / cellHeight + (height % cellHeight == 0 ? 0 : 1);
	MatConstIterator_<float> pmag = mag.begin<float>();
	MatConstIterator_<float> emag = mag.end<float>();
	float max = 0;
	for (; pmag!=emag; pmag++){
		if (*pmag > max) max = *pmag;
	}
	Mat hist(1,histbins,CV_32SC1);
	float histmax = max + max/histbins;
	hist2f(mag,hist,0,histmax);
	float minval = histquant2u(hist,width*height,(100-roiPercent)*0.01) * histmax / histbins;
	MatIterator_<uchar> smask = mask.begin<uchar>();
	pmag = mag.begin<float>();
	for (; pmag!=emag; pmag++, smask++){
		*smask = (*pmag >= minval) ? 255 : 0;
	}
	int stepx = (gradX.step/sizeof(float));
	int stepy = (gradY.step/sizeof(float));
	int stepmag = (mag.step/sizeof(float));
	for (int y = 0; y < gridHeight; y++) {
		for (int x = 0; x < gridWidth; x++) {
			int cX = x*cellWidth;
			int cY = y*cellHeight;
			int cWidth = min(cellWidth, width - x*cellWidth);
			int cHeight = min(cellHeight, height - y*cellHeight);
			float * pgradX = ((float *) (gradX.data)) + stepx*cY + cX;
			float * pgradY = ((float *) (gradY.data)) + stepy*cY + cX;
			float * pmag = ((float *) (mag.data)) + stepmag*cY + cX;
			int gradXCellOffset = stepx - cWidth;
			int gradYCellOffset = stepy - cWidth;
			int magoffset = stepmag - cWidth;
			double sumX = 0;
			double sumY = 0;
			double sumMag = 0;
			for (int b=0; b < cHeight; b++, pmag += magoffset, pgradX += gradXCellOffset, pgradY+= gradYCellOffset){
			  for (int a=0; a < cWidth; a++, pmag++, pgradX++, pgradY++){
				  if (*pmag >= minval) {
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
			bool is_significant = ((sumX * sumX + sumY * sumY) > 0.5);
			uchar * pmask = mask.data + (mask.step)*cY + cX;
			int maskoffset = mask.step - cWidth;
			for (int b=0; b < cHeight; b++, pmask += maskoffset){
			  for (int a=0; a < cWidth; a++, pmask++){
				  if (!is_significant && *pmask > 0) *pmask = 0;
			  }
			}
		}
	}
}

/** ------------------------------- Center detection ------------------------------- **/

/**
 * Type for a bi-directional ray with originating point and direction
 *
 * x: x-coordinate of origin
 * y: y-coordinate of origin
 * fx: x-direction
 * fy: y-direction
 * mag: ray weight (magnitude)
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
 * Calculates determinant of vectors (x1, y1) and (x2, y2)
 *
 * x1: first vector's x-coordinate
 * y1: first vector's x-coordinate
 * x2: second vector's y-coordinate
 * y2: second vector's y-coordinate
 *
 * returning: determinant
 */
inline float det(const float &x1, const float &y1, const float &x2, const float &y2) {
	return x1*y2 - y1*x2;
}

/**
 * Intersects two lines (x1, y1) + s*(fx1, fy1) and (x2, y2) + t*(fx2, fy2)
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
 *
 * returning: 1 if they intersect, 0 if they are parallel, -1 is they are equal
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
 * Intersects a line (x1, y1) + s*(fx1, fy1) with an axis parallel to the x-axis
 *
 * x1: x-coordinate of point on line 1
 * y1: y-coordinate of point on line 1
 * fx1: direction-vector x-coordinate of line 1
 * fy1: direction-vector y-coordinate of line 1
 * y2: y-coordinate of point on axis parallel to x-axis
 * sx: intersection point x-coordinate (sy is always equal y2)
 *
 * returning:  1 if the line intersects, 0 if it is parallel to the x-axis, -1 if it is the x-axis
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
 * Intersects a line (x1, y1) + s*(fx1, fy1) with an axis parallel to the y-axis
 *
 * x1: x-coordinate of point on line 1
 * y1: y-coordinate of point on line 1
 * fx1: direction-vector x-coordinate of line 1
 * fy1: direction-vector y-coordinate of line 1
 * x2: x-coordinate of point on axis parallel to y-axis
 * sy: intersection point x-coordinate (sx is always equal x2)
 *
 * returning:  1 if the line intersects, 0 if it is parallel to the y-axis, -1 if it is the y-axis
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
 * right: right coordinate of rectangle
 * bottom: bottom coordinate of rectangle
 * px: first intersection point x-coordinate
 * py: first intersection point y-coordinate
 * qx: first intersection point x-coordinate
 * qy: first intersection point y-coordinate
 *
 * returning:  1 if the line intersects in 2 points, 0 if it does not intersect, -1 if it corresponds to a side of the rectangle
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
 * Draws a line onto accumulator matrix using Bresenham's algorithm:
 * Increases a rectangular accumulator by adding a given value to all points on a line
 *
 * line: line to be drawn
 * accu: floating point canvas (accumulator)
 * border: outer accu boundary rectangle in user space coordinates
 *
 * returning: true, if values are added to the accu
 */
bool drawLine(const BidRay& line, Mat_<float>& accu, const float borderX, const float borderY, const float borderWidth, const float borderHeight){
	// intersect line with border
	float cellWidth = borderWidth/accu.cols;
	float cellHeight = borderHeight/accu.rows;
	float lx = borderX, ly = borderY;
	float rx = borderX+borderWidth, ry = borderY+borderHeight;
	float px, py, qx, qy;
	float incValue = line.mag / 1000;
	int accuLine = (accu.step/sizeof(float));

	int res = intersectRect(line.x,line.y,line.fx,line.fy,lx,ly,rx,ry,px,py,qx,qy);
	if (res != 0){
	  int x1 = min(max(cvRound((px-lx)/cellWidth),0),accu.cols-1);
	  int y1 = min(max(cvRound((py-ly)/cellHeight),0),accu.rows-1);
	  int x2 = min(max(cvRound((qx-lx)/cellWidth),0),accu.cols-1);
	  int y2 = min(max(cvRound((qy-ly)/cellHeight),0),accu.rows-1);
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

/**
 * Returns a gaussian 2D kernel
 * kernel: output CV_32FC1 image of specific size
 * sigma: gaussian sigma parameter
 */
void gaussianKernel(cv::Mat& kernel,float sigma = 1.4){
	CV_Assert(kernel.type() == CV_32FC1);
	CV_Assert(kernel.cols%2==1 && kernel.rows%2==1);
	float * p = (float *)kernel.data;
	int width = kernel.cols;
	int height = kernel.rows;
	int offset = kernel.step/sizeof(float) - width;
	int rx = width/2;
	int ry = height/2;
	float sqrsigma = sigma*sigma;
	for (int y=-ry,i=0;i<height;y++,i++,p+=offset){
		for (int x=-rx,j=0;j<width;x++,j++,p++){
			*p = std::exp((x*x+y*y)/(-2*sqrsigma))/(2*M_PI*sqrsigma);
		}
	}
}

/*
 * Calculates circle center in source image.
 *
 * gradX: CV_32FC1 image, gradient in x direction
 * gradY: CV_32FC1 image, gradient in y direction
 * mask: CV_8UC1 mask image to exclude wrong points for gradient extraction (same size as gradX, gradY)
 * center: center point of main circle in source image
 * accuPrecision: stop condition for accuracy of center
 * accuSize: size of the accumulator array
 */
void detectEyeCenter(const Mat& gradX, const Mat& gradY, const Mat& mag, const Mat& mask, float& centerx, float& centery, const float accuPrecision = .5, const int accuSize = 10){
	// initial declarations
	int width = mask.cols;
	int height = mask.rows;
	int accuScaledSize = (accuSize+1)/2;
	float rectX = -0.5, rectY = -0.5, rectWidth = width, rectHeight = height;
	Mat gauss(accuScaledSize,accuScaledSize,CV_32FC1);
	gaussianKernel(gauss,accuScaledSize/3);
	Mat_<float> accu(accuSize,accuSize);
	Mat_<float> accuScaled(accuScaledSize,accuScaledSize);
	// create candidates list
	list<BidRay> candidates;
	float * px = (float *)(gradX.data);
	float * py = (float *)(gradY.data);
	float * pmag = (float *)(mag.data);
	uchar * pmask = (uchar *)(mask.data);
	int xoffset = gradX.step/sizeof(float) - width;
	int yoffset = gradY.step/sizeof(float) - width;
	int magoffset = mag.step/sizeof(float) - width;
	int maskoffset = mask.step - width;

	for (int y=0; y < height; y++, px += xoffset, py += yoffset,pmask += maskoffset,pmag += magoffset){
		for (int x=0; x < width; x++, px++, py++, pmask++, pmag++){
			if (*pmask > 0){
				candidates.push_back(BidRay(x,y,*px,*py,*pmag));
			}
		}
	}
	//int tempi = 0;
	while (rectWidth > accuPrecision || rectHeight > accuPrecision){
		accu.setTo(0);
		bool isIn = true;
		if (candidates.size() > 0){
			for (list<BidRay>::iterator it = candidates.begin(); it != candidates.end();(isIn) ? ++it : it = candidates.erase(it)){
				isIn = drawLine(*it,accu,rectX,rectY,rectWidth,rectHeight);
			}
		}
		pyrDown(accu,accuScaled);
		multiply(accuScaled,gauss,accuScaled,1,CV_32FC1);

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
		rectX += (((maxCellX + 0.5) * rectWidth) / accuScaledSize) - (rectWidth * 0.25); //std::min( accuRect.width * 0.5,((maxCellX * accuRect.width + 0.5) / accuScaledSize) - (accuRect.width * 0.25));
		rectY += (((maxCellY + 0.5) * rectHeight) / accuScaledSize) - (rectHeight * 0.25); //std::min( accuRect.height * 0.5,((maxCellY * accuRect.height + 0.5) / accuScaledSize) - (accuRect.height * 0.25));
		rectWidth /= 2;
		rectHeight /= 2;

	}
	centerx = rectX + rectWidth / 2;
	centery = rectY + rectHeight / 2;
}

/** ------------------------------- Rubbersheet transform ------------------------------- **/

/*
 * interpolation mode for rubbersheet repeating the last pixel for a given angle if no values are available
 * (otherwise behaves like INTER_LINEAR)
 */
static const int INTER_LINEAR_REPEAT = 82;

/*
 * Calculates the mapped (polar) image of source using two transformation contours.
 *
 * src: CV_8U (cartesian) source image (possibly multi-channel)
 * dst:	CV_8U (polar) destination image (possibly multi-channel, 2 times the col size of inner)
 * inner: CV_32FC2 inner cartesian coordinates
 * outer: CV_32FC2 outer cartesian coordinates
 * interpolation: interpolation mode (INTER_NEAREST, INTER_LINEAR or INTER_LINEAR_REPEAT)
 * fill: fill value for pixels out of the image
 */
void rubbersheet(const Mat& src, Mat& dst, const Mat& inner, const Mat& outer, const int interpolation = INTER_LINEAR, const uchar fill = 0) {
	int nChannels = src.channels();
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
				float a = *pinner + r * (*pouter - *pinner);
				pinner++; pouter++;
				float b =  *pinner + r * (*pouter - *pinner);
				int coordX = cvRound(a);
				int coordY = cvRound(b);
				if (coordX < 0 || coordY < 0 || coordX >= srcwidth || coordY >= srcheight){
					for (int i=0; i< nChannels; i++,pdst++){
						*pdst = fill;
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
	else if (interpolation == INTER_LINEAR){
		for (int y=0; y < dstheight; y++, pdst+= dstoffset, r+= roffset){
			float * pinner = (float *) inner.data;
			float * pouter = (float *) outer.data;
			for (int x=0; x < dstwidth; x++, pinner++, pouter++){
				float a = *pinner + r * (*pouter - *pinner);
				pinner++; pouter++;
				float b =  *pinner + r * (*pouter - *pinner);
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
							else if (coordY == srcheight-1){ // bottom out
								float dx = a-coordX;
								float dy = b-coordY;
								uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
								for (int i=0; i< nChannels; i++,pdst++,psrc++){
									*pdst = saturate_cast<uchar>((1-dy)*((1-dx)*((float)(*psrc)) + dx*(float)(psrc[nChannels])) + dy * ((float) fill));
								}
							}
							else {
								for (int i=0; i< nChannels; i++,pdst++){
									*pdst = fill;
								}
							}
						}
						else if (coordX == srcwidth-1){
							if (coordY < srcheight-1){// right out
								float dx = a-coordX;
								float dy = b-coordY;
								uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
								for (int i=0; i< nChannels; i++,pdst++,psrc++){
									*pdst = saturate_cast<uchar>((1-dx)*((1-dy)*((float)(*psrc))+ dy*((float)(psrc[srcstep]))) + dx * ((float) fill));
								}
							}
							else if (coordY == srcheight-1){ // bottom right out
								float dx = a-coordX;
								float dy = b-coordY;
								uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
								for (int i=0; i< nChannels; i++,pdst++,psrc++){
									*pdst = saturate_cast<uchar>((1-dy)*((1-dx)*((float)(*psrc)) + dx * ((float)fill)) + dy * ((float) fill));
								}
							}
							else {
								for (int i=0; i< nChannels; i++,pdst++){
									*pdst = fill;
								}
							}
						}
						else {
							for (int i=0; i< nChannels; i++,pdst++){
								*pdst = fill;
							}
						}
					}
					else if (coordY == -1){
						if (coordX < srcwidth-1){// top out
								float dx = a-coordX;
								float dy = b-coordY;
								uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
								for (int i=0; i< nChannels; i++,pdst++,psrc++){
									*pdst = saturate_cast<uchar>((dy)*((1-dx)*((float)(psrc[srcstep])) + dx*(float)(psrc[nChannels+srcstep])) + (1-dy) * ((float) fill));
								}
						}
						else if (coordX == srcwidth-1){// top right out
								float dx = a-coordX;
								float dy = b-coordY;
								uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
								for (int i=0; i< nChannels; i++,pdst++,psrc++){
									*pdst = saturate_cast<uchar>((dy)*((1-dx)*((float)(psrc[srcstep])) + dx * ((float)fill)) + (1-dy) * ((float) fill));
								}
						}
						else {
							for (int i=0; i< nChannels; i++,pdst++){
								*pdst = fill;
							}
						}
					}
					else {
						for (int i=0; i< nChannels; i++,pdst++){
							*pdst = fill;
						}
					}
				}
				else if (coordX == -1){
					if (coordY >= 0){
						if (coordY < srcheight-1){// left out
							float dx = a-coordX;
							float dy = b-coordY;
							uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
							for (int i=0; i< nChannels; i++,pdst++,psrc++){
								*pdst = saturate_cast<uchar>(dx*((1-dy)*(float)(psrc[nChannels]) + dy*(float)(psrc[nChannels+srcstep])) + (1-dx) * ((float) fill));
							}
						}
						else if (coordY == srcheight-1){ // left bottom out
							float dx = a-coordX;
							float dy = b-coordY;
							uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
							for (int i=0; i< nChannels; i++,pdst++,psrc++){
								*pdst = saturate_cast<uchar>((1-dy)*((dx)*((float)(psrc[nChannels])) + (1-dx) * ((float) fill)) + dy * ((float) fill));
							}
						}
						else {
							for (int i=0; i< nChannels; i++,pdst++){
								*pdst = fill;
							}
						}
					}
					else if (coordY == -1){ // left top out
						float dx = a-coordX;
						float dy = b-coordY;
						uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
						for (int i=0; i< nChannels; i++,pdst++,psrc++){
							*pdst = saturate_cast<uchar>((dy)*((dx)*((float)(psrc[nChannels+srcstep])) + (1-dx) * ((float) fill)) + (1-dy) * ((float) fill));
						}
					}
					else {
						for (int i=0; i< nChannels; i++,pdst++){
							*pdst = fill;
						}
					}
				}
				else {
					for (int i=0; i< nChannels; i++,pdst++){
						*pdst = fill;
					}
				}
			}
		}
	}
	else { // INTER_LINEAR_REPEAT (repeats last pixel value)
		uchar * firstLine = pdst + dst.step;
		int step = dst.step;
		for (int y=0; y < dstheight; y++, pdst+= dstoffset, r+= roffset){
			float * pinner = (float *) inner.data;
			float * pouter = (float *) outer.data;
			for (int x=0; x < dstwidth; x++, pinner++, pouter++){
				float a = *pinner + r * (*pouter - *pinner);
				pinner++; pouter++;
				float b =  *pinner + r * (*pouter - *pinner);
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
							else if (coordY == srcheight-1){ // bottom out
								float dx = a-coordX;
								uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
								for (int i=0; i< nChannels; i++,pdst++,psrc++){
									*pdst = saturate_cast<uchar>(((1-dx)*((float)(*psrc)) + dx*(float)(psrc[nChannels])));
								}
							}
							else if (pdst >= firstLine){
								for (int i=0; i< nChannels; i++,pdst++){
									*pdst = *(pdst - step); // one row above
								}
							}
							else {
								for (int i=0; i< nChannels; i++,pdst++){
									*pdst = fill;
								}
							}
						}
						else if (coordX == srcwidth-1){
							if (coordY < srcheight-1){// right out
								float dy = b-coordY;
								uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
								for (int i=0; i< nChannels; i++,pdst++,psrc++){
									*pdst = saturate_cast<uchar>(((1-dy)*((float)(*psrc))+ dy*((float)(psrc[srcstep]))));
								}
							}
							else if (coordY == srcheight-1){ // bottom right out
								uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
								for (int i=0; i< nChannels; i++,pdst++,psrc++){
									*pdst = *psrc;
								}
							}
							else if (pdst >= firstLine){
								for (int i=0; i< nChannels; i++,pdst++){
									*pdst = *(pdst - step); // one row above
								}
							}
							else {
								for (int i=0; i< nChannels; i++,pdst++){
									*pdst = fill;
								}
							}
						}
						else {
							for (int i=0; i< nChannels; i++,pdst++){
								*pdst = *(pdst - step); // one row above
							}
						}
					}
					else if (coordY == -1){
						if (coordX < srcwidth-1){// top out
								float dx = a-coordX;
								uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
								for (int i=0; i< nChannels; i++,pdst++,psrc++){
									*pdst = saturate_cast<uchar>(((1-dx)*((float)(psrc[srcstep])) + dx*(float)(psrc[nChannels+srcstep])));
								}
						}
						else if (coordX == srcwidth-1){// top right out
								uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
								for (int i=0; i< nChannels; i++,pdst++,psrc++){
									*pdst = psrc[srcstep];
								}
						}
						else if (pdst >= firstLine){
							for (int i=0; i< nChannels; i++,pdst++){
								*pdst = *(pdst - step); // one row above
							}
						}
						else {
							for (int i=0; i< nChannels; i++,pdst++){
								*pdst = fill;
							}
						}
					}
					else if (pdst >= firstLine){
						for (int i=0; i< nChannels; i++,pdst++){
							*pdst = *(pdst - step); // one row above
						}
					}
					else {
						for (int i=0; i< nChannels; i++,pdst++){
							*pdst = fill;
						}
					}
				}
				else if (coordX == -1){
					if (coordY >= 0){
						if (coordY < srcheight-1){// left out
							float dy = b-coordY;
							uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
							for (int i=0; i< nChannels; i++,pdst++,psrc++){
								*pdst = saturate_cast<uchar>(((1-dy)*(float)(psrc[nChannels]) + dy*(float)(psrc[nChannels+srcstep])));
							}
						}
						else if (coordY == srcheight-1){ // bottom left out
							uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
							for (int i=0; i< nChannels; i++,pdst++,psrc++){
								*pdst = psrc[nChannels];
							}
						}
						else if (pdst >= firstLine){
							for (int i=0; i< nChannels; i++,pdst++){
								*pdst = *(pdst - step); // one row above
							}
						}
						else {
							for (int i=0; i< nChannels; i++,pdst++){
								*pdst = fill;
							}
						}
					}
					else if (coordY == -1){ // top left out
						uchar * psrc = (uchar*) (src.data+coordY*srcstep+coordX*nChannels);
						for (int i=0; i< nChannels; i++,pdst++,psrc++){
							*pdst = psrc[nChannels+srcstep];
						}
					}
					else if (pdst >= firstLine){
						for (int i=0; i< nChannels; i++,pdst++){
							*pdst = *(pdst - step); // one row above
						}
					}
					else {
						for (int i=0; i< nChannels; i++,pdst++){
							*pdst = fill;
						}
					}
				}
				else if (pdst >= firstLine){
					for (int i=0; i< nChannels; i++,pdst++){
						*pdst = *(pdst - step); // one row above
					}
				}
				else {
					for (int i=0; i< nChannels; i++,pdst++){
						*pdst = fill;
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
 * src: CV_8UC1 (cartesian) source image
 * dst: CV_8UC1 (polar) destination image (same size as src)
 * ellipse: unit ellipse located in the source image (size equals stretching coefficients)
 * radius: radius in pixels of the source image (to map whole image, this should be the maximum of distances between origin and corners)
 * interpolation: interpolation mode (INTER_NEAREST, INTER_LINEAR or INTER_LINEAR_REPEAT)
 * fill: fill value for pixels out of the image
 *
 * returning: polar resolution
 */
float ellipsopolarTransform(const Mat& src, Mat& dst, const RotatedRect& ellipse, const float radius = -1, const int interpolation = INTER_LINEAR, const uchar fill = 0) {
	// first: translate point to origin
	// then: rotate points against ellipse angle
	// then: scale points' axes
	// finally: polar transform
	int dstheight = dst.rows;
	int dstwidth = dst.cols;
	int srcheight = src.rows;
	int srcwidth = src.cols;
	float rad = radius;
	float centerX = ellipse.center.x;
	float centerY = ellipse.center.y;
	float ellA = ellipse.size.width/2;
	float ellB = ellipse.size.height/2;
	double alpha = (ellipse.angle)*M_PI/180; // inclination angle of the ellipse
	if (alpha > M_PI) alpha -= M_PI; // normalize alpha to [0,M_PI]
	double omega = 2 * M_PI - alpha; // angle for reverse transformation
	const double cosAlpha = cos(alpha);
	const double sinAlpha = sin(alpha);
	const double cosOmega = cos(omega);
	const double sinOmega = sin(omega);
	if (rad < 0){
		const float x1 = cosOmega * (-centerX) - sinOmega * (-centerY); // center x-coordinate in ellipse coords
		const float y1 = sinOmega * (-centerX) + cosOmega * (-centerY); // center x-coordinate in ellipse coords
		const float x2 = cosOmega * (srcwidth-centerX) - sinOmega * (-centerY); // center x-coordinate in ellipse coords
		const float y2 = sinOmega * (srcwidth-centerX) + cosOmega * (-centerY); // center x-coordinate in ellipse coords
		const float x3 = cosOmega * (srcwidth-centerX) - sinOmega * (srcheight-centerY); // center x-coordinate in ellipse coords
		const float y3 = sinOmega * (srcwidth-centerX) + cosOmega * (srcheight-centerY); // center x-coordinate in ellipse coords
		const float x4 = cosOmega * (-centerX) - sinOmega * (srcheight-centerY); // center x-coordinate in ellipse coords
		const float y4 = sinOmega * (-centerX) + cosOmega * (srcheight-centerY); // center x-coordinate in ellipse coords
		const float ellASquare = ellA*ellA;
		const float ellBSquare = ellB*ellB;
		rad = max(max(sqrt(x1*x1/ellASquare+y1*y1/ellBSquare),sqrt(x2*x2/ellASquare+y2*y2/ellBSquare)),max(sqrt(x3*x3/ellASquare+y3*y3/ellBSquare),sqrt(x4*x4/ellASquare+y4*y4/ellBSquare)));
	}
	uchar * pdst = dst.data;
	uchar * psrc = src.data;
	int dstoffset = dst.step - dstwidth;
	int srcstep = src.step;
	float roffset = rad/(dstheight-1);
	float r = 0;
	float thetaoffset = 2.f * M_PI / dstwidth;
	if (interpolation == INTER_NEAREST){
		for (int y=0; y < dstheight; y++, pdst+= dstoffset, r+= roffset){
			float theta = 0;
			for (int x=0; x < dstwidth; x++, theta += thetaoffset,pdst++){
				float beta = (alpha <= theta) ? theta - alpha : 2 * M_PI + theta - alpha; // angle of polar ray in ellipse coords (alpha + beta = theta)
				float s = r * ellA * cos(beta), t = r * ellB * sin(beta);
				float a = centerX + cosAlpha * s - sinAlpha * t, b = centerY + sinAlpha * s + cosAlpha * t;
				int coordX = cvRound(a);
				int coordY = cvRound(b);
				if (coordX < 0 || coordY < 0 || coordX >= srcwidth || coordY >= srcheight){
					*pdst = fill;
				}
				else {
					*pdst = psrc[coordY*srcstep+coordX];
				}
			}
		}
	}
	else if (interpolation == INTER_LINEAR){
		for (int y=0; y < dstheight; y++, pdst+= dstoffset, r+= roffset){
			float theta = 0;
			for (int x=0; x < dstwidth; x++, theta += thetaoffset,pdst++){
				float beta = (alpha <= theta) ? theta - alpha : 2 * M_PI + theta - alpha; // angle of polar ray in ellipse coords (alpha + beta = theta)
				float s = r * ellA * cos(beta), t = r * ellB * sin(beta);
				float a = centerX + cosAlpha * s - sinAlpha * t, b = centerY + sinAlpha * s + cosAlpha * t;
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
							else if (coordY == srcheight-1){ // bottom out
								float dx = a-coordX;
								float dy = b-coordY;
								int offset = coordY*srcstep+coordX;
								*pdst = saturate_cast<uchar>((1-dy)*((1-dx)*((float)(psrc[offset])) + dx*(float)(psrc[offset+1])) + dy * ((float)fill));
							}
							else {
								*pdst = fill;
							}
						}
						else if (coordX == srcwidth-1){
							if (coordY < srcheight-1){// right out
								float dx = a-coordX;
								float dy = b-coordY;
								int offset = coordY*srcstep+coordX;
								*pdst = saturate_cast<uchar>((1-dx)*((1-dy)*((float)(psrc[offset]))+ dy*((float)(psrc[offset+srcstep]))) + dx * ((float)fill));
							}
							else if (coordY == srcheight-1){ // bottom right out
								float dx = a-coordX;
								float dy = b-coordY;
								int offset = coordY*srcstep+coordX;
								*pdst = saturate_cast<uchar>((1-dy)*((1-dx)*((float)(psrc[offset])) + dx * ((float) fill)) + dy * ((float) fill));
							}
							else {
								*pdst = fill;
							}
						}
						else {
							*pdst = fill;
						}
					}
					else if (coordY == -1){
						if (coordX < srcwidth-1){// top out
							float dx = a-coordX;
							float dy = b-coordY;
							int offset = coordY*srcstep+coordX;
							*pdst = saturate_cast<uchar>((dy)*((1-dx)*((float)(psrc[offset+srcstep])) + dx*(float)(psrc[offset+1+srcstep])) + (1-dy) * ((float) fill));

						}
						else if (coordX == srcwidth-1){// top right out
							float dx = a-coordX;
							float dy = b-coordY;
							int offset = coordY*srcstep+coordX;
							*pdst = saturate_cast<uchar>((dy)*((1-dx)*((float)(psrc[offset+srcstep])) + dx * ((float)fill)) + (1-dy) * ((float) fill));
						}
						else {
							*pdst = fill;
						}
					}
					else {
						*pdst = fill;
					}
				}
				else if (coordX == -1){
					if (coordY >= 0){
						if (coordY < srcheight-1){// left out
							float dx = a-coordX;
							float dy = b-coordY;
							int offset = coordY*srcstep+coordX;
							*pdst = saturate_cast<uchar>(dx*((1-dy)*(float)(psrc[offset+1]) + dy*(float)(psrc[offset+1+srcstep])) + (1-dx) * ((float) fill));
						}
						else if (coordY == srcheight-1){ // bottom left out
							float dx = a-coordX;
							float dy = b-coordY;
							int offset = coordY*srcstep+coordX;
							*pdst = saturate_cast<uchar>((1-dy)*((dx)*((float)(psrc[offset+1])) + (1-dx) * ((float) fill)) + dy * ((float) fill));
						}
						else {
							*pdst = fill;
						}
					}
					else if (coordY == -1){ // top left out
						float dx = a-coordX;
						float dy = b-coordY;
						int offset = coordY*srcstep+coordX;
						*pdst = saturate_cast<uchar>((dy)*((dx)*((float)(psrc[offset+1+srcstep])) + (1-dx) * ((float) fill)) + (1-dy) * ((float)fill));
					}
					else {
						*pdst = fill;
					}
				}
				else {
					*pdst = fill;
				}
			}
		}
	}
	else { // INTER_LINEAR_REPEAT (repeats last pixel value)
		uchar * firstLine = pdst + dst.step;
		for (int y=0; y < dstheight; y++, pdst+= dstoffset, r+= roffset){
			float theta = 0;
			for (int x=0; x < dstwidth; x++, theta += thetaoffset,pdst++){
				float beta = (alpha <= theta) ? theta - alpha : 2 * M_PI + theta - alpha; // angle of polar ray in ellipse coords (alpha + beta = theta)
				float s = r * ellA * cos(beta), t = r * ellB * sin(beta);
				float a = centerX + cosAlpha * s - sinAlpha * t, b = centerY + sinAlpha * s + cosAlpha * t;
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
							else if (coordY == srcheight-1){ // bottom out
								float dx = a-coordX;
								int offset = coordY*srcstep+coordX;
								*pdst = saturate_cast<uchar>(((1-dx)*((float)(psrc[offset])) + dx*(float)(psrc[offset+1])));
							}
							else if (pdst >= firstLine){
								*pdst = *(pdst - dst.step); // one row above;
							}
							else {
								*pdst = fill;
							}
						}
						else if (coordX == srcwidth-1){
							if (coordY < srcheight-1){// right out
								float dy = b-coordY;
								int offset = coordY*srcstep+coordX;
								*pdst = saturate_cast<uchar>(((1-dy)*((float)(psrc[offset]))+ dy*((float)(psrc[offset+srcstep]))));
							}
							else if (coordY == srcheight-1){ // bottom right out
								int offset = coordY*srcstep+coordX;
								*pdst = psrc[offset];
							}
							else if (pdst >= firstLine){
								*pdst = *(pdst - dst.step); // one row above;
							}
							else {
								*pdst = fill;
							}
						}
						else {
							*pdst = *(pdst - dst.step); // one row above;
						}
					}
					else if (coordY == -1){
						if (coordX < srcwidth-1){// top out
							float dx = a-coordX;
							int offset = coordY*srcstep+coordX;
							*pdst = saturate_cast<uchar>(((1-dx)*((float)(psrc[offset+srcstep])) + dx*(float)(psrc[offset+1+srcstep])));

						}
						else if (coordX == srcwidth-1){// top right out
							int offset = coordY*srcstep+coordX;
							*pdst = psrc[offset+srcstep];
						}
						else if (pdst >= firstLine){
							*pdst = *(pdst - dst.step);
						}
						else {
							*pdst = fill; // one row above;
						}
					}
					else {
						*pdst = *(pdst - dst.step); // one row above;
					}
				}
				else if (coordX == -1){
					if (coordY >= 0){
						if (coordY < srcheight-1){// left out
							float dy = b-coordY;
							int offset = coordY*srcstep+coordX;
							*pdst = saturate_cast<uchar>(((1-dy)*(float)(psrc[offset+1]) + dy*(float)(psrc[offset+1+srcstep])));
						}
						else if (coordY == srcheight-1){ // bottom left out
							int offset = coordY*srcstep+coordX;
							*pdst = psrc[offset+1];
						}
						else if (pdst >= firstLine){
							*pdst = *(pdst - dst.step); // one row above;
						}
						else {
							*pdst = fill;
						}
					}
					else if (coordY == -1){ // top left out
						int offset = coordY*srcstep+coordX;
						*pdst = psrc[offset+1+srcstep];
					}
					else if (pdst >= firstLine){
						*pdst = *(pdst - dst.step); // one row above;
					}
					else {
						*pdst = fill;
					}
				}
				else if (pdst >= firstLine){
					*pdst = *(pdst - dst.step); // one row above;
				}
				else {
					*pdst = fill;
				}
			}
		}
	}
	return roffset;
}

/*
 * Calculates the mapped (polar) image of source using transformation center (polar origin) and radius.
 *
 * src: CV_8UC1 (cartesian) source image
 * dst: CV_8UC1 (polar) destination image (same size as src)
 * centerX: x-coordinate of polar origin in (floating point) pixels of the source image
 * centerY: y-coordinate of polar origin in (floating point) pixels of the source image
 * radius: radius in pixels of the source image (to map whole image, this should be the maximum of distances between origin and corners)
 * interpolation: interpolation mode (INTER_NEAREST, INTER_LINEAR or INTER_LINEAR_REPEAT)
 * fill: fill value for pixels out of the image
 *
 * returning: polar resolution
 */
float polarTransform(const Mat& src, Mat& dst, const float centerX = 0, const float centerY = 0, const float radius = -1, const int interpolation = INTER_LINEAR, const uchar fill = 0) {
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
	float roffset = rad/(dstheight-1);
	float r = 0;
	float thetaoffset = 2.f * M_PI / dstwidth;
	if (interpolation == INTER_NEAREST){
		for (int y=0; y < dstheight; y++, pdst+= dstoffset, r+= roffset){
			float theta = 0;
			for (int x=0; x < dstwidth; x++, theta += thetaoffset,pdst++){
				float a = centerX + r * cos(theta);//std::cos(theta);
				float b = centerY + r * sin(theta);//std::sin(theta);
				int coordX = cvRound(a);
				int coordY = cvRound(b);
				if (coordX < 0 || coordY < 0 || coordX >= srcwidth || coordY >= srcheight){
					*pdst = fill;
				}
				else {
					*pdst = psrc[coordY*srcstep+coordX];
				}
			}
		}
	}
	else if (interpolation == INTER_LINEAR){
		for (int y=0; y < dstheight; y++, pdst+= dstoffset, r+= roffset){
			float theta = 0;
			for (int x=0; x < dstwidth; x++, theta += thetaoffset,pdst++){
				float a = centerX + r * cos(theta);//std::cos(theta);
				float b = centerY + r * sin(theta);//std::sin(theta);
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
							else if (coordY == srcheight-1){ // bottom out
								float dx = a-coordX;
								float dy = b-coordY;
								int offset = coordY*srcstep+coordX;
								*pdst = saturate_cast<uchar>((1-dy)*((1-dx)*((float)(psrc[offset])) + dx*(float)(psrc[offset+1])) + dy * ((float)fill));
							}
							else {
								*pdst = fill;
							}
						}
						else if (coordX == srcwidth-1){
							if (coordY < srcheight-1){// right out
								float dx = a-coordX;
								float dy = b-coordY;
								int offset = coordY*srcstep+coordX;
								*pdst = saturate_cast<uchar>((1-dx)*((1-dy)*((float)(psrc[offset]))+ dy*((float)(psrc[offset+srcstep]))) + dx * ((float)fill));
							}
							else if (coordY == srcheight-1){ // bottom right out
								float dx = a-coordX;
								float dy = b-coordY;
								int offset = coordY*srcstep+coordX;
								*pdst = saturate_cast<uchar>((1-dy)*((1-dx)*((float)(psrc[offset])) + dx * ((float) fill)) + dy * ((float) fill));
							}
							else {
								*pdst = fill;
							}
						}
						else {
							*pdst = fill;
						}
					}
					else if (coordY == -1){
						if (coordX < srcwidth-1){// top out
							float dx = a-coordX;
							float dy = b-coordY;
							int offset = coordY*srcstep+coordX;
							*pdst = saturate_cast<uchar>((dy)*((1-dx)*((float)(psrc[offset+srcstep])) + dx*(float)(psrc[offset+1+srcstep])) + (1-dy) * ((float) fill));

						}
						else if (coordX == srcwidth-1){// top right out
							float dx = a-coordX;
							float dy = b-coordY;
							int offset = coordY*srcstep+coordX;
							*pdst = saturate_cast<uchar>((dy)*((1-dx)*((float)(psrc[offset+srcstep])) + dx * ((float)fill)) + (1-dy) * ((float) fill));
						}
						else {
							*pdst = fill;
						}
					}
					else {
						*pdst = fill;
					}
				}
				else if (coordX == -1){
					if (coordY >= 0){
						if (coordY < srcheight-1){// left out
							float dx = a-coordX;
							float dy = b-coordY;
							int offset = coordY*srcstep+coordX;
							*pdst = saturate_cast<uchar>(dx*((1-dy)*(float)(psrc[offset+1]) + dy*(float)(psrc[offset+1+srcstep])) + (1-dx) * ((float) fill));
						}
						else if (coordY == srcheight-1){ // bottom left out
							float dx = a-coordX;
							float dy = b-coordY;
							int offset = coordY*srcstep+coordX;
							*pdst = saturate_cast<uchar>((1-dy)*((dx)*((float)(psrc[offset+1])) + (1-dx) * ((float) fill)) + dy * ((float) fill));
						}
						else {
							*pdst = fill;
						}
					}
					else if (coordY == -1){ // top left out
						float dx = a-coordX;
						float dy = b-coordY;
						int offset = coordY*srcstep+coordX;
						*pdst = saturate_cast<uchar>((dy)*((dx)*((float)(psrc[offset+1+srcstep])) + (1-dx) * ((float) fill)) + (1-dy) * ((float)fill));
					}
					else {
						*pdst = fill;
					}
				}
				else {
					*pdst = fill;
				}
			}
		}
	}
	else { // INTER_LINEAR_REPEAT (repeats last pixel value)
		uchar * firstLine = pdst + dst.step;
		for (int y=0; y < dstheight; y++, pdst+= dstoffset, r+= roffset){
			float theta = 0;
			for (int x=0; x < dstwidth; x++, theta += thetaoffset,pdst++){
				float a = centerX + r * cos(theta);//std::cos(theta);
				float b = centerY + r * sin(theta);//std::sin(theta);
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
							else if (coordY == srcheight-1){ // bottom out
								float dx = a-coordX;
								int offset = coordY*srcstep+coordX;
								*pdst = saturate_cast<uchar>(((1-dx)*((float)(psrc[offset])) + dx*(float)(psrc[offset+1])));
							}
							else if (pdst >= firstLine){
								*pdst = *(pdst - dst.step); // one row above;
							}
							else {
								*pdst = fill;
							}
						}
						else if (coordX == srcwidth-1){
							if (coordY < srcheight-1){// right out
								float dy = b-coordY;
								int offset = coordY*srcstep+coordX;
								*pdst = saturate_cast<uchar>(((1-dy)*((float)(psrc[offset]))+ dy*((float)(psrc[offset+srcstep]))));
							}
							else if (coordY == srcheight-1){ // bottom right out
								int offset = coordY*srcstep+coordX;
								*pdst = psrc[offset];
							}
							else if (pdst >= firstLine){
								*pdst = *(pdst - dst.step); // one row above;
							}
							else {
								*pdst = fill;
							}
						}
						else {
							*pdst = *(pdst - dst.step); // one row above;
						}
					}
					else if (coordY == -1){
						if (coordX < srcwidth-1){// top out
							float dx = a-coordX;
							int offset = coordY*srcstep+coordX;
							*pdst = saturate_cast<uchar>(((1-dx)*((float)(psrc[offset+srcstep])) + dx*(float)(psrc[offset+1+srcstep])));

						}
						else if (coordX == srcwidth-1){// top right out
							int offset = coordY*srcstep+coordX;
							*pdst = psrc[offset+srcstep];
						}
						else if (pdst >= firstLine){
							*pdst = *(pdst - dst.step);
						}
						else {
							*pdst = fill; // one row above;
						}
					}
					else {
						*pdst = *(pdst - dst.step); // one row above;
					}
				}
				else if (coordX == -1){
					if (coordY >= 0){
						if (coordY < srcheight-1){// left out
							float dy = b-coordY;
							int offset = coordY*srcstep+coordX;
							*pdst = saturate_cast<uchar>(((1-dy)*(float)(psrc[offset+1]) + dy*(float)(psrc[offset+1+srcstep])));
						}
						else if (coordY == srcheight-1){ // bottom left out
							int offset = coordY*srcstep+coordX;
							*pdst = psrc[offset+1];
						}
						else if (pdst >= firstLine){
							*pdst = *(pdst - dst.step); // one row above;
						}
						else {
							*pdst = fill;
						}
					}
					else if (coordY == -1){ // top left out
						int offset = coordY*srcstep+coordX;
						*pdst = psrc[offset+1+srcstep];
					}
					else if (pdst >= firstLine){
						*pdst = *(pdst - dst.step); // one row above;
					}
					else {
						*pdst = fill;
					}
				}
				else if (pdst >= firstLine){
					*pdst = *(pdst - dst.step); // one row above;
				}
				else {
					*pdst = fill;
				}
			}
		}
	}
	return roffset;
}

/**
 * Constructs a gabor 2D kernel
 *
 * kernel: CV_32FC1 image of specific size (kernel.cols%2 and kernel.rows%2 should be 1)
 * lambda: wavelength of cosine factor
 * theta: orientation of normal to parallel stripes
 * psi: phase offset
 * sigma: gaussian sigma parameter
 * gamma: spatial aspect ratio
 *
 * returning: gabor kernel
 */
void gaborKernel(Mat& kernel,const float lambda,const float theta,const float psi,const float sigma,const float gamma){
	MatIterator_<float> p = kernel.begin<float>();
	int width = kernel.cols;
	int height = kernel.rows;
	int rx = width/2;
	int ry = height/2;
	float sigma_x = sigma;
	float sigma_y = sigma/gamma;
	for (int y=-ry,i=0;i<height;y++,i++){
		for (int x=-rx,j=0;j<width;x++,j++,p++){
			float x_theta = x*std::cos(theta)+y*std::sin(theta);
			float y_theta =-x*std::sin(theta)+y*std::cos(theta);
			*p = 1 / (2*M_PI*sigma_x*sigma_y) * std::exp(-.5f*((x_theta*x_theta)/(sigma_x*sigma_x) + (y_theta*y_theta)/(sigma_y*sigma_y)))*std::cos(2*M_PI/lambda*x_theta+psi);
		}
	}
}

/**
 * Normalizes a kernel, such that the area under the kernel is 1
 *
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
 *
 * kernel: output CV_32FC1 image of specific size
 */
void kernelNormPeak(Mat& kernel){
	float peak = 0;
	MatIterator_<float> it;
	for (it = kernel.begin<float>(); it < kernel.end<float>(); it++){
		float val = std::abs(*it);
		if (val > peak) peak = val;
	}
	if (peak != 0){
		for (it = kernel.begin<float>(); it < kernel.end<float>(); it++){
			*it /= peak;
		}
	}
}

/**
 * Interpolates an array by linearly filling all values less than zero
 * arr: CV_32FC1 array to be normalized
 */
void interpolateArr(Mat& arr){
	float *c = (float *) arr.data;
	float *p = (float *) arr.data;
	int width = arr.cols;
	int pos = width-1;
	// repair beginning, if needed
	p += pos;
	while (pos > 0 && *p < 0) {p--; pos--;} // find previous intact point
	if (pos == 0) return; // nothing, we can do
	int startx = pos;
	float starty = c[pos];
	pos = 0;
	p = c;
	while (pos < width && *p < 0) {p++; pos++;} // find next intact point
	int endx = pos;
	float endy = c[pos];
	int diffx1 = width - startx;
	int diffx = diffx1 + endx;
	if (diffx > 1){ // repairing needed
		p = (c + startx + 1);
		float delta = (endy - starty) / diffx;
		for (int i=1;i<diffx1;i++,p++){
			*p = starty + i * delta;
		}
		p = c;
		for (int i=diffx1; i<diffx; i++, p++){
			*p = starty + i * delta;
		}
	}
	startx = endx;
	starty = endy;
	p = c + startx + 1;
	pos = startx + 1;
	while (pos < width && *p >= 0) {p++; pos++;} // find next damaged point
	if (pos == width) return; // nothing more, we can do
	else { // intact point is the point before
		startx = pos - 1;
		starty = c[pos-1];
	}
	pos++;p++;
	while (pos < width){
		// find next intact point and interpolate
		while (pos < width && *p < 0) {p++; pos++;} // find next intact point
		endx = pos;
		endy = c[pos];
		diffx = endx-startx;
		float delta = (endy - starty) / diffx;
		p = (c + startx + 1);
		for (int i=1; i<diffx; i++, p++){
			*p = starty + i * delta;
		}
		startx = endx;
		starty = endy;
		p = c + startx + 1;
		pos = startx + 1;
		// and then look for next non-intact point
		while (pos < width && *p >= 0) {p++; pos++;} // find next damaged point
	}
}

/**
 * Filters polar image for edges
 *
 * src: source image
 * filtered: result image (src.rows, src.cols, CV_32FC1)
 * fast: if true employs only one filter direction
 */
void findHorizontalEdges(const Mat& src, const Mat& reflect, Mat& filtered){
		int fsize = 21;
		Mat filter(fsize,fsize,CV_32FC1);
		gaborKernel(filter,8*M_PI,-M_PI/2,M_PI/2,6,0.5);
		kernelNormPeak(filter);
		Mat src2(src.rows,src.cols+2*fsize,CV_8UC1);
		Mat p1(src2,cv::Rect(0,0,fsize,src.rows));
		Mat p2(src2,cv::Rect(fsize,0,src.cols,src.rows));
		Mat p3(src2,cv::Rect(src.cols+fsize,0,fsize,src.rows));
		Mat p4(src,cv::Rect(0,0,fsize,src.rows));
		Mat p5(src,cv::Rect(src.cols-fsize,0,fsize,src.rows));
		p5.copyTo(p1);
		src.copyTo(p2);
		p4.copyTo(p3);

		//kernelWrite(filter,"kernel.png");
		Mat filtered2(filtered.rows,filtered.cols+2*fsize,CV_32FC1);
		Mat p6(filtered2,cv::Rect(fsize,0,filtered.cols,filtered.rows));
		filter2D(src2,filtered2,filtered2.depth(),filter,cvPoint(-1,-1),0,BORDER_CONSTANT);
		maskValue(p6,filtered,reflect,255,0);
		/*
		Mat filter(21,21,CV_32FC1);
		gaborKernel(filter,8*M_PI,-M_PI/2,M_PI/2,6,0.5);
		kernelNormPeak(filter);
		//kernelWrite(filter,"kernel.png");
		filter2D(src,filtered,filtered.depth(),filter,cvPoint(-1,-1),0,BORDER_REPLICATE);
		mask2f(filtered,filtered,reflect,255,0);*/
}

/**
 * Interpolates a contour by linearly filling all values less than zero
 * contour: contour to be normalized
 */
void contInterpolate(Mat& cont){
	int *c = (int *) cont.data;
	int *p = (int *) cont.data;
	int width = cont.cols;
	int pos = width-1;
	// repair beginning, if needed
	p += pos;
	while (pos > 0 && *p < 0) {p--; pos--;} // find previous intact point
	if (pos == 0) return; // nothing, we can do
	int startx = pos;
	int starty = c[pos];
	pos = 0;
	p = c;
	while (pos < width && *p < 0) {p++; pos++;} // find next intact point
	int endx = pos;
	int endy = c[pos];
	int diffx1 = width - startx;
	int diffx = diffx1 + endx;
	if (diffx > 1){ // repairing needed
		p = (c + startx + 1);
		float delta = ((float)(endy - starty)) / diffx;
		for (int i=1;i<diffx1;i++,p++){
			*p = cvRound(starty + i * delta);
		}
		p = c;
		for (int i=diffx1; i<diffx; i++, p++){
			*p = cvRound(starty + i * delta);
		}
	}
	startx = endx;
	starty = endy;
	p = c + startx + 1;
	pos = startx + 1;
	while (pos < width && *p >= 0) {p++; pos++;} // find next damaged point
	if (pos == width) return; // nothing more, we can do
	else { // intact point is the point before
		startx = pos - 1;
		starty = c[pos-1];
	}
	pos++;p++;
	while (pos < width){
		// find next intact point and interpolate
		while (pos < width && *p < 0) {p++; pos++;} // find next intact point
		endx = pos;
		endy = c[pos];
		diffx = endx-startx;
		float delta = ((float)(endy - starty)) / diffx;
		p = (c + startx + 1);
		for (int i=1; i<diffx; i++, p++){
			*p = cvRound(starty + i * delta);
		}
		startx = endx;
		starty = endy;
		p = c + startx + 1;
		pos = startx + 1;
		// and then look for next non-intact point
		while (pos < width && *p >= 0) {p++; pos++;} // find next damaged point
	}
}

/**
 * Conducts a gradient fit cont onto gradient image
 * cont: contour (1,width,CV_32SC1)
 * gradient: edge image (CV_32FC1)
 */
void gradientFit(Mat& cont, const Mat& gradient, const int maxrange, const int from = 0, const int to = -1){
	CV_Assert(cont.type() == CV_32FC1);
	CV_Assert(gradient.type() == CV_32FC1);
	int width = gradient.cols;
	int height = gradient.rows;
	MatIterator_<float> p = cont.begin<float>();
	float * g = (float *)gradient.data;
	int gstride = gradient.step / sizeof(float);
	for (int x=0; x<width; x++, g++, p++){
		int val = min(max(0,cvRound(*p)),height-1);
		if (*(g + val*gstride) > 0){ // not masked
			int pvalfrom = max(max(0,from),min(height-1,val - maxrange));
			int pvalto = max(0,min((to < 0) ? height-1 : to,val + maxrange));
			float * row = g + pvalfrom * gstride;
			int maxy = pvalfrom;
			float maxrow = - FLT_MAX;
			for (int y = pvalfrom; y < pvalto; y++, row += gstride) {
				if (*row > maxrow){
					maxrow = *row;
					maxy = y;
				}
			}
			*p = maxy;
		}
		else {
			*p = val;
		}
	}
}

/**
 * Modifies a contour such that only a certain amount of best edges are kept - other contour points are linearly interpolated.
 * cont: contour (1,width,CV_32SC1)
 * gradient: edge image (CV_32FC1)
 */
void contKeepBestEdges(Mat& cont, const Mat& gradient){
	CV_Assert(cont.type() == CV_32SC1);
	CV_Assert(gradient.type() == CV_32FC1);
	int width = cont.cols;
	int histbins = 100;
	Mat hist(1,histbins,CV_32SC1);
	Mat edges(1,width,CV_32SC1);
	MatIterator_<int> p = cont.begin<int>();
	MatIterator_<int> e = cont.end<int>();
	MatIterator_<float> pg = edges.begin<float>();
	float * g = (float *)gradient.data;
	int gstride = gradient.step / sizeof(float);
	float min = FLT_MAX;
	float max = -FLT_MAX;
	for (; p != e; pg++, g++, p++){
		float val = *(g + (*p) * gstride);
		*pg = val;
		if (val > max) max = val;
		if (val < min) min = val;
	}
	float histmax = max + (max-min)/histbins;
	hist2f(edges,hist,min,histmax);
	float low = histquant2u(hist,width,0.33) * (histmax-min) * 1. / histbins + min;
	// eliminate outliers
	p = cont.begin<int>();
	pg = edges.begin<float>();
	for (; p != e; p++, pg++){
		if (*pg < low) *p = -1;
	}
	contInterpolate(cont);
}

/**
 * Calculates the energy of a Boundary
 *
 * src: CV_32FC1 source image
 * cartBoundary: CV_32FC2 contour in cartesian coordinates (x and y coordinates)
 * polarBoundary: CV_32FC1 polar values
 * my: gaussian mean
 * sigma: gaussian sigma
 * returning: resulting energy (sum)
 */
double boundaryEnergyWeighted(const Mat& src, const Mat& cartBoundary, const RotatedRect& ellipse, const float resolution, const float my, const float sigma){
	double sum = 0;
	int width = src.cols;
	int height = src.rows;
	float centerX = ellipse.center.x;
	float centerY = ellipse.center.y;
	float ellA = ellipse.size.width/2;
	float ellB = ellipse.size.height/2;
	float corrA = ellA * resolution;
	float corrB = ellB * resolution;
	double alpha = (ellipse.angle)*M_PI/180; // inclination angle of the ellipse
	if (alpha > M_PI) alpha -= M_PI; // normalize alpha to [0,M_PI]
	double omega = 2 * M_PI - alpha; // angle for reverse transformation
	const double cosOmega = cos(omega);
	const double sinOmega = sin(omega);
	float * c = (float *)cartBoundary.data;
	double sigma2 = 2 * sigma * sigma;
	for (float * e = c + 2*cartBoundary.cols; c != e; c+=2) {
		int x =  std::max(0,std::min(width-1,cvRound(c[0])));
		int y = std::max(0,std::min(height-1,cvRound(c[1])));
		float a = c[0] - centerX;
		float b = c[1] - centerY;
		float s = (cosOmega * a - sinOmega * b) / corrA;
		float t = (sinOmega * a + cosOmega * b) / corrB;
		float r = sqrt(s*s+t*t);
		double z = r - my;
		double w = cv::exp(-z*z / sigma2);
		sum += src.at<float>(y,x) * w;
	}
	return sum;
}

/**
 * Calculates the energy of a Boundary
 *
 * src: CV_32FC1 source image
 * cartCont: CV_32FC1 contour in cartesian coordinates (x and y coordinates)
 *
 * returning: resulting energy (sum)
 */
double boundaryEnergy(const Mat& src, const Mat& cartBoundary){
	double sum = 0;
	int width = src.cols;
	int height = src.rows;
	float* b = (float*)cartBoundary.data;
	float* e = b + 2*cartBoundary.cols;
	while (b != e){
		int x = std::max(0, std::min(width - 1, cvRound(*b++)));
		int y = std::max(0, std::min(height - 1, cvRound(*b++)));
		sum += src.at<float>(y, x);
	}
	return sum;
}

/**
 * Computes the maximum sum of a sliding window for a given line range
 * src: CV_32FC1 source image
 * line: offset, line
 * wsize: size of the window
 * from: inclusive starting index
 * to: exclusive ending index
 *
 * resulting: energy (line sum)
 */
double lineSumWindowed(const Mat& src, const int line = 0, const int wsize = -1, const int from = 0, const int to = -1){
	MatConstIterator_<float> p = src.begin<float>();
	MatConstIterator_<float> e;
	MatConstIterator_<float> m;
	int end = (to < 0) ? src.cols : min(src.cols,to);
	int start = max(0,min(src.cols-1,from));
	int y = max(0,min(src.rows-1,line));
	int windowsize = (wsize < 0) ? ((start <= end) ? end-start : (src.cols + end - start)) : min(wsize,((start <= end) ? end-start : (src.cols + end - start)));
	double sum = 0;
	p += y * src.cols + start;
	double maxsum = -FLT_MAX;
	if (start <= end){
		m = p;
		// build up first window
		for(e = p + windowsize; p != e; p++){
			sum += *p;
		}
		maxsum = sum;
		// now window has energy, lets shift window and compare energy
		for(e = p + (end - start - windowsize); p != e; p++, m++){
			sum = sum + *p - *m;
			if (sum > maxsum) maxsum = sum;
		}
	}
	else if (windowsize >= src.cols - start){
		m = p;
		int size2 = src.cols - start;
		for(e = p + size2; p != e; p++){
			sum += *p;
		}
		p = src.begin<float>() + (y * src.cols);
		for(e = p + (windowsize-size2); p != e; p++){
			sum += *p;
		}
		maxsum = sum;
		for(e = p + min(end-windowsize+size2,size2); p != e; p++, m++){
			sum = sum + *p - *m;
			if (sum > maxsum) maxsum = sum;
		}
		m = src.begin<float>() + (y * src.cols);
		for(e = p + max(0,end - windowsize); p != e; p++, m++){
			sum = sum + *p - *m;
			if (sum > maxsum) maxsum = sum;
		}
	}
	else { // (wsize < cont.cols - from){
		m = p;
		int size2 = src.cols - start;
		for(e = p + windowsize; p != e; p++){
			sum += *p;
		}
		maxsum = sum;
		for(e = p + (size2 - windowsize); p != e; p++, m++){
			sum = sum + *p - *m;
			if (sum > maxsum) maxsum = sum;
		}
		p = src.begin<float>() + (y * src.cols);
		for(e = p + min(end,windowsize); p != e; p++, m++){
			sum = sum + *p - *m;
			if (sum > maxsum) maxsum = sum;
		}
		m = src.begin<float>() + (y * src.cols);
		for(e = p + max(0,end-windowsize); p != e; p++, m++){
			sum = sum + *p - *m;
			if (sum > maxsum) maxsum = sum;
			if (sum > maxsum) maxsum = sum;
		}
	}
	return maxsum;
}

/**
 * Samples an ellipse from center point using polar rays
 *
 * ellipse: rotated rectangle ellipse representation
 * cont: CV_32FC1 1 x size sampled contour (360/size is used as sample angle theta)
 * centerX: polar center x-coordinate
 * centerY: polar center y-coordinate
 */
void ellipse2polar(const RotatedRect& ellipse, Mat& cont, const float centerX, const float centerY){
	const int width = cont.cols;
	float theta = 0; // angle of the polar ray
	const float thetaoffset = 2.f * M_PI / width;
	const float a = ellipse.size.width/2; // big half axis of the ellipse
	const float aSquare = a*a;
	const float b = ellipse.size.height/2; // small half axis of the ellipse
	const float bSquare = b*b;
	double alpha = (ellipse.angle)*M_PI/180; // inclination angle of the ellipse
	if (alpha > M_PI) alpha -= M_PI; // normalize alpha to [0,M_PI]
	double omega = 2 * M_PI - alpha; // angle for reverse transformation
	const double cosAlpha = cos(alpha);
	const double sinAlpha = sin(alpha);
	const double cosOmega = cos(omega);
	const double sinOmega = sin(omega);
	const float cXunrot = centerX - ellipse.center.x;
	const float cYunrot = centerY - ellipse.center.y;
	const float cX = cosOmega * cXunrot - sinOmega * cYunrot; // center x-coordinate in ellipse coords
	const float cY = sinOmega * cXunrot + cosOmega * cYunrot; // center x-coordinate in ellipse coords
	float *p = (float *)cont.data;
	for (float *e = p+width; p != e; p++, theta += thetaoffset){
		float beta = (alpha <= theta) ? theta - alpha : 2 * M_PI + theta - alpha; // angle of polar ray in ellipse coords (alpha + beta = theta)
		//cout << "Shooting ray in direction: " << beta / M_PI * 180 << " ( alpha is " << alpha / M_PI * 180 << " theta is: " << theta / M_PI * 180 << ")" << endl;
		float x=0, y=0;
		if (abs(beta-M_PI/2) < 0.00001){
			// return positive y value of ellipse at cX
			x=cX;
			y=b*sqrt(aSquare-cX*cX)/a;
		}
		else if(abs(beta-3*M_PI/2) < 0.00001){
			// return negative y value of ellipse at cX
			x=cX;
			y=-b*sqrt(aSquare-cX*cX)/a;
		}
		else {
			float m = tan(beta); // steigung
			//cout << "Steigung: " << m << endl;
			float b1 = cY - m*cX; // verschiebung
			float mSquare = m*m;
			float D = aSquare*mSquare + bSquare - b1*b1;
			if (D <= 0) { // ERROR, entering desperate mode
				*p = 0; continue;
			}
			else { // Formula see Bartsch, p. 262
				D = sqrt(D);
				float N = bSquare + aSquare * mSquare;
				float T1 = -aSquare*m*b1;
				float T2 = a*b*D;
				float U1 = bSquare * b1;
				float U2 = a*b*m*D;
				// determine quadrant
				float y1 = (U1 + U2)/N;
				if (beta < M_PI){ // Q1 x: + y: + or Q2 x: - y: + let y decide
					if (y1-cY >= 0){
						x = (T1 + T2)/N;
						y = y1;

					} else {
						x = (T1 - T2)/N;
						y = (U1 - U2)/N;
					}
				}
				else { // Q3 x: - y: - or Q4 x: + y: -
					if (y1-cY < 0){
						x = (T1 + T2)/N;
						y = y1;
					} else {
						x = (T1 - T2)/N;
						y = (U1 - U2)/N;
					}
				}
			}

		}
		// rotate back to polar coords
		float fX = cosAlpha * x - sinAlpha * y - cXunrot;
		float fY = sinAlpha * x + cosAlpha * y - cYunrot;
		*p = sqrt(fX*fX+fY*fY);
	}
}



/**
 * Normalizes a function based on DFT by keeping the first numberCoeffs fourier coefficients only
 * f:            original signal, should be Mat (1, size, CV_32FC1);
 * norm:         normalized signal, should be Mat (1, size, CV_32FC1);
 * energy:		 total energy of the first numberCoeffs Fourier coefficients
 * numberCoeffs: number of fourrier coefficients to keep
 */
void fourierNormalize(Mat& cont, Mat& norm, float& energy, const int numberCoeffs = -1){
	CV_Assert(cont.size() == norm.size());
	int size = cont.cols;
	int dftSize = getOptimalDFTSize(size);
	if (dftSize != size){
		printf("WARNING ERROR");
		Mat g (1,dftSize,CV_32FC1);
		float mean = cv::mean(cont)[0];
		int * src = (int *) cont.data;
		MatIterator_<float> dst = g.begin<float>();
		double inc = ((double)size) / dftSize;
		double idx = 0;
		for (MatConstIterator_<float> e = g.end<float>(); dst != e; dst++, idx += inc){
			*dst = src[min(size-1,cvRound(idx))] - mean;
		}
		Mat gFourier (1,dftSize,CV_32FC1);
		dft(g,gFourier,CV_DXT_FORWARD);
		if (numberCoeffs >= 0 && (1 + 2*numberCoeffs) < dftSize){
			Mat roi(gFourier,Rect(1 + 2*numberCoeffs,0,dftSize - (1 + 2*numberCoeffs),1));
			roi.setTo(0);
			roi = Mat(gFourier,Rect(1,0,2*numberCoeffs,1));
			energy = cv::norm(roi);
		}
		dft(gFourier,g,CV_DXT_INV_SCALE);
		float * isrc = (float *) g.data;
		MatIterator_<int> idst = norm.begin<int>();
		inc = ((double)dftSize) / size;
		idx = 0;
		for (MatConstIterator_<int> e = norm.end<int>(); idst != e; idst++, idx += inc){
			*idst = saturate_cast<int>(isrc[min(dftSize-1,cvRound(idx))] + mean);
		}
	}
	else {
		Mat g (1,dftSize,CV_32FC1);
		float mean = cv::mean(cont)[0];
		g = cont - mean;
		Mat gFourier (1,dftSize,CV_32FC1);
		dft(g,gFourier,CV_DXT_FORWARD);
		if (numberCoeffs >= 0 && (1 + 2*numberCoeffs) < dftSize){
			Mat roi(gFourier,Rect(1 + 2*numberCoeffs,0,dftSize - (1 + 2*numberCoeffs),1));
			roi.setTo(0);
			roi = Mat(gFourier,Rect(1,0,2*numberCoeffs,1));
			energy = cv::norm(roi);
		}
		dft(gFourier,g,CV_DXT_INV_SCALE);
		norm = g + mean;
	}
}

/**
 * Maps an ellipsopolar contour to cartesian coordinates
 * polar: CV_32FC1 1 x size array of radius values from center position
 * cart CV_32FC2 1 x size array of x- and y-positions from upper left corner
 * centerX: x-coordinate of center in cartesian coordinate system (upper-left corner)
 * centerY: y-coordinate of center in cartesian coordinate system (upper left corner)
 * resolution: polar resolution (i.e. pixels per unit in polar coords before considering axis stretch factors)
 * offset: optional offset added to the contour before the conversion
 */
void ellipsopolar2Cart(const Mat& polar, Mat& cart, const RotatedRect& ellipse, const float resolution = 1, const float offset = 0) {
	CV_Assert(polar.type() == CV_32FC1);
	CV_Assert(cart.type() == CV_32FC2);
	CV_Assert(polar.cols == cart.cols);
	float centerX = ellipse.center.x;
	float centerY = ellipse.center.y;
	float ellA = ellipse.size.width/2;
	float ellB = ellipse.size.height/2;
	double alpha = (ellipse.angle)*M_PI/180; // inclination angle of the ellipse
	if (alpha > M_PI) alpha -= M_PI; // normalize alpha to [0,M_PI]
	const double cosAlpha = cos(alpha);
	const double sinAlpha = sin(alpha);
	float * d = (float *)cart.data;
	int width = polar.cols;
	const float thetaoffset = 2 * M_PI / polar.cols;
	float theta = 0;
	float * p = (float *)polar.data;
	for (int x=0; x<width; x++,p++,d++, theta += thetaoffset){
		float beta = (alpha <= theta) ? theta - alpha : 2 * M_PI + theta - alpha; // angle of polar ray in ellipse coords (alpha + beta = theta)
		float s = (*p + offset) * resolution * ellA * cos(beta), t = (*p + offset) * resolution * ellB * sin(beta);
		*d = centerX + cosAlpha * s - sinAlpha * t; // x coordiante
		d++;
		*d = centerY + sinAlpha * s + cosAlpha * t; // y coordinate
	}
}

/**
 * Maps a cartesian contour to ellipsopolar coordinates (more precisely, only radius values are computed)
 */
void cart2Ellipsopolar(const Mat& cart, Mat& polar, const RotatedRect& ellipse, const float resolution = 1) {
	CV_Assert(polar.type() == CV_32FC1);
	CV_Assert(cart.type() == CV_32FC2);
	CV_Assert(polar.cols == cart.cols);
	float centerX = ellipse.center.x;
	float centerY = ellipse.center.y;
	float ellA = ellipse.size.width/2;
	float ellB = ellipse.size.height/2;
	float corrA = ellA * resolution;
	float corrB = ellB * resolution;
	double alpha = (ellipse.angle)*M_PI/180; // inclination angle of the ellipse
	if (alpha > M_PI) alpha -= M_PI; // normalize alpha to [0,M_PI]
	double omega = 2 * M_PI - alpha; // angle for reverse transformation
	const double cosOmega = cos(omega);
	const double sinOmega = sin(omega);
	float * c = (float *)cart.data;
	int width = cart.cols;
	float * p = (float *)polar.data;
	for (int x=0; x<width; x++,p++,c++){
		// rotate to ellipse coords
		float a = *c - centerX; c++;
		float b = *c - centerY;
		float s = (cosOmega * a - sinOmega * b) / corrA;
		float t = (sinOmega * a + cosOmega * b) / corrB;
		*p = sqrt(s*s+t*t);
	}
}



/**
 * Samples an ellipse in cartesian coordinates
 * cart: CV_32FC2 1 x size cartesian coordinates
 * ellipse: ellipse to be sampled
 */
void ellipse2Cart(Mat& cart, const RotatedRect& ellipse) {
	CV_Assert(cart.type() == CV_32FC2);
	float centerX = ellipse.center.x;
	float centerY = ellipse.center.y;
	float ellA = ellipse.size.width/2;
	float ellB = ellipse.size.height/2;
	double alpha = (ellipse.angle)*M_PI/180; // inclination angle of the ellipse
	if (alpha > M_PI) alpha -= M_PI; // normalize alpha to [0,M_PI]
	const double cosAlpha = cos(alpha);
	const double sinAlpha = sin(alpha);
	float * d = (float *)cart.data;
	int width = cart.cols;
	const float thetaoffset = 2 * M_PI / cart.cols;
	float theta = 0;
	for (int x=0; x<width; x++, d++, theta += thetaoffset){
		float beta = (alpha <= theta) ? theta - alpha : 2 * M_PI + theta - alpha; // angle of polar ray in ellipse coords (alpha + beta = theta)
		float s = ellA * cos(beta), t = ellB * sin(beta);
		*d = centerX + cosAlpha * s - sinAlpha * t; // x coordiante
		d++;
		*d = centerY + sinAlpha * s + cosAlpha * t; // y coordinate
	}
}

/**
 * Maps a polar contour to cartesian coordinates
 * polar: CV_32FC1 1 x size array of radius values from center position
 * cart CV_32FC2 1 x size array of x- and y-positions from upper left corner
 * centerX: x-coordinate of center in cartesian coordinate system (upper-left corner)
 * centerY: y-coordinate of center in cartesian coordinate system (upper left corner)
 * resolution: polar resolution (i.e. pixels per unit in polar coords, use maxDistToCorner() / (polar.rows-1))
 * offset: optional offset added to the contour before the conversion
 */
void polar2Cart(const Mat& polar, Mat& cart, const float centerX = 0, const float centerY = 0, const float resolution = 1, const float offset = 0) {
	CV_Assert(polar.type() == CV_32FC1);
	CV_Assert(cart.type() == CV_32FC2);
	CV_Assert(polar.cols == cart.cols);
	float * d = (float *)cart.data;
	int width = polar.cols;
	const float thetaoffset = 2 * M_PI / polar.cols;
	float theta = 0;
	float * s = (float *)polar.data;
	if (offset != 0){
		for (int x=0; x<width; x++,s++,d++, theta += thetaoffset){
			*d = centerX + (*s + offset) * resolution * cos(theta); // x coordiante
			d++;
			*d = centerY + (*s + offset) * resolution * sin(theta);; // y coordinate
		}
	}
	else if (centerX == 0 && centerY == 0 && resolution == 1){ // fast version
		for (int x=0; x<width; x++,s++,d++, theta += thetaoffset){
			*d = *s * cos(theta); // x coordiante
			d++;
			*d = *s * sin(theta);; // y coordinate
		}
	}
	else {
		for (int x=0; x<width; x++,s++,d++, theta += thetaoffset){
			*d = centerX + *s * resolution * cos(theta); // x coordiante
			d++;
			*d = centerY + *s * resolution * sin(theta);; // y coordinate
		}
	}
}

void ellipseNormalize(Mat& cont, Mat& norm){
	CV_Assert(cont.size() == norm.size());
	int width = cont.cols;
	Mat cart (1,width,CV_32FC2);
	polar2Cart(cont, cart, 0, 0, 1, 2.f * M_PI / width);
	RotatedRect ell = fitEllipse(cart);
	ellipse2polar(ell,norm,0,0);
}

/**
 * Initializes a contour in polar or ellipsopolar transformed domain by first looking for the highest horizontal energy window.
 *
 * src: CV_32FC1 height x width polar or ellipsopolar gradient image
 * cont CV_32FC1 1 x width polar or ellipsopolar contour
 * min: minimum inclusive index
 * max: maximum exclusive index
 * useSectors: if true, look in left and right windows, otherwise look in entire range
 * sigma: fuzzy gaussian sigma (use < 0 to avoid fuzzy computation, default)
 * my: gaussian my
 */
void initContour(const Mat& src, Mat& cont, const int min = 0, const int max = -1, const bool useSectors = false, const float sigma = -1, const float my = -1){
	const int miny = std::max(0,std::min(min,src.rows - 1));
	const int maxy = (max < 0) ? src.rows : std::min(src.rows,max);
	const int width = cont.cols;
	int y = miny;
	int besty = y;
	double energy = -FLT_MAX;
	double sigma2 = 2 * sigma * sigma;
	double z = y - my;
	if (sigma > 0){ // with fuzzy logic
		if (useSectors){
			const int windowSize = width/8;
			const int windowRightFrom = 15*width / 16;
			const int windowRightTo = 3*width / 16;
			const int windowLeftFrom = 5*width / 16;
			const int windowLeftTo = 9*width / 16;
			for (; y < maxy; y++, z++){
				double w = cv::exp(-z*z / sigma2);
				double eng = (lineSumWindowed(src,y,windowSize,windowRightFrom,windowRightTo) + lineSumWindowed(src,y,windowSize,windowLeftFrom,windowLeftTo)) * w; //cv::exp(-z*z / sigma2);
				if (eng > energy) {
					energy = eng;
					besty = y;
				}
			}
		}
		else {
			for (; y < maxy; y++, z++){
				double w = cv::exp(-z*z / sigma2);
				float eng = lineSumWindowed(src,y) * w;
				if (eng > energy) {
					energy = eng;
					besty = y;
				}
			}
		}
	}
	else { // without fuzzy logic
		if (useSectors){
			const int windowSize = width/8;
			const int windowRightFrom = 15*width / 16;
			const int windowRightTo = 3*width / 16;
			const int windowLeftFrom = 5*width / 16;
			const int windowLeftTo = 9*width / 16;
			for (; y < maxy; y++, z++){
				double eng = (lineSumWindowed(src,y,windowSize,windowRightFrom,windowRightTo) + lineSumWindowed(src,y,windowSize,windowLeftFrom,windowLeftTo));
				if (eng > energy) {
					energy = eng;
					besty = y;
				}
			}
		}
		else {
			for (; y < maxy; y++){
				float eng = lineSumWindowed(src,y);
				if (eng > energy) {
					energy = eng;
					besty = y;
				}
			}
		}
	}
	cont.setTo(besty);
}

/**
 * Returns a subwindow of cart
 *
 * cart: CV_32FC2 1 x size cartesian contour
 *
 * returning sectorized contour
 */
void cartSectors(const Mat& cart, Mat& sub){
	int width = cart.cols;
	const int windowRightFrom = 15*width / 16;
	const int windowRightTo = 3*width / 16;
	const int windowLeftFrom = 5*width / 16;
	const int windowLeftTo = 9*width / 16;
	if (sub.empty() || sub.type() != CV_32FC2 || sub.rows != 1 || sub.cols != width-windowRightFrom+windowRightTo+windowLeftTo-windowLeftFrom)
		sub.create(1,width-windowRightFrom+windowRightTo+windowLeftTo-windowLeftFrom,CV_32FC2);
	Mat s1(cart,Rect(windowRightFrom,0,width-windowRightFrom,1));
	Mat s2(cart,Rect(0,0,windowRightTo,1));
	Mat s3(cart,Rect(windowLeftFrom,0,windowLeftTo-windowLeftFrom,1));
	Mat d1(sub,Rect(0,0,width-windowRightFrom,1));
	Mat d2(sub,Rect(width-windowRightFrom,0,windowRightTo,1));
	Mat d3(sub,Rect(width-windowRightFrom+windowRightTo,0,windowLeftTo-windowLeftFrom,1));
	s1.copyTo(d1);
	s2.copyTo(d2);
	s3.copyTo(d3);
}

/** ------------------------------- Contour refinement ------------------------------- **/




/**
 * Maps a contour in rubbersheet coordinates back to cartesian coordinates given the two cartesian boundaries
 * cartInner: CV_32FC2 1 x size inner cartesian boundary
 * cartOuter: CV_32FC2 1 x size outer cartesian boundary
 * cont: CV_32FC1 1 x size array of radius values from inner contour
 * cartCont: CV_32FC2 1 x size contour in cartesian coordinates
 * height: total height of the rubbersheet model
 *
 */
void rubbersheet2Cart(const Mat& cartInner, const Mat& cartOuter, const Mat& cont, Mat& cartCont, const int height, const float pixeloffset = 0) {

	float * i = (float *)cartInner.data, * o = (float *)cartOuter.data, *c = (float *)cont.data, *dst = (float *)cartCont.data;
	for (float *e = c + cont.cols;c != e; c++, i++, o++, dst++){
		*dst = *i + (*c + pixeloffset) * (*o - *i) / height;
		i++; o++; dst++;
		*dst = *i + (*c + pixeloffset) * (*o - *i) / height;
	}
}

/**
 * Resamples a closed contour from its new nucleus.
 * Hint: take care to divide dx and dy by polar resolution resY, if not equals to 1
 *
 * contOld: CV_32FC1 old polar contour, even spacing of 2 Pi (1,width,CV_32FC1), stretch-normalized (divide by resolution resY)
 * dx: center x-coordinate difference to old center (old_cx + dx = new_cx)
 * dy: center y-coordinate difference to old center (old_cy + dy = new_cy)
 * contourNew: CV_32FC1 new polar contour, even spacing of 2 Pi (1,width,CV_32FC1)
 */
void resampleContour(const Mat& contOld, const float dx, const float dy, Mat& contNew){
	CV_Assert(contOld.type() == CV_32FC1);
	CV_Assert(contNew.type() == CV_32FC1);
	int width = contOld.cols;
	Mat angles(1,width,CV_32FC1);
	Mat points(1,width,CV_32FC2);
	Mat angleIdx(1,width,CV_32SC1);
	float *p = (float *)contOld.data;
	float *q = (float *)angles.data;
	float *v = (float *)points.data;
	const float thetaoffset = 2.f * M_PI / width;
	float theta = 0;
	float thetafactor = M_PI/180.f;
	// calculate angles with respect to new center
	for (float *e = p+width; p != e; p++, q++, v+=2, theta += thetaoffset){
		*v = (*p) * cos(theta) - dx;
		v[1] = (*p) * sin(theta) - dy;
		*q = fastAtan2(v[1], *v) * thetafactor;
	}
	// angles are sorted
	sortIdx(angles,angleIdx,CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
	q = (float *)angles.data;
	v = (float *)points.data;
	int * idx = (int *) angleIdx.data;
	theta = 0;
	float *dst = (float *)contNew.data;
	int j=0;
	for (int i=0; i<width; i++, theta += thetaoffset){
		for (; j<width && q[idx[j]] < theta; j++);
		int idxPrev, idxSucc; // interesting indices are j-1 und j (wrt. to ring buffer)
		bool desperate = false;
		if (j== width || j == 0){
			// jumping over 360 deg
			idxPrev = idx[width-1];
			idxSucc = idx[0];
			// (det(x1,y1,x2,y2) < 0) i.e. negative orientation: new origin is outside
			if (q[idxSucc] + M_PI - q[idxPrev] > 0) desperate = true;
		}
		else {
			idxPrev = idx[j-1];
			idxSucc = idx[j];
			if (q[idxSucc] - q[idxPrev] > M_PI) desperate = true;
		}
		// build equation of line
		// y = k*x + m in polar coordinates: r(t) = m / (sin(t) + k*cos(t))
		float x1 = v[2*idxPrev], y1 = v[2*idxPrev+1], x2 = v[2*idxSucc], y2 = v[2*idxSucc+1];
		if (!desperate) {
			// Calculate intersection points
			if  (abs(x2-x1) < 0.00001){
				dst[i] = x1 / cos(theta);
			}
			else {
				float k = (y2-y1)/(x2-x1);
				float m = ((y2-k*x2) + (y1-k*x1))/2;
				dst[i] = m / (sin(theta) - k * cos(theta));
			}
		}
		else { // force validity
			dst[i] = 0;
		}
	}
}

/**
 * Computes the contour center
 * warning: multiply coordinates with y-resolution to get correct values!
 */
void pullAndPush(const Mat& polarCont, Vec2f& offset){
	int width = polarCont.cols;
	Mat cart(1,width,CV_32FC2);
	Mat polar(1,width,CV_32FC1);
	polarCont.convertTo(polar,CV_32FC1);
	//copy2f(polarCont,polar);
	float fx = 0, fy = 0, dx = 0, dy = 0;
	int iter = 0;
	do {
		fx = 0; fy = 0;
		polar2Cart(polar,cart,0,0,1,0);
		for (MatIterator_<float> s = cart.begin<float>(), e = cart.end<float>(); s != e; s++){
			fx += (*s);
			s++;
			fy += (*s);
		}
		fx /= width;
		fy /= width;
		dx += fx;
		dy += fy;
		resampleContour(polarCont,dx,dy,polar);
		iter++;
	}
	while ((abs(fx) > 0.1 || abs(fy) > 0.1) && iter < 1000);
	offset[0] = dx;
	offset[1] = dy;
}

/*** Mask lids ****/


void adjust_luminance(Mat& image, int black, int white)
{
    /*
     * Adjust the luminance of the image in-place so that the result has at least
     * black completely black and white completely white pixels (if possible). */
	Mat hist(1,256,CV_32SC1);
	hist2u(image,hist);
	int * histogram = (int*)hist.data;
	int blackpoint, whitepoint;
    int i;
    if (black)
    {
        int bn = 0;
        for (i = 0; i < 256; i++)
        {
            bn += histogram[i];
            if (bn >= black)
            {
                break;
            }
        }
        blackpoint = i;
    }
    else
    {
        blackpoint = 0;
    }
    if (white)
    {
        int wn = 0;
        for (i = 255; i >= 0; i--)
        {
            wn += histogram[i];
            if (wn >= white)
            {
                break;
            }
        }
        whitepoint = i;
    }
    else
    {
        whitepoint = 255;
    }
	for (MatIterator_<uchar> it = image.begin<uchar>(); it < image.end<uchar>(); it++){
		int c = *it;
		int c_ = c - blackpoint;
		c_ = c_ * 255 / (whitepoint - blackpoint);
		if (c_ < 0)
		{
			c_ = 0;
		}
		if (c_ > 255)
		{
			c_ = 255;
		}
		*it = c_;
	}
}

/*
 * Make light areas lighter
 */
void cumulate(const Mat& image, Mat& result, int cw, int ch)
{
	CV_Assert(image.size() == result.size());

    int w = image.cols;
    int h = image.rows;
    int step = image.step;
    uchar* data = image.data;
    uchar* a = result.data;
    int ox = cw / 2;
    int oy = ch / 2;
    for (int y = 0; y < h; y++)
    {
        int ay, by;
        if (y < oy)
        {
            ay = oy - y;
        }
        else
        {
            ay = 0;
        }
        if (y + ch - oy > h)
        {
            by = h + oy - y;
        }
        else
        {
            by = ch;
        }
        for (int x = 0; x < w; x++)
        {
            int sum = 0;
            int ax, bx;
            if (x < ox)
            {
                ax = ox - x;
            }
            else
            {
                ax = 0;
            }
            if (x + cw - ox > w)
            {
                bx = w + ox - x;
            }
            else
            {
                bx = cw;
            }
            for (int j = ay; j < by; j++)
            {
                for (int i = ax; i < bx; i++)
                {
                    sum += data[(y + j - oy) * step + (x + i - ox)];
                }
            }
            if (sum > 255)
            {
                sum = 255;
            }
            a[y * step + x] = sum;
        }
    }
}

/*
 * Looks for longest horizontal run
 */
int longest_horizontal_run(const Mat& image, int border, int threshold)
{
    int w = image.cols;
    int h = image.rows;
    int step = image.step;
    int maxrun = 0;
    uchar * data = image.data;
    for (int j = border; j < h - border; j++)
    {
        int run = 0;
        for (int i = border; i < w - border; i++)
        {
            if (data[i + j * step] <= threshold)
            {
                run++;
                if (run > maxrun)
                {
                    maxrun = run;
                }
            }
            else
            {
                run = 0;
            }
        }
    }
    return maxrun;
}

/*
 * Looks for longest vertical run
 */
int longest_vertical_run(const Mat& image, int border, int threshold)
{
    int w = image.cols;
    int h = image.rows;
    int step = image.step;
    int maxrun = 0;
    uchar * data = image.data;
    for (int i = border; i < w - border; i++)
    {
        int run = 0;
        for (int j = border; j < h - border; j++)
        {
            if (data[i + j * step] <= threshold)
            {
                run++;
                if (run > maxrun)
                {
                    maxrun = run;
                }
            }
            else
            {
                run = 0;
            }
        }
    }
    return maxrun;
}



/*
 * Returns the result of convoluting image with kernel.
 * The border is handled by treating all outside pixels as the nearest picture
 * pixel.
 */
void convolution_brute(const Mat& image, Mat& res, const Mat& kernel)
{
    int w = image.cols;
    int h = image.rows;
    int step = image.step / sizeof (float);
    int kw = kernel.cols;
    int kh = kernel.rows;
    int kstep = kernel.step / sizeof (float);
    float *data = (float*)image.data;
    float *kdata = (float*)kernel.data;
    int ox = -kw / 2;
    int oy = -kh / 2;
    float *result = (float*)res.data;
    int rstep = res.step / sizeof (float);
    for (int y = oy; y < h + oy; y++)
    {
        for (int x = ox; x < w + ox; x++)
        {
            float sum = 0;
            for (int j = 0; j < kh; j++)
            {
                int yj;
                if (y + j >= 0 && y + j < h)
                {
                    yj = y + j;
                }
                else
                {
                    yj = y - oy;
                }
                for (int i = 0; i < kw; i++)
                {
                    int xi;
                    if (x + i >= 0 && x + i < w)
                    {
                        xi = x + i;
                    }
                    else
                    {
                        xi = x - ox;
                    }
                    float v = data[step * yj + xi];
                    float k = kdata[j * kstep + i];
                    sum += v * k;
                }
            }
            result[(y - oy) * rstep + x - ox] = sum;
        }
    }
}

/*
 * Returns a new image of size w times h with the result of a gaussian
 * smoothing operation.
 * TODO: Should do this in frequency domain, especially for big radius.
 */
void gaussian_smooth(Mat& image, Mat& res, double sigma, int size)
{
    double sigma2 = sigma * sigma;
    int kw = size;
    int kh = size;
    Mat kernel(kh,kw,CV_32FC1);
    float *pos = (float *) kernel.data;
    int off = (kernel.step / sizeof(float)) - kernel.cols;
    int o = -size / 2;
    for (int j = o; j < kh + o; j++, pos+=off)
    {
        for (int i = o; i < kw + o; i++, pos++)
        /* This is 2D (for 1D, sqrt the factor) */
        {
            *pos = exp(-(i * i + j * j) / (2 * sigma2)) / (2 * M_PI * sigma2);
        }
    }
    convolution_brute(image, res, kernel);
}

/*
 * Generates horizontal Sobel Kernel
 */
void horizontal_sobel(Mat& image, Mat& res)
{
    float kdata[] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    Mat kernel(3, 3, CV_32FC1, &kdata);
    convolution_brute(image, res, kernel);
}

/*
 * Generates vertical Sobel Kernel
 */
void vertical_sobel(Mat& image, Mat& res)
{
	float kdata[] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };
    Mat kernel(3, 3, CV_32FC1, &kdata);
    convolution_brute(image, res, kernel);
}

/*
 * Use vertical and horizontal sobel operators to estimate the gradient, and
 * return it as  magnitude an orientation maps.
 */
void get_gradient(Mat& image, Mat& mag, Mat& orient, float hfactor, float vfactor)
{
    int w = image.cols;
    int h = image.rows;
    Mat imsobelh(h,w,CV_32FC1);
    Mat imsobelv(h,w,CV_32FC1);
    int w_sobelh = imsobelh.step / sizeof(float);
    int w_sobelv = imsobelv.step / sizeof(float);
    int w_mag = mag.step / sizeof(float);
    int w_orient = orient.step / sizeof(float);
    horizontal_sobel(image,imsobelh);
    vertical_sobel(image,imsobelv);

    float* sobelh = (float*) imsobelh.data;
    float* sobelv = (float*) imsobelv.data;
    float* magnitude = (float*) mag.data;
    float* orientation = (float*) orient.data;
    for (int j = 0; j < h; j++)
    {
        for (int i = 0; i < w; i++)
        {
            float x = sobelh[i + w_sobelh * j];
            float y = sobelv[i + w_sobelv * j];
            float m = sqrt(x * x * hfactor + y * y * vfactor);
            /* if m > 255: m = 255 */
            m /= 4;
            magnitude[i + w_mag * j] = m;
            double a = atan2(y, x);
            orientation[i + w_orient * j] = a;
        }
    }
}

/*
 * Returns gradient neighbors
 */
static inline void get_gradient_neighbors(float o, int i, int j, int *i_, int *j_, int *i__, int *j__)
{
    if (o == 0)
    {
        *i_ = i + 1;
        *i__ = i - 1;
        *j_ = *j__ = j;
    }
    else if (o == 1)
    {
        *i_ = i + 1;
        *i__ = i - 1;
        *j_ = j - 1;
        *j__ = j + 1;
    }
    else if (o == 2)
    {
        *i_ = *i__ = i;
        *j_ = j - 1;
        *j__ = j + 1;
    }
    else if (o == 3)
    {
        *i_ = i - 1;
        *i__ = i + 1;
        *j_ = j - 1;
        *j__ = j + 1;
    }
    else
    {
        assert(0);
    }
}

/*
 * Performs Border check
 */
static inline void suppress_gradient_with_border_check(float* gradient, float* orientation, int i, int j, float* result, int gradstep, int orientstep, int resstep, int w, int h)
{
    float g = gradient[i + gradstep * j];
    float o = orientation[i + orientstep * j];
    int i_, j_, i__, j__;
    get_gradient_neighbors(o, i, j, &i_, &j_, &i__, &j__);
    float g1 = g, g2 = g;
    if (i_ >= 0 && i_ < w && j_ >= 0 && j_ < h)
    {
        g1 = gradient[i_ + gradstep * j_];
    }
    if (i__ >= 0 && i__ < w && j__ >= 0 && j__ < h)
    {
        g2 = gradient[i__ + gradstep * j__];
    }
    if (g < g1 || g < g2)
    {
        result[i + resstep * j] = 0;
    }
    else
    {
        result[i + resstep * j] = g;
    }
}

/*
 * For each pixel, if either the left or right neighbor is brighter, set it to
 * 0. That way, only thinned areas of maximum brightness are left. Because
 * left/right neighbors are used, this is biased to create vertical lines.
 */
void non_maximum_suppression(Mat& gradient, Mat& orientation, Mat& res)
{
    int w = gradient.cols;
    int h = gradient.rows;
    float* result = (float*) res.data;
    float * gdata = (float*) gradient.data;
    float * odata = (float*) orientation.data;
    int gstep = gradient.step/sizeof(float);
    int ostep = orientation.step/sizeof(float);
    int rstep = res.step/sizeof(float);

    /* handle border pixels in a special way first */
    for (int j = 0; j < h; j += h - 1)
    {
        for (int i = 0; i < w; i++)
        {
            suppress_gradient_with_border_check(gdata, odata, i, j, result,gstep,ostep,rstep,w,h);
        }
    }
    for (int i = 0; i < w; i += w - 1)
    {
        for (int j = 1; j < h - 1; j++)
        {
            suppress_gradient_with_border_check(gdata, odata, i, j, result,gstep,ostep,rstep,w,h);
            /* now remaining pixels, without check */
        }
    }
    for (int j = 1; j < h - 1; j++)
    {
        for (int i = 1; i < w - 1; i++)
        {
            float g = gdata[i + gstep * j];
            float o = odata[i + ostep * j];
            int i_, j_, i__, j__;
            get_gradient_neighbors(o, i, j, &i_, &j_, &i__, &j__);
            float g1 = gdata[i_ + gstep * j_];
            float g2 = gdata[i__ + gstep * j__];
            if (g < g1 || g < g2)
            {
                result[i + rstep * j] = 0;
            }
            else
            {
                result[i + rstep * j] = g;
            }
        }
    }
}

/*
 * Quantize the orientations into 4 bins:
 * 0 right
 * 1 right up
 * 2 up
 * 3 left up
 * Those are all we need for pixel-edge-tracking  */
void edge_directions(Mat& orientation, Mat& res)
{
    int w = orientation.cols;
    int h = orientation.rows;
    float *e = (float*)res.data;
    float *o = (float*)orientation.data;
    int estep = res.step / sizeof(float);
    int ostep = orientation.step / sizeof(float);

    for (int j = 0; j < h; j++)
    {
        for (int i = 0; i < w; i++)
        {
            float a = o[i + ostep * j];
            float c;
            /* not 2*pi, left/right both is horizontal */
            if (a < 0)
            {
                a += M_PI;
            }
            if (a > M_PI * 0.875)
            {
                c = 0;
            }
            else if (a > M_PI * 0.625)
            {
                c = 3;
            }
            else if (a > M_PI * 0.375)
            {
                c = 2;
            }
            else if (a > M_PI * 0.125)
            {
                c = 1;
            }
            else
            {
                c = 0;
            }
            e[i + estep * j] = c;
        }
    }
}

/*
 * Returns a new image of size w times h with the result of a canny edge
 * detection. The implementation here uses the following algorithm:
 * - Apply gaussian smoothing to the image.
 * - Approximate gradient for each pixel (we use H and V Sobel here right now,
 *   should compare to e.g. differential gaussian instead)
 * - Localize edges.
 * - fix edges
 */
void canny(Mat& raster, Mat& magnitude, Mat& orientation, Mat& res, double gauss_sigma, int gauss_size, double hfactor, double vfactor)
{
	Mat image(raster.rows,raster.cols,CV_32FC1);
	raster.convertTo(image,CV_32FC1);
	Mat smooth(raster.rows,raster.cols,CV_32FC1);
    gaussian_smooth(image, smooth, gauss_sigma, gauss_size);
    //Mat magnitude(raster.rows,raster.cols,CV_32FC1);
    //Mat orientation(raster.rows,raster.cols,CV_32FC1);
    get_gradient(smooth, magnitude, orientation, hfactor, vfactor);
    Mat dirs(raster.rows,raster.cols,CV_32FC1);
    edge_directions(orientation,dirs);
    Mat thinned(raster.rows,raster.cols,CV_32FC1);
    non_maximum_suppression(magnitude, dirs,thinned);
    /* binarize(thinned, w, h, 10) */
    thinned.convertTo(res,CV_8UC1);
}

/*
 * Follows a streak
 */
int follow_streak(Mat& image, int x, int y, int *dirs, int *mark, int tag, int kill, int *ax, int *ay, int *bx, int *by)
{
	vector<Point> stack;
    int w = image.cols;
    int h = image.rows;
    uchar * data = image.data;
    int dx[] = {+1, +1,  0, -1, -1, -1,  0, +1, +1};
    int dy[] = { 0, -1, -1, -1,  0, +1, +1, +1,  0};
    /*  */
    /* xxxxx */
    /* x...x */
    /* x.x.x */
    /* x...x */
    /* xxxxx */
    /*  */
    unsigned int first = 0;
    stack.push_back(Point(x,y));
    if (kill)
    {
        data[x + w * y] = 0;
    }
    else
    {
        mark[x + w * y] = tag;
    }
    *ax = x;
    *ay = y;
    *bx = x;
    *by = y;
    while (first < stack.size())
    {
    	Point n = stack[first++];
        for (int i = 0; i < 8; i++)
        {
            int nx = n.x + dx[i];
            int ny = n.y + dy[i];
            if (nx >= 0 && ny >= 0 && nx < w && ny < h)
            {
                int ok;
                if (kill)
                {
                    ok = (mark[nx + w * ny] == tag);
                }
                else
                {
                    ok = (mark[nx + w * ny] == 0);
                }
                if (ok && data[nx + w * ny])
                {
                    if (kill)
                    {
                        data[nx + w * ny] = 0;
                    }
                    else
                    {
                        mark[nx + w * ny] = tag;
                    }
                    dirs[i]++;
                    stack.push_back(Point(nx,ny));
                    if (nx < *ax)
                    {
                        *ax = nx;
                    }
                    if (nx > *bx)
                    {
                        *bx = nx;
                    }
                    if (ny < *ay)
                    {
                        *ay = ny;
                    }
                    if (ny > *by)
                    {
                        *by = ny;
                    }
                }
            }
        }
    }
    return stack.size();
}

/*
 * We trace some edges and remove them if they certainly aren't part of the
 * wanted feature, but could confuse the hough transform later.
 */
void remove_streaks(Mat& image, int minpixels, int min_w_pixels, int min_h_pixels, int max_horizontal, int max_vertical, int max_slash, int max_backslash, int max_bound_w, int max_bound_h)
{
	int w = image.cols;
    int h = image.rows;
    vector<int> mark(w*h);
    uchar * data = image.data;
    int tag = 1;
    for (int j = 0; j < h; j++)
    {
        for (int i = 0; i < w; i++)
        {
            if (! mark[i + j * w] && data[i + j * w])
            {
                int dirs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                int ax, ay, bx, by;
                int size = follow_streak(image, i, j, dirs, &mark[0], tag, 0, &ax, &ay, &bx, &by);
                int kill = 0;
                /* Seems, using just the bounding box is enough (so we can get */
                /* rid of all the dirs counting). And using the bounding box, */
                /* instead of just the ratio, we could also already do the test for */
                /* min/max radius. */
                if (size >= minpixels && bx - ax >= min_w_pixels && by - ay >= min_h_pixels)
                /* int ho = dirs[0] + dirs[4] */
                /* int ve = dirs[2] + dirs[6] */
                /* int d1 = dirs[1] + dirs[5] */
                /* int d2 = dirs[3] + dirs[7] */
                /* printf("%d/%d: %d %d %d %d\n", i, j, ho, ve, d1, d2) */
                /* We don't want to remove the pupil here under no */
                /* circumstances, even if it is degenerated to just a small */
                /* arc - so must be quite conservative. */
                /* if ho > ve * max_horizontal or ve > ho * max_vertical: */
                /* d1 > d2 * max_slash or d2 > d1 * max_backslash or\ */
                {
                    if ((bx - ax) > max_bound_w * (by - ay) || (by - ay) > max_bound_h * (bx - ax))
                    {
                        kill = 1;
                    }
                }
                else
                {
                    kill = 1;
                }
                if (kill)
                {
                    follow_streak(image, i, j, dirs, &mark[0], tag, 1, &ax, &ay, &bx, &by);
                }
                tag++;
            }
        }
    }
}

/*
 * thresholds
 */
int bounds_bigger_than_threshold(Mat& image, int threshold, int *x, int *y, int *x_, int *y_)
{
    int w = image.cols;
    int step = image.step;
    int h = image.rows;
    uchar* data = image.data;
    int n= 0;
    *x = w - 1;
    *y = h - 1;
    *x_ = 0;
    *y_ = 0;
    for (int j = 0; j < h ; j++)
    {
        for (int i = 0; i < w; i++)
        {
            uchar c = data[j * step + i];
            if (c > threshold)
            {
                if (i < *x)
                {
                    *x = i;
                }
                if (i > *x_)
                {
                    *x_ = i;
                }
                if (j < *y)
                {
                    *y = j;
                }
                if (j > *y_)
                {
                    *y_ = j;
                }
                n++;
            }
        }
    }
    return n;
}


/*
 * Blacks out pupil
 */
void black_out_circle(Mat& image, int x, int y, int r1, int r2)
{
    int w = image.cols;
    int step = image.step;
    int h = image.rows;
    uchar* data = image.data;
    int sum = 0, n = 0;
    for (int j = 0; j < h; j++)
    {
        for (int i = 0; i < w ; i++)
        {
            int dx = i - x;
            int dy = j - y;
            int d = sqrt(dx * dx + dy * dy);
            if (d <= r2 && d >= r1)
            {
                int c = data[i + j * step];
                sum += c;
                n++;
            }
        }
    }
    sum /= n;
    for (int j = 0; j < h; j++)
    {
        for (int i = 0; i < w ; i++)
        {
            int dx = i - x;
            int dy = j - y;
            int d = sqrt(dx * dx + dy * dy);
            if (d <= r2)
            {
                int alpha = r2 * (d - r1) / (r2 - r1);
                if (alpha < 0)
                {
                    alpha = 0;
                }
                int c = data[i + j * step];
                data[i + j * step] = c * alpha / r2 + sum * (r2 - alpha) / r2;
            }
        }
    }
}

/*
 * horizon fade out
 */
void dim_above_horizon(Mat& image, int y, float a)
{
    int w = image.cols;
    int step = image.step;
    uchar* data = image.data;
    for (int j = 0; j < y; j++)
    {
        float val = (j - y) * (j - y) * a;
        val = 255 - val;
        if (val < 0)
        {
            val = 0;
        }
        for (int i = 0; i < w ; i++)
        {
            int c = data[i + j * step];
            data[i + j * step] = c * val / 255;
        }
    }
}

/*
 * Any pixels at or below the threshold are set to the given value.
 */
void threshold_below(Mat& image, int threshold, int value)
{
    int w = image.cols;
    int step = image.step;
    int h = image.rows;
    uchar* data = image.data;
    for (int j = 0; j < h; j++)
    {
        for (int i = 0; i < w; i++)
        {
            int c = data[i + j * step];
            if (c <= threshold)
            {
                data[i + j * step] = value;
            }
        }
    }
}

/*
 * Any pixels ar or above the given threshold are set to the given value.
 */
void threshold_above(Mat& image, int threshold, int value)
{
    int w = image.cols;
    int step = image.step;
    int h = image.rows;
    uchar* data = image.data;
    for (int j = 0; j < h; j++)
    {
        for (int i = 0; i < w; i++)
        {
            int c = data[i + j * step];
            if (c >= threshold)
            {
                data[i + j * step] = value;
            }
        }
    }
}

/** ------------------------------- Iris mask ------------------------------- **/

/*
 * Starting at the given location, created a mask from edges.
 */
void scan_edge(Mat& image, int x, int y, int dx, int dy, int n)
{
	int step = image.step;
    uchar * data = image.data;
    int tox = (dx > 0) ? min(x+n,image.cols) : max(x-n,-1);
    int toy = (dy > 0) ? image.rows: -1;
    //printf("x from %i to %i inc %i, y from %i to %i inc %i\n",x,tox,dx,y,toy,dy);
    double avg = 0;
    int count = 0;
    for (int i=x; i != tox; i+=dx)
    {
    	bool mask = false;
    	int j=y;
    	for (; j != toy; j+= dy){
    		if (mask) {
    			data[i + j * step] = 255;
    		}
    		else {
    		     int c = data[i + j * step];
    		     if (c > 0) {
    		    	 mask = true;
    		    	 avg += j;
    		    	 count++;
    		     }
    		}
    	}
    }
    if (count > 0){
    	int y = avg / count;
		for (int i=x; i != tox; i+=dx)
		{
			int j=y;
			for (; j != toy; j+= dy){
				data[i + j * step] = 255;
			}
		}
    }
}

/*
 * px, py, pr: Pupil position and radius
 * ix, iy, ir: Iris position and radius
 *
 * This creates a black&white image, where black pixels are considered iris
 * texture and all other pixels are considered belonging to lids.
 *
 * Right now, this is a somewhat over-zealous algorithm, often cutting away
 * more than needed, especially in the presence of eyelashes.
 */
void mask_lids(const Mat& orig_image, Mat& mask, int px, int py, int pr, int ix, int iy, int ir)
{

    int oxl = ix - ir;
    int oyl = iy - ir;
    int oxr = ix+ir;
    int oyr = iy+ir;
    int xl = min(max(0,oxl),orig_image.cols-1);
    int yl = min(max(0,oyl),orig_image.rows-1);
    int xr = min(max(0,oxr),orig_image.cols-1);
    int yr = min(max(0,oyr),orig_image.rows-1);
    Mat image(orig_image,Rect(xl,yl,xr-xl,yr-yl));
    Mat img(yr-yl,xr-xl,CV_8UC1);
    image.copyTo(img);
    Mat mag(yr-yl,xr-xl,CV_32FC1);
    Mat orient(yr-yl,xr-xl,CV_32FC1);
    Mat edges(yr-yl,xr-xl,CV_8UC1);
    int min_r = pr * 1.5;
    /* Blackout the pupil */
    black_out_circle(img, px-xl, py-yl, pr, min_r);
    canny(img,mag,orient,edges, 3, 15, 0, 1);
    threshold_below(edges, 6, 0);
    threshold_above(edges, 7, 255);
    remove_streaks(edges, 30, 30, 0, 1, 1, 1, 1, 100, 1);
    scan_edge(edges, 0, iy-yl, 1, -1, ir);
    scan_edge(edges, edges.cols - 1, iy-yl, -1, -1, ir);
    scan_edge(edges, 0, iy-yl, 1, 1, ir);
    scan_edge(edges, edges.cols - 1, iy-yl, -1, 1, ir);
    mask.setTo(255);
    Mat maskcrop(mask,Rect(xl,yl,xr-xl,yr-yl));
    edges.copyTo(maskcrop);
}


/** ------------------------------- commandline functions ------------------------------- **/

/**
 * Parses a command line
 * This routine should be called for parsing command lines for executables.
 * Note, that all options require '-' as prefix and may contain an arbitrary
 * number of optional arguments.
 *
 * cmd: commandline representation
 * argc: number of parameters
 * argv: string array of argument values
 */
void cmdRead(map<string ,vector<string> >& cmd, int argc, char *argv[]){
	for (int i=1; i< argc; i++){
		char * argument = argv[i];
		if (strlen(argument) > 1 && argument[0] == '-' && (argument[1] < '0' || argument[1] > '9')){
			cmd[argument]; // insert
			char * argument2;
			while (i + 1 < argc && (strlen(argument2 = argv[i+1]) <= 1 || argument2[0] != '-'  || (argument2[1] >= '0' && argument2[1] <= '9'))){
				cmd[argument].push_back(argument2);
				i++;
			}
		}
		else {
			CV_Error(CV_StsBadArg,"Invalid command line format");
		}
	}
}

/**
 * Checks, if each command line option is valid, i.e. exists in the options array
 *
 * cmd: commandline representation
 * validOptions: list of valid options separated by pipe (i.e. |) character
 */
void cmdCheckOpts(map<string ,vector<string> >& cmd, const string validOptions){
	vector<string> tokens;
	const string delimiters = "|";
	string::size_type lastPos = validOptions.find_first_not_of(delimiters,0); // skip delimiters at beginning
	string::size_type pos = validOptions.find_first_of(delimiters, lastPos); // find first non-delimiter
	while (string::npos != pos || string::npos != lastPos){
		tokens.push_back(validOptions.substr(lastPos,pos - lastPos)); // add found token to vector
		lastPos = validOptions.find_first_not_of(delimiters,pos); // skip delimiters
		pos = validOptions.find_first_of(delimiters,lastPos); // find next non-delimiter
	}
	sort(tokens.begin(), tokens.end());
	for (map<string, vector<string> >::iterator it = cmd.begin(); it != cmd.end(); it++){
		if (!binary_search(tokens.begin(),tokens.end(),it->first)){
			CV_Error(CV_StsBadArg,"Command line parameter '" + it->first + "' not allowed.");
			tokens.clear();
			return;
		}
	}
	tokens.clear();
}

/*
 * Checks, if a specific required option exists in the command line
 *
 * cmd: commandline representation
 * option: option name
 */
void cmdCheckOptExists(map<string ,vector<string> >& cmd, const string option){
	map<string, vector<string> >::iterator it = cmd.find(option);
	if (it == cmd.end()) CV_Error(CV_StsBadArg,"Command line parameter '" + option + "' is required, but does not exist.");
}

/*
 * Checks, if a specific option has the appropriate number of parameters
 *
 * cmd: commandline representation
 * option: option name
 * size: appropriate number of parameters for the option
 */
void cmdCheckOptSize(map<string ,vector<string> >& cmd, const string option, const unsigned int size = 1){
	map<string, vector<string> >::iterator it = cmd.find(option);
	if (it->second.size() != size) CV_Error(CV_StsBadArg,"Command line parameter '" + option + "' has unexpected size.");
}

/*
 * Checks, if a specific option has the appropriate number of parameters
 *
 * cmd: commandline representation
 * option: option name
 * min: minimum appropriate number of parameters for the option
 * max: maximum appropriate number of parameters for the option
 */
void cmdCheckOptRange(map<string ,vector<string> >& cmd, string option, unsigned int min = 0, unsigned int max = 1){
	map<string, vector<string> >::iterator it = cmd.find(option);
	unsigned int size = it->second.size();
	if (size < min || size > max) CV_Error(CV_StsBadArg,"Command line parameter '" + option + "' is out of range.");
}

/*
 * Returns the list of parameters for a given option
 *
 * cmd: commandline representation
 * option: name of the option
 */
vector<string> * cmdGetOpt(map<string ,vector<string> >& cmd, const string option){
	map<string, vector<string> >::iterator it = cmd.find(option);
	return (it != cmd.end()) ? &(it->second) : 0;
}

/*
 * Returns number of parameters in an option
 *
 * cmd: commandline representation
 * option: name of the option
 */
unsigned int cmdSizePars(map<string ,vector<string> >& cmd, const string option){
	map<string, vector<string> >::iterator it = cmd.find(option);
	return (it != cmd.end()) ? it->second.size() : 0;
}

/*
 * Returns a specific parameter type (int) given an option and parameter index
 *
 * cmd: commandline representation
 * option: name of option
 * param: name of parameter
 */
int cmdGetParInt(map<string ,vector<string> >& cmd, string option, unsigned int param = 0){
	map<string, vector<string> >::iterator it = cmd.find(option);
	if (it != cmd.end()) {
		if (param < it->second.size()) {
			return atoi(it->second[param].c_str());
		}
	}
	return 0;
}

/*
 * Returns a specific parameter type (float) given an option and parameter index
 *
 * cmd: commandline representation
 * option: name of option
 * param: name of parameter
 */
float cmdGetParFloat(map<string ,vector<string> >& cmd, const string option, const unsigned int param = 0){
	map<string, vector<string> >::iterator it = cmd.find(option);
	if (it != cmd.end()) {
		if (param < it->second.size()) {
			return atof(it->second[param].c_str());
		}
	}
	return 0;
}

/*
 * Returns a specific parameter type (string) given an option and parameter index
 *
 * cmd: commandline representation
 * option: name of option
 * param: name of parameter
 */
string cmdGetPar(map<string ,vector<string> >& cmd, const string option, const unsigned int param = 0){
	map<string, vector<string> >::iterator it = cmd.find(option);
	if (it != cmd.end()) {
		if (param < it->second.size()) {
			return it->second[param];
		}
	}
	return 0;
}

/** ------------------------------- timing functions ------------------------------- **/

/**
 * Class for handling timing progress information
 */
class Timing{
public:
	/** integer indicating progress with respect tot total **/
	int progress;
	/** total count for progress **/
	int total;

	/*
	 * Default constructor for timing initializing time.
	 * Automatically calls init()
	 *
	 * seconds: update interval in seconds
	 * eraseMode: if true, outputs sends erase characters at each print command
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
		string erase(eraseCount,'\r');
		erase.append(eraseCount,' ');
		erase.append(eraseCount,'\r');
		printf("%s",erase.c_str());
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
		float percent = 100.f * progress / total;
		boost::posix_time::time_duration passed = (current - start);
		boost::posix_time::time_duration togo = passed * (total - progress) / max(1,progress);
		if (erase) {
			string erase(eraseCount,'\r');
			printf("%s",erase.c_str());
			int newEraseCount = (progress != total) ? printf("Progress ... %3.2f%% (%i/%i Total %i:%02i:%02i.%03i Remaining ca. %i:%02i:%02i.%03i)",percent,progress,total,passed.hours(),passed.minutes(),passed.seconds(),(int)(passed.total_milliseconds()%1000),togo.hours(),togo.minutes(),togo.seconds(),(int)(togo.total_milliseconds() % 1000)) : printf("Progress ... %3.2f%% (%i/%i Total %i:%02i:%02i.%03d)",percent,progress,total,passed.hours(),passed.minutes(),passed.seconds(),(int)(passed.total_milliseconds()%1000));
			if (newEraseCount < eraseCount) {
				string erase(newEraseCount-eraseCount,' ');
				erase.append(newEraseCount-eraseCount,'\r');
				printf("%s",erase.c_str());
			}
			eraseCount = newEraseCount;
		}
		else {
			eraseCount = (progress != total) ? printf("Progress ... %3.2f%% (%i/%i Total %i:%02i:%02i.%03i Remaining ca. %i:%02i:%02i.%03i)\n",percent,progress,total,passed.hours(),passed.minutes(),passed.seconds(),(int)(passed.total_milliseconds()%1000),togo.hours(),togo.minutes(),togo.seconds(),(int)(togo.total_milliseconds() % 1000)) : printf("Progress ... %3.2f%% (%i/%i Total %i:%02i:%02i.%03d)\n",percent,progress,total,passed.hours(),passed.minutes(),passed.seconds(),(int)(passed.total_milliseconds()%1000));
		}
	}
private:
	long updateInterval;
	boost::posix_time::ptime start;
	boost::posix_time::ptime current;
	boost::posix_time::ptime lastPrint;
	int eraseCount;
	bool erase;
};

/** ------------------------------- file pattern matching functions ------------------------------- **/


/*
 * Formats a given string, such that it can be used as a regular expression
 * I.e. escapes special characters and uses * and ? as wildcards
 *
 * pattern: regular expression path pattern
 * pos: substring starting index
 * n: substring size
 *
 * returning: escaped substring
 */
string patternSubstrRegex(string& pattern, size_t pos, size_t n){
	string result;
	for (size_t i=pos, e=pos+n; i < e; i++ ) {
		char c = pattern[i];
		if ( c == '\\' || c == '.' || c == '+' || c == '[' || c == '{' || c == '|' || c == '(' || c == ')' || c == '^' || c == '$' || c == '}' || c == ']') {
			result.append(1,'\\');
			result.append(1,c);
		}
		else if (c == '*'){
			result.append("([^/\\\\]*)");
		}
		else if (c == '?'){
			result.append("([^/\\\\])");
		}
		else {
			result.append(1,c);
		}
	}
	return result;
}

/*
 * Converts a regular expression path pattern into a list of files matching with this pattern by replacing wildcards
 * starting in position pos assuming that all prior wildcards have been resolved yielding intermediate directory path.
 * I.e. this function appends the files in the specified path according to yet unresolved pattern by recursive calling.
 *
 * pattern: regular expression path pattern
 * files: the list to which new files can be applied
 * pos: an index such that positions 0...pos-1 of pattern are already considered/matched yielding path
 * path: the current directory (or empty)
 */
void patternToFiles(string& pattern, vector<string>& files, const size_t& pos, const string& path){
	size_t first_unknown = pattern.find_first_of("*?",pos); // find unknown * in pattern
	if (first_unknown != string::npos){
		size_t last_dirpath = pattern.find_last_of("/\\",first_unknown);
		size_t next_dirpath = pattern.find_first_of("/\\",first_unknown);
		if (next_dirpath != string::npos){
			boost::regex expr((last_dirpath != string::npos && last_dirpath > pos) ? patternSubstrRegex(pattern,last_dirpath+1,next_dirpath-last_dirpath-1) : patternSubstrRegex(pattern,pos,next_dirpath-pos));
			boost::filesystem::directory_iterator end_itr; // default construction yields past-the-end
			try {
				for ( boost::filesystem::directory_iterator itr( ((path.length() > 0) ? path + pattern[pos-1] : (last_dirpath != string::npos && last_dirpath > pos) ? "" : "./") + ((last_dirpath != string::npos && last_dirpath > pos) ? pattern.substr(pos,last_dirpath-pos) : "")); itr != end_itr; ++itr )
				{
					if (boost::filesystem::is_directory(itr->path())){
						boost::filesystem::path p = itr->path().filename();
						string s =  p.string();
						if (boost::regex_match(s.c_str(), expr)){
							patternToFiles(pattern,files,(int)(next_dirpath+1),((path.length() > 0) ? path + pattern[pos-1] : "") + ((last_dirpath != string::npos && last_dirpath > pos) ? pattern.substr(pos,last_dirpath-pos) + pattern[last_dirpath] : "") + s);
						}
					}
				}
			}
			catch (boost::filesystem::filesystem_error &e){}
		}
		else {
			boost::regex expr((last_dirpath != string::npos && last_dirpath > pos) ? patternSubstrRegex(pattern,last_dirpath+1,pattern.length()-last_dirpath-1) : patternSubstrRegex(pattern,pos,pattern.length()-pos));
			boost::filesystem::directory_iterator end_itr; // default construction yields past-the-end
			try {
				for ( boost::filesystem::directory_iterator itr(((path.length() > 0) ? path +  pattern[pos-1] : (last_dirpath != string::npos && last_dirpath > pos) ? "" : "./") + ((last_dirpath != string::npos && last_dirpath > pos) ? pattern.substr(pos,last_dirpath-pos) : "")); itr != end_itr; ++itr )
				{
					boost::filesystem::path p = itr->path().filename();
					string s =  p.string();
					if (boost::regex_match(s.c_str(), expr)){
						files.push_back(((path.length() > 0) ? path + pattern[pos-1] : "") + ((last_dirpath != string::npos && last_dirpath > pos) ? pattern.substr(pos,last_dirpath-pos) + pattern[last_dirpath] : "") + s);
					}
				}
			}
			catch (boost::filesystem::filesystem_error &e){}
		}
	}
	else { // no unknown symbols
		boost::filesystem::path file(((path.length() > 0) ? path + "/" : "") + pattern.substr(pos,pattern.length()-pos));
		if (boost::filesystem::exists(file)){
			files.push_back(file.string());
		}
	}
}

/**
 * Converts a regular expression path pattern into a list of files matching with this pattern
 *
 * pattern: regular expression path pattern
 * files: the list to which new files can be applied
 */
void patternToFiles(string& pattern, vector<string>& files){
	patternToFiles(pattern,files,0,"");
}

/*
 * Renames a given filename corresponding to the actual file pattern using a renaming pattern.
 * Wildcards can be referred to as ?1, ?2, ... in the order they appeared in the file pattern.
 *
 * pattern: regular expression path pattern
 * renamePattern: renaming pattern using ?1, ?2, ... as placeholders for wildcards
 * infile: path of the file (matching with pattern) to be renamed
 * outfile: path of the renamed file
 * par: used parameter (default: '?')
 */
void patternFileRename(string& pattern, const string& renamePattern, const string& infile, string& outfile, const char par = '?'){
	size_t first_unknown = renamePattern.find_first_of(par,0); // find unknown ? in renamePattern
	if (first_unknown != string::npos){
		string formatOut = "";
		for (size_t i=0, e=renamePattern.length(); i < e; i++ ) {
			char c = renamePattern[i];
			if ( c == par && i+1 < e) {
				c = renamePattern[i+1];
				if (c > '0' && c <= '9'){
					formatOut.append(1,'$');
					formatOut.append(1,c);
				}
				else {
					formatOut.append(1,par);
					formatOut.append(1,c);
				}
				i++;
			}
			else {
				formatOut.append(1,c);
			}
		}
		boost::regex patternOut(patternSubstrRegex(pattern,0,pattern.length()));
		outfile = boost::regex_replace(infile,patternOut,formatOut,boost::match_default | boost::format_perl);
	} else {
		outfile = renamePattern;
	}
}

/** ------------------------------- Program ------------------------------- **/

/*
 * Main program
 */
int main(int argc, char *argv[])
{
	int mode = MODE_HELP;
	map<string,vector<string> > cmd;
	try {
		cmdRead(cmd,argc,argv);
    	if (cmd.size() == 0 || cmdGetOpt(cmd,"-h") != 0) mode = MODE_HELP;
    	else mode = MODE_MAIN;
    	if (mode == MODE_MAIN){
			// validate command line
			cmdCheckOpts(cmd,"-i|-o|-s|-e|-q|-t|-m|-rm|-rr|-em|-gr|-ic|-po|-fb|-ep|-ib|-ob|-sr|-bm|-lt|-so|-si|-tr|-l");
			cmdCheckOptExists(cmd,"-i");
			cmdCheckOptSize(cmd,"-i",1);
			string inFiles = cmdGetPar(cmd,"-i");
			cmdCheckOptExists(cmd,"-o");
			cmdCheckOptSize(cmd,"-o",1);
			string outFiles = cmdGetPar(cmd,"-o");
			int outWidth = 512, outHeight = 64;
			if (cmdGetOpt(cmd,"-s") != 0){
				cmdCheckOptSize(cmd,"-s",2);
				outWidth = cmdGetParInt(cmd,"-s",0);
				outHeight = cmdGetParInt(cmd,"-s",1);
			}
			bool enhance = false;
			if (cmdGetOpt(cmd,"-e") != 0){
				cmdCheckOptSize(cmd,"-e",0);
				enhance = true;
			}
			bool quiet = false;
			if (cmdGetOpt(cmd,"-q") != 0){
				cmdCheckOptSize(cmd,"-q",0);
				quiet = true;
			}
			bool time = false;
			if (cmdGetOpt(cmd,"-t") != 0){
				cmdCheckOptSize(cmd,"-t",0);
				time = true;
			}
			string maskFiles;
			if (cmdGetOpt(cmd,"-m") != 0){
				cmdCheckOptSize(cmd,"-m",1);
				maskFiles = cmdGetPar(cmd,"-m");
			}
			string rmaskFiles;
			if (cmdGetOpt(cmd,"-rm") != 0){
				cmdCheckOptSize(cmd,"-rm",1);
				rmaskFiles = cmdGetPar(cmd,"-rm");
			}
			string rremovFiles;
			if (cmdGetOpt(cmd,"-rr") != 0){
				cmdCheckOptSize(cmd,"-rr",1);
				rremovFiles = cmdGetPar(cmd,"-rr");
			}
			string emaskFiles;
			if (cmdGetOpt(cmd,"-em") != 0){
				cmdCheckOptSize(cmd,"-em",1);
				emaskFiles = cmdGetPar(cmd,"-em");
			}
			string gradFiles;
			if (cmdGetOpt(cmd,"-gr") != 0){
				cmdCheckOptSize(cmd,"-gr",1);
				gradFiles = cmdGetPar(cmd,"-gr");
			}
			string icentFiles;
			if (cmdGetOpt(cmd,"-ic") != 0){
				cmdCheckOptSize(cmd,"-ic",1);
				icentFiles = cmdGetPar(cmd,"-ic");
			}
			string polarFiles;
			if (cmdGetOpt(cmd,"-po") != 0){
				cmdCheckOptSize(cmd,"-po",1);
				polarFiles = cmdGetPar(cmd,"-po");
			}
			string fboundFiles;
			if (cmdGetOpt(cmd,"-fb") != 0){
				cmdCheckOptSize(cmd,"-fb",1);
				fboundFiles = cmdGetPar(cmd,"-fb");
			}
			string ellpolFiles;
			if (cmdGetOpt(cmd,"-ep") != 0){
				cmdCheckOptSize(cmd,"-ep",1);
				ellpolFiles = cmdGetPar(cmd,"-ep");
			}
			string iboundFiles;
			if (cmdGetOpt(cmd,"-ib") != 0){
				cmdCheckOptSize(cmd,"-ib",1);
				iboundFiles = cmdGetPar(cmd,"-ib");
			}
			string oboundFiles;
			if (cmdGetOpt(cmd,"-ob") != 0){
				cmdCheckOptSize(cmd,"-ob",1);
				oboundFiles = cmdGetPar(cmd,"-ob");
			}
			string segresFiles;
			if (cmdGetOpt(cmd,"-sr") != 0){
				cmdCheckOptSize(cmd,"-sr",1);
				segresFiles = cmdGetPar(cmd,"-sr");
			}
			string binmaskFiles;
			if (cmdGetOpt(cmd,"-bm") != 0){
				cmdCheckOptSize(cmd,"-bm",1);
				binmaskFiles = cmdGetPar(cmd,"-bm");
			}
			int lt = 1;
			if (cmdGetOpt(cmd,"-lt") != 0){
				cmdCheckOptSize(cmd,"-lt",1);
				lt = cmdGetParInt(cmd,"-lt");
			}
            float outer_scale = 1;
            if (cmdGetOpt(cmd,"-so") != 0){
                cmdCheckOptSize(cmd, "-so", 1);
                outer_scale = cmdGetParFloat(cmd, "-so", 0);
                if ( outer_scale <=0){
                    cerr << "outer_scale factor (-so) <=0, this will result in an error" << endl;
                }
                outer_scale = sqrt(outer_scale);
            }
            float inner_scale = 1;
            if (cmdGetOpt(cmd,"-si") != 0){
                cmdCheckOptSize(cmd, "-si", 1);
                inner_scale = cmdGetParFloat(cmd, "-si", 0);
                if ( inner_scale <=0){
                    cerr << "inner_scale factor (-si) <=0, this will result in an error" << endl;
                }
                inner_scale = sqrt(inner_scale);
            }
            float translate = 0.;
            if (cmdGetOpt(cmd,"-tr") != 0){
                cmdCheckOptSize(cmd, "-tr", 1);
                translate = cmdGetParFloat(cmd, "-tr", 0);
            }
			ofstream logFile;
			if (cmdGetOpt(cmd,"-l") != 0){
				cmdCheckOptSize(cmd,"-l",1);
                string logFileName = cmdGetPar(cmd,"-l");
				logFile.open(logFileName.c_str()); 
                if( !logFile) cerr << "Failed to open logfile ("<<logFileName<<") for writing"<<endl;
			}
            if( logFile.is_open()){ 
                    logFile << "# Filename" << ", " 
                            << "inner.x" << ", " 
                            << "inner.y" << ", " 
                            << "inner.width" << ", " 
                            << "inner.height" << ", " 
                            << "inner.angle" << ", " 
                            << "outer.x" << ", "
                            << "outer.y" << ", "
                            << "outer.width" << ", "
                            << "outer.height" << ", "
                            << "outer.angle" << endl;
            }
			// starting routine
			Timing timing(1,quiet);
			vector<string> files;
			patternToFiles(inFiles,files);
			CV_Assert(files.size() > 0);
			timing.total = files.size();
			for (vector<string>::iterator inFile = files.begin(); inFile != files.end(); ++inFile, timing.progress++){
				if (!quiet) printf("Loading image '%s' ...\n", (*inFile).c_str());;
				// MODIFICATION TB, June 3rd, 2014
				// additional conversion step to enable direct JP2K processing (Bug in CV)
				// Loading of JP2k in color is supported, loading as grayscale, however, not
				Mat imgCol = imread(*inFile, CV_LOAD_IMAGE_COLOR);			
				Mat img;			
				cvtColor(imgCol,img,CV_BGR2GRAY);
				CV_Assert(img.data != 0);
				Mat orig;
				if (!segresFiles.empty()) img.copyTo(orig);
				int width = img.cols;
				int height = img.rows;
				int cx1 = 0, cy1 = 0, cr1 = 0;
				int cx2 = 0, cy2 = 0, cr2 = 0;
				int cx3 = 0, cy3 = 0, cr3 = 0;
				if (!quiet) printf("Removing reflections ...\n");
				Mat mask(height,width,CV_8UC1);
				createReflectionMask(img, mask);
				if (!rmaskFiles.empty()){
					string rmaskFile;
					patternFileRename(inFiles,rmaskFiles,*inFile,rmaskFile);
					if (!quiet) printf("Storing reflection mask '%s' ...\n",rmaskFile.c_str());
					if (!imwrite(rmaskFile,mask)) CV_Error(CV_StsError,"Could not save image '" + rmaskFile + "'");
				}
				inpaint(img,mask,img,10,INPAINT_NS);
				if (!rremovFiles.empty()){
					string nonreflectFile;
					patternFileRename(inFiles,rremovFiles,*inFile,nonreflectFile);
					if (!quiet) printf("Storing reflection-removed image '%s' ...\n",nonreflectFile.c_str());
					if (!imwrite(nonreflectFile,img)) CV_Error(CV_StsError,"Could not save image '" + nonreflectFile + "'");
				}
				if (!quiet) printf("Estimating gradient information ...\n");
				Mat gradX(height,width,CV_32FC1);
				Mat gradY(height,width,CV_32FC1);
				Mat mag(height,width,CV_32FC1);
				Sobel(img,gradX,gradX.depth(),1,0,7);
				Sobel(img,gradY,gradY.depth(),0,1,7);
				maskValue(gradX,gradX,mask,255,0);
				maskValue(gradY,gradY,mask,255,0);
				magnitude(gradX,gradY,mag);
				if (!gradFiles.empty()){
					Mat visual;
					Mat gradPhase(height,width,CV_32FC1);
					phase(gradX,gradY,gradPhase,true);
					vector<Mat> planes(3);
					for (int i=0; i<3; i++) planes[i].create(height,width,CV_8UC1);
					planes[1].setTo(255);
					double minVal, maxVal;
					minMaxLoc(mag,&minVal,&maxVal);
					mag.convertTo(planes[2],CV_8UC1,255.f/maxVal);
					gradPhase.convertTo(planes[0],CV_8UC1,0.5f);
					merge(planes,visual);
					cvtColor(visual,visual,CV_HSV2BGR);
					string gradFile;
					patternFileRename(inFiles,gradFiles,*inFile,gradFile);
					if (!quiet) printf("Storing gradient image '%s' ...\n",gradFile.c_str());
					if (!imwrite(gradFile,visual)) CV_Error(CV_StsError,"Could not save image '" + gradFile + "'");
				}
				Mat boundaryEdges(height,width,CV_8UC1);
				createBoundaryMask(img,boundaryEdges,gradX,gradY,mag);
				if (!emaskFiles.empty()){
					string emaskFile;
					patternFileRename(inFiles,emaskFiles,*inFile,emaskFile);
					if (!quiet) printf("Storing edges mask '%s' ...\n",emaskFile.c_str());
					if (!imwrite(emaskFile,boundaryEdges)) CV_Error(CV_StsError,"Could not save image '" + emaskFile + "'");
				}
				if (!quiet) printf("Detecting initial center ...\n");
				float centerX, centerY;
				detectEyeCenter(gradX,gradY,mag,boundaryEdges,centerX, centerY,.5f,101);
				if (centerX < 0 || centerY < 0 || centerX >= width || centerY >= height){
					if (!quiet) printf("Warning: Center not in bounds, correcting to image center.\n");
					centerX = width/2;
					centerY = height/2;
				}
				else {
					if (!quiet) printf("Initial center: (x,y) = (%f,%f)\n", centerX, centerY);
				}
				if (!icentFiles.empty()){
					Mat visual;
					cvtColor(img,visual,CV_GRAY2BGR);
					line(visual,Point2f(centerX,0),Point2f(centerX,img.rows),Scalar(0,0,255,0),lt);
					line(visual,Point2f(0,centerY),Point2f(img.cols,centerY),Scalar(0,0,255,0),lt);
					string icentFile;
					patternFileRename(inFiles,icentFiles,*inFile,icentFile);
					if (!quiet) printf("Storing initial center '%s' ...\n", icentFile.c_str());
					if (!imwrite(icentFile,visual)) CV_Error(CV_StsError,"Could not save image '" + icentFile + "'");
				}
				if (!quiet) printf("Detecting first boundary ...\n");
				int polarWidth = outWidth;
				int polarHeight = cvRound(polarWidth * height / ((float)(width)));
				Mat polar (polarHeight,polarWidth,CV_8UC1);
				Mat polarMask (polarHeight,polarWidth,CV_8UC1);
				Mat polarGrad (polarHeight,polarWidth,CV_32FC1);
				Mat cont(1,polarWidth,CV_32FC1);
				Mat cart (1,polarWidth,CV_32FC2);
				Mat sub;
				float resolution = polarTransform(img,polar,centerX,centerY,-1,INTER_LINEAR_REPEAT);
				if (!polarFiles.empty()){
					string polarFile;
					patternFileRename(inFiles,polarFiles,*inFile,polarFile);
					if (!quiet) printf("Storing polar image '%s' ...\n", polarFile.c_str());
					if (!imwrite(polarFile,polar)) CV_Error(CV_StsError,"Could not save image '" + polarFile + "'");
				}
				polarTransform(mask,polarMask,centerX,centerY,-1,INTER_NEAREST);
				findHorizontalEdges(polar,polarMask,polarGrad);
				initContour(polarGrad,cont,12,polarHeight); // CHANGED, REMOVE 115 15
				gradientFit(cont,polarGrad,15);
				float feng = 0;
				fourierNormalize(cont,cont,feng,1);
				gradientFit(cont,polarGrad,5);
				fourierNormalize(cont,cont,feng,3);
				if (!fboundFiles.empty()){
					Mat visual;
					cvtColor(polar,visual,CV_GRAY2BGR);
					MatIterator_<float> it, ite;
					int i=0;
					for (it = cont.begin<float>(), ite = (cont.end<float>()-1); it < ite; it++, i++){
						line(visual,Point2f(i,*it),Point2f(i+1, it[1]),Scalar(0,0,255,0),lt);
					}
					string fboundFile;
					patternFileRename(inFiles,fboundFiles,*inFile,fboundFile);
					if (!quiet) printf("Storing first boundary image '%s' ...\n", fboundFile.c_str());
					if (!imwrite(fboundFile,visual)) CV_Error(CV_StsError,"Could not save image '" + fboundFile + "'");
				}
				polar2Cart(cont, cart, 0, 0, resolution);
				//cartSectors(cart, sub);
				//RotatedRect boundary = fitEllipse(sub);
				RotatedRect boundary = fitEllipse(cart);
				boundary.center.x += centerX;
				boundary.center.y += centerY;
				Mat cartRefBoundary(1,polarWidth,CV_32FC2);
				ellipse2Cart(cartRefBoundary,boundary);
                
                RotatedRect boundary_cartref_outer_scale = boundary; //scale
                float translate_from_outer_ref = boundary_cartref_outer_scale.size.width * translate; // translate
                boundary_cartref_outer_scale.center.x += translate_from_outer_ref;
                boundary_cartref_outer_scale.size.width *= outer_scale; // scale
                boundary_cartref_outer_scale.size.height *= outer_scale; //scale
                
                RotatedRect boundary_cartref_inner_scale = boundary; //scale
                boundary_cartref_inner_scale.center.x += translate_from_outer_ref;
                boundary_cartref_inner_scale.size.width *= inner_scale; // scale
                boundary_cartref_inner_scale.size.height *= inner_scale; //scale

				float enRefBoundary = boundaryEnergy(mag,cartRefBoundary);
				cx1 = centerX;
				cy1 = centerY;
				cr1 = (boundary.size.width + boundary.size.height) / 2;
				if (!quiet) printf("Boundary found with energy: %f\n", enRefBoundary);
				if (!quiet) printf("Refined center: (x,y) = (%f,%f)\n", centerX, centerY);
				if (abs(boundary.size.width) < 0.0001 || abs(boundary.size.height) < 0.0001){
					boundary.size.width = 1; boundary.size.height = 1;
				}
				if (!quiet) printf("Ellipsopolar transform and image enhancement ...\n");
				Mat ellipsopolar (polarHeight,polarWidth,CV_8UC1);
				Mat ellipsopolarMask (polarHeight,polarWidth,CV_8UC1);
				float ellResolution = ellipsopolarTransform(img,ellipsopolar,boundary,-1,INTER_LINEAR_REPEAT);
				if (!ellpolFiles.empty()){
					string ellpolFile;
					patternFileRename(inFiles,ellpolFiles,*inFile,ellpolFile);
					if (!quiet) printf("Storing polar image '%s' ...\n", ellpolFile.c_str());
					if (!imwrite(ellpolFile,ellipsopolar)) CV_Error(CV_StsError,"Could not save image '" + ellpolFile + "'");
				}
				ellipsopolarTransform(mask,ellipsopolarMask,boundary,-1,INTER_NEAREST);
				// enhance images and refer to subimages for inner/outer boundary detection
				int heightInner = max(1,cvRound(1.f/ellResolution));
				int heightOuter = polarHeight - heightInner;
				Mat ellipsopolarInner (ellipsopolar,Rect(0,0,polarWidth,heightInner));
				Mat ellipsopolarOuter (ellipsopolar,Rect(0,heightInner,polarWidth,heightOuter));
				equalizeHist(ellipsopolarInner,ellipsopolarInner);
				clahe(ellipsopolarOuter,ellipsopolarOuter,polarWidth,min(heightOuter, heightInner * 3));
				// calculate gradient
				Mat ellipsopolarGrad (polarHeight,polarWidth,CV_32FC1);
				findHorizontalEdges(ellipsopolar,ellipsopolarMask,ellipsopolarGrad);
				RotatedRect innerEll, outerEll;
				float enInner = - FLT_MAX, enOuter = - FLT_MAX;
				Mat cartInner (1,polarWidth,CV_32FC2);
				Mat cartOuter (1,polarWidth,CV_32FC2);
				if (!quiet) printf("Detecting inner boundary candidate ...\n");
				// detect boundary candidates
				double myInner = heightInner * 0.66;//.66
				double sigmaInner = heightInner * 0.4;//.4
				double miny = 21;
				double maxy = heightInner-21;
                RotatedRect fromInner, fromOuter; // keep the source of the inner and outer iris for printout
				if (miny < maxy){
					initContour(ellipsopolarGrad,cont,miny,maxy,false,sigmaInner,myInner);// was sigmaouter
					gradientFit(cont,ellipsopolarGrad,15,miny,maxy);
					fourierNormalize(cont,cont,feng,1);
					gradientFit(cont,ellipsopolarGrad,5,miny,maxy);
					fourierNormalize(cont,cont,feng,3);
					if (!iboundFiles.empty()){
						Mat visual;
						cvtColor(ellipsopolar,visual,CV_GRAY2BGR);
						MatIterator_<float> it, ite;
						int i=0;
						for (it = cont.begin<float>(), ite = (cont.end<float>()-1); it < ite; it++, i++){
							line(visual,Point2f(i,*it),Point2f(i+1, it[1]),Scalar(0,0,255,0),lt);
						}
						string iboundFile;
						patternFileRename(inFiles,iboundFiles,*inFile,iboundFile);
						if (!quiet) printf("Storing i border image '%s' ...\n", iboundFile.c_str());
						if (!imwrite(iboundFile,visual)) CV_Error(CV_StsError,"Could not save image '" + iboundFile + "'");
					}
					ellipsopolar2Cart(cont, cart, boundary, ellResolution);
					//cartSectors(cart,sub); // this usually took a different sector
					//innerEll = fitEllipse(sub);
					innerEll = fitEllipse(cart);
					ellipse2Cart(cartInner,innerEll);
					enInner = boundaryEnergyWeighted(mag,cartInner,boundary,ellResolution,myInner, sigmaInner);//boundaryEnergy(mag,cartInner);//
					if (!quiet) printf("Inner boundary candidate found with energy: %f\n", enInner);
					RotatedRect innerEll_scale = innerEll;
                    innerEll_scale.center.x += translate_from_outer_ref;
                    innerEll_scale.size.width *= inner_scale;
                    innerEll_scale.size.height *= inner_scale;
					ellipse2Cart(cartInner,innerEll_scale);
                    fromInner = innerEll_scale;
					cx2 = innerEll_scale.center.x;
					cy2 = innerEll_scale.center.y;
					cr2 = (innerEll_scale.size.width + innerEll_scale.size.height) / 2;
				}
				if (!quiet) printf("Detecting outer boundary candidate ...\n");
				double myOuter = heightInner * 2.5;//2.5
				double sigmaOuter = heightInner * 1;// 1
				miny = heightInner+21;
				maxy = polarHeight-21;
				if (miny < maxy){
					initContour(ellipsopolarGrad,cont,miny,maxy,true,sigmaOuter,myOuter);// was sigmaouter
					gradientFit(cont,ellipsopolarGrad,15,miny,maxy);
					fourierNormalize(cont,cont,feng,1);
					gradientFit(cont,ellipsopolarGrad,5,miny,maxy);
					fourierNormalize(cont,cont,feng,3);
					if (!oboundFiles.empty()){
						Mat visual;
						cvtColor(ellipsopolar,visual,CV_GRAY2BGR);
						MatIterator_<float> it, ite;
						int i=0;
						for (it = cont.begin<float>(), ite = (cont.end<float>()-1); it < ite; it++, i++){
							line(visual,Point2f(i,*it),Point2f(i+1, it[1]),Scalar(0,0,255,0),lt);
						}
						string oboundFile;
						patternFileRename(inFiles,oboundFiles,*inFile,oboundFile);
						if (!quiet) printf("Storing border image '%s' ...\n", oboundFile.c_str());
						if (!imwrite(oboundFile,visual)) CV_Error(CV_StsError,"Could not save image '" + oboundFile + "'");
					}
					ellipsopolar2Cart(cont, cart, boundary, ellResolution);
					//cartSectors(cart,sub); // this usually took a different sector
					//outerEll = fitEllipse(sub);
					outerEll = fitEllipse(cart);
					ellipse2Cart(cartOuter,outerEll);
					enOuter = boundaryEnergyWeighted(mag,cartOuter,boundary,ellResolution,myOuter, sigmaOuter);//boundaryEnergy(mag,cartInner);//
					if (!quiet) printf("Outer boundary candidate found with energy: %f\n", enOuter);
                    //scaling outer boundary
					RotatedRect outerEll_scale = outerEll;
                    outerEll_scale.center.x += translate_from_outer_ref;
                    outerEll_scale.size.width *= outer_scale;
                    outerEll_scale.size.height *= outer_scale;
					ellipse2Cart(cartOuter,outerEll_scale);
                    fromOuter = outerEll_scale;
					cx3 = outerEll_scale.center.x;
					cy3 = outerEll_scale.center.y;
					cr3 = (outerEll_scale.size.width + outerEll_scale.size.height) / 2;
				}
				if (!quiet) printf("Selecting better candidate ...\n");
				int px = 0, py = 0, pr = 0, ix = 0, iy = 0, ir = 0;
				if (enInner > enOuter){
                    ellipse2Cart(cartRefBoundary,boundary_cartref_outer_scale);
					cartOuter = cartRefBoundary;
                    fromOuter = boundary_cartref_outer_scale;
					px = cx2;
					py = cy2;
					pr = cr2;
					ix = cx1;
					iy = cy1;
					ir = cr1;
				}
				else {
                    ellipse2Cart(cartRefBoundary,boundary_cartref_inner_scale);
					cartInner = cartRefBoundary;
                    fromInner = boundary_cartref_inner_scale;
					px = cx1;
					py = cy1;
					pr = cr1;
					ix = cx3;
					iy = cy3;
					ir = cr3;
				}
				Mat imask;
				if (!maskFiles.empty()){
					imask.create(height,width,CV_8UC1);
					mask_lids(img, imask, px, py, pr, ix, iy, ir);
					threshold(imask,imask,1,255,CV_THRESH_BINARY_INV);
				}
				if (!segresFiles.empty()){
					Mat visual;
					cvtColor(orig,visual,CV_GRAY2BGR);
					for (float * it = (float *)cartInner.data, * ite = it + (2*polarWidth - 2); it < ite; it+=2){
						line(visual,Point2f(*it,it[1]),Point2f(it[2], it[3]),Scalar(0,0,255,0),lt);
					}
					for (float * it = (float *) cartOuter.data, * ite = it + (2*polarWidth - 2); it < ite; it+=2){
						line(visual,Point2f(*it,it[1]),Point2f(it[2], it[3]),Scalar(0,255,0,0),lt);
					}
					string vsegmentfile;
					patternFileRename(inFiles,segresFiles,*inFile,vsegmentfile);
					if (!quiet) printf("Storing segmentation image '%s' ...\n", vsegmentfile.c_str());
					if (!imwrite(vsegmentfile,visual)) CV_Error(CV_StsError,"Could not save image '" + vsegmentfile + "'");
				}
				if (!binmaskFiles.empty()){
					Mat bw(height, width, CV_8UC1, Scalar(0));
					vector<Point> iris_points;
					float * it = (float *)cartOuter.data;
					for (int i = 0; i < outWidth; i++){
						iris_points.push_back(Point2i(cvRound(*it), cvRound(it[1])));
						it += 2;
					}
					const Point* irisp[1] = { &iris_points[0] };
					vector<Point> pupil_points;
					it = (float *)cartInner.data;
					for (int i = 0; i < outWidth; i++){
						pupil_points.push_back(Point2f(cvRound(*it), cvRound(it[1])));
						it += 2;
					}
					const Point* pupilp[1] = { &pupil_points[0] };
					fillPoly(bw, irisp, &outWidth, 1, Scalar(255, 255, 255));
					fillPoly(bw, pupilp, &outWidth, 1, Scalar(0, 0, 0));
					if (!binmaskFiles.empty()) {
						string binmaskFile;
						patternFileRename(inFiles, binmaskFiles, *inFile, binmaskFile);
						if (!quiet) printf("Storing binary mask '%s' ...\n", binmaskFile.c_str());
						if (!imwrite(binmaskFile, bw)) CV_Error(CV_StsError, "Could not save image '" + binmaskFile + "'");
					}
				}
				if (!quiet) printf("Creating final texture ...\n");
				Mat out (outHeight,outWidth,CV_8UC1);
                if (!quiet) printf("Inner RotatedRect at (%f,%f) size (%f,%f) angle (%f)\n", fromInner.center.x, fromInner.center.y, fromInner.size.width, fromInner.size.height, fromInner.angle);
                if (!quiet) printf("Outer RotatedRect at (%f,%f) size (%f,%f) angle (%f)\n", fromOuter.center.x, fromOuter.center.y, fromOuter.size.width, fromOuter.size.height, fromOuter.angle);
                if( logFile.is_open()){ 
                    logFile << inFile->c_str() << ", " 
                            << fromInner.center.x << ", " 
                            << fromInner.center.y << ", " 
                            << fromInner.size.width << ", " 
                            << fromInner.size.height << ", " 
                            << fromInner.angle << ", " 
                            << fromOuter.center.x << ", "
                            << fromOuter.center.y << ", "
                            << fromOuter.size.width << ", "
                            << fromOuter.size.height << ", "
                            << fromOuter.angle << endl;
                }
				rubbersheet(img, out, cartInner, cartOuter, INTER_LINEAR);
				if (enhance){
					if (!quiet) printf("Enhancing texture ...\n");
					clahe(out,out,width/8,height/2);
				}
				if (!maskFiles.empty()){
					Mat maskout (outHeight,outWidth,CV_8UC1);
					rubbersheet(imask, maskout, cartInner, cartOuter, INTER_NEAREST);
					string maskfile;
					patternFileRename(inFiles,maskFiles,*inFile,maskfile);
					if (!quiet) printf("Storing mask image '%s' ...\n", maskfile.c_str());
					if (!imwrite(maskfile,maskout)) CV_Error(CV_StsError,"Could not save image '" + maskfile + "'");
				}
				string outfile;
				patternFileRename(inFiles,outFiles,*inFile,outfile);
				if (!quiet) printf("Storing image '%s' ...\n", outfile.c_str());
				if (!imwrite(outfile,out)) CV_Error(CV_StsError,"Could not save image '" + outfile + "'");
				if (time && timing.update()) timing.print();
			}
			if (time && quiet) timing.clear();
    	}
    	else if (mode == MODE_HELP){
			// validate command line
			cmdCheckOpts(cmd,"-h");
			if (cmdGetOpt(cmd,"-h") != 0) cmdCheckOptSize(cmd,"-h",0);
			// starting routine
			printUsage();
    	}
    }
	catch (cv::Exception e){
	   	printf("Exit with errors.\n");
	   	exit(EXIT_FAILURE);
	}
	catch (...){
	   	printf("Exit with errors.\n");
	   	exit(EXIT_FAILURE);
	}
    return EXIT_SUCCESS;
}
