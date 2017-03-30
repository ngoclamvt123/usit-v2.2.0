/*
 * cr.cpp
 *
 * Author: C. Rathgeb (crathgeb@cosy.sbg.ac.at), P. Wild (pwild@cosy.sbg.ac.at)
 *
 * Generates an iris code from iris texture using the Rathgeb algorithm
 *
 */
#include "version.h"
#include <cstdio>
#include <map>
#include <vector>
#include <string>
#include <cstring>
#include <ctime>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/regex.hpp>
#include <boost/filesystem.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

using namespace std;
using namespace cv;

#define CR_FEATHEIGHT 3

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
    printf("| cr - Iris-code generation (feature extraction) using the CR algorithm       |\n");
    printf("|                                                                             |\n");
    printf("| Rathgeb C., Uhl A.: Secure Iris Recognition based on Local Intensity        |\n");
    printf("| Variations. In Proc. of the Int'l Conf. on Image Analysis and Recognition   |\n");
    printf("| (ICIAR'10), pp. 266-275, Springer LNCS, 6112, (2010)                        |\n");
    printf("|                                                                             |\n");
    printf("| MODES                                                                       |\n");
    printf("|                                                                             |\n");
    printf("| (# 1) CR iris code extraction from iris textures                            |\n");
    printf("| (# 2) usage                                                                 |\n");
    printf("|                                                                             |\n");
    printf("| ARGUMENTS                                                                   |\n");
    printf("|                                                                             |\n");
    printf("+------+------------+---+---+-------------------------------------------------+\n");
    printf("| Name | Parameters | # | ? | Description                                     |\n");
    printf("+------+------------+---+---+-------------------------------------------------+\n");
    printf("| -i   | infile     | 1 | N | input iris texture (use * as wildcard, all other|\n");
    printf("|      |            |   |   | file parameters may refer to n-th * with ?n)    |\n");
    printf("| -o   | outfile    | 1 | N | output iris code image                          |\n");
    printf("| -q   |            | 1 | Y | quiet mode on (off)                             |\n");
    printf("| -t   |            | 1 | Y | time progress on (off)                          |\n");
    printf("| -h   |            | 2 | N | prints usage                                    |\n");
    printf("+------+------------+---+---+-------------------------------------------------+\n");
    printf("|                                                                             |\n");
    printf("| EXAMPLE USAGE                                                               |\n");
    printf("|                                                                             |\n");
    printf("| -i s1.tiff -o s1.png                                                        |\n");
    printf("| -i *.tiff -o ?1.png -q -t                                                   |\n");
    printf("|                                                                             |\n");
    printf("| AUTHORS                                                                     |\n");
    printf("|                                                                             |\n");
    printf("| Christian Rathgeb (crathgeb@cosy.sbg.ac.at)                                 |\n");
    printf("| Peter Wild (pwild@cosy.sbg.ac.at)                                           |\n");
    printf("|                                                                             |\n");
    printf("| COPYRIGHT                                                                   |\n");
    printf("|                                                                             |\n");
    printf("| (C) 2012 All rights reserved. Do not distribute without written permission. |\n");
    printf("+-----------------------------------------------------------------------------+\n");
}

/** ------------------------------- image processing functions ------------------------------- **/

void setBitTo1(uchar* code, int bitpos){
    code[bitpos / 8] |= (1 << (7-(bitpos % 8)));
}

void setBitTo0(uchar* code, int bitpos){
    code[bitpos / 8] &= (0xff ^ 1 << (7-(bitpos % 8)));
}

/* 
 * Smooth the extracted intensity variations
 * 
 * features: Intensity variations
 * rows: Number of rows
 * cols: Number of columns
 *
 */
int * cr_smooth(int *features, int rows, int cols)
{
    int i;
    
    /* iteratively smooth the feature vector (outlier of length 1) */
    for (i = 1; i < cols * rows * 2 - 1; i++)
    {
        if (features[i - 1] == features[i + 1])
        {
            features[i] = features[i - 1];
        }
    }
    /* iteratively smooth the feature vector (outlier of length 2) */
    for (i = 2; i < cols * rows * 2 - 2; i++)
    {
        if (features[i - 2] == features[i + 2])
        {
            features[i - 1] = features[i - 2];
            features[i] = features[i - 2];
            features[i + 1] = features[i - 2];
        }
    }
    return features;
}

/*
 * The Rathgeb feature extraction algorithm
 *
 * code: Code matrix
 * texture: texture matrix
 */
void featureExtract(Mat& code, const Mat& texture)
{
    int i, j;
    int pos, feat_pos;
    /* number of processed rows */
    int rows = texture.rows / CR_FEATHEIGHT;
    uchar* textureData = texture.data;
    uchar* iris_code = code.data;
    /* features (for light/ dark intension variations) */
    Mat features(1,texture.cols * rows  * 2,CV_32FC1);
    int * featuresData = (int *) features.data;
    
    /* process each row  for light intension variations */
    for (i = 0; i < texture.rows / CR_FEATHEIGHT; i++)
    {
        /* get the starting position  (center of the stripe) */
        pos = (i * CR_FEATHEIGHT + CR_FEATHEIGHT / 2) * texture.cols;
        /* initial feature to start with */
        featuresData[i * texture.cols] = CR_FEATHEIGHT / 2;
        
        /* process a single row and encode variations */
        for (j = 0; j < texture.cols; j++)
        {
            if (featuresData[i * texture.cols + j] == 0)
            {
                if (textureData[pos + 1 + texture.cols] > textureData[pos + 1])
                {
                    featuresData[i * texture.cols + j + 1] = featuresData[i * texture.cols + j] + 1;
                    pos = pos + 1 + texture.cols;
                }
                else
                {
                    featuresData[i * texture.cols + j + 1] = featuresData[i * texture.cols + j];
                    pos = pos + 1;
                }
            }
            else if (featuresData[i * texture.cols + j] == CR_FEATHEIGHT-1)
            {
                if (textureData[pos + 1 - texture.cols] > textureData[pos + 1])
                {
                    featuresData[i * texture.cols + j + 1] = featuresData[i * texture.cols + j] - 1;
                    pos = pos + 1 - texture.cols;
                }
                else
                {
                    featuresData[i * texture.cols + j + 1] = featuresData[i * texture.cols + j];
                    pos = pos + 1;
                }
            }
            else
            {
                /* proceed with upper row */
                if ((textureData[pos + 1 - texture.cols] > textureData[pos + 1]) && (textureData[pos + 1 - texture.cols] > textureData[pos + 1 + texture.cols]))
                {
                    featuresData[i * texture.cols + j + 1] = featuresData[i * texture.cols + j] - 1;
                    pos = pos + 1 - texture.cols;
                }
                /* proceed with lower row */
                else if ((textureData[pos + 1 + texture.cols] > textureData[pos + 1]) && (textureData[pos + 1 + texture.cols] > textureData[pos + 1 - texture.cols]))
                {
                    featuresData[i * texture.cols + j + 1] = featuresData[i * texture.cols + j] + 1;
                    pos = pos + 1 + texture.cols;
                }
                /* proceed with current row */
                else 
                {
                    featuresData[i * texture.cols + j + 1] = featuresData[i * texture.cols + j];
                    pos = pos + 1;
                }
            }
        }
    } 
    /* reset feature position */
    feat_pos = texture.cols * (texture.rows / CR_FEATHEIGHT);
    
    /* process each row  for light intension variations */
    for (i = 0; i < texture.rows / CR_FEATHEIGHT; i++)
    {
        /* get the starting position  (center of the stripe) */
        pos = (i * CR_FEATHEIGHT + CR_FEATHEIGHT / 2) * texture.cols;
        /* initial feature to start with */
        featuresData[i * texture.cols + feat_pos] = CR_FEATHEIGHT/2;
        
        /* process a single row and encode variations */
        for (j = 0; j < texture.cols; j++)
        {
            if (featuresData[i * texture.cols + j + feat_pos] == 0)
            {
                if (textureData[pos + 1 + texture.cols] < textureData[pos + 1])
                {
                    featuresData[i * texture.cols + j + 1 + feat_pos] = featuresData[i * texture.cols + j + feat_pos] + 1;
                    pos = pos + 1 + texture.cols;
                }
                else
                {
                    featuresData[i * texture.cols + j + 1 + feat_pos] = featuresData[i * texture.cols + j + feat_pos];
                    pos = pos + 1;
                }
            }
            else if (featuresData[i * texture.cols + j + feat_pos] == CR_FEATHEIGHT-1)
            {
                if (textureData[pos + 1 - texture.cols] < textureData[pos + 1])
                {
                    featuresData[i * texture.cols + j + 1 + feat_pos] = featuresData[i * texture.cols + j + feat_pos] - 1;
                    pos = pos + 1 - texture.cols;
                }
                else
                {
                    featuresData[i * texture.cols + j + 1 + feat_pos] = featuresData[i * texture.cols + j + feat_pos];
                    pos = pos + 1;
                }
            }
            else
            {
                /* proceed with upper row */
                if ((textureData[pos + 1 - texture.cols] < textureData[pos + 1]) && (textureData[pos + 1 - texture.cols] < textureData[pos + 1 + texture.cols]))
                {
                    featuresData[i * texture.cols + j + 1 + feat_pos] = featuresData[i * texture.cols + j + feat_pos] - 1;
                    pos = pos + 1 - texture.cols;
                }
                /* proceed with lower row */
                else if ((textureData[pos + 1 + texture.cols] < textureData[pos + 1]) && (textureData[pos + 1 + texture.cols] < textureData[pos + 1 - texture.cols]))
                {
                    featuresData[i * texture.cols + j + 1 + feat_pos] = featuresData[i * texture.cols + j + feat_pos] + 1;
                    pos = pos + 1 + texture.cols;
                }
                /* proceed with current row */
                else
                {
                    featuresData[i * texture.cols + j + 1 + feat_pos] = featuresData[i * texture.cols + j + feat_pos];
                    pos = pos + 1;
                }
            }
        }
    }
    
    /* smooth the obtained feature vector */
    featuresData=cr_smooth(featuresData, rows, texture.cols);
    
    /* map the feature vector to a binary code */
    for (i = 0; i< texture.cols * rows  * 2; i++)
    {
        if (featuresData[i] == 0)
        {
            setBitTo0(iris_code,i);
            setBitTo0(iris_code,texture.cols * rows  * 2 + i);
        }
        else if (featuresData[i] == 1)
        {
            setBitTo0(iris_code,i);
            setBitTo1(iris_code,texture.cols * rows  * 2 + i);
        }
        else
        {
            setBitTo1(iris_code,i);
            setBitTo1(iris_code,texture.cols * rows  * 2 + i);
        }
    }
}

/**
 * Calculate a standard uniform upper exclusive lower inclusive 256-bin
 *histogram for range [0,256]
 *
 * src: CV_8UC1 image
 * histogram: CV_32SC1 1 x 256 histogram matrix
 */
void hist2u(const Mat &src, Mat &histogram) {
  histogram.setTo(0);
  MatConstIterator_<uchar> s = src.begin<uchar>();
  MatConstIterator_<uchar> e = src.end<uchar>();
  int *p = (int *)histogram.data;
  for (; s != e; s++) {
    p[*s]++;
  }
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
uchar interp(const double x, const double y, const double r, const double s,
             const uchar b1, const uchar b2, const uchar b3, const uchar b4) {
  double w1 = (x + y);
  double w2 = x / w1;
  w1 = y / w1;
  double w3 = (r + s);
  double w4 = r / w3;
  w3 = s / w3;
  return saturate_cast<uchar>(w3 * (w1 * b1 + w2 * b2) +
                              w4 * (w1 * b3 + w2 * b4));
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
 * Inplace histogram clipping according to Zuiderveld (counts excess and
 *redistributes excess by adding the average increment)
 *
 * hist: CV_32SC1 1 x 256 histogram matrix
 * clipFactor: between 0 (maximum slope M/N, where M #pixel in window, N #bins)
 *and 1 (maximum slope M)
 * pixelCount: number of pixels in window
 */
void clipHistogram(Mat &hist, const float clipFactor, const int pixelCount) {
  double minSlope = ((double)pixelCount) / 256;
  int clipLimit = std::min(
      pixelCount,
      std::max(1, cvCeil(minSlope + clipFactor * (pixelCount - minSlope))));
  int distributeCount = 0;
  MatIterator_<int> p = hist.begin<int>();
  MatIterator_<int> e = hist.end<int>();
  for (; p != e; p++) {
    int binsExcess = *p - clipLimit;
    if (binsExcess > 0) {
      distributeCount += binsExcess;
      *p = clipLimit;
    }
  }
  int avgInc = distributeCount / 256;
  int maxBins = clipLimit - avgInc;
  for (p = hist.begin<int>(); p != e; p++) {
    if (*p <= maxBins) {
      distributeCount -= avgInc;
      *p += avgInc;
    } else if (*p < clipLimit) {
      distributeCount -= (clipLimit - *p);
      *p = clipLimit;
    }
  }
  while (distributeCount > 0) {
    for (p = hist.begin<int>(); p != e && distributeCount > 0; p++) {
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

void clahe(const Mat &src, Mat &dst, const int cellWidth = 10,
           const int cellHeight = 10, const float clipFactor = 1.) {
  Mat hist(1, 256, CV_32SC1);
  Mat roi;
  uchar *sp, *dp;
  int height = src.rows;
  int width = src.cols;
  int gridWidth = width / cellWidth + (width % cellWidth == 0 ? 0 : 1);
  int gridHeight = height / cellHeight + (height % cellHeight == 0 ? 0 : 1);
  int bufSize = (gridWidth + 2) * 256;
  int bufOffsetLeft = bufSize - 256;
  int bufOffsetTop = bufSize - gridWidth * 256;
  int bufOffsetTopLeft = bufSize - (gridWidth + 1) * 256;
  Mat buf(1, bufSize, CV_8UC1);
  MatIterator_<uchar> pbuf = buf.begin<uchar>(), ebuf = buf.end<uchar>();
  MatIterator_<int> phist, ehist = hist.end<int>();
  uchar *curr, *topleft, *top, *left;
  int pixelCount, cX, cY, cWidth, cHeight, cellOrigin, cellOffset;
  double sum;
  // process first row, first cell
  cX = 0;
  cY = 0;
  cWidth = min(cellWidth, width);
  cHeight = min(cellHeight, height);
  pixelCount = cWidth * cHeight;
  sum = 0;
  roi = Mat(src, Rect(cX, cY, cWidth, cHeight));
  hist2u(roi, hist);
  if (clipFactor < 1) clipHistogram(hist, clipFactor, pixelCount);
  // equalization
  for (phist = hist.begin<int>(); phist != ehist; phist++, pbuf++) {
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
  for (int b = 0; b < cHeight; b++, sp += cellOffset, dp += cellOffset) {
    for (int a = 0; a < cWidth; a++, sp++, dp++) {
      *dp = curr[*sp];
    }
  }
  // process first row, other cells
  for (int x = 1; x < gridWidth; x++) {
    cX = x * cellWidth;
    cWidth = min(cellWidth, width - x * cellWidth);
    cHeight = min(cellHeight, height);
    pixelCount = cWidth * cHeight;
    sum = 0;
    roi.release();
    roi = Mat(src, Rect(cX, cY, cWidth, cHeight));
    hist2u(roi, hist);
    if (clipFactor < 1) clipHistogram(hist, clipFactor, pixelCount);
    // equalization
    for (phist = hist.begin<int>(); phist != ehist; phist++, pbuf++) {
      sum += *phist;
      *pbuf = saturate_cast<uchar>(sum * 255 / pixelCount);
    }
    // paint first row, other cells
    cX += cellWidth / 2 - cellWidth;
    cWidth = min(cellWidth, width - x * cellWidth + cellWidth / 2);
    cHeight = min(cellHeight / 2, height);
    cellOrigin = src.step * cY + cX;
    cellOffset = src.step - cWidth;
    sp = (uchar *)(src.data + cellOrigin);
    dp = (uchar *)(dst.data + cellOrigin);
    curr = buf.data + (curr - buf.data + 256) % bufSize;
    left = buf.data + (curr - buf.data + bufOffsetLeft) % bufSize;
    for (int b = 0; b < cHeight; b++, sp += cellOffset, dp += cellOffset) {
      for (int a = 0; a < cWidth; a++, sp++, dp++) {
        *dp = interp(a, cWidth - a, left[*sp], curr[*sp]);
      }
    }
  }
  // process (i.e. paint) first row, last cell (only if necessary)
  if (width % cellWidth > cellWidth / 2 || width % cellWidth == 0) {
    cWidth = (width - cellWidth / 2) % cellWidth;
    cHeight = min(cellHeight / 2, height);
    cX = width - cWidth;
    cellOrigin = src.step * cY + cX;
    cellOffset = src.step - cWidth;
    sp = (uchar *)(src.data + cellOrigin);
    dp = (uchar *)(dst.data + cellOrigin);
    for (int b = 0; b < cHeight; b++, sp += cellOffset, dp += cellOffset) {
      for (int a = 0; a < cWidth; a++, sp++, dp++) {
        *dp = curr[*sp];
      }
    }
  }
  // process rest of rows
  for (int y = 1; y < gridHeight; y++) {
    // process other rows, first cell
    cX = 0;
    cY = y * cellHeight;
    cWidth = min(cellWidth, width);
    cHeight = min(cellHeight, height - y * cellHeight);
    pixelCount = cWidth * cHeight;
    sum = 0;
    roi.release();
    roi = Mat(src, Rect(cX, cY, cWidth, cHeight));
    hist2u(roi, hist);
    if (clipFactor < 1) clipHistogram(hist, clipFactor, pixelCount);
    // equalization
    if (pbuf == ebuf) pbuf = buf.begin<uchar>();
    for (phist = hist.begin<int>(); phist != ehist; phist++, pbuf++) {
      sum += *phist;
      *pbuf = saturate_cast<uchar>(sum * 255 / pixelCount);
    }
    // paint other rows, first cell
    cY += cellHeight / 2 - cellHeight;
    cWidth = min(cellWidth / 2, width);
    cHeight = min(cellHeight, height - y * cellHeight + cellHeight / 2);
    cellOrigin = src.step * cY + cX;
    cellOffset = src.step - cWidth;
    sp = (uchar *)(src.data + cellOrigin);
    dp = (uchar *)(dst.data + cellOrigin);
    curr = buf.data + (curr - buf.data + 256) % bufSize;
    top = buf.data + (curr - buf.data + bufOffsetTop) % bufSize;
    for (int b = 0; b < cHeight; b++, sp += cellOffset, dp += cellOffset) {
      for (int a = 0; a < cWidth; a++, sp++, dp++) {
        *dp = interp(b, cHeight - b, top[*sp], curr[*sp]);
      }
    }
    // process other rows, rest of cells
    for (int x = 1; x < gridWidth; x++) {
      cX = x * cellWidth;
      cY = y * cellHeight;
      cWidth = min(cellWidth, width - x * cellWidth);
      cHeight = min(cellHeight, height - y * cellHeight);
      pixelCount = cWidth * cHeight;
      sum = 0;
      roi.release();
      roi = Mat(src, Rect(cX, cY, cWidth, cHeight));
      hist2u(roi, hist);
      if (clipFactor < 1) clipHistogram(hist, clipFactor, pixelCount);
      // equalization
      if (pbuf == ebuf) pbuf = buf.begin<uchar>();
      for (phist = hist.begin<int>(); phist != ehist; phist++, pbuf++) {
        sum += *phist;
        *pbuf = saturate_cast<uchar>(sum * 255 / pixelCount);
      }
      // paint other rows, rest of cells
      cX += cellWidth / 2 - cellWidth;
      cY += cellHeight / 2 - cellHeight;
      cWidth = min(cellWidth, width - x * cellWidth + cellWidth / 2);
      cHeight = min(cellHeight, height - y * cellHeight + cellHeight / 2);
      cellOrigin = src.step * cY + cX;
      cellOffset = src.step - cWidth;
      sp = (uchar *)(src.data + cellOrigin);
      dp = (uchar *)(dst.data + cellOrigin);
      curr = buf.data + (curr - buf.data + 256) % bufSize;
      top = buf.data + (curr - buf.data + bufOffsetTop) % bufSize;
      topleft = buf.data + (curr - buf.data + bufOffsetTopLeft) % bufSize;
      left = buf.data + (curr - buf.data + bufOffsetLeft) % bufSize;
      for (int b = 0; b < cHeight; b++, sp += cellOffset, dp += cellOffset) {
        for (int a = 0; a < cWidth; a++, sp++, dp++) {
          *dp = interp(a, cWidth - a, b, cHeight - b, topleft[*sp], top[*sp],
                       left[*sp], curr[*sp]);
        }
      }
    }
    // process (i.e. paint) other rows, last cell (only if necessary)
    if (width % cellWidth > cellWidth / 2 || width % cellWidth == 0) {
      cWidth = (width - cellWidth / 2) % cellWidth;
      cHeight = min(cellHeight, height - y * cellHeight + cellHeight / 2);
      cX = width - cWidth;
      cellOrigin = src.step * cY + cX;
      cellOffset = src.step - cWidth;
      sp = (uchar *)(src.data + cellOrigin);
      dp = (uchar *)(dst.data + cellOrigin);
      top = buf.data + (curr - buf.data + bufOffsetTop) % bufSize;
      for (int b = 0; b < cHeight; b++, sp += cellOffset, dp += cellOffset) {
        for (int a = 0; a < cWidth; a++, sp++, dp++) {
          *dp = interp(b, cHeight - b, top[*sp], curr[*sp]);
        }
      }
    }
  }
  // process (i.e. paint) last row (only if necessary)
  if (height % cellHeight > cellHeight / 2 || height % cellHeight == 0) {
    // paint last row, first cell
    cWidth = min(cellWidth / 2, width);
    cHeight = (height - cellHeight / 2) % cellHeight;
    cX = 0;
    cY = height - cHeight;
    cellOrigin = src.step * cY + cX;
    cellOffset = src.step - cWidth;
    sp = (uchar *)(src.data + cellOrigin);
    dp = (uchar *)(dst.data + cellOrigin);
    curr = buf.data + (curr - buf.data + bufOffsetTop + 256) % bufSize;
    for (int b = 0; b < cHeight; b++, sp += cellOffset, dp += cellOffset) {
      for (int a = 0; a < cWidth; a++, sp++, dp++) {
        *dp = curr[*sp];
      }
    }
    // paint last row, other cells
    for (int x = 1; x < gridWidth; x++) {
      cX = (x - 1) * cellWidth + cellWidth / 2;
      cWidth = min(cellWidth, width - x * cellWidth + cellWidth / 2);
      cHeight = (height - cellHeight / 2) % cellHeight;
      cellOrigin = src.step * cY + cX;
      cellOffset = src.step - cWidth;
      sp = (uchar *)(src.data + cellOrigin);
      dp = (uchar *)(dst.data + cellOrigin);
      left = curr;
      curr = buf.data + (curr - buf.data + 256) % bufSize;
      for (int b = 0; b < cHeight; b++, sp += cellOffset, dp += cellOffset) {
        for (int a = 0; a < cWidth; a++, sp++, dp++) {
          *dp = interp(a, cWidth - a, left[*sp], curr[*sp]);
        }
      }
    }
    // paint last row, last cell (only if necessary)
    if (width % cellWidth > cellWidth / 2 || width % cellWidth == 0) {
      cWidth = (width - cellWidth / 2) % cellWidth;
      cHeight = (height - cellHeight / 2) % cellHeight;
      cX = width - cWidth;
      cellOrigin = src.step * cY + cX;
      cellOffset = src.step - cWidth;
      sp = (uchar *)(src.data + cellOrigin);
      dp = (uchar *)(dst.data + cellOrigin);
      for (int b = 0; b < cHeight; b++, sp += cellOffset, dp += cellOffset) {
        for (int a = 0; a < cWidth; a++, sp++, dp++) {
          *dp = curr[*sp];
        }
      }
    }
  }
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
            cmdCheckOpts(cmd,"-i|-o|-q|-t");
            cmdCheckOptExists(cmd,"-i");
            cmdCheckOptSize(cmd,"-i",1);
            string inFiles = cmdGetPar(cmd,"-i");
            cmdCheckOptExists(cmd,"-o");
            cmdCheckOptSize(cmd,"-o",1);
            string outFiles = cmdGetPar(cmd,"-o");
            string imaskFiles, omaskFiles;
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
            // starting routine
            Timing timing(1,quiet);
            vector<string> files;
            patternToFiles(inFiles,files);
            CV_Assert(files.size() > 0);
            timing.total = files.size();
            for (vector<string>::iterator inFile = files.begin(); inFile != files.end(); ++inFile, timing.progress++){
                if (!quiet) printf("Loading texture '%s' ...\n", (*inFile).c_str());;
                Mat img = imread(*inFile, CV_LOAD_IMAGE_GRAYSCALE);
                Mat img_e(64, 512, CV_8UC1);
                CV_Assert(img.data != 0);
                Mat out;
                if (img.rows != 64 || img.cols != 512)
                {
                 printf("Input texture has to be of size 512 x 64.\n");
                 exit(EXIT_FAILURE);
                }
                int w = img.cols*2;
                int h = (img.rows/CR_FEATHEIGHT)*2;
                if (!quiet) printf("Creating %d x %d iris-code ...\n", w, h);
                Mat code (1,(w*h)/8,CV_8UC1);
                code.setTo(0);
                GaussianBlur(img, img_e, Size(3, 3), 0, 0);
                clahe(img_e, img, 512 / 8, 64 / 2);
                featureExtract(code, img);
                out = code;
                string outfile;
                patternFileRename(inFiles,outFiles,*inFile,outfile);
                if (!quiet) printf("Storing code '%s' ...\n", outfile.c_str());
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
    catch (...){
        printf("Exit with errors.\n");
        exit(EXIT_FAILURE);
    }
    return EXIT_SUCCESS;
}
