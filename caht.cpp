/*
 * caht.cpp
 *
 * Author: E. Pschernig (epschern@cosy.sbg.ac.at), P. Wild (pwild@cosy.sbg.ac.at)
 *
 * Iris segmentation tool based on Contrast-adjusted Hough Transform
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
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
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
	printf("| caht - Contrast-adjusted Hough Transform                                    |\n");
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
    printf("| -po  | polarfile  | 1 | Y | write polar image (off)                         |\n");
    printf("| -bm  | binmaskfile| 1 | Y | write binary segmentation mask (off)            |\n");
    printf("| -sr  | segresfile | 1 | Y | write segmentation result (off)                 |\n");
    printf("| -lt  | thickness  | 1 | Y | thickness for lines to be drawn (1)             |\n");
    printf("| -h   |            | 2 | N | prints usage                                    |\n");
    printf("| -so  | scale fac. | 1 | N | scale the area of the outer ellipse (def.=1.0)  |\n");
    printf("| -si  | scale fac. | 1 | N | scale the area of the inner ellipse (def.=1.0)  |\n");
    printf("| -tr  | translate  | 1 | N | horizontal translation of ellipses (def.=0.0)   |\n");
    printf("|      |            |   |   | the factor is by iris radius (-1 would be a     |\n");
    printf("|      |            |   |   | translation by iris radius to the left)         |\n");
    printf("| -tc  | translate  | 1 | N | horizontal translation of unrolling center.0)   |\n");
    printf("|      |            |   |   | works as per translate (but is based on pupil   |\n");
    printf("|      |            |   |   | radius, moust be in -1 to 1) (default=0.0).     |\n");
    printf("| -l   | logfile    | 1 | N | log parameters for unrolling to this file.      |\n");
    printf("+------+------------+---+---+-------------------------------------------------+\n");
    printf("|                                                                             |\n");
    printf("| EXAMPLE USAGE                                                               |\n");
    printf("|                                                                             |\n");
    printf("| -i *.tiff -o ?1_texture.png -s 512 32 -e -q -t                              |\n");
    printf("| -i *.tiff -o ?1_texture.png -m ?1_mask.tiff -q -t                           |\n");
    printf("|                                                                             |\n");
    printf("| AUTHOR                                                                      |\n");
    printf("|                                                                             |\n");
    printf("| Peter Wild (pwild@cosy.sbg.ac.at)                                           |\n");
    printf("| Elias Pschernig (epschern@cosy.sbg.ac.at)                                   |\n");
    printf("| Heinz Hofbauer (hhofbaue@cosy.sbg.ac.at)                                    |\n");
    printf("|                                                                             |\n");
    printf("| JPEG2000 Hack                                                               |\n");
    printf("| Thomas Bergmueller (thomas.bergmueller@authenticvision.com)                 |\n");
    printf("|                                                                             |\n");
    printf("| COPYRIGHT                                                                   |\n");
    printf("|                                                                             |\n");
    printf("| (C) 2012 All rights reserved. Do not distribute without written permission. |\n");
    printf("+-----------------------------------------------------------------------------+\n");
}

/** ------------------------------- OpenCV helpers ------------------------------- **/

/*
 * Performs the AND operation on pixel values of the source image
 *
 * left:      left operand (single-channel 8-bit source image)
 * right:     right operand (single-channel 8-bit source image)
 * result:    left AND right
 */
void maskAnd(const Mat& left, const Mat& right, Mat& result){
	MatConstIterator_<uchar> li = left.begin<uchar>();
	MatConstIterator_<uchar> ri = right.begin<uchar>();
	MatIterator_<uchar> p = result.begin<uchar>();
	MatConstIterator_<uchar> e = result.end<uchar>();
	for (;p!=e;p++,li++,ri++){
		*p = (*li & *ri);
	}
}

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
    if( blackpoint >= whitepoint){ //This should never happen -- prevent div by zero or nonesense
        printf("ERROR in %s (%s:%d): blackpoint >= whitepoint\n", __func__, __FILE__,__LINE__);
        if( blackpoint == whitepoint){
            if ( whitepoint < 255){
                whitepoint = blackpoint+1;
            } else { 
                blackpoint = whitepoint-1;
            }
        } else if( blackpoint < 128){
            whitepoint = blackpoint+1;
        } else if (whitepoint > 128) {
            blackpoint = whitepoint -1;
        } else {
            blackpoint = 127;
            whitepoint = 128;
        }
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
 * Inverts an image
 */
void invert(Mat& image)
{
	for(MatIterator_<uchar> it = image.begin<uchar>(), end = image.end<uchar>(); it < end; it++){
		*it = (255 - *it);
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
            /* not 2*M_PI, left/right both is horizontal */
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
    int mark[w * h];
    uchar * data = image.data;
    memset(mark, 0, w * h * sizeof(int));
    int tag = 1;
    for (int j = 0; j < h; j++)
    {
        for (int i = 0; i < w; i++)
        {
            if (! mark[i + j * w] && data[i + j * w])
            {
                int dirs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                int ax, ay, bx, by;
                int size = follow_streak(image, i, j, dirs, mark, tag, 0, &ax, &ay, &bx, &by);
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
                    follow_streak(image, i, j, dirs, mark, tag, 1, &ax, &ay, &bx, &by);
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

/** ------------------------------- Hough transform ------------------------------- **/

struct HoughCircle
{
    int x, y;
    int radius;
    int pixels;
    int accum;
    /* value the accumulator would have if all pixels of the circle */
    int max;
    /* existed at maximum intensity */
};

static int ** lookup_table(int min_r, int max_r)
{
    int nr = 1 + max_r - min_r;
    int **lut = (int**) malloc(nr * sizeof *lut);
    for (int r = min_r; r <= max_r; r++)
    {
        int ir = r - min_r;
        /* we use pixel resolution for the circumference of the circle */
        int na = 2 * M_PI * r;
        lut[ir] = (int*) malloc(1 + na * 2 * sizeof *lut[ir]);
        lut[ir][0] = na;
        double a = 0, da = 2 * M_PI / na;
        for (int ia = 0; ia < na; ia++)
        {
            int rcos = round(r * cos(a));
            int rsin = round(r * sin(a));
            lut[ir][1 + ia * 2 + 0] = rcos;
            lut[ir][1 + ia * 2 + 1] = rsin;
            a += da;
        }
    }
    return lut;
}

/*
 * We use a three dimensional parameter space a, b, r. Pixels on a circle are
 * given by:
 * x = a + r * cos(theta)
 * y = b + r * sin(theta)
 * theta [0..360)
 * That is, a and b are the center ofthe circle, and r is the radius.
 *
 * Now, if we find that a point x/y in our image lies on a circle, then we
 * accumulate each possible a/b/r position in hough space.
 *
 * For example, we search for a circle with radius 5. We have a circle pixel at
 * 0/0. Which possible a/b combinations would this contribute to? In Hough
 * space it are all a/b where a circle with center a/b would include the point.
 * For example, a circle at 5/0 would include 0/0, as would a circle at 0/5. In
 * short, we draw a circle around 0/0 with radius 5 in Hough space.
 *
 * The min/max parameters are ranges for the radius and the center position
 * relative to the image. The radius as well as the center range may also
 * extend to outside the image area.
 */
void hough_circle(Mat& image, vector<HoughCircle>& res, int min_r, int max_r, int min_cx, int max_cx, int min_cy, int max_cy, int threshold, int hint_r, int hint_x, int hint_y)
{

    int w = image.cols;
    int h = image.rows;
    uchar *data = image.data;
    int **lut = lookup_table(min_r, max_r);
    int nx = 1 + max_cx - min_cx;
    int ny = 1 + max_cy - min_cy;
    Mat houghAccu(1,nx * ny,CV_32SC1);
    int *hough = (int*)houghAccu.data;
    for (int r = min_r; r <= max_r; r++)
    {
    	houghAccu.setTo(0);
        int ir = r - min_r;
        int *rlut = lut[ir];
        /* ok, draw some circles */
        for (int j = 0; j < h; j++)
        {
            for (int i = 0; i < w; i++)
            {
                uchar c = data[j * w + i];
                if (c)
                {
                    int na = rlut[0];
                    for (int ia = 0; ia < na; ia++)
                    {
                        int rcos = rlut[1 + 2 * ia + 0];
                        int rsin = rlut[1 + 2 * ia + 1];
                        int a = i - rcos;
                        int b = j - rsin;
                        if (a >= min_cx && a <= max_cx && b >= min_cy && b <= max_cy)
                        {
                            hough[a - min_cx + nx * (b - min_cy)] += c;
                            /* Since we draw non-antialiased and rather crude circles, pixels likely */
                            /* can be off by a pixel or so. Therefore we collect neighbor */
                            /* pixels for each pixel as well. */
                        }
                    }
                }
            }
        }
        for (int j = 1; j < ny - 1; j++)
        {
            for (int i = 3; i < nx - 3; i++)
            {
                int sum = 0;
                for (int k = -1; k <= 1; k++)
                {
                    for (int l = -3; l <= 3; l++)
                    {
                        sum += hough[(i + l) + (j + k) * nx];
                    }
                }
                hough[i + j * nx] = sum / 21;
                /* If we have a hint circle, anything outside is discarded. */
            }
        }
        if (hint_r)
        {
            int rr = hint_r * hint_r;
            for (int j = 0; j < ny; j++)
            {
                for (int i = 0; i < nx; i++)
                {
                    int dx = min_cx + i - hint_x;
                    int dy = min_cy + j - hint_y;
                    if (dx * dx + dy * dy > rr)
                    {
                        hough[i + j * nx] = 0;
                        /* find maximum accumulator value */
                    }
                }
            }
        }
        int max = 1;
        for (int j = 0; j < ny; j++)
        {
            for (int i = 0; i < nx; i++)
            {
                if (hough[i + j * nx] > max)
                {
                    max = hough[i + j * nx];
                    /* TODO: As we actually will get slight ellipses most of the time, instead */
                    /* of non-maximum suppression, some kind of averaging would be better. */
                    /* _ _ */
                    /* / X \ */
                    /* | | | | */
                    /* \_X_/ */
                    /* A B */
                    /* _ */
                    /* / \ */
                    /* |   | */
                    /* \_/ */
                    /* C */
                    /*  */
                    /* Instead of choosing either one of found circles A and B at the two */
                    /* ends of the ellipse, we want C averaged by them (maybe with the */
                    /* radius slightly compensated). */
                    /* create a grayscale image, with non-maxima suppressed and faint circles */
                    /* thresholded */
                }
            }
        }
        for (int j = 0; j < ny; j++)
        {
            for (int i = 0; i < nx; i++)
            {
                int v = hough[i + j * nx];
                if (i < nx - 1 && hough[i + 1 + j * nx] > v)
                {
                    v = 0;
                }
                if (j > 0 && hough[i + (j - 1) * nx] > v)
                {
                    v = 0;
                }
                if (i > 0 && hough[i - 1 + j * nx] > v)
                {
                    v = 0;
                }
                if (j < ny - 1 && hough[i + (j + 1) * nx] > v)
                {
                    v = 0;
                }
                int cv = v * 255 / max;
                if (cv < threshold)
                {
                    v = 0;
                    /* cv = 0 */
                }
                if (v)
                {
                    HoughCircle c = {min_cx + i, min_cy + j, r, (int)(2 * M_PI * r), v, (int)(2 * M_PI * r * 255)};
                    //printf("found x: %i y: %i r: %i pix: %i acc: %i max: %i\n",c.x,c.y,c.radius,c.pixels,c.accum,c.max);
                    res.push_back(c);
                }
            }
        }
    }
}

/*
 * Finds the best Hough circle
 */
int find_best_circle(vector<HoughCircle>& circles, int* x, int* y, int* r, int hint, int hint_x, int hint_y, int hint_r, int hint_weight)
{
    int FAIL = -1000;
    int best_accum = FAIL;
    if (! circles.size())
    {
        return 0;
    }
    int mindist = 0;
    int maxdist = 0;
    if (hint)
    /* Find minimum and maximum distance from hint */
    {
        for (unsigned int i = 0; i < circles.size(); i++)
        {
            HoughCircle c = circles.at(i);
            //printf("processing x: %i y: %i r: %i pix: %i acc: %i max: %i\n",c.x,c.y,c.radius,c.pixels,c.accum,c.max);

            int dx = c.x - hint_x;
            int dy = c.y - hint_y;
            int dist =sqrt(dx * dx + dy * dy);
            if (mindist == 0 || dist < mindist)
            {
                mindist = dist;
            }
            if (dist > maxdist)
            {
                maxdist = dist;
            }
        }
    }
    int middist = (mindist + maxdist) / 2;
    int medium_coverage = 0;
    for (unsigned int i = 0; i < circles.size(); i++)
    {
        HoughCircle c = circles.at(i);
        int coverage = c.accum * 1000 / c.max;
        medium_coverage += coverage;
    }
    medium_coverage /= circles.size();
    /* Square of radius where distance from hint-circle is unscaled. */
    int qdist = 5 * 5;
    for (unsigned int i = 0; i < circles.size(); i++)
    {
        HoughCircle c = circles.at(i);
        int score;
        int dist = 0;
        int coverage = c.accum * 1000 / c.max;
        if (hint)
        /* Prefer circles with center closer to the hint position */
        {
            int dx = c.x - hint_x;
            int dy = c.y - hint_y;
            dist = sqrt(dx * dx + dy * dy);
            if (dist > middist)
            {
                score /= 2;
            }
            if (dist > hint_r)
            {
                continue;
                /* This gives bigger circles an advantage, as they have more pixels in */
                /* total. */
            }
        }
        score = c.accum;
        /* this means for bigger circles, distance weighs more */
        score -= dist * dist * hint_weight * c.radius / qdist;
        /* this usually has no effect, but for completely identical scores */
        /* otherwise, we pefer the "outer" outline */
        score += c.radius;
        if (0)
        {
            if (coverage < medium_coverage)
            {
                score /= 2;
            }
        }
        else
        {
            score *= coverage;
        }
        if (score > best_accum)
        {
            best_accum = score;
            *x = c.x;
            *y = c.y;
            *r = c.radius;
        }
    }
    if (best_accum == FAIL)
    {
        return 0;
    }
    return 1;
}

/** ------------------------------- Iris boundary fitting ------------------------------- **/


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
            if (d <= r2 && r2 > 0)
            {
                int alpha = 0;
                if( r2 -r1 != 0)
                {
                    alpha = r2 * (d - r1) / (r2 - r1);
                }
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

/** ------------------------------- Rubbersheet transform ------------------------------- **/

/*
 * interpolation mode for rubbersheet repeating the last pixel for a given angle if no values are available
 * (otherwise behaves like INTER_LINEAR)
 */
//static const int INTER_LINEAR_REPEAT = 82;

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


/** 
 * Set cont to off center radi to describe a circle with radius ir  from the point with x-offset tr
 */
void setOffCenterRadi(Mat& dest, const float r, const float t){ //translate2
	CV_Assert(dest.type() == CV_32FC1);
    if( t == 0){//no off center unrolling
        dest.setTo(r);
        return;
    }
    float * d = (float *) dest.data;
	const float thetaoffset = 2 * M_PI / dest.cols;
    const float threshold = tanf( M_PI/2 - M_PI/2/(dest.cols/4.*10)) ; 
    float thetab = 0, theta;
    float ac_y, ac_x, x, y, a, b, c, s, ss;
    int count;
    for( count = 0; count < dest.cols; count++, d++, thetab+=thetaoffset){
        //angular correction stuff
        ac_y = (theta > M_PI) ? -1 : 1;
        theta = ( thetab > M_PI ) ? ( 2*M_PI-thetab ) : thetab;
        ac_x = (theta > M_PI/2.) ? -1 : 1;

        s = tanf( theta);
        if(abs(s) > threshold){ //special case M_PI/2 
            printf("**");
            x = t;
            c = t*t - r*r;
            y = sqrt( -4*c)/2.;
        } else {
            ss = s*s;
            a = 1+ss;
            b = -1 * (  2*ss*t ) ;
            c = ss*t*t - r*r ;
            x = ( -b + ac_x*sqrt( b*b - 4*a*c ) ) / ( 2*a ) ;
            y =  s*( x - t ) ;
        }
        y *= ac_y;
        *d = sqrt( pow( (x-t), 2) + pow(y, 2) );
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
			cmdCheckOpts(cmd,"-i|-o|-m|-s|-e|-q|-t|-po|-bm|-sr|-lt|-so|-si|-tr|-tc|-l");
			cmdCheckOptExists(cmd,"-i");
			cmdCheckOptSize(cmd,"-i",1);
			string inFiles = cmdGetPar(cmd,"-i");
			cmdCheckOptExists(cmd,"-o");
			cmdCheckOptSize(cmd,"-o",1);
			string outFiles = cmdGetPar(cmd,"-o");
			string maskFiles;
			if (cmdGetOpt(cmd,"-m") != 0){
				cmdCheckOptSize(cmd,"-m",1);
				maskFiles = cmdGetPar(cmd,"-m");
			}
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
			string polarFiles;
			if (cmdGetOpt(cmd,"-po") != 0){
				cmdCheckOptSize(cmd,"-po",1);
				polarFiles = cmdGetPar(cmd,"-po");
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
                /* 
                 * A = M_PI a b
                 * xA = M_PI sa sb = s²ab M_PI = s²A
                 * => x = sqrt(s)
                 */
                if ( outer_scale <=0){
                    cerr << "scale factor (-so) <=0, this will result in an error" << endl;
                }
                outer_scale = sqrt(outer_scale);
            }
            float inner_scale = 1;
            if (cmdGetOpt(cmd,"-si") != 0){
                cmdCheckOptSize(cmd, "-si", 1);
                inner_scale = cmdGetParFloat(cmd, "-si", 0);
                if ( inner_scale <=0){
                    cerr << "scale factor (-si) <=0, this will result in an error" << endl;
                }
                inner_scale = sqrt(inner_scale);
            }
            float translate = 0;
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
                            << "inner.x" << ", " << "inner.y" << ", " << "inner.r" << ", " 
                            << "outer.x" << ", " << "outer.y" << ", " << "outer.r" << endl;
            }
            float translate_unroll = 0;
            if (cmdGetOpt(cmd,"-tc") != 0){
                cmdCheckOptSize(cmd, "-tc", 1);
                translate_unroll = cmdGetParFloat(cmd, "-tc", 0);
                if( abs(translate_unroll) >= 1){
                    cerr << "translating the unrolling center outside the iris will result in errors" << endl;
                }
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
				int px = 0, py = 0, pr = 0;
				if (!quiet) printf("Finding pupil ...\n");
				int PUPIL_HOUGH_THRESHOLD = 200;
				Mat temp(height,width,CV_8UC1);
				img.copyTo(temp);
				adjust_luminance(temp, width * height * 0.02, width * height * 0.04);
				Mat accum(height,width,CV_8UC1);
				cumulate(temp, accum, 3, 3);
				invert(accum);
				cumulate(accum, temp, 3, 3);
				invert(temp);
				int hrun = longest_horizontal_run(temp, width * 0.25, 127);
				int vrun = longest_vertical_run(temp, height * 0.25, 127);
				Mat mag(height,width,CV_32FC1);
				Mat orient(height,width,CV_32FC1);
				canny(temp, mag, orient, accum, 3, 15, 1, 1);
				remove_streaks(accum, 6, 1, 1, 5, 5, 10, 10, 3, 3);
				int ax, ay, bx, by;
				bounds_bigger_than_threshold(accum, 0, &ax, &ay, &bx, &by);
				int min_r = min(hrun / 2, vrun / 2);
				/* This can be far off if there's large reflections on the pupil */
				int max_r = min_r + max(hrun / 2, vrun / 2);
				/* else where would we fit the iris? */
				int border = min_r * 1.5;
				if (ax < border)
				{
					ax = border;
				}
				if (ay < border)
				{
					ay = border;
				}
				if (bx > width - 1 - border)
				{
					bx = width - 1 - border;
				}
				if (by > height - 1 - border)
				{
					by = height - 1 - border;
				}
				vector<HoughCircle> circles;
				hough_circle(accum, circles,min_r, max_r, ax, bx, ay, by, PUPIL_HOUGH_THRESHOLD, 0, 0, 0);
				find_best_circle(circles, &px, &py, &pr, 1, width / 2, height / 2, height / 2,0);
				if (!quiet) printf("Pupil circle: (x,y,r) = (%i,%i,%i)\n", px, py,pr);
				if (!quiet) printf("Finding iris ...\n");
				int ix = 0, iy = 0, ir = 0;
				img.copyTo(temp);
				/* Blackout the pupil first */
				int min_ir = pr * 1.5;
				black_out_circle(temp, px, py, pr, min_ir);
				int IRIS_HOUGH_THRESHOLD = 200;
				adjust_luminance(temp, min_ir * min_ir * M_PI, width * height * 0.01);
				threshold_above(temp, 128, 128);
				hrun = longest_horizontal_run(temp, width * 0.1, 191);
				vrun = longest_vertical_run(temp, height * 0.1, 191);
				canny(temp, mag, orient, accum, 3, 15, 1, 0);
				threshold_below(accum, 6, 0);
				threshold_above(accum, 7, 255);
				remove_streaks(accum, 6, 6, 6, 10, 10, 10, 10, 2, 10);
				bounds_bigger_than_threshold(accum, 0, &ax, &ay, &bx, &by);
				/* make sure the pupil area is covered */
				/* TODO: in fact, the area to use for the accum (the above) and possible */
				/* iris center should be separated.. */
				if (px - pr < ax)
				{
					ax = px - pr;
				}
				if (py - pr < ay)
				{
					ay = py - pr;
				}
				if (px + pr > bx)
				{
					bx = px + pr;
				}
				if (py - pr > by)
				{
					by = py + pr;
				}
				//printf("Iris must be within: %d %d %d %d\n", ax, bx, ay, by);
				if (bx - ax >= min_ir && by - ay >= min_ir)
				/* actually, the area for the "center" is much smaller. There's no */
				/* point to e.g. let it lie outside the pupil */
				{
					int max_ir = pr * 8;
					if (max_ir > (bx - ax) / 2)
					{
						max_ir = (bx - ax) / 2;
						/* No point making the radius bigger than half the biggest run */
					}
					int run = max(hrun, vrun);
					if (max_ir > run / 2)
					{
						max_ir = run / 2;
						/* OTOH, it shouldn't be too small.. blah.. */
					}
					run = min(hrun, vrun);
					if (max_ir < run / 2)
					{
						max_ir = run / 2;
					}
					/* Hack: We dim pixels far above the horizontal line through the pupil */
					/* center, since the actual circle outline mostly can contribute from */
					/* the center. */
					/* 240 at double radius */
					dim_above_horizon(accum, py, 240.0 / pr / pr / 4);
					vector<HoughCircle> irisCircles;
					hough_circle(accum, irisCircles,min_ir, max_ir, ax, bx, ay, by, IRIS_HOUGH_THRESHOLD, pr * 1.3, px, py);
					find_best_circle(irisCircles, &ix, &iy, &ir, 1, px, py, pr, 10);
				}
				if (!quiet) printf("Iris circle: (x,y,r) = (%i,%i,%i)\n", ix, iy,ir);
                //scale
                int tr = (int) (ir*translate);
                int tc = (int) (pr*translate_unroll);
                ir = (int) (ir*outer_scale);
                pr = (int) (pr*inner_scale);
                ix += tr; //full translate
                px += tr; //full translate 
				if (!quiet) printf("Scaled Iris circle: (x,y,r) = (%i,%i,%i)\n", ix, iy,ir);
				if (!quiet) printf("Scaled Pupil circle: (x,y,r) = (%i,%i,%i)\n", px, py,pr);
                if( logFile.is_open()){ 
                    logFile << inFile->c_str() << ", " 
                            << ix << ", " << iy << ", " << ir << ", " 
                            << px << ", " << py << ", " << pr << endl;
                }
                //scale off
				Mat mask;
				if (!maskFiles.empty()){
					mask.create(height,width,CV_8UC1);
					mask_lids(img, mask, px, py, pr, ix, iy, ir);
					threshold(mask,mask,1,255,CV_THRESH_BINARY_INV);
				}
				Mat pupilCart (1,outWidth,CV_32FC2);
				Mat irisCart (1,outWidth,CV_32FC2);
				Mat cont(1,outWidth,CV_32FC1);
				//cont.setTo(pr);
                setOffCenterRadi( cont, pr, tc); //unroll_translate
				polar2Cart(cont, pupilCart, px+tc, py);//unroll_translate
				//cont.setTo(ir);
                setOffCenterRadi( cont, ir, tc); //unroll_translate
				polar2Cart(cont, irisCart, ix+tc, iy);//unroll_translate
				if (!segresFiles.empty()){
					Mat visual;
					cvtColor(orig,visual,CV_GRAY2BGR);
					for (float * it = (float *) pupilCart.data, * ite = it + (2*outWidth - 2); it < ite; it+=2){
						line(visual,Point2f(*it,it[1]),Point2f(it[2], it[3]),Scalar(0,0,255,0),lt);
					}
					for (float * it = (float *) irisCart.data, * ite = it + (2*outWidth - 2); it < ite; it+=2){
						line(visual,Point2f(*it,it[1]),Point2f(it[2], it[3]),Scalar(0,255,0,0),lt);
					}
                    line(visual,Point2f(ix+3*lt,iy),Point2f(ix-3*lt, iy),Scalar(0,0,255,0),lt);
                    line(visual,Point2f(ix,iy+3*lt),Point2f(ix, iy-3*lt),Scalar(0,0,255,0),lt);
                    circle(visual, Point2f(ix+tc,iy),lt*7/2,Scalar(0,0,255,0),lt);
                    circle(visual, Point2f(ix+tc,iy),max(1,lt/2),Scalar(0,0,255,0),lt);
					string vsegmentfile;
					patternFileRename(inFiles,segresFiles,*inFile,vsegmentfile);
					if (!quiet) printf("Storing segmentation image '%s' ...\n", vsegmentfile.c_str());
					if (!imwrite(vsegmentfile,visual)) CV_Error(CV_StsError,"Could not save image '" + vsegmentfile + "'");
				}
				if (!binmaskFiles.empty()){
					Mat bw(height,width,CV_8UC1,Scalar(0));
					Point iris_points[1][outWidth];
					float * it = (float *) irisCart.data;
					for (int i=0; i < outWidth; i++){
						iris_points[0][i] = Point2i(cvRound(*it),cvRound(it[1]));
						it+=2;
					}
					const Point* irispoints[1] = { iris_points[0] };
					Point pupil_points[1][outWidth];
					it = (float *) pupilCart.data;
					for (int i=0; i < outWidth; i++){
						pupil_points[0][i] = Point2f(cvRound(*it),cvRound(it[1]));
						it+=2;
					}
					const Point* pupilpoints[1] = { pupil_points[0] };
					fillPoly(bw,irispoints,&outWidth,1,Scalar(255,255,255));
					fillPoly(bw,pupilpoints,&outWidth,1,Scalar(0,0,0));
					if (!maskFiles.empty()){
						maskAnd(bw,mask,bw);
					}
					string binmaskFile;
					patternFileRename(inFiles,binmaskFiles,*inFile,binmaskFile);
					if (!quiet) printf("Storing binary mask '%s' ...\n", binmaskFile.c_str());
					if (!imwrite(binmaskFile,bw)) CV_Error(CV_StsError,"Could not save image '" + binmaskFile + "'");
				}
				if (!quiet) printf("Creating final texture ...\n");
				Mat out (outHeight,outWidth,CV_8UC1);
				rubbersheet(img, out, pupilCart, irisCart, INTER_LINEAR);
				if (enhance){
					if (!quiet) printf("Enhancing texture ...\n");
					clahe(out,out,width/8,height/2);
				}
				if (!maskFiles.empty()){
					Mat maskout (outHeight,outWidth,CV_8UC1);
					rubbersheet(mask, maskout, pupilCart, irisCart, INTER_NEAREST);
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
	catch (...){
	   	printf("Exit with errors.\n");
	   	exit(EXIT_FAILURE);
	}
    return EXIT_SUCCESS;
}
