/*
 * dct.cpp
 *
 * Author: M. Karnutsch (mkarnut@cosy.sbg.ac.at), C. Rathgeb (crathgeb@cosy.sbg.ac.at), P. Wild (pwild@cosy.sbg.ac.at)
 *
 * Generates an iris code from iris texture using the Monro et al. algorithm
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

#define pi 3.1415926535897931
#define M_PI 3.14159265358979323846
#define pihalf 1.57079632679489661923

/*
 * default options
 */
#define DCT_CUTOFF 3
#define DCT_DEGREE 65
#define DCT_PATCHHEIGHT 8
#define DCT_PATCHWIDTH 12
#define DCT_PATCHES 85
#define DCT_ROWS 12
#define DCT_WNDHEIGHT 2
#define DCT_WNDWIDTH 1
#define DCT_COEFFSIZE 3

struct b_d{
    unsigned char **data;
    unsigned char **bits;
    int size;
    struct b_d *next;
};

struct allshifts_and_bits{
    int size;
    struct b_d *bitsdata;
};

struct window_functions
{
    double *window_w;
    double *window_h;
    int sizew;
    int sizeh;
};

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
	printf("| dct - Iris-code generation (feature extraction) using the Monro algorithm   |\n");
	printf("|                                                                             |\n");
	printf("| Monro D. M., Rakshit S., Zhang D.: DCT-based Iris Recognition,              |\n");
	printf("| IEEE Trans Pattern Anal Mach Intell. 2007 Apr;29(4):586-95.                 |\n");
	printf("|                                                                             |\n");
	printf("| MODES                                                                       |\n");
	printf("|                                                                             |\n");
    printf("| (# 1) DCT iris code extraction from iris textures                           |\n");
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
    printf("| Michael Karnutsch (mkarnut@cosy.sbg.ac.at)                                  |\n");
    printf("| Christian Rathgeb (crathgeb@cosy.sbg.ac.at)                                 |\n");
    printf("| Peter Wild (pwild@cosy.sbg.ac.at)                                           |\n");
    printf("|                                                                             |\n");
    printf("| COPYRIGHT                                                                   |\n");
    printf("|                                                                             |\n");
    printf("| (C) 2012 All rights reserved. Do not distribute without written permission. |\n");
    printf("+-----------------------------------------------------------------------------+\n");
}

/** ------------------------------- image processing functions ------------------------------- **/

unsigned char **data, **left4, **left8, **left12, **right4, **right8, **right12;

struct allshifts_and_bits *ab = NULL;
struct window_functions *window = NULL;

int h;
int w;
int rotateDCT[3] = {4,8,12};
int degree0[64] = {0};
int degree10[64] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15};
int degree17[64] = {0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 20, 20, 20, 21, 21, 21, 22, 22, 23, 23, 23};
int degree25[64] = {0, 0,  1, 1, 2, 2,  3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31};
int degree32[64] = {0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8, 8, 9, 10, 11, 11, 12, 12, 13, 13, 14, 15, 16, 16, 17, 17, 18, 18, 19, 20, 21, 21, 22, 22, 23, 23, 24, 25, 26, 26, 27, 27, 28, 28, 29, 30, 31, 31, 32, 32, 33, 33, 34, 35, 36, 36, 37, 37, 38, 38, 39};
int degree38[64] = {0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12, 13, 13, 14, 15, 16, 16, 17, 18, 19, 19, 20, 21, 22, 22, 23, 24, 25, 25, 26, 27, 28, 28, 29, 30, 31, 31, 32, 33, 34, 34, 35, 36, 37, 37, 38, 39, 40, 40, 41, 42, 43, 43, 44, 45, 46, 46, 47};
int degree41[64] = {0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16, 17, 17, 18, 19, 20, 21, 22, 23, 24, 24, 25, 26, 27, 28, 29, 30, 31, 31, 32, 33, 34, 35, 36, 37, 38, 38, 39, 40, 41, 42, 43, 44, 45, 45, 46, 47, 48, 49, 50, 51, 52, 52, 53, 54, 55};
int degree45[64] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
int degree50[64] = {0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39,  41, 42, 43, 44, 45,  46, 47, 48,  50, 51,  52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 64, 65, 66, 68, 69, 70, 71};
int degree55[64] = {0, 1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24, 25, 26, 28, 29, 30, 31, 33, 34, 35, 36, 38, 39, 40, 41, 43, 44, 45, 46, 48, 49, 50, 51, 53, 54, 55, 56, 58, 59, 60, 61, 63, 64, 65, 66, 68, 69, 70, 71, 73, 74, 75, 76, 78, 79};
int degree60[64] = {0, 1, 3, 5, 7, 9, 11, 12, 13, 14, 16, 18, 20, 22, 24, 25, 26, 27, 29, 31, 33, 35, 37, 38, 39, 40, 42, 44, 46, 48, 50, 51, 52, 53, 55, 57, 59, 61, 63, 64, 65, 66, 67, 69, 71, 73, 75, 77, 78, 79, 81, 83, 85, 87, 89, 90, 91, 92, 93, 95, 97, 99, 101, 102};
int degree65[64] = {0, 2, 4, 6, 8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88,  90,  92,  94,  96,  98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126};
int degree70[64] = {0, 2, 5, 7, 10, 12, 15, 17, 18, 20, 23, 25, 28, 30, 33, 35, 36, 38, 41, 43, 45, 47, 50, 52, 53, 55, 58, 60, 63, 65, 68, 70, 71, 73,  76,  78,  81,  83,  85,  87,  88,  90,  93,  95,  98, 100, 103, 105, 106, 108, 111, 113, 116, 118, 121, 123, 124, 126, 129, 131, 134, 136, 139, 141};
int degree75[64] = {0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 120, 123, 126, 129, 132, 135, 138, 141, 145, 148, 151, 154, 157, 160, 163, 166, 169, 172, 175, 178, 181, 184, 187, 190};

void setBitTo1(uchar* code, int bitpos){
	code[bitpos / 8] |= (1 << (bitpos % 8));
}

void setBitTo0(uchar* code, int bitpos){
	code[bitpos / 8] &= (0xff ^ 1 << (bitpos % 8));
}

int signum(double i)
{
    if(i > 0.0) return 1;
    else return -1;
}

double *mydct(double *i, int n)
{
    long bin,k;
    double *d = (double*)malloc(sizeof(double) * n);
    double arg;
    double one = sqrt((float)n);
    one = 1/one;
    double two = 2/(float)n;
    two = sqrt(two);
    for (bin = 1; bin <= n; bin++)
    {
        d[bin - 1] = 0.;
        for (k = 1; k <= n; k++)
        {
            arg = (2*k - 1) * (bin - 1)  / (float)n;
            arg = arg * pihalf;
            d[bin - 1] += i[k - 1] * cos(arg);
        }
        if(bin == 1)
        {
            d[bin - 1] = d[bin - 1] * one;
        }
        else
        {
            d[bin - 1] = d[bin - 1] * two;
        }
    }
    return d;
}


unsigned char **truncate_image(const Mat& texture)
{
    h = (DCT_ROWS + 1)*(DCT_PATCHHEIGHT/2);
    w = (DCT_PATCHES + 9)*(DCT_PATCHWIDTH/2);
    unsigned char **image;
    image = (unsigned char**)malloc( h * sizeof(unsigned char*));
    int front = DCT_PATCHWIDTH;
    int i = (DCT_CUTOFF)*(texture.cols);
    int diff = w - texture.cols - front;
    uchar* textureData = texture.data;
    for(int q = 0; q < h; q++)
    {
        image[q] = (unsigned char*)malloc( w * sizeof(unsigned char));
    }

    if(diff > 0)
    {
      for(int r = 0; r < h*texture.cols; r++)
      {
          image[(r/512)][(r%512) + front] = textureData[r + i];
      }
        for(int q = 0; q < h; q++)
        {
            for(int e = 0; e < diff; e++)
            {
                image[q][texture.cols + e + front] = image[q][e + front];
                //printf("(%i,%i) = %d \n", q, img->w + e, image[q][e]);
            }
            for(int t = 0; t < front; t++)
            {
                image[q][t] = image[q][texture.cols + t];
            }
        }
    }
    else
    {
        for(int q = 0; q < h; q++)
        {
            for(int e = 0; e < w; e++)
            {
                image[q][e] = textureData[q*texture.cols + i + e];
            }
        }
    }
    return image;
}


unsigned char **shift_image(unsigned char **img)
{
    int *shiftpointer;
    switch(DCT_DEGREE)
    {
        case 0:
          shiftpointer = degree0;
          break;
        case 10:
          shiftpointer = degree10;
          break;
        case 17:
          shiftpointer = degree17;
          break;
        case 25:
          shiftpointer = degree25;
          break;
        case 32:
          shiftpointer = degree32;
          break;
        case 38:
          shiftpointer = degree38;
          break;
        case 41:
          shiftpointer = degree41;
          break;
        case 45:
          shiftpointer = degree45;
          break;
        case 50:
          shiftpointer = degree50;
          break;
        case 55:
          shiftpointer = degree55;
          break;
        case 60:
          shiftpointer = degree60;
          break;
        case 65:
          shiftpointer = degree65;
          break;
        case 70:
          shiftpointer = degree70;
          break;
        case 75:
          shiftpointer = degree75;
          break;
        default:
          shiftpointer = degree45;
          printf("degree: %i not possible, 45 used instead\n" , DCT_DEGREE);
          break;
    }
    unsigned char **shifted = (unsigned char**)malloc( h * sizeof(unsigned char*));
    int y = 0;
    while(y < h)
    {
        shifted[y] = (unsigned char*)malloc( w * sizeof(unsigned char));
        int i;
        for(i = 0; i < w; i++)
        {
            shifted[y][(i + shiftpointer[y])%w] = img[y][i];
        }
        y++;
    }
    return shifted;
}

void rotate_image()
{
    left4 = (unsigned char**)malloc( h * sizeof(unsigned char*));
    left8 = (unsigned char**)malloc( h * sizeof(unsigned char*));
    left12 = (unsigned char**)malloc( h * sizeof(unsigned char*));
    right4 = (unsigned char**)malloc( h * sizeof(unsigned char*));
    right8 = (unsigned char**)malloc( h * sizeof(unsigned char*));
    right12 = (unsigned char**)malloc( h * sizeof(unsigned char*));
    int y = 0;
    while(y < h)
    {
        left4[y] = (unsigned char*)malloc( w * sizeof(unsigned char));
        left8[y] = (unsigned char*)malloc( w * sizeof(unsigned char));
        left12[y] = (unsigned char*)malloc( w * sizeof(unsigned char));
        right4[y] = (unsigned char*)malloc( w * sizeof(unsigned char));
        right8[y] = (unsigned char*)malloc( w * sizeof(unsigned char));
        right12[y] = (unsigned char*)malloc( w * sizeof(unsigned char));
        int i;
        for(i = 0; i < w; i++)
        {
          left4[y][i] = data[y][(i + rotateDCT[0])%w];
          left8[y][i] = data[y][(i + rotateDCT[1])%w];
          left12[y][i] = data[y][(i + rotateDCT[2])%w];
          right4[y][i] = data[y][(i + w - rotateDCT[0])%w];
          right8[y][i] = data[y][(i + w - rotateDCT[1])%w];
          right12[y][i] = data[y][(i + w -rotateDCT[2])%w];
        }
        y++;
    }
    ab = (allshifts_and_bits*)malloc(sizeof(struct allshifts_and_bits ));
    ab->size = 7;
    struct b_d *left12bd = (b_d*)malloc(sizeof(struct b_d));
    struct b_d *left8bd = (b_d*)malloc(sizeof(struct b_d));
    struct b_d *left4bd = (b_d*)malloc(sizeof(struct b_d));
    struct b_d *origbd = (b_d*)malloc(sizeof(struct b_d));
    struct b_d *right4bd = (b_d*)malloc(sizeof(struct b_d));
    struct b_d *right8bd = (b_d*)malloc(sizeof(struct b_d));
    struct b_d *right12bd = (b_d*)malloc(sizeof(struct b_d));
    origbd->data = data;
    origbd->next = right4bd;
    right4bd->data = right4;
    right4bd->next = right8bd;
    right8bd->data = right8;
    right8bd->next = right12bd;
    right12bd->data = right12;
    right12bd->next = left4bd;
    left4bd->data = left4;
    left4bd->next = left8bd;
    left8bd->data = left8;
    left8bd->next = left12bd;
    left12bd->data = left12;
    left12bd->next = NULL;
    ab->bitsdata = origbd;
}

double *compute_dct(double *patch)
{
    double *d, *all;
    d= (double*)malloc(DCT_COEFFSIZE * sizeof(double));
    all = mydct(patch, DCT_PATCHHEIGHT);
    for(int i = 0; i < DCT_COEFFSIZE; i++)
    {
        d[i] = all[i];
    }
    return d;
}

unsigned char *generate_codes()
{
    struct b_d *actualstruct = ab->bitsdata;
    unsigned char **actualdata;
    double actualpatch[DCT_PATCHHEIGHT];
    double *actualcoeff;
    double dctvalues[DCT_COEFFSIZE][DCT_PATCHES + 2 + 3];
    double diffs[DCT_COEFFSIZE][DCT_PATCHES + 1 + 3];
    unsigned char **bits;
    int howmanybits = DCT_PATCHES * DCT_ROWS;
    int extrabits = 0;
    int size = howmanybits;
    if(howmanybits%8 > 0)
    {
        howmanybits = howmanybits + 8 - howmanybits%8;
        size = howmanybits;
        extrabits = 1;
    }
    int  r, p, ah, aw, c, d, halfpatchsizeheight = DCT_PATCHHEIGHT/2, halfpatchsizewidth = DCT_PATCHWIDTH/2;
    while(actualstruct)
    {
        actualstruct->size = size;
	
        bits = (uchar**)malloc(sizeof(unsigned char*) * DCT_COEFFSIZE);
        int q;
        for(q = 0; q < DCT_COEFFSIZE; q++)
        {
            bits[q] = (unsigned char*)malloc(sizeof(unsigned char) * size);
        }
        actualdata = actualstruct->data;
        for(r = 0; r < DCT_ROWS; r++)
        {
            for(p = 0; p < DCT_PATCHES + 2 + 3; p++)
            {
                for(ah = 0; ah < DCT_PATCHHEIGHT; ah++)
                {
                    actualpatch[ah] = 0.0;
                    for(aw = 0; aw < DCT_PATCHWIDTH; aw++)
                    {
                        actualpatch[ah] += (window->window_w[aw] * actualdata[halfpatchsizeheight*r + ah][halfpatchsizewidth*p + aw]);
                    }
                    actualpatch[ah] = actualpatch[ah] * window->window_h[ah];
                }
                actualcoeff = compute_dct(actualpatch);
                for(c = 0; c < DCT_COEFFSIZE; c++)
                {
                    dctvalues[c][p] = actualcoeff[c];
                }
            }
            for(c = 0; c < DCT_COEFFSIZE; c++)
            {
                for(d = 0; d < DCT_PATCHES + 1 + 3; d++)
                {
                    diffs[c][d] = dctvalues[c][d + 1] - dctvalues[c][d];
                }
                for(d = 0; d < DCT_PATCHES; d++)
                {
                    if(signum(diffs[c][d + 3])==signum(diffs[c][d + 4]))
                    {
                        bits[c][r*DCT_PATCHES + d] = 0;
                    }
                    else
                    {
                        bits[c][r*DCT_PATCHES + d] = 1;
                    }
                }
            }
        }
        if(extrabits == 1)
        {
            for(c = 0; c < DCT_COEFFSIZE; c++)
            {
                for(int e = 0; e < size - DCT_PATCHES * DCT_ROWS; e++)
                {
                    int randbit = rand()%10 + 1;
                    if(randbit > 5)
                    {
                        bits[c][DCT_PATCHES * DCT_ROWS + e] = 1;
                    }
                    else
                    {
                        bits[c][DCT_PATCHES * DCT_ROWS + e] = 0;
                    }
                }
            }
        }
        actualstruct->bits = bits;
        actualstruct = actualstruct->next;
    }
    
    int idx_1,i;
    unsigned char *code = (unsigned char*)malloc(512*21*2 * sizeof(unsigned char));
    int cnt = 0;
    actualstruct = ab->bitsdata;
    while(actualstruct)
    {
        for(i = 0; i < DCT_COEFFSIZE; i++)
        {
            for (idx_1 = 0; idx_1 < actualstruct->size; idx_1++)
            {
                code[cnt*DCT_COEFFSIZE*actualstruct->size+i*actualstruct->size + idx_1] = actualstruct->bits[i][idx_1];
            }
        }
        cnt++;
        actualstruct = actualstruct->next;
    }
    return code;
}

double *hann(int size, int add)
{
    double *hh = (double*)malloc(sizeof(double) * (size + 2*add));
    double *r = (double*)malloc(sizeof(double) * size);
    for(int j = 0; j < size; j++) r[j] = 0.0;

    int L = size + 2*add + 2;
    int N = L - 1;
    for(int i = 1; i <= size + 2*add ; i++)
    {
        hh[i-1] = (double)0.5 * (1.0 - cos(2.0*pi*(double)(i) / (double)N));
    }
    if(add >= 0)
    {
        for(int j = 0; j < size; j++)
        {
            r[j] = 0.0;
            r[j] = hh[j + add];
        }
    }
    else
    {
        for(int i = 0; i < size; i++)
        {
            r[i] = (double)0.0;
        }

        for(int k = -add; k < size + add; k++)
        {
            r[k] = hh[k + add];
        }
    }
    return r;
}

struct window_functions  *compute_window()
{
    struct window_functions *we  = (window_functions*)malloc(sizeof(struct window_functions));
    we->sizeh = DCT_PATCHHEIGHT + DCT_WNDHEIGHT;
    we->sizew = DCT_PATCHWIDTH + DCT_WNDWIDTH;
    we->window_h = hann(DCT_PATCHHEIGHT , DCT_WNDHEIGHT);
    we->window_w = hann(DCT_PATCHWIDTH , DCT_WNDWIDTH);
    return we;
}

void freeall()
{
    struct b_d *actualstruct = ab->bitsdata, *next;
    unsigned char **actualdata;
    unsigned char **actualbits;
    while(actualstruct)
    {
        actualdata = actualstruct->data;
        actualbits = actualstruct->bits;
        next = actualstruct->next;
        free(actualdata);
        free(actualbits);
        free(actualstruct);
        actualstruct = next;
    }
    free(ab);
    free(window);
}

/*
 * The Monro feature extraction algorithm
 *
 * code: Code matrix
 * texture: texture matrix
 */
void featureExtract(Mat& code, const Mat& texture)
{
    /*truncate to size*/
    unsigned char **temp;
    uchar* features;
    uchar* iris_code = code.data;
    
    //imagei = img;
    temp = truncate_image(texture);
    /*shift image by degree x*/
    data = shift_image(temp);
    /*rotates the image to the left and to the right and store in struct*/
    rotate_image();
    /*compute windowfunction*/
    window = compute_window();
    
    /*generates the codes for the 7 shifted images*/
    features = generate_codes();
    
    for (int i=0; i < texture.cols*(texture.rows/3)*2; i++)
    {
		if (features[i] == 0) setBitTo0(iris_code,i);
		else setBitTo1(iris_code,i);
    }
    //freeall();
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
				CV_Assert(img.data != 0);
				Mat out;
				int w = img.cols;
				int h = (img.rows/DCT_COEFFSIZE)*2;
				if (!quiet) printf("Creating %d x %d iris-code ...\n", w, h);
				Mat code (1,(w*h)/8,CV_8UC1);
				code.setTo(0);
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
