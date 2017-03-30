/*
 * hd.cpp
 *
 * Author: P. Wild (pwild@cosy.sbg.ac.at)
 *
 * Calculates hamming distance of iris codes
 *
 */
#include "version.h"
#include <cstdio>
#include <map>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/regex.hpp>
#include <boost/filesystem.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

using namespace std;
using namespace cv;

/** no globbing in win32 mode **/
int _CRT_glob = 0;

/** Algorithms **/
static const int ALG_MINHD = 0, ALG_MAXHD = 1, ALG_SSF = 2;
/** Program modes **/
static const int MODE_MAIN = 1, MODE_HELP = 2;

static const int htlut[256] = {0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8};

/*
 * Print command line usage for this program
 */
void printUsage() {
    printVersion();
	printf("+-----------------------------------------------------------------------------+\n");
	printf("| hd - calculates hamming distance of iris codes                              |\n");
	printf("|                                                                             |\n");
	printf("| MODES                                                                       |\n");
	printf("|                                                                             |\n");
	printf("| (# 1) HD calculation of the input images (cross comparison)                 |\n");
	printf("| (# 2) usage                                                                 |\n");
	printf("|                                                                             |\n");
	printf("| ARGUMENTS                                                                   |\n");
	printf("|                                                                             |\n");
	printf("+------+------------+---+---+-------------------------------------------------+\n");
	printf("| Name | Parameters | # | ? | Description                                     |\n");
	printf("+------+------------+---+---+-------------------------------------------------+\n");
	printf("| -i   | infile1    | 1 | N | source/reference iris codes (use * as wildcard, |\n");
	printf("|      | infile2    |   |   | all other files may refer to n-th * with ?n)    |\n");
	printf("| -s   | param+     | 1 | Y | min max: min/max (0 0) number of bit shifts     |\n");
	printf("|      |            |   | Y | img: shifted src (?n = n-th * in infile1,* =any)|\n");
    printf("| -ss  | shiftstep  |   | Y | Number of grouped bits to shift for one step of |\n");
    printf("|      |            |   |   | -s shift.                                       |\n");
	printf("| -m   | maskfile1  | 1 | Y | source/reference iris masks (?n/!n = n-th * in  |\n");
	printf("|      | maskfile2  |   |   | infile1/img, ?n = n-th * in infile2)            |\n");
	printf("| -a   | algorithm  | 1 | Y | HD-based algorithm (minhd)                      |\n");
	printf("|      |            |   |   | minhd: minimum HD for all shifts                |\n");
	printf("|      |            |   |   | maxhd: 1-maximum HD for all shifts              |\n");
	printf("|      |            |   |   | ssf: shift score fusion                         |\n");
	printf("| -n   | from to    | 1 | Y | starting (0) and ending bit (MAX)               |\n");
	printf("| -o   | outfile    | 1 | Y | target text                                     |\n");
	printf("| -owp |            | 1 | Y | write full paths in outfile instead of file only|\n");
	printf("| -q   |            | 1 | Y | quiet mode on (off)                             |\n");
	printf("| -t   |            | 1 | Y | time progress on (off)                          |\n");
	printf("| -#   |            | 1 | N | use memoization if memory is not a concern.     |\n");
	printf("| -b   |            | 1 | N | also record the bit shift at which the HD occurs|\n");
	printf("| -h   |            | 2 | N | prints usage                                    |\n");
	printf("+------+------------+---+---+-------------------------------------------------+\n");
	printf("|                                                                             |\n");
	printf("| EXAMPLE USAGE                                                               |\n");
    printf("|                                                                             |\n");
    printf("| -i s1.png s2.png -m s1_mask.png s2_mask.png -o compare.txt                  |\n");
    printf("| -i *.png *.png -s -7 7 -o compare.txt -q -t                                 |\n");
    printf("| -i *.png *.png -a ssf -s ?1_shifted_*.png -7 7 -o compare.txt -q -t         |\n");
    printf("|                                                                             |\n");
	printf("| AUTHOR                                                                      |\n");
	printf("|                                                                             |\n");
	printf("| Peter Wild (pwild@cosy.sbg.ac.at)                                           |\n");
	printf("|                                                                             |\n");
	printf("| COPYRIGHT                                                                   |\n");
	printf("|                                                                             |\n");
	printf("| (C) 2012 All rights reserved. Do not distribute without written permission. |\n");
	printf("+-----------------------------------------------------------------------------+\n");
}

/**
 * Fast hamming distance estimation
 * a: sample iris code
 * b: reference iris code
 * start8: starting 8-bit block
 * stop8: ending 8-bit block
 */
unsigned int hd(const Mat a, const Mat b, const unsigned int start8, const unsigned int stop8, const Mat mask = Mat()){
	unsigned int dist = 0;
	MatConstIterator_<uchar> pa = a.begin<uchar>();
	MatConstIterator_<uchar> pb = b.begin<uchar>();
	MatConstIterator_<uchar> enda = pa + stop8;
	pa += start8;
	pb += start8;
	if (!mask.empty()){
		MatConstIterator_<uchar> pm = mask.begin<uchar>();
		//unsigned int * pm = (unsigned int*) mask.data;
		pm += start8;
		for (;pa<enda; pa++, pb++, pm++){
			dist += htlut[(*pa ^*pb) & *pm];

			/*unsigned int val = (*pa ^*pb) & *pm;
			while(val)
			{
				++dist;
				val &= val - 1;
			}*/
		}
	}
	else {
		for (;pa<enda; pa++, pb++){
			//unsigned int val = *pa ^*pb;
			dist += htlut[*pa ^*pb];
			/*
			while(val)
			{
				++dist;
				val &= val - 1;
			}*/
		}
	}
	return dist;

}

/**
 * Shifts source by a given shift count
 * src: source iris code
 * dst: destination (shifted) iris code
 * shifts: shift count (positive values indicate left shifts)
 */
void shift(const Mat src, Mat dst, const int shifts){
	int size = src.cols*src.rows; // size in bytes
	MatConstIterator_<uchar> psrc = src.begin<uchar>();
	MatConstIterator_<uchar> endsrc = src.end<uchar>();
	MatIterator_<uchar> pdst = dst.begin<uchar>();
	MatIterator_<uchar> enddst = dst.end<uchar>();

	if (shifts >= 0){ // left shift
		unsigned int offset = shifts / 8;
		psrc += offset;
		unsigned int shiftCount = shifts % 8;
		unsigned int ishiftCount = 8 - shiftCount;
		for (;pdst<enddst; pdst++){
			*pdst = (*psrc << shiftCount);
			psrc++;
			if (psrc == endsrc) psrc = src.begin<uchar>();
			*pdst |= (*psrc >> (ishiftCount));
		}
	}
	else { // right shift
		unsigned int offset = (-shifts) / 8;
		offset = (size-offset-1) % size;
		psrc += offset;
		unsigned int shiftCount = (-shifts) % 8;
		unsigned int ishiftCount = 8 - shiftCount;
		for (;pdst<enddst; pdst++){
			*pdst = (*psrc << ishiftCount);
			psrc++;
			if (psrc == endsrc) psrc = src.begin<uchar>();
			*pdst |= (*psrc >> (shiftCount));
		}
	}
}

/**
 * Shifts matrix a by a given shift count and intersects result with b
 * a: sample source iris mask
 * b: reference source iris mask
 * dst: destination (shifted) iris mask
 * shifts: shift count (positive values indicate left shifts)
 */
void intersectShifted(const Mat a, const Mat b, Mat dst, const int shifts){
	uchar * pb = b.data;
	uchar * pa = a.data;
	uchar * pdst = dst.data;
	unsigned int size = dst.cols;
	unsigned int offset;
	if (shifts >= 0){ // left shift
		offset = shifts / 8;
		uchar * enddst = pdst + dst.cols;
		unsigned int shiftCount = shifts % 8;
		unsigned int ishiftCount = 8 - shiftCount;
		offset = (size + offset) % size;
		for (;pdst<enddst; pdst++, pb++){
			*pdst = (pa[offset] << shiftCount);
			offset = ((offset+1)%size);
			*pdst |= (pa[offset] >> (ishiftCount));
			*pdst &= *pb;
		}
	}
	else { // right shift
		offset = (-shifts) / 8;
		uchar * enddst = pdst + dst.cols;
		unsigned int shiftCount = (-shifts) % 8;
		unsigned int ishiftCount = 8 - shiftCount;
		offset = (size-offset-1) % size;
		for (;pdst<enddst; pdst++, pb++){
			*pdst = (pa[offset] << ishiftCount);
			offset = ((offset+1)%size);
			*pdst |= (pa[offset] >> (shiftCount));
			*pdst &= *pb;
		}
	}
}

/**
 * determines the best fractional Hamming Distance of two iris codes
 * a: first iris code
 * b: second iris code
 * start8: starting 8-bit block (inclusive)
 * stop8: ending 8-bit block (exclusive)
 * shifts: number of shifts
 * aMask: mask for first iris code
 * bMask: mask for second iris code
 */
std::pair<double,int> minHD(const Mat& a, const Mat& b, const unsigned int start8, const unsigned int stop8, const int minShifts, const int maxShifts,const int shiftStep, const Mat aMask, const Mat bMask = Mat()){
    std::pair<double, int> result = {1,0};
	if (!aMask.empty() && !bMask.empty()){
		Mat mask(b.rows,b.cols,CV_8UC1);
		Mat zero(b.rows,b.cols,CV_8UC1);
		zero.setTo(Scalar(0));
		double hamdist = 1;
		Mat imgSmplShifted(a.rows,a.cols,CV_8UC1);
		for (int ss=minShifts; ss<=maxShifts; ss++){
            int s = ss*shiftStep;
			shift(a,imgSmplShifted,s);
			intersectShifted(aMask,bMask,mask,s);
			int codeLengthBits = hd(mask, zero, start8, stop8);
			double shiftedHamdist = (codeLengthBits == 0) ? 0 : ((double)hd(imgSmplShifted,b,start8,stop8,mask)) / codeLengthBits;
			if (shiftedHamdist < hamdist){
                hamdist = shiftedHamdist;
                result.second = ss;
            }
		}
		result.first = hamdist;
	}
	else {
		int codeLengthBits = 8*(stop8-start8);
		unsigned int hamdist = codeLengthBits;
		Mat imgSmplShifted(a.rows,a.cols,CV_8UC1);
		for (int ss=minShifts; ss<=maxShifts; ss++){
            int s = ss*shiftStep;
			shift(a,imgSmplShifted,s);
			unsigned int shiftedHamdist = hd(imgSmplShifted,b,start8,stop8);
			if (shiftedHamdist < hamdist){
                hamdist = shiftedHamdist;
                result.second = ss;
            }
		}
		result.first = (((double)hamdist) / (codeLengthBits));
	}
	return result;
}

/**
 * determines the best fractional Hamming Distance of an iris code with a list of shifted versions of this code
 * a: shifted versions of first iris code
 * b: second iris code
 * start8: starting 8-bit block (inclusive)
 * stop8: ending 8-bit block (exclusive)
 * aMask: masks for corresponding shifted versions of first iris code
 * bMask: mask for second iris code
 */
std::pair<double, int> minHD(const vector<Mat>& a, const Mat& b, const unsigned int start8, const unsigned int stop8, const vector<Mat>& aMask, const Mat bMask = Mat()){
    std::pair<double, int> result = {1,0};
	if (!aMask.empty() && !bMask.empty()){
		CV_Assert(aMask.size() == a.size());
		Mat mask(b.rows,b.cols,CV_8UC1);
		Mat zero(b.rows,b.cols,CV_8UC1);
		zero.setTo(Scalar(0));
		double hamdist = 1;
		for (unsigned int i=0; i<a.size();i++){
			intersectShifted(aMask[i],bMask,mask,0);
			int codeLengthBits = hd(mask, zero, start8, stop8);
			double shiftedHamdist = (codeLengthBits == 0) ? 0 : ((double)hd(a[i],b,start8,stop8,mask)) / codeLengthBits;
			if (shiftedHamdist < hamdist){
                hamdist = shiftedHamdist;
                result.second = i;
            }
		}
		result.first = hamdist;
	} else {
		int codeLengthBits = 8*(stop8-start8);
		unsigned int hamdist = codeLengthBits;
		for (unsigned int i=0; i<a.size();i++){
			unsigned int shiftedHamdist = hd(a[i],b,start8,stop8);
			if (shiftedHamdist < hamdist){
                hamdist = shiftedHamdist;
                result.second = i;
            }
		}
		result.first = (((double)hamdist) / (codeLengthBits));
	}
	return result;
}

/**
 * determines the worst fractional Hamming Distance of two iris codes
 * a: first iris code
 * b: second iris code
 * start8: starting 8-bit block (inclusive)
 * stop8: ending 8-bit block (exclusive)
 * shifts: number of shifts
 * aMask: mask for first iris code
 * bMask: mask for second iris code
 */
std::pair<double, int> maxHD(const Mat& a, const Mat& b, const unsigned int start8, const unsigned int stop8, const int minShifts, const int maxShifts, const int shiftStep, const Mat aMask, const Mat bMask = Mat()){
    std::pair<double, int> result = {0,0};
	if (!aMask.empty() && !bMask.empty()){
		Mat mask(b.rows,b.cols,CV_8UC1);
		Mat zero(b.rows,b.cols,CV_8UC1);
		zero.setTo(Scalar(0));
		double hamdist = 0;
		Mat imgSmplShifted(a.rows,a.cols,CV_8UC1);
		for (int ss=minShifts; ss<=maxShifts; ss++){
            int s = ss*shiftStep;
			shift(a,imgSmplShifted,s);
			intersectShifted(aMask,bMask,mask,s);
			int codeLengthBits = hd(mask, zero, start8, stop8);
			double shiftedHamdist = (codeLengthBits == 0) ? 0 : ((double)hd(imgSmplShifted,b,start8,stop8,mask)) / codeLengthBits;
			if (shiftedHamdist > hamdist){
                hamdist = shiftedHamdist;
                result.second = ss;
            }
		}
		result.first = 1 - hamdist;
	}
	else {

		int codeLengthBits = 8*(stop8-start8);
		unsigned int hamdist = 0;
		Mat imgSmplShifted(a.rows,a.cols,CV_8UC1);
		for (int ss=minShifts; ss<=maxShifts; ss++){
            int s = ss*shiftStep;
			shift(a,imgSmplShifted,s);
			unsigned int shiftedHamdist = hd(imgSmplShifted,b,start8,stop8);
			if (shiftedHamdist > hamdist){
                hamdist = shiftedHamdist;
                result.second = ss;
            }
		}
		result.first = 1-(((double)hamdist) / (codeLengthBits));
	}
	return result;
}

/**
 * determines the worst fractional Hamming Distance of an iris code with a list of shifted versions of this code
 * a: shifted versions of first iris code
 * b: second iris code
 * start8: starting 8-bit block (inclusive)
 * stop8: ending 8-bit block (exclusive)
 * aMask: masks for corresponding shifted versions of first iris code
 * bMask: mask for second iris code
 */
std::pair<double, int> maxHD(const vector<Mat>& a, const Mat& b, const unsigned int start8, const unsigned int stop8, const vector<Mat>& aMask, const Mat bMask = Mat()){
	std::pair<double,int>  result = {1,0};
	if (!aMask.empty() && !bMask.empty()){
		CV_Assert(aMask.size() == a.size());
		Mat mask(b.rows,b.cols,CV_8UC1);
		Mat zero(b.rows,b.cols,CV_8UC1);
		zero.setTo(Scalar(0));
		double hamdist = 0;
		for (unsigned int i=0; i<a.size();i++){
			intersectShifted(aMask[i],bMask,mask,0);
			int codeLengthBits = hd(mask, zero, start8, stop8);
			double shiftedHamdist = (codeLengthBits == 0) ? 0 : ((double)hd(a[i],b,start8,stop8,mask)) / codeLengthBits;
			if (shiftedHamdist > hamdist){
                hamdist = shiftedHamdist;
                result.second = i;
            }
		}
		result.first = 1 - hamdist;
	}
	else {
		int codeLengthBits = 8*(stop8-start8);
		unsigned int hamdist = 0;
		for (unsigned int i=0; i<a.size();i++){
			unsigned int shiftedHamdist = hd(a[i],b,start8,stop8);
			if (shiftedHamdist > hamdist){
                hamdist = shiftedHamdist;
                result.second = i;
            }
		}
		result.first = 1- (((double)hamdist) / (codeLengthBits));
	}
	return result;
}

/**
 * determines Shifting Score Fusion of two iris codes
 * a: first iris code
 * b: second iris code
 * start8: starting 8-bit block (inclusive)
 * stop8: ending 8-bit block (exclusive)
 * shifts: number of shifts
 * aMask: mask for first iris code
 * bMask: mask for second iris code
 */
std::pair<double,int>  ssf(const Mat& a, const Mat& b, const unsigned int start8, const unsigned int stop8, const int minShifts, const int maxShifts, const int shiftStep, const Mat aMask, const Mat bMask = Mat()){
	std::pair<double,int>  result = {0,666};
	if (!aMask.empty() && !bMask.empty()){
		Mat mask(b.rows,b.cols,CV_8UC1);
		Mat zero(b.rows,b.cols,CV_8UC1);
		zero.setTo(Scalar(0));
		intersectShifted(aMask,bMask,mask,0);
		double hamdist = 1;
		double maxhamdist = 0;
		Mat imgSmplShifted(a.rows,a.cols,CV_8UC1);
		for (int ss=minShifts; ss<=maxShifts; ss++){
            int s = ss*shiftStep;
			shift(a,imgSmplShifted,s);
			intersectShifted(aMask,bMask,mask,s);
			int codeLengthBits = hd(mask, zero, start8, stop8);
			double shiftedHamdist = (codeLengthBits == 0) ? 0 : ((double)hd(imgSmplShifted,b,start8,stop8,mask)) / codeLengthBits;
			if (shiftedHamdist < hamdist){
                hamdist = shiftedHamdist;
                result.second = ss;
            }
			if (shiftedHamdist > maxhamdist) maxhamdist = shiftedHamdist;
		}
		result.first = ((1-maxhamdist) +  hamdist)/2;
	}
	else {
		int codeLengthBits = 8*(stop8-start8);
		unsigned int hamdist = codeLengthBits;
		unsigned int maxhamdist = 0;
		Mat imgSmplShifted(a.rows,a.cols,CV_8UC1);
		for (int ss=minShifts; ss<=maxShifts; ss++){
            int s = ss*shiftStep;
			shift(a,imgSmplShifted,s);
			unsigned int shiftedHamdist = hd(imgSmplShifted,b,start8,stop8);
			if (shiftedHamdist < hamdist){
                hamdist = shiftedHamdist;
                result.second = ss;
            }
			if (shiftedHamdist > maxhamdist) maxhamdist = shiftedHamdist;
		}
		result.first = ((1-(((double)maxhamdist) / (codeLengthBits))) + (((double)hamdist) / (codeLengthBits)))/2;
	}
	return result;
}

/**
 * determines Shifting Score Fusion of an iris code with a list of shifted versions of this code
 * a: shifted versions of first iris code
 * b: second iris code
 * start8: starting 8-bit block (inclusive)
 * stop8: ending 8-bit block (exclusive)
 * aMask: masks for corresponding shifted versions of first iris code
 * bMask: mask for second iris code
 */
std::pair<double,int>  ssf(const vector<Mat>& a, const Mat& b, const unsigned int start8, const unsigned int stop8, const vector<Mat>& aMask, const Mat bMask = Mat()){
	std::pair<double,int>  result = {0,0};
	if (!aMask.empty() && !bMask.empty()){
		CV_Assert(aMask.size() == a.size());
		Mat mask(b.rows,b.cols,CV_8UC1);
		Mat zero(b.rows,b.cols,CV_8UC1);
		zero.setTo(Scalar(0));
		int codeLengthBits = 0;
		double hamdist = 1;
		double maxhamdist = 0;
		for (unsigned int i=0; i<a.size();i++){
			intersectShifted(aMask[i],bMask,mask,0);
			codeLengthBits = hd(mask, zero, start8, stop8);
			double shiftedHamdist = (codeLengthBits == 0) ? 0 : ((double)hd(a[i],b,start8,stop8,mask)) / codeLengthBits;
			if (shiftedHamdist < hamdist){
                hamdist = shiftedHamdist;
                result.second = i;
            }
			if (shiftedHamdist > maxhamdist) maxhamdist = shiftedHamdist;
		}
		result.first = ((1-maxhamdist) +  hamdist)/2;
	}
	else {
		int codeLengthBits = 8*(stop8-start8);
		unsigned int hamdist = codeLengthBits;
		unsigned int maxhamdist = 0;
		for (unsigned int i=0; i<a.size();i++){
			unsigned int shiftedHamdist = hd(a[i],b,start8,stop8);
			if (shiftedHamdist < hamdist){
                hamdist = shiftedHamdist;
                result.second = i;
            }
			if (shiftedHamdist > maxhamdist) maxhamdist = shiftedHamdist;
		}
		result.first = ((1-(((double)maxhamdist) / (codeLengthBits))) + (((double)hamdist) / (codeLengthBits)))/2;
	}
	return result;
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

string skipPath(string from){
    string::size_type idx = from.find_last_of("/\\"); // / is for linux, \\ is for windows
    if( idx == string::npos ) return from; // not found, do nothing
    return from.substr(idx+1); // return after path
}

/** ------------------------------- Program ------------------------------- **/

bool use_mem = false;
Mat imread_mem( const string& filename, int flags=1){
    if( !use_mem) return imread(filename, flags);
    static map< std::pair<string, int>, Mat> memmap;
    auto key = std::make_pair(filename, flags);
    auto memitem = memmap.find( key );
    Mat ret;
    if( memitem != memmap.end()){
        ret =  memitem->second;
    } else{
        memmap[ key ]=imread(filename, flags);
        ret = memmap[ key ];
    }
    return ret;
}


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
			cmdCheckOpts(cmd,"-i|-m|-s|-ss|-a|-n|-o|-owp|-q|-t|-#|-b");
			cmdCheckOptExists(cmd,"-i");
			cmdCheckOptSize(cmd,"-i",2);
			string infilesSmpl = cmdGetPar(cmd,"-i",0);
			string infilesRef = cmdGetPar(cmd,"-i",1);
			bool masks = (cmdGetOpt(cmd,"-m") != 0);
			if (masks) cmdCheckOptSize(cmd,"-m",2);
			string masksSmpl = ((masks) ? cmdGetPar(cmd,"-m",0) : "");
			string masksRef = ((masks) ? cmdGetPar(cmd,"-m",1) : "");
			bool shiftedfiles = (cmdGetOpt(cmd,"-s") != 0 && cmdSizePars(cmd,"-s") == 1);
			string shiftfiles = ((shiftedfiles) ? cmdGetPar(cmd,"-s") : "");
			int minShifts = 0;
			int maxShifts = 0;
			if (cmdGetOpt(cmd,"-s") != 0 && !shiftedfiles){
				cmdCheckOptSize(cmd,"-s",2);
				minShifts = cmdGetParInt(cmd,"-s",0);
				maxShifts = cmdGetParInt(cmd,"-s",1);
			}
			int shiftStep = 1;
			if (cmdGetOpt(cmd,"-ss") != 0 && !shiftedfiles){
				cmdCheckOptSize(cmd,"-ss",1);
				shiftStep = cmdGetParInt(cmd,"-ss",0);
			}
			int alg = ALG_MINHD;
			if (cmdGetOpt(cmd,"-a") != 0){
				cmdCheckOptSize(cmd,"-a",1);
				string algo = cmdGetPar(cmd,"-a");
				if (algo == "maxhd"){
					alg = ALG_MAXHD;
				}
				else if (algo == "ssf"){
					alg = ALG_SSF;
				}
			}
			unsigned int from = 0;
			unsigned int to = INT_MAX;
			if (cmdGetOpt(cmd,"-n") != 0){
				cmdCheckOptSize(cmd,"-n",2);
				from = cmdGetParInt(cmd,"-n",0);
				CV_Assert(from % 8 == 0);
				from /= 8;
				to = cmdGetParInt(cmd,"-n",1);
				CV_Assert(to % 8 == 0);
				to /= 8;
			}
			string outfile;
            bool outfile_with_path = false;
			if (cmdGetOpt(cmd,"-o") != 0){
				cmdCheckOptSize(cmd,"-o",1);
				outfile = cmdGetPar(cmd,"-o");
                if( cmdGetOpt(cmd, "-owp") != 0){
				    cmdCheckOptSize(cmd,"-owp",0);
                    outfile_with_path = true;
                }
			}
			bool quiet = false;
			if (cmdGetOpt(cmd,"-q") != 0){
				cmdCheckOptSize(cmd,"-q",0);
				quiet = true;
			}
            bool writebitshift = false;
			if (cmdGetOpt(cmd,"-b") != 0){
				cmdCheckOptSize(cmd,"-b",0);
                writebitshift = true;
			}
            //global use_mem
			if (cmdGetOpt(cmd,"-#") != 0){
				cmdCheckOptSize(cmd,"-#",0);
				use_mem = true;
			}
			bool time = false;
			if (cmdGetOpt(cmd,"-t") != 0){
				cmdCheckOptSize(cmd,"-t",0);
				time = true;
			}
			// starting routine
			Timing timing(1,quiet);
			vector<string> filesSmpl;
			patternToFiles(infilesSmpl,filesSmpl);
			vector<string> filesRef;
			patternToFiles(infilesRef,filesRef);
			CV_Assert(filesSmpl.size() > 0);
			CV_Assert(filesRef.size() > 0);
			timing.total = filesSmpl.size() * filesRef.size();
			ofstream cfile;
			if (!outfile.empty()){
				if (!quiet) printf("Opening result file '%s' ...\n", outfile.c_str());;
				cfile.open(outfile.c_str(),ios::out | ios::trunc);
				if (!(cfile.is_open())) {
					CV_Error(CV_StsError,"Could not open result file '" + outfile + "'");
				}
			}
			for (vector<string>::iterator infileSmpl = filesSmpl.begin(); infileSmpl != filesSmpl.end(); ++infileSmpl){
				vector<Mat> imgSmpl;
				vector<Mat> maskSmpl;
				if (shiftedfiles){
					string shiftfile;
					patternFileRename(infilesSmpl,shiftfiles,*infileSmpl,shiftfile);
					vector<string> shiftsSmpl;
					patternToFiles(shiftfile,shiftsSmpl);
					// now load virtual files
					for (vector<string>::iterator shiftSmpl = shiftsSmpl.begin(); shiftSmpl != shiftsSmpl.end(); ++shiftSmpl){
						Mat img = imread_mem(*shiftSmpl, CV_LOAD_IMAGE_UNCHANGED);
						CV_Assert(img.data != 0);
						CV_Assert(img.type() == CV_8UC1);
						if (imgSmpl.size() > 0) { CV_Assert(imgSmpl.back().size() == img.size());}
						imgSmpl.push_back(img);
						if (masks){
							string maskfile1, maskfile2;
							patternFileRename(infilesSmpl,masksSmpl,*infileSmpl,maskfile1);
							patternFileRename(infilesSmpl,maskfile1,*infileSmpl,maskfile2,'!');
							Mat msk = imread_mem(maskfile2, CV_LOAD_IMAGE_UNCHANGED);
							CV_Assert(msk.data != 0);
							CV_Assert(msk.type() == CV_8UC1);
							CV_Assert(img.size() == msk.size());
							maskSmpl.push_back(msk);
						}
					}
				}
				else {
					Mat img = imread(*infileSmpl, CV_LOAD_IMAGE_UNCHANGED);
					CV_Assert(img.data != 0);
					CV_Assert(img.type() == CV_8UC1);
					imgSmpl.push_back(img);
					if (masks){
						string maskSmplFile;
						patternFileRename(infilesSmpl,masksSmpl,*infileSmpl,maskSmplFile);
						Mat msk = imread_mem(maskSmplFile, CV_LOAD_IMAGE_UNCHANGED);
						CV_Assert(msk.data != 0);
						CV_Assert(msk.type() == CV_8UC1);
						CV_Assert(msk.size() == img.size());
						maskSmpl.push_back(msk);
					}
				}
				Size codeSize = imgSmpl[0].size();
				unsigned int codeLength = codeSize.height * codeSize.width;
				unsigned int bitStop = min(to,codeLength);
				//CV_Assert(codeLength % sizeof(int) == 0);
				for (vector<string>::iterator infileRef = filesRef.begin(); infileRef != filesRef.end(); ++infileRef, timing.progress++){
					Mat imgRef = imread_mem(*infileRef, CV_LOAD_IMAGE_UNCHANGED);
					CV_Assert(imgRef.data != 0);
					CV_Assert(imgRef.type() == CV_8UC1);
					CV_Assert(imgRef.size() == codeSize);
					Mat maskRef;
					if (masks){
						string maskRefFile;
						patternFileRename(infilesRef,masksRef,*infileRef,maskRefFile);
						maskRef = imread_mem(maskRefFile, CV_LOAD_IMAGE_UNCHANGED);
						CV_Assert(maskRef.data != 0);
						CV_Assert(maskRef.type() == CV_8UC1);
						CV_Assert(maskRef.size() == codeSize);
					}
					else {
						maskRef = Mat();
					}
                    std::pair<double, int> score = (alg == ALG_MINHD) ? (shiftedfiles) ? minHD(imgSmpl,imgRef,from,bitStop, maskSmpl, maskRef) : minHD(imgSmpl[0],imgRef,from,bitStop,minShifts, maxShifts, shiftStep, (maskSmpl.size() > 0) ? maskSmpl[0] : Mat(), maskRef) :
								(alg == ALG_MAXHD) ? (shiftedfiles) ? maxHD(imgSmpl,imgRef,from,bitStop, maskSmpl, maskRef) : maxHD(imgSmpl[0],imgRef,from,bitStop,minShifts, maxShifts, shiftStep, (maskSmpl.size() > 0) ? maskSmpl[0] : Mat(), maskRef) :
								(shiftedfiles) ? ssf(imgSmpl,imgRef,from,bitStop,maskSmpl, maskRef) : ssf(imgSmpl[0],imgRef,from,bitStop,minShifts, maxShifts, shiftStep, (maskSmpl.size() > 0) ? maskSmpl[0] : Mat(), maskRef);
					if (!quiet){
                        if (writebitshift)
                            printf("hd(%s,%s) = %f at %d bits\n",(*infileSmpl).c_str(), (*infileRef).c_str(), score.first, score.second);
                        else
                            printf("hd(%s,%s) = %f\n",(*infileSmpl).c_str(), (*infileRef).c_str(), score.first);
                    }

					if (!outfile.empty() && cfile.is_open()){
						if( outfile_with_path){
                            cfile << *infileSmpl << " " << *infileRef;
                        } else {
    						cfile << skipPath(*infileSmpl) << " " << skipPath(*infileRef);
                        }
                        cfile  << " " << score.first;
                        if( writebitshift) cfile << " " << score.second;
                        cfile << endl;
					}
					if (time && timing.update()) timing.print();
				}
			}
			if (time && quiet) timing.clear();
			if (!outfile.empty() && cfile.is_open()){
				cfile.close();
			}
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
