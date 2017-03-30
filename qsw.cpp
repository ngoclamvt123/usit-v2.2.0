/*
 * qsw.cpp
 *
 * Author: E. Pschernig (epschern@cosy.sbg.ac.at), P. Wild (pwild@cosy.sbg.ac.at)
 *
 * Generates an iris code from iris texture using the QSW algorithm
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

#define QSW_SCALE1 0x011
#define QSW_SCALE2 0x009


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
	printf("| qsw - Iris-code generation (feature extraction) using the QSW algorithm     |\n");
	printf("|                                                                             |\n");
	printf("| Ma, L., Tan, T., Wang, Y., Zhang, D.: Personal identification based on iris |\n");
	printf("| texture analysis. IEEE Trans. Pattern Anal. Machine Intell. 25(12), 1519 –  |\n");
	printf("| 1533 (2003). doi: 10.1109/TPAMI.2003.1251145                                |\n");
	printf("|                                                                             |\n");
	printf("| MODES                                                                       |\n");
	printf("|                                                                             |\n");
    printf("| (# 1) QSW iris code extraction from iris textures                           |\n");
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
    printf("| -m   | inmaskfile | 1 | Y | noise pixelmask (0: noise, 255: noise-free)     |\n");
    printf("|      | outmaskfile|   |   | code bitmask (0: noise, 1: noise-free, off)     |\n");
    printf("| -q   |            | 1 | Y | quiet mode on (off)                             |\n");
    printf("| -t   |            | 1 | Y | time progress on (off)                          |\n");
    printf("| -h   |            | 2 | N | prints usage                                    |\n");
    printf("+------+------------+---+---+-------------------------------------------------+\n");
    printf("|                                                                             |\n");
    printf("| EXAMPLE USAGE                                                               |\n");
    printf("|                                                                             |\n");
    printf("| -i s1.tiff -o s1.png                                                        |\n");
    printf("| -i *.tiff -o ?1.png -q -t                                                   |\n");
    printf("| -i *.tiff -im ?1_mask.png -o ?1_code.png -om ?1_codemask.png -q -t          |\n");
    printf("|                                                                             |\n");
    printf("| AUTHORS                                                                     |\n");
    printf("|                                                                             |\n");
    printf("| Elias Pschernig (epschern@cosy.sbg.ac.at)                                   |\n");
    printf("| Peter Wild (pwild@cosy.sbg.ac.at)                                           |\n");
    printf("|                                                                             |\n");
    printf("| COPYRIGHT                                                                   |\n");
    printf("|                                                                             |\n");
    printf("| (C) 2012 All rights reserved. Do not distribute without written permission. |\n");
    printf("+-----------------------------------------------------------------------------+\n");
}

/** ------------------------------- image processing functions ------------------------------- **/

/*
 * Calculates qsw transform
 *
 * indata: pointer to input data
 * n: number of samples
 * instep: use 1
 * outlow: pointer to low subband
 * lowstep: use 1
 * outhigh: pointer to high subband
 * highstep: use 1
 * scale: scale parameter
 */
void qsw_step_scaled(float const *indata, int n, int instep, float *outlow, int lowstep, float *outhigh, int highstep, int scale)
{
    /* This version has a dyadic scale parameter for the filters. */
    double S = sqrt(2);
    double Hu[] = {0.125 * S, 0.375 * S, 0.375 * S, 0.125 * S};
    double Gu[]= {0.5 * S, -0.5 * S};
    int hn = 4 * (1 << scale);
    int gn = 2 * (1 << scale);
    double H[hn];
    double G[gn];
    for (int i = 0; i < hn; i++)
    {
        H[i] = 0;
    }
    H[0 << scale] = Hu[0];
    H[1 << scale] = Hu[1];
    H[2 << scale] = Hu[2];
    H[3 << scale] = Hu[3];
    for (int i = 0; i < gn; i++)
    {
        G[i] = 0;
    }
    G[0 << scale] = Gu[0];
    G[1 << scale] = Gu[1];
    float const *s = indata;
    /* Highpass filter (with cyclic border handling) */
    int hi = 0;
    for (int i = 0; i < n; i++)
    {
        outhigh[hi] = 0;
        for (int j = 0; j < gn; j++)
        {
            outhigh[hi] += s[instep * ((unsigned)(i + gn / 2 - j) % n)] * G[j];
        }
        hi += highstep;
        /* Lowpass filter (with cyclic border handling) */
    }
    int li = 0;
    for (int i = 0; i < n; i++)
    {
        outlow[li] = 0;
        for (int j = 0; j < hn; j++)
        {
            outlow[li] += s[instep * ((unsigned)(i + hn / 2 - j) % n)] * H[j];
        }
        li += lowstep;
    }
}

/*
 * Performs QSW wavelet transform: Returns an array with twice as many signals as dyadic steps, with
 * alternating (full-length) wavelet transformation results for each scale, always lowpass followed by
 * highpass.
 *
 * dst: pointer to destination array
 * src: pointer to source array
 * n: number of samples (use 512)
 * which: the subband to get. The highest bit tells the depth, the remaining bits select the subband
 *  (e.g. 9, binary 1010, means third decomposition depth, and the remaining 010 means for low, then
 *  high, then again lowpass. Starting with high bits, going left to low bits. Since our signals have
 *  512 samples, we have at most 9 subdivisions.
 */
void qsw(float * dst, float const *src, int n, uint32_t which)
{
    /* int l = ceil(log2(n)); */
    float temp[n];
    uint32_t l = 1;
    while ((1u << l) <= which)
    {
        l++;
    }
    l -= 1;
    which &= (1 << l) - 1;
    float const *p = src;
    int read = 1 << (l - 1);
    for (uint32_t i = 0; i < l; i++)
    {
        float result_lo[n];
        float result_hi[n];
        qsw_step_scaled(p, n, 1, result_lo, 1, result_hi, 1, i);
        if (which & read)
        {
            // High
            memcpy(temp, result_hi, n * sizeof(float));
        }
        else
        {
            // Low
            memcpy(temp, result_lo, n * sizeof(float));
        }
        p = temp;
        read >>= 1;
    }
    memcpy(dst, p, n * sizeof(float));
}

/*
 * List all local extremum points in the given signal of n samples. A
 * threshold using the relative distance is used.
 * s: source signal
 * n: number of samples
 * out: destination signal
 */
int extrema_relative(float *s, int n, int *out)
{
    int first = 1;
    int first_mm = 0;
    int pos[n];
    int minmax = 0;
    int found = 0;
    for (int i = 0; i < n; i++)
    {
        float d1 = s[i] - s[(i - 1 + n) % n];
        float d2 = s[i] - s[(i + 1) % n];
        /* a flat point */
        if (d1 == 0 && d2 == 0)
        {
            continue;
        }
        int mm;
        /* a maximum */
        if (d1 >= 0 && d2 >= 0)
        {
            mm = 1;
            /* a minimum */
        }
        else if (d1 <= 0 && d2 <= 0)
        {
            mm = 0;
        }
        else
        {
            continue;
        }
        if (! first && mm == minmax)
        /* double extremum - we ignore it */
        {
            /* pass */
        }
        else
        {
            if (first)
            {
                first_mm = mm;
            }
            first = 0;
            minmax = mm;
            pos[found] = i;
            found++;
        }
    }
    int mm = !first_mm;
    float threshold = 0.1;
    int j = 1;
    int prev = found - 1;
    for (int i = 0; i < found; i++)
    {
        mm = !mm;
        int next = (i + 1) % found;
        int next2 = (next + 1) % found;
        int p1 = pos[prev];
        int p2 = pos[i];
        int p3 = pos[next];
        int p4 = pos[next2];
        prev = i;
        float dx = p3 - p2;
        if (dx < 0)
        {
            dx += 512;
        }
        float dy = s[p2] - s[p3];
        if (mm)
        {
            if (dy * dx < threshold)
            {
               if (s[p1] < s[p3] && s[p2] < s[p4])
               {
                i++;
                mm = !mm;
                continue;
               }
            }
        }
        else
        {
            if (dy * dx > -threshold)
            {
                if (s[p1] > s[p3] && s[p2] > s[p4])
                {
                    i++;
                    mm = !mm;
                    continue;
                }
            }
        }
        out[j++] = pos[i];
        /* element 0 is 1 for maximum, 0 for minimum */
    }
    out[0] = first_mm;
    return j;
}

/*
 * The QSW feature extraction algorithm
 *
 * code: Code matrix
 * texture: texture matrix
 * M: row height (use 5)
 * N: number of rows (use 10); we usually expect the texture to be sized 512x64, M = 5 and N = 10.
 *  This means we should end up with 10 1-D vectors of 512 samples.
 */
void featureExtract(Mat& code, Mat& codeMask, const Mat& texture, const Mat& textureMask, int M, int N)
{
	CV_Assert(texture.size() == Size(512,64));
	CV_Assert(code.size() == Size(512 * N * 2 / 8,1));
	CV_Assert(textureMask.empty() || codeMask.empty() || codeMask.size() == code.size());
	bool useMask = !textureMask.empty() && !codeMask.empty();
	uchar * codeData = code.data;
	uchar * textureData = texture.data;
	int tOffset = texture.step;
	int w = texture.cols;
	uchar * textureMaskData = textureMask.data;
	int tmOffset = textureMask.step;
	Mat noise;
	uchar * codeMaskData = codeMask.data;
	vector<int>::iterator itMask;
	if (useMask) codeMask.setTo(255);
	Mat s(1,N*w, CV_32FC1);
	float * sData = (float *) s.data;
    for (int x = 0; x < w; x++)
    {
        for (int i = 0; i < N; i++)
        {
            float sum = 0;
            int noisefree = 0;
            for (int j = 0; j < M; j++)
            {
            	int y = (j + i * M);
            	if (!useMask || textureMaskData[ y * tmOffset + x] != 0){
            		sum += textureData[y * tOffset + x];
            		noisefree++;
            	}
            }
            sData[i * w + x] = (noisefree != 0) ? sum / noisefree : 128;
            if (useMask && noisefree == 0) {
            	int idx = (2*i*w + x)/8;
            	int bit = (2*i*w + x) % 8;
            	codeMaskData[idx] &= (0xff ^ 1 << (7-bit));
            	idx = (2*i*w + w + x)/8;
            	bit = (2*i*w + w + x) % 8;
            	//if (bit == 8) { idx++; bit = 0; }
            	codeMaskData[idx] &= (0xff ^ 1 << (7-bit));
            }
            /* normalize the signals from 0..255 to -1..1 range, probably not necessary */
        }
    }
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < w; j++)
        {
        	sData[i * w + j] /= 255;
        	sData[i * w + j] -= 0.5;
        }
    }
    Mat dwts(1,2*N*w,CV_32FC1);
    float * dwtsData = (float *)dwts.data;
	for (int i = 0; i < N; i++)
	{
		int which;
		which = QSW_SCALE1;
		qsw(dwtsData + i*w*2,sData + i * w, w, which);
		which = QSW_SCALE2;
		qsw(dwtsData + i*w*2 + w,sData + i * w, w, which);
	}
	vector<int> result;
	int pos[w];
	for (int i = 0; i < N; i++)
	{
		int n = extrema_relative(dwtsData+i*w*2, w, pos);
		result.push_back(n);
		result.push_back(pos[0]);
		int prev = 0;
		for (int j = 1; j < n; j++)
		{
			int d = pos[j] - prev;
		    CV_Assert(d >= 0);
		    result.push_back(d);
		    prev = pos[j];
		}
		n = extrema_relative(dwtsData+i*w*2+w, w, pos);
		result.push_back(n);
		result.push_back(pos[0]);
		prev = 0;
		for (int j = 1; j < n; j++)
		{
			int d = pos[j] - prev;
			CV_Assert(d >= 0);
			result.push_back(d);
			prev = pos[j];
		}
	}
	/*
	 * Bits are added to the code starting to the left and with the lowest bit in
	 * a byte. */
	int bitpos = 0;
	for (vector<int>::iterator it = result.begin(); it < result.end();) {
		int sl = *it;
		it++; /* length of minmax string */
		int minmax = *it;
		it++; /* is the first extremum a minimum (0) or maximum (1)? */
		minmax = !minmax;
		int x = 0, ex = 0;
		for (int k = 1; k < sl; k++)
		{
			int offset = 0;
			offset = *it;
			it++;
			ex += offset;
			while (x < ex)
			{
				if (minmax) *codeData |= 1 << (7-bitpos); bitpos++; if (bitpos == 8) { codeData++; bitpos = 0; }
				x++;
			}
			minmax = !minmax;
		}
		while (x < 512)
		{
			if (minmax) *codeData |= 1 << (7-bitpos); bitpos++; if (bitpos == 8) { codeData++; bitpos = 0; }
			x++;
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
			cmdCheckOpts(cmd,"-i|-o|-m|-q|-t");
			cmdCheckOptExists(cmd,"-i");
			cmdCheckOptSize(cmd,"-i",1);
			string inFiles = cmdGetPar(cmd,"-i");
			cmdCheckOptExists(cmd,"-o");
			cmdCheckOptSize(cmd,"-o",1);
			string outFiles = cmdGetPar(cmd,"-o");
			string imaskFiles, omaskFiles;
			if (cmdGetOpt(cmd,"-m") != 0){
				cmdCheckOptSize(cmd,"-m",2);
				imaskFiles = cmdGetPar(cmd,"-m", 0);
				omaskFiles = cmdGetPar(cmd,"-m", 1);
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
				Mat mask, codeMask;
				if (!imaskFiles.empty()) {
					string imaskfile;
					patternFileRename(inFiles,imaskFiles,*inFile,imaskfile);
					if (!quiet) printf("Loading mask image '%s' ...\n", imaskfile.c_str());;
					mask = imread(imaskfile, CV_LOAD_IMAGE_GRAYSCALE);
					CV_Assert(img.data != 0);
				}
				Mat out;
				if (!quiet) printf("Creating iris-code ...\n");
				CV_Assert(img.size() == Size(512,64));
				int m = 5, n = 10;
				Mat code (1,512*n*2/8,CV_8UC1);

				if (!imaskFiles.empty()) {
					codeMask.create(1,512*n*2/8,CV_8UC1);
					codeMask.setTo(255);
				}
				code.setTo(0);
				featureExtract(code, codeMask, img, mask, m, n);
				out = code;
				string outfile;
				patternFileRename(inFiles,outFiles,*inFile,outfile);
				if (!quiet) printf("Storing code '%s' ...\n", outfile.c_str());
				if (!imwrite(outfile,out)) CV_Error(CV_StsError,"Could not save image '" + outfile + "'");
				if (!imaskFiles.empty()) {
					string omaskfile;
					patternFileRename(inFiles,omaskFiles,*inFile,omaskfile);
					if (!quiet) printf("Storing code-mask '%s' ...\n", omaskfile.c_str());
					if (!imwrite(omaskfile,codeMask)) CV_Error(CV_StsError,"Could not save image '" + omaskfile + "'");
				}
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
