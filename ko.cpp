/*
 * ko.cpp
 *
 * Author: E. Pschernig (epschern@cosy.sbg.ac.at), P. Wild (pwild@cosy.sbg.ac.at)
 *
 * Generates an iris code from iris texture using the Ko algorithm
 *
 */
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
#include "version.h"

using namespace std;
using namespace cv;

#define KO_CELLHEIGHT 3
#define KO_CELLWIDTH 10
#define KO_GROUPSIZE 4

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
	printf("| ko - Iris-code generation (feature extraction) using the KO algorithm       |\n");
	printf("|                                                                             |\n");
	printf("| Ko, J.G., Gil, Y.H., Yoo, J.H., Chung, K.I.: A novel and efficient feature  |\n");
	printf("| extraction method for iris recognition. ETRI Journal 29(3), 399–401 (2007)  |\n");
	printf("|                                                                             |\n");
	printf("| MODES                                                                       |\n");
	printf("|                                                                             |\n");
    printf("| (# 1) KO iris code extraction from iris textures                            |\n");
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
    printf("| -y   |            | 1 | N | Use texture size of 512xy (default y=64)        |\n");
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
    printf("| Elias Pschernig (epschern@cosy.sbg.ac.at)                                   |\n");
    printf("| Peter Wild (pwild@cosy.sbg.ac.at)                                           |\n");
    printf("|                                                                             |\n");
    printf("| COPYRIGHT                                                                   |\n");
    printf("|                                                                             |\n");
    printf("| (C) 2012 All rights reserved. Do not distribute without written permission. |\n");
    printf("+-----------------------------------------------------------------------------+\n");
}

/** ------------------------------- image processing functions ------------------------------- **/

void setBitTo1(uchar* code, int bitpos){
	code[bitpos / 8] |= (1 << (bitpos % 8));
}

void setBitTo0(uchar* code, int bitpos){
	code[bitpos / 8] &= (0xff ^ 1 << (bitpos % 8));
}

/* calculate the iris code of given cumulative sums */
void calculate_iris_code (Mat& sums, Mat& code, int w, int h)
{
    int i, j, k;
    int min, max, min_pos, max_pos;
    int up = 0;
    int down = 0;

    int code_size = w * h * 4;
    uchar* iris_code = code.data;
    int * cum_sums = (int *)sums.data;

    for (i = 0; i < code_size / 2; i += KO_GROUPSIZE)
    {
        /* the initial values and positions */
        max = cum_sums[i];
        min = cum_sums[i];
        max_pos = 0;
        min_pos = 0;

        /* fist calculate the position of the maximal and minimal value */
        for (j = i; j < i + KO_GROUPSIZE; j++)
        {
            if ( cum_sums[j] < min )
            {
                min = cum_sums[j];
                min_pos = j - i;
            }
            if ( cum_sums[j] >= max )
            {
                max = cum_sums[j];
                max_pos = j - i;
            }
        }
        /* check wether there is a up- of a downward slope */
        if (min_pos <= max_pos)
        {
            for (k = min_pos + i; k < max_pos + i - 1; k++)
            {
                if (cum_sums[k] > cum_sums[k + 1])
                {
                    up = 0;
                    break;
                }
                else
                {
                    up = 1;
                }
            }
        }
        else if (min_pos > max_pos)
        {
            for (k = max_pos + i; k < min_pos + i - 1; k++)
            {
                if (cum_sums[k] < cum_sums[k + 1])
                {
                    down = 0;
                    break;
                }
                else
                {
                    down = 1;
                }
            }
        }
        if (abs(min_pos - max_pos) < 2 || up == down)
        {
            up = 0;
            down = 0;
        }
        /* now generate the iris code */
        for (j = i; j < i + KO_GROUPSIZE; j++)
        {
            if ((up == 0) && (down == 0))
            {
            	setBitTo0(iris_code,j);
            	setBitTo0(iris_code,j + code_size / 2);
            }
            /* check wether values are outside the borders */
            else if ((j - i) < MIN(min_pos, max_pos) || (j - i) > MAX(min_pos, max_pos))
            {
            	setBitTo0(iris_code,j);
            	setBitTo0(iris_code,j + code_size / 2);
            }

            else
            {
                /* check wether it is the last value or the right border  and take the previous value are reference */
                if ((j - i) == MAX(min_pos, max_pos))
                {
                    /* greater value: assign '1' */
                    if ((cum_sums[j] >= cum_sums[j - 1]) && (up == 1))
                    {
                    	setBitTo1(iris_code,j);
                    	setBitTo0(iris_code,j + code_size / 2);
                    }
                    /* smaller value: assign '2' */
                    else if ((cum_sums[j] < cum_sums[j - 1]) && (down == 1))
                    {
                    	setBitTo0(iris_code,j);
                    	setBitTo1(iris_code,j + code_size / 2);
                    }
                    else
                    {
                    	setBitTo0(iris_code,j);
                    	setBitTo0(iris_code,j + code_size / 2);
                    }
                }
                else
                {
                    /* the common case, where just the next value is checked to apply the code */
                    /* greater value: assign '1' */
                    /* same value: assign this value */
                    if ( cum_sums[j] == cum_sums[j + 1] )
                    {
                        if (up == 1)
                        {
                        	setBitTo1(iris_code,j);
                        	setBitTo0(iris_code,j + code_size / 2);
                        }
                        else if (down == 1)
                        {
                        	setBitTo0(iris_code,j);
                        	setBitTo1(iris_code,j + code_size / 2);
                        }
                        else
                        {
                        	setBitTo0(iris_code,j);
                        	setBitTo0(iris_code,j + code_size / 2);
                        }
                    }
                    else if (up == 1)
                    {
                    	setBitTo1(iris_code,j);
                    	setBitTo0(iris_code,j + code_size / 2);
                    }
                    /* smaller value: assign '2' */
                    else if (down == 1)
                    {
                    	setBitTo0(iris_code,j);
                    	setBitTo1(iris_code,j + code_size / 2);
                    }
                }
            }
        }
    }
}

/*
 * The Ko feature extraction algorithm
 *
 * code: Code matrix
 * texture: texture matrix
 */
void featureExtract(Mat& code, const Mat& texture)
{
	//CV_Assert(code.size() == Size(512 * N * 2 / 8,1));
	int w = (texture.cols / (KO_CELLWIDTH * KO_GROUPSIZE)) * KO_GROUPSIZE;
	int h = (texture.rows / (KO_CELLHEIGHT * KO_GROUPSIZE)) * KO_GROUPSIZE;
	Mat means(1,h*w,CV_32FC1);
	int * meansData = (int *) means.data;
	uchar * textureData = texture.data;
	int textureStep = texture.step;
	// original code
	int mean_idx = 0;
	for (int i = 0; i < texture.rows; i += KO_CELLHEIGHT)
	{
		for (int j = 0; j < texture.cols; j += KO_CELLWIDTH)
		{
			int curr_mean = 0;
			/* now process one cell of given width and height */
			for (int k = j; k < j + KO_CELLWIDTH; k++)
			{
				for (int l = 0; l < KO_CELLHEIGHT; l++)
				{
					curr_mean += textureData[k + (l + i) * textureStep];
				}
			}
			curr_mean /= KO_CELLWIDTH * KO_CELLHEIGHT;
			if (mean_idx >= (int)(w * h))
			{
				break;

			}
			meansData[mean_idx] = curr_mean;
			mean_idx++;
		}
	}
	// better use this code
//  int pOffset = textureStep - KO_CELLWIDTH;
//	for (int i = 0; i < h; i++)
//	{
//		for (int j = 0; j < w; j++)
//		{
//			//printf("i: %i j: %i",i*KO_CELLHEIGHT,j*KO_CELLWIDTH);
//			int sum = 0;
//			/* now process one cell of given width and height */
//			uchar * p = textureData + i*KO_CELLHEIGHT*textureStep+j*KO_CELLWIDTH;
//			for (int k = 0; k < KO_CELLHEIGHT; k++, p += pOffset)
//			{
//				for (int l = 0; l < KO_CELLWIDTH; l++, p++)
//				{
//					sum += *p;
//				}
//			}
//			*meansData = sum / (KO_CELLWIDTH * KO_CELLHEIGHT);
//			//printf("INSERT: %i\n",*meansData);
//			meansData++;
//			//printf("Sum: %i\n",sum);
//		}
//	}
	// horizontal group sums
	Mat sums(1,w*h*2,CV_32SC1);
	int* sumsData = (int *) sums.data;
	meansData = (int *) means.data;
	int start = 0;
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j+=KO_GROUPSIZE)
		{
			int mean = 0;
			/* calculate the mean of all gryscale values */
			for (int k = j; k < j + KO_GROUPSIZE; k++)
			{
				mean += meansData[k + i * w];
			}
			mean /= KO_GROUPSIZE;
			sumsData[start] = meansData[start] - mean;
			for (int k = start + 1; k < start + KO_GROUPSIZE; k++)
			{
				sumsData[k] = sumsData[k - 1];
				sumsData[k] += meansData[k] - mean;
			}
			start += KO_GROUPSIZE;
		}
	}
	// vertical group sums
	start = w*h;
	meansData = (int *) means.data;
	for (int i = 0; i < h; i+=KO_GROUPSIZE)
	{
		for (int j = 0; j < w; j++)
		{
			int mean = 0;
			/* calculate the mean of all gryscale values */
			for (int k = j; k < j + KO_GROUPSIZE; k++)
			{
				//printf("index: %i\n",j + (i + k - j) * w);
				mean += meansData[j + (i + k - j) * w];
			}
			mean /= KO_GROUPSIZE;
			sumsData[start] = meansData[j + i * w] - mean;
			for (int k = start + 1; k < start + KO_GROUPSIZE; k++)
			{
				sumsData[k] = sumsData[k - 1];
				sumsData[k] += meansData[j + (i + k - start) * w] - mean;
			}
			start += KO_GROUPSIZE;

			//printf("i: %i j: %i",i,j);
			//printf("Sum: %i\n",mean);
		}
	}
	calculate_iris_code(sums,code,w,h);
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
			cmdCheckOpts(cmd,"-i|-o|-m|-q|-t|-y");
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
            int YSIZE = 64;
			if (cmdGetOpt(cmd,"-y") != 0){
				cmdCheckOptSize(cmd,"-y",1);
                YSIZE = cmdGetParInt(cmd,"-y",0);
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
				if (img.rows != YSIZE || img.cols != 512)
				{
				 printf("Input texture has to be of size 512 x %d.\n", YSIZE);
				 exit(EXIT_FAILURE);
				}
				int w = (img.cols / (KO_CELLWIDTH * KO_GROUPSIZE)) * KO_GROUPSIZE;
				int h = (img.rows / (KO_CELLHEIGHT * KO_GROUPSIZE)) * KO_GROUPSIZE;
				if (!quiet) printf("Creating %d x %d iris-code ...\n", w, h);
				Mat code (1,w*h*4/8,CV_8UC1);
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
