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
#include <chrono>

using namespace std;
using namespace boost::filesystem;
using namespace cv;

typedef struct IrisCode
{
  string name;
  unsigned char **data;
  int stepsize;
} IrisCode;

static const int CODE_WIDTH = 512;
static const int CODE_HEIGHT = 20;

static const int MIN_SHIFT = -16;
static const int MAX_SHIFT = 16;

static const int htlut[256] = {0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8};


static const int ALG_MINHD = 0, ALG_3A = 1, ALG_3AL = 2, ALG_3AS = 3;
static const int MODE_MAIN = 1, MODE_HELP = 2;
enum MAIN_MODE { STATIC, DYNAMIC};

void printUsage() {
	printf("+-----------------------------------------------------------------------------+\n");
	printf("| TripleA - calculates Hamming distance of iris-codes                         |\n");
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
	printf("| -f   | folder     | 1 | N | folder of iris-codes generated using LG or QSW  |\n");
	printf("| -o   | file       | 1 | N | output file for the results                     |\n");
	printf("| -s   | step-size  | 1 | Y | step-size used in static mode (5)               |\n");
	printf("| -c   | constant   | 1 | Y | constant used in dynamic mode (1)               |\n");
	printf("| -a   | algorithm  | 1 | Y | TripleA algorithm (3a) (default)                |\n"); 
	printf("|      |            |   |   | TripleA-limited algorithm (3al)                 |\n");
	printf("|      |            |   |   | TripleA-single-sided algorithm (3as)            |\n");
	printf("| -h   |            | 2 | N | prints usage                                    |\n");
	printf("+------+------------+---+---+-------------------------------------------------+\n");
	printf("|                                                                             |\n");
	printf("| The default is to run the program in dynamic mode. When a step size (-s) is |\n");
	printf("| set the program automatically switches to static-mode.                      |\n");
	printf("|                                                                             |\n");
	printf("| EXAMPLE USAGE                                                               |\n");
    printf("|                                                                             |\n");
    printf("| -f codes -s 4 -c 0.33 -a 3a -o compare.txt                                  |\n");
    printf("|                                                                             |\n");
	printf("| AUTHOR                                                                      |\n");
	printf("|                                                                             |\n");
	printf("| Christian Rathgeb                                                           |\n");
	printf("|                                                                             |\n");
	printf("| COPYRIGHT                                                                   |\n");
	printf("|                                                                             |\n");
	printf("| (C) 2016 All rights reserved. Do not distribute without written permission. |\n");
	printf("+-----------------------------------------------------------------------------+\n");
}

// estimate the Hamming distance between two given codes using the TripleA algorithm
void TripleA(IrisCode code1, IrisCode code2, int start, int stop, ofstream& log, int stepsize){
	
	float mindist = CODE_HEIGHT*CODE_WIDTH;
	unsigned int dist;
	int pos_min = 0;
	
	int w = CODE_WIDTH;
	int h;
	int border;
	
	if (CODE_HEIGHT%8 == 0) h = CODE_HEIGHT/8;
	else h = CODE_HEIGHT/8 + 1;

	auto start_time = chrono::high_resolution_clock::now();
	
	// alignmnet with step-size
	border = stop%stepsize;
	if (border != 0){
		int k=start;
		dist = 0;
		for (int i=0; i < h; i++){
			for (int j=0; j < w; j++){
				dist+= htlut[code1.data[j][i] ^ code2.data[(j+k+CODE_WIDTH)%CODE_WIDTH][i]];
			}
		}
		if (dist < mindist){
			mindist = dist;
			pos_min = k;
		}
		start+=border;
	}
	
	for (int k = start; k <= stop; k+=stepsize){
		dist = 0;
		for (int i=0; i < h; i++){
			for (int j=0; j < w; j++){
				dist+= htlut[code1.data[j][i] ^ code2.data[(j+k+CODE_WIDTH)%CODE_WIDTH][i]];
			}
		}
		if (dist < mindist){
			mindist = dist;
			pos_min = k;
		}
	}

	if (border != 0){
		int k=stop;
		dist = 0;
		for (int i=0; i < h; i++){
			for (int j=0; j < w; j++){
				dist+= htlut[code1.data[j][i] ^ code2.data[(j+k+CODE_WIDTH)%CODE_WIDTH][i]];
			}
		}
		if (dist < mindist){
			mindist = dist;
			pos_min = k;
		}
	}
	
	start = max(pos_min - stepsize + 1, start); 
	stop = min(pos_min + stepsize - 1, stop);
	
	for (int k = start; k <= stop; k++){
		dist = 0;
		for (int i=0; i < h; i++){
			for (int j=0; j < w; j++){
				dist+= htlut[code1.data[j][i] ^ code2.data[(j+k+CODE_WIDTH)%CODE_WIDTH][i]];
			}
		}
		if (dist < mindist){
			mindist = dist;
			pos_min = k;
		}
	}

	auto end_time = chrono::high_resolution_clock::now();
	
	log << "hd(" << code1.name << "," << code2.name << "):" <<  mindist/(CODE_HEIGHT*CODE_WIDTH) << "\t time: " << chrono::duration_cast<chrono::microseconds>(end_time - start_time).count() << " micro seconds \t step-size: " << stepsize << endl; 
	log.flush();
}

// estimate the Hamming distance between two given codes using the TripleA-limited algorithm
void TripleA_limited(IrisCode code1, IrisCode code2, int start, int stop, ofstream& log, int stepsize){
	
	float mindist = CODE_HEIGHT*CODE_WIDTH;
	unsigned int dist;
	int pos_min = 0;
	
	int w = CODE_WIDTH;
	int h;
	int border;
	
	if (CODE_HEIGHT%8 == 0) h = CODE_HEIGHT/8;
	else h = CODE_HEIGHT/8 + 1;

	auto start_time = chrono::high_resolution_clock::now();
	
	// limited alignment with step-size
	border = stop%stepsize;
	if (border != 0){
		int k=start;
		dist = 0;
		for (int j=0; j < w; j++){
			dist+= htlut[code1.data[j][0] ^ code2.data[(j+k+CODE_WIDTH)%CODE_WIDTH][0]];
		}
		if (dist < mindist){
			mindist = dist;
			pos_min = k;
		}
		start+=border;
	}
	
	for (int k = start; k <= stop; k+=stepsize){
		dist = 0;
		for (int j=0; j < w; j++){
			dist+= htlut[code1.data[j][0] ^ code2.data[(j+k+CODE_WIDTH)%CODE_WIDTH][0]];
		}
		if (dist < mindist){
			mindist = dist;
			pos_min = k;
		}
	}

	if (border != 0){
		int k=stop;
		dist = 0;
		for (int j=0; j < w; j++){
			dist+= htlut[code1.data[j][0] ^ code2.data[(j+k+CODE_WIDTH)%CODE_WIDTH][0]];
		}
		if (dist < mindist){
			mindist = dist;
			pos_min = k;
		}
	}
	mindist = CODE_HEIGHT*CODE_WIDTH;
	
	start = max(pos_min - stepsize + 1, start); // note that we already tested the boundaries
	stop = min(pos_min + stepsize - 1, stop);
	
	for (int k = start; k <= stop; k++){
		dist = 0;
		for (int i=0; i < h; i++){
			for (int j=0; j < w; j++){
				dist+= htlut[code1.data[j][i] ^ code2.data[(j+k+CODE_WIDTH)%CODE_WIDTH][i]];
			}
		}
		if (dist < mindist){
			mindist = dist;
			pos_min = k;
		}
	}

	auto end_time = chrono::high_resolution_clock::now();
	
	log << "hd(" << code1.name << "," << code2.name << "):" <<  mindist/(CODE_HEIGHT*CODE_WIDTH) << "\t time: " << chrono::duration_cast<chrono::microseconds>(end_time - start_time).count() << " micro seconds \t step-size: " << stepsize << endl;
	log.flush();
}

// estimate the Hamming distance between two given codes using the TripleA-single-sided algorithm
void TripleA_singlesided(IrisCode code1, IrisCode code2, int start, int stop, ofstream& log, int stepsize){
	
	float mindist = CODE_HEIGHT*CODE_WIDTH;
	unsigned int dist;
	int pos_min = 0;
	int idx_min = 0;
	
	int w = CODE_WIDTH;
	int h;
	int border;
	int off = 0;
	
	if (CODE_HEIGHT%8 == 0) h = CODE_HEIGHT/8;
	else h = CODE_HEIGHT/8 + 1;
	
	border = stop%stepsize;
	if (border > 0) off=2;
	
	int cnt=0;
	int scores[(abs(start)+stop)/stepsize + 1 + off];
	
	auto start_time = chrono::high_resolution_clock::now();
	
	if (border != 0){
		int k=start;
		dist = 0;
		for (int j=0; j < w; j++){
			dist+= htlut[code1.data[j][0] ^ code2.data[(j+k+CODE_WIDTH)%CODE_WIDTH][0]];
		}
		scores[cnt] = dist;
		if (dist < mindist){
			mindist = dist;
			pos_min = k;
			idx_min = cnt;
		}
		cnt++;
		start+=border;
	}
	
	for (int k = start; k <= stop; k+=stepsize){
		dist = 0;

		for (int j=0; j < w; j++){
			dist+= htlut[code1.data[j][0] ^ code2.data[(j+k+CODE_WIDTH)%CODE_WIDTH][0]];
		}
		scores[cnt] = dist;
		if (dist < mindist){
			mindist = dist;
			pos_min = k;
			idx_min = cnt;
		}
		cnt++;
	}

	if (border != 0){
		int k=stop;
		dist = 0;
		for (int j=0; j < w; j++){
			dist+= htlut[code1.data[j][0] ^ code2.data[(j+k+CODE_WIDTH)%CODE_WIDTH][0]];
		}
		scores[cnt] = dist;
		if (dist < mindist){
			mindist = dist;
			pos_min = k;
			idx_min = cnt;
		}
	}
	
	mindist = CODE_HEIGHT*CODE_WIDTH;
	if (idx_min == 0){
		start = pos_min;
		stop = pos_min + stepsize - 1;
	}
	else if (idx_min == (abs(start)+stop)/stepsize + off){
		start = pos_min - stepsize + 1;
		stop = pos_min;
	}
	else if (scores[idx_min +1] > scores[idx_min -1]){
		start = pos_min - stepsize + 1;
		stop = pos_min;
	}
	else{
		start = pos_min;
		stop = pos_min + stepsize - 1;
	}
	
	for (int k = start; k <= stop; k++){
		dist = 0;
		for (int i=0; i < h; i++){
			for (int j=0; j < w; j++){
				dist+= htlut[code1.data[j][i] ^ code2.data[(j+k+CODE_WIDTH)%CODE_WIDTH][i]];
			}
		}
		if (dist < mindist){
			mindist = dist;
			pos_min = k;
		}
	}
	auto end_time = chrono::high_resolution_clock::now();
	log << "hd(" << code1.name << "," << code2.name << "):" <<  mindist/(CODE_HEIGHT*CODE_WIDTH) << "\t time: " << chrono::duration_cast<chrono::microseconds>(end_time - start_time).count() << " micro seconds \t step-size: " << stepsize << endl;
	log.flush();
}

// read an iris-code form a given file name, extract the (dynamic) step size
// and map it to a more efficient representation
IrisCode getCode(string filename, float k)
{
	IrisCode iriscode;
	Mat img = imread(filename, CV_LOAD_IMAGE_UNCHANGED);
	CV_Assert(img.data != 0);
	CV_Assert(img.type() == CV_8UC1);

	int w = CODE_WIDTH;
	int h;

	iriscode.name=filename;
	
	if (CODE_HEIGHT%8 == 0) h = CODE_HEIGHT/8;
	else h = CODE_HEIGHT/8 + 1;

	unsigned char c;
	unsigned char bincode[CODE_WIDTH][CODE_HEIGHT];
	
	// initialize with 0s
	for (int i=0; i < CODE_HEIGHT; i++){
		for (int j=0; j < CODE_WIDTH; j++){
			bincode[j][i] = 0;
		}
	}
	
	// now map the given codes to 0/1 representation
	for (int i=0; i < CODE_HEIGHT; i++){
		for (int j=0; j < CODE_WIDTH; j++){
			c=pow(2,7-(j%8));
			bincode[j][i]=htlut[img.data[j/8+i*CODE_WIDTH/8]&c];
		}
	}
	
	float mu = 0;
	int length;
	int count=0;
	for (int i=0; i < CODE_HEIGHT; i++){
		length = 1;
		for (int j=1; j < CODE_WIDTH; j++){
			if (bincode[j][i] == bincode[j-1][i]) length++;
			else {
				mu+=length;
				length=1;
				count++;
			}
		}
		mu+=length;
		length=1;
		count++;
	}
	mu/=count;
	
	// assign the average dynamic step-size according to the given constant k
	iriscode.stepsize = max((int)(mu*k),1);
	
	unsigned char **code;
	code = (unsigned char **)malloc(w * sizeof(unsigned char *)); 
	for (int i = 0; i < w; i++) code[i] = (unsigned char *)malloc(h * sizeof(unsigned char));
	
	// initialize with 0s
	for (int i=0; i < h; i++){
		for (int j=0; j < w; j++){
			code[j][i] = 0;
		}
	}
	
	// now map the given codes to a more efficient representation
	// create bytes out of 8 adjacent row-bits
	for (int i=0; i < h; i++){
		for (int j=0; j < w; j++){
			c=pow(2,7-(j%8));
			for (int k = 0; k < 8; k++){
				if (i*8+k >=CODE_HEIGHT) break;
				code[j][i]+=htlut[img.data[j/8+i*w+k*w/8]&c] * pow(2,7-k);
			}
		}
	}

	iriscode.data=code;
	return iriscode;
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
	std::vector<IrisCode> iriscodes;
	
	int mode = MODE_HELP;
	map<string,vector<string> > cmd;
	try {
		cmdRead(cmd,argc,argv);
		if (cmd.size() == 0 || cmdGetOpt(cmd,"-h") != 0) mode = MODE_HELP;
		else mode = MODE_MAIN;
		if (mode == MODE_MAIN){
			// validate command line
			cmdCheckOpts(cmd,"-f|-s|-a|-c|-o|-h");
			string folder;
			if (cmdGetOpt(cmd,"-f") != 0){
				cmdCheckOptSize(cmd,"-f",1);
				folder = cmdGetPar(cmd,"-f");
			}
			int alg = ALG_3A;
			if (cmdGetOpt(cmd,"-a") != 0){
				cmdCheckOptSize(cmd,"-a",1);
				string algo = cmdGetPar(cmd,"-a");
				if (algo == "3al"){
					alg = ALG_3AL;
				}
				if (algo == "3as"){
					alg = ALG_3AS;
				}
			}
            MAIN_MODE main_mode = DYNAMIC;
            string outfilename = "dynamic.txt";			
			// parameter for static mode
			int s = 5;
			if (cmdGetOpt(cmd,"-s") != 0){
				cmdCheckOptSize(cmd,"-s",1);
				s = cmdGetParInt(cmd,"-s");               
                main_mode =  STATIC;
                outfilename="static.txt";
            }
			float c = 0.33;
			if (cmdGetOpt(cmd,"-c") != 0){
                cmdCheckOptSize(cmd,"-c",1);
                c = cmdGetParFloat(cmd,"-c");
                main_mode = DYNAMIC;
                outfilename="dynamic.txt";
            }

            if (cmdGetOpt(cmd,"-o") != 0){
                
                cmdCheckOptSize(cmd,"-o",1);
                outfilename = cmdGetPar(cmd,"-o");
            }

			// starting routine
			path p(folder);
			cout << "loading iriscodes..." << endl;
			for (auto i = directory_iterator(p); i != directory_iterator(); i++){
				if (!is_directory(i->path())){
					iriscodes.push_back(getCode(i->path().string(), c));
				}
				else continue;
			}
			cout << "loaded " << (int)iriscodes.size() << " iriscodes" << endl;
				
			ofstream log;
            log.open(outfilename, std::ios_base::app);                    
            cout << "output written to " << outfilename << endl;
            
            if (  main_mode == STATIC){
                // evaluation with static step size
                cout << "evaluation with static step size of " << s << " bit" << endl;
                for (int i=0; i<(int)iriscodes.size()-1; i++){
                    for (int j=i+1; j<(int)iriscodes.size(); j++){
                        if (i!=j) {
                            if (alg == ALG_3A){
                                TripleA(iriscodes.at(i), iriscodes.at(j), MIN_SHIFT, MAX_SHIFT, log, s);
                            }
                            else if (alg == ALG_3AL){
                                TripleA_limited(iriscodes.at(i), iriscodes.at(j), MIN_SHIFT, MAX_SHIFT, log, s);
                            }
                            else {
                                TripleA_singlesided(iriscodes.at(i), iriscodes.at(j), MIN_SHIFT, MAX_SHIFT, log, s);
                            }
                        }
                    }
                }
            } else if ( main_mode == DYNAMIC){
                // evaluation with dynamic step size
                cout << "evaluation with dynamic step size and constant of " << c << " bit" << endl;
                for (int i=0; i<(int)iriscodes.size()-1; i++){
                    for (int j=i+1; j<(int)iriscodes.size(); j++){
                        if (i!=j) {
                            if (alg == ALG_3A){
                                TripleA(iriscodes.at(i), iriscodes.at(j), MIN_SHIFT, MAX_SHIFT, log, iriscodes.at(i).stepsize);
                            }
                            else if (alg == ALG_3AL){
                                TripleA_limited(iriscodes.at(i), iriscodes.at(j), MIN_SHIFT, MAX_SHIFT, log, iriscodes.at(i).stepsize);
                            }
                            else {
                                TripleA_singlesided(iriscodes.at(i), iriscodes.at(j), MIN_SHIFT, MAX_SHIFT, log, iriscodes.at(i).stepsize);
                            }
                        }
                    }
                }
            }
			log.close();
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
