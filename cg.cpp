/*
 * cg.cpp
 *
 * Author: H. Hofbauer (hhofbaue@cosy.sbg.ac.at), E. Pschernig (epschern@cosy.sbg.ac.at), P. Wild (pwild@cosy.sbg.ac.at)
 *
 * Generates an iris code from iris texture using complex gabor filters with different filter size and wavelength.
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
#include <boost/lexical_cast.hpp>
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

int FILTER_HEIGHT=31;

/*
 * Print command line usage for this program
 */
void printUsage() {
    printVersion();
	printf("+-----------------------------------------------------------------------------+\n");
	printf("| cg - Iris-code generation (feature extraction) using complex gabor filters  |\n");
	printf("|      with varying wavelengths.                                              |\n");
	printf("|                                                                             |\n");
	printf("| John Daugman. The importance of being random: statistical principles of     |\n");
	printf("| iris recognition. Pattern Recognition 36(2003) 279--291.                    |\n");
	printf("|                                                                             |\n");
	printf("| MODES                                                                       |\n");
	printf("|                                                                             |\n");
    printf("| (# 1) cg iris code extraction from iris textures                            |\n");
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
//    printf("| -m   | inmaskfile | 1 | Y | noise pixelmask (0: noise, 255: noise-free)     |\n");
//    printf("|      | outmaskfile|   |   | code bitmask (0: noise, 1: noise-free, off)     |\n");
    printf("| -q   |            | 1 | Y | quiet mode on (off)                             |\n");
    printf("| -t   |            | 1 | Y | time progress on (off)                          |\n");
    printf("| -h   |            | 2 | N | prints usage                                    |\n");
    printf("|      |            |   |   |                                                 |\n");
    printf("| -pwl | wavelength | 1 | N | Base wavelength of the gabor filter in pixel.   |\n");
    printf("|      |            |   |   | Size of base filter will then be 2*wl+1.        |\n");
    printf("|      |            |   |   | (default 6)                                     |\n");
    printf("| -pbp | borderpower| 1 | N | Remaining power of the gauss-part of the filter |\n");
    printf("|      |            |   |   | relative to the center (default 0.01).          |\n");
    printf("| -psx | x-samples  | 1 | N | Number of horizontal samples (default 256)      |\n");
    printf("| -x   | width      | 1 | N | Width of input texture (default 512).           |\n");
    printf("| -y   | height     | 1 | N | Height of input texture (default 64).           |\n");
    printf("|      |            |   |   |                                                 |\n");
    printf("| -wf  |            | 1 | N | Write filterbank images                         |\n");
    printf("| -ws  |            | 1 | N | Write image sample with filtered versions       |\n");
    printf("| -wp  |            | 1 | N | Write iris image with extraction points         |\n");
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
    printf("| Heinz Hofbauer (hhofbaue@cosy.sbg.ac.at)                                    |\n");
    printf("| Elias Pschernig (epschern@cosy.sbg.ac.at)                                   |\n");
    printf("| Peter Wild (pwild@cosy.sbg.ac.at)                                           |\n");
    printf("|                                                                             |\n");
    printf("| COPYRIGHT                                                                   |\n");
    printf("|                                                                             |\n");
    printf("| (C) 2012 All rights reserved. Do not distribute without written permission. |\n");
    printf("+-----------------------------------------------------------------------------+\n");
}


/** --------------- writerstuff for debugging/control --------------- **/

void writeNormMatToFile(const Mat &what, const string &where, const string &txtmessage = "", bool write_sign=false){
    if( write_sign){
        Mat img = ( what >= 0);
        imwrite( where, img);
    } else {
        Mat img(what.rows, what.cols, CV_8U);
        normalize( what, img, 0, 255, NORM_MINMAX);
        imwrite( where, img);
    }
    if( !txtmessage.empty()) cout << txtmessage << " file("<<where<<")"<<endl;
}

void writeExtractionsequence( const Mat &img, const vector<Point> &extractionsequence, bool &writeagain){
    Mat pointoverlay = img.clone();
    for( vector<Point>::const_iterator eit = extractionsequence.begin(); eit != extractionsequence.end(); advance(eit, 1) )
        circle( pointoverlay, *eit, 1, Scalar(255,128,128));
    writeagain = false;
    imwrite( "extractionpointlocation.png", pointoverlay);
    cout << "Extraction point locations written to: extractionpointlocation.png"<<endl;
}
void writeFilterbank( const vector<Mat> &filterbank){
    for( vector<Mat>::const_iterator it = filterbank.begin(); it != filterbank.end(); advance(it,1)){
        string filename = "filter_";
        int idx = distance( filterbank.begin(), it);
        if ( idx % 2 == 0 ) filename+="re";
        else filename+="im";
        filename += boost::lexical_cast<string>(idx/2+1) + "_wl" + boost::lexical_cast<string>((it->cols-1)/2);
        filename += ".png";
        writeNormMatToFile( *it, filename);
        cout << "filterbank written: "<<filename<<endl;
    }
}
/** ------------------------------- image processing functions ------------------------------- **/


Mat extendByWrap (const Mat &src,  int ext_width,  int  ext_height=0){
    CV_Assert(ext_width >=0 && ext_height >= 0);
    int w = src.cols + ext_width;
    int woff = ext_width/2;
    int wrest = ext_width-woff;
    int h = src.rows + ext_height;
    int hoff = ext_height/2;
    int hrest = ext_height-hoff;

    Mat dst(h,w,src.type());
    copyMakeBorder( src, dst, hoff, hrest, woff, wrest, BORDER_WRAP);
    return dst;
}


void featureExtract(Mat &code, Mat &codeMask, const Mat &img, const Mat &mask, const vector<Mat> &filterbank, const vector<Point> &extractionsequence, bool &write_samples){
    CV_Assert(code.size() == Size( filterbank.size()  * extractionsequence.size()/8,1));
    CV_Assert(mask.empty() || codeMask.empty() || codeMask.size() == code.size());
    bool useMask = !mask.empty() && !codeMask.empty();

    int filter_w=0, filter_h=0;
    for( vector<Mat>::const_iterator it=filterbank.begin(); it != filterbank.end(); advance(it,1)){
        if( it->cols > filter_w) filter_w = it->cols;
        if( it->rows > filter_h) filter_h = it->rows;
    }
    Mat extimg = extendByWrap( img, filter_w/* , filter_h */);
    Point extoff(filter_w/2, 0);

    vector<Mat> filteredExtImg;
    for( unsigned int i=0; i < filterbank.size(); ++i){
        filteredExtImg.push_back(extimg.clone());
        filter2D(extimg, filteredExtImg[i], CV_32F, filterbank[i]);
        if( write_samples){
            writeNormMatToFile(filteredExtImg[i], "filtered_ext_fb"+boost::lexical_cast<string>(i)+".png", "sample written:");
            writeNormMatToFile(filteredExtImg[i], "filtered_ext_fb"+boost::lexical_cast<string>(i)+"_bitmap.png", "sample written:", true);
        }
    }
    if( write_samples) writeNormMatToFile(extimg, "filtered_ext_orig.png", "sample written");
    

    uchar *codeMaskData = NULL;
	if (useMask){
        codeMask.setTo(255);
        codeMaskData = codeMask.data;
    }
	code.setTo(0);
    uchar * codeData = code.data;
    int bitpos = 0;
    float fval;
    for( vector<Mat>::const_iterator fit = filteredExtImg.begin(); fit != filteredExtImg.end(); advance( fit, 1) ){
        for( vector<Point>::const_iterator eit = extractionsequence.begin(); eit != extractionsequence.end(); advance( eit,1) ){
            // true = 1 for sign, 0 is default
            fval = fit->at<float>(*eit + extoff);
            if( fval >= 0) codeData[0] |= 1<<(7-bitpos);
            if( useMask ){
                if( ( fval*fval < 0.000000001 ) ||    // values are very small, discard as possible error
                    ( codeMask.at<uchar>(*eit) > 0 ) )     // texture is masked
                            codeMaskData[0] &= 0xff ^ 1 << (7-bitpos);
            }
            bitpos++;
            if( bitpos == 8){
                codeData++;
                bitpos = 0;
                if (useMask) codeMaskData++;
            }
        }
    }
    write_samples = false; // only write one set 
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
void cmdCheckOpts(map<string ,vector<string> >& cmd, const string &validOptions){
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
	for (map<string, vector<string> >::iterator it = cmd.begin(); it != cmd.end(); std::advance(it,1)){
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
void cmdCheckOptExists(map<string ,vector<string> >& cmd, const string &option){
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
void cmdCheckOptSize(map<string ,vector<string> >& cmd, const string &option, const unsigned int size = 1){
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
vector<string> * cmdGetOpt(map<string ,vector<string> >& cmd, const string &option){
	map<string, vector<string> >::iterator it = cmd.find(option);
	return (it != cmd.end()) ? &(it->second) : 0;
}

/*
 * Returns number of parameters in an option
 *
 * cmd: commandline representation
 * option: name of the option
 */
unsigned int cmdSizePars(map<string ,vector<string> >& cmd, const string &option){
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
float cmdGetParFloat(map<string ,vector<string> >& cmd, const string &option, const unsigned int param = 0){
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
string cmdGetPar(map<string ,vector<string> >& cmd, const string &option, const unsigned int param = 0){
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


/**
 * First filter ist square with width wl*2+1, for the other filter horizontal
 * wavelength increase by one octave for each filter. 
 * A total of 8 filters are generated, 3 octaves from base with real and
 * imaginary parts of the complex gabor wavelet as separate filter.
 *
 * base_wavelength:       Is the base wavelength of the underlying cosine/sine.
 * border_gauss_residual: Adjust gauss sigma such that this percentage of
 *                        central energy is left at the border.
 */
void generateFilterbank( vector<Mat> &filterbank, const int &base_wavelength, const float &border_gauss_residual, int max_octave=3){
    float amp = -1*M_PI;
    float borderfactor = sqrtf(amp/logf(border_gauss_residual));
    int hr = FILTER_HEIGHT; // height radius and filter height
    int h = 2*hr+1;
    float alpha = hr*borderfactor;

    for( int octave =0, wavelength = base_wavelength; octave <= max_octave; wavelength*=2, octave++){
        int wr = wavelength;
        int w = wr*2+1;
        float beta = wr*borderfactor;
        cout << "generating filter for wavelength: "<<wavelength<<" and size "<<w<<"x"<<h ; 
        cout <<" alpha: "<<alpha<<" beta: "<<beta<<endl;
        Mat filter_re( h,w,CV_32F), filter_im( h,w,CV_32F);
        float G,F_RE,F_IM;
        int xp,yp;
        for( int y = 0; y < h; ++y)
        for( int x = 0; x < w; ++x){
            xp = x-wr; 
            yp = y-hr;
            G = exp( amp*( powf( float(xp)/beta,2) + pow(float(yp)/alpha,2)) );
            F_RE = cos( xp * 2*M_PI/wavelength );
            F_IM = sin( xp * 2*M_PI/wavelength );
            filter_re.at<float>(y,x) = G*F_RE;
            filter_im.at<float>(y,x) = G*F_IM;
        }
        filter_re -= sum(filter_re)[0]/(w*h); // correct for non/zero response  of real filter
        filterbank.push_back(filter_re);
        filterbank.push_back(filter_im);
    }
}


/**
 * Specifies which points to use for the iris code, points are taken as given each point is taken from each filter, then the next point is taken.
 *
 * This function specifies and equally spaced grid for extraction
 *
 * grid_x: number of points in x-direction to extract
 *
 * returns the number of grouped points for a single angle
 */
void generateExtractsequenceGrid( vector<Point> &extractsequence, const int &base_wavelength, const int &grid_x,  const int size_x, const int size_y){
    int sample_x = (grid_x >= 1) ? grid_x : size_x/base_wavelength ;
    int sample_y = size_y/(FILTER_HEIGHT*2+1) ;
    int dx = size_x/sample_x; // horizontal equally spaced, image is rotationally extended for filtering
    float offy = FILTER_HEIGHT+0.5;
    float resty= size_y - 2.*offy;
    float dy = resty/(sample_y-1); // keep filter in image for vertical
    if (dy!=dy || sample_y == 1) dy=0; //NaN comparissons are always false or smaple_y ==1 ,i.e., dy = inf
    for( int y=0; y<sample_y; ++y)
    for( int x=0; x<sample_x; ++x)
        extractsequence.push_back( Point( int(dx/2. + x*dx),int(offy + y*dy)));
    
    cout << "Extractsequence samples: "<<sample_x<<"x"<<sample_y<<endl;
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
    	if (cmd.empty() || cmdGetOpt(cmd,"-h") != 0) mode = MODE_HELP;
    	else mode = MODE_MAIN;
    	if (mode == MODE_MAIN){
			// validate command line
			cmdCheckOpts(cmd,"-i|-o|-q|-t|-wf|-ws|-wp|-pwl|-pbp|-psx|-x|-y");
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
            /** parmeters **/
            int base_wavelength = 6;
			if (cmdGetOpt(cmd,"-pwl") != 0){
				cmdCheckOptSize(cmd,"-pwl",1);
				base_wavelength = cmdGetParInt(cmd,"-pwl", 0);
			}
            float border_gauss_residual = 0.01;
			if (cmdGetOpt(cmd,"-pbp") != 0){
				cmdCheckOptSize(cmd,"-pbp",1);
				border_gauss_residual = cmdGetParFloat(cmd,"-pbp", 0);
                if( border_gauss_residual <= 0 || border_gauss_residual >= 1){
                    cout << "WARNING: Border power does not make much sense, reverting to 0.01"<<endl;
                    border_gauss_residual = 0.01;
                }
			}
            int grid_x=256;
			if (cmdGetOpt(cmd,"-psx") != 0){
				cmdCheckOptSize(cmd,"-psx",1);
				grid_x = cmdGetParInt(cmd,"-psx", 0);
                if( grid_x <= 0){
                    cout << "WARNING: grid_x does not make much sense, setting to sampling with 50% base filter overlap" << endl;
                    grid_x = -1;
                }
			}
            cout <<" grid_x from parameters" << grid_x << endl;
            int size_x=512;
			if (cmdGetOpt(cmd,"-x") != 0){
				cmdCheckOptSize(cmd,"-x",1);
				size_x = cmdGetParInt(cmd,"-x", 0);
			}
            int size_y=64;
			if (cmdGetOpt(cmd,"-y") != 0){
				cmdCheckOptSize(cmd,"-y",1);
				size_y = cmdGetParInt(cmd,"-y", 0);
			}
            FILTER_HEIGHT=(size_y-1)/2 ;
            /** options **/
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
			bool write_filterbank = false;
			if (cmdGetOpt(cmd,"-wf") != 0){
				cmdCheckOptSize(cmd,"-wf",0);
				write_filterbank = true;
			}
			bool write_samples = false;
			if (cmdGetOpt(cmd,"-ws") != 0){
				cmdCheckOptSize(cmd,"-ws",0);
				write_samples = true;
			}
			bool write_points = false;
			if (cmdGetOpt(cmd,"-wp") != 0){
				cmdCheckOptSize(cmd,"-wp",0);
				write_points = true;
			}
			// starting routine
			Timing timing(1,quiet);
			vector<string> files;
			patternToFiles(inFiles,files);
			CV_Assert(files.size() > 0);
			timing.total = files.size();
            vector<Mat> filterbank;
            generateFilterbank( filterbank, base_wavelength, border_gauss_residual);
            if( write_filterbank) writeFilterbank(filterbank);
            vector<Point> extractsequence;
            generateExtractsequenceGrid( extractsequence, base_wavelength, grid_x,  size_x, size_y);
			Mat code (1,extractsequence.size()*filterbank.size()/8,CV_8UC1);
            cout << "Code size (bytes): "<< code.size()<<endl;
			Mat codeMask(1,extractsequence.size()*filterbank.size()/8,CV_8UC1);
			for (vector<string>::iterator inFile = files.begin(); inFile != files.end(); std::advance( inFile, 1), timing.progress++){
				if (!quiet) printf("Loading texture '%s' ...\n", (*inFile).c_str());;
				Mat img = imread(*inFile, CV_LOAD_IMAGE_GRAYSCALE);
                if( write_points) writeExtractionsequence(img, extractsequence, write_points); // write_points is reset after one use
				CV_Assert(img.data != 0);
				CV_Assert(img.size() == Size(size_x,size_y));
				Mat mask = Mat();
				if (!imaskFiles.empty()) {
					string imaskfile;
					patternFileRename(inFiles,imaskFiles,*inFile,imaskfile);
					if (!quiet) printf("Loading mask image '%s' ...\n", imaskfile.c_str());;
					mask = imread(imaskfile, CV_LOAD_IMAGE_GRAYSCALE);
					CV_Assert(mask.data != 0);
					CV_Assert(mask.size() == Size(size_x,size_y));
				}
				if (!quiet) printf("Creating iris-code ...\n");

				//featureExtract(code, codeMask, img, mask, m, n);
				featureExtract(code, codeMask, img, mask, filterbank, extractsequence, write_samples);
				string outfile;
				patternFileRename(inFiles,outFiles,*inFile,outfile);
				if (!quiet) printf("Storing code '%s' ...\n", outfile.c_str());
				if (!imwrite(outfile,code)) CV_Error(CV_StsError,"Could not save image '" + outfile + "'");
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
