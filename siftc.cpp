/*
 * cbc.cpp
 *
 * Author: J. Wagner (johannes.wagner@cased.de)
 *
 * Calculates similarity of two sets of SIFT keypoints
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
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <boost/regex.hpp>
#include <boost/filesystem.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

using namespace std;
using namespace cv;

/* xy-tolerances for the geometric constrains */
#define OPT_PARAM_X 20
#define OPT_PARAM_Y 5

/** no globbing in win32 mode **/
int _CRT_glob = 0;

/** Program modes **/
static const int MODE_MAIN = 1, MODE_HELP = 2;

/**
 * Print command line usage for this program
 */
void printUsage() {
    printVersion();
	printf( "+-----------------------------------------------------------------------------+\n");
	printf( "| SIFTC - Calculates similarity of two sets of SIFT keypoints                 |\n");
	printf( "|                                                                             |\n");
	printf( "|                                                                             |\n");
	printf( "| MODES                                                                       |\n");
	printf( "|                                                                             |\n");
	printf( "| (# 1) distance calculation of the input images (cross comparison)           |\n");
	printf( "| (# 2) usage                                                                 |\n");
	printf( "|                                                                             |\n");
	printf( "| ARGUMENTS                                                                   |\n");
	printf( "|                                                                             |\n");
	printf( "+------+------------+---+---+-------------------------------------------------+\n");
	printf( "| Name | Parameters | # | ? | Description                                     |\n");
	printf( "+------+------------+---+---+-------------------------------------------------+\n");
	printf( "| -i   | infile1    | 1 | N | source/reference .sift files (use * as wildcard,|\n");
	printf( "|      | infile2    |   |   | all other files may refer to n-th * with ?n)    |\n");
	printf( "| -o   | outfile    | 1 | Y | target text                                     |\n");
	printf( "| -q   |            | 1 | Y | quiet mode on (off)                             |\n");
	printf( "| -t   |            | 1 | Y | time progress on (off)                          |\n");
	printf( "| -h   |            | 2 | N | prints usage                                    |\n");
	printf( "+------+------------+---+---+-------------------------------------------------+\n");
	printf( "|                                                                             |\n");
	printf( "| EXAMPLE USAGE                                                               |\n");
	printf( "|                                                                             |\n");
	printf( "| -i s1.sift s2.sift -o compare.txt                                           |\n");
	printf( "| -i *.sift *.sift -s -o compare.txt -q -t                                    |\n");
	printf( "|                                                                             |\n");
	printf( "| AUTHOR                                                                      |\n");
	printf( "|                                                                             |\n");
	printf( "| Johannes Wagner (johannes.wagner@cased.de)                                  |\n");
	printf( "|                                                                             |\n");
	printf( "| COPYRIGHT                                                                   |\n");
	printf( "|                                                                             |\n");
	printf( "| (C) 2015 All rights reserved. Do not distribute without written permission. |\n");
	printf( "+-----------------------------------------------------------------------------+\n");
}

/**
 * Read the keypoints from the .sift file
 */

vector<KeyPoint> readKeypointsFromFile(string ymlfile_kp) {
	vector<KeyPoint> newKeypoints;
	FileStorage filestorage(ymlfile_kp, FileStorage::READ);
	FileNode kptFileNode = filestorage["keypoints"];
	read(kptFileNode, newKeypoints);
	filestorage.release();

	return newKeypoints;
}

/**
 * Read the feature descriptors from the .sift file
 */

Mat readDescriptorsFromFile(string ymlfile_desc) {
	Mat desc;
	FileStorage filestorage(ymlfile_desc, FileStorage::READ);
	FileNode descFileNode = filestorage["descriptors"];
	read(descFileNode, desc);
	filestorage.release();

	return desc;
}

/**
 * Read the image from the .sift file
 * to use this function make sure writing the image in line 129
 * from the feature extractor is commented in
 */

Mat readImageFromFile(string ymlfile_img) {
	Mat img;
	FileStorage filestorage(ymlfile_img, FileStorage::READ);
	FileNode descFileNode = filestorage["image"];
	read(descFileNode, img);
	filestorage.release();

	return img;
}

/**
 * Distance estimation for geometric constrains
 * a: sample sift features
 * b: reference sift features
 */

float siftCompare(string reference_file, string probe_file) {

	vector<KeyPoint> keypoints_reference = readKeypointsFromFile(
			reference_file);
	Mat descriptor_reference = readDescriptorsFromFile(reference_file);

	vector<KeyPoint> keypoints_probe = readKeypointsFromFile(probe_file);
	Mat descriptor_probe = readDescriptorsFromFile(probe_file);
	
	BFMatcher matcher(NORM_L2, true);
	vector<DMatch> matches;

	matcher.match(descriptor_reference, descriptor_probe, matches);
	float match_score = 0;
	for (vector<DMatch>::size_type i = 0; i < matches.size(); i++) {
	//Match only the keypoints with in a certain distance defined by OPT_PARAM_X; OPT_PARAM_Y
	        if (abs(keypoints_reference[matches[i].queryIdx].pt.y - keypoints_probe[matches[i].trainIdx].pt.y) < OPT_PARAM_Y && 
	            abs(keypoints_reference[matches[i].queryIdx].pt.x - keypoints_probe[matches[i].trainIdx].pt.x) < OPT_PARAM_X) {
			match_score++;
		}
        }
        return 1 - (match_score / float((keypoints_reference.size() + keypoints_probe.size()) / 2));
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
	void cmdRead(map<string ,vector<string> >& cmd, int argc, char *argv[]) {
		for (int i=1; i< argc; i++) {
			char * argument = argv[i];
			if (strlen(argument) > 1 && argument[0] == '-' && (argument[1] < '0' || argument[1] > '9')) {
				cmd[argument]; // insert
				char * argument2;
				while (i + 1 < argc && (strlen(argument2 = argv[i+1]) <= 1 || argument2[0] != '-' || (argument2[1] >= '0' && argument2[1] <= '9'))) {
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
	void cmdCheckOpts(map<string ,vector<string> >& cmd, const string validOptions) {
		vector<string> tokens;
		const string delimiters = "|";
		string::size_type lastPos = validOptions.find_first_not_of(delimiters,0); // skip delimiters at beginning
		string::size_type pos = validOptions.find_first_of(delimiters, lastPos);// find first non-delimiter
		while (string::npos != pos || string::npos != lastPos) {
			tokens.push_back(validOptions.substr(lastPos,pos - lastPos)); // add found token to vector
			lastPos = validOptions.find_first_not_of(delimiters,pos);// skip delimiters
			pos = validOptions.find_first_of(delimiters,lastPos);// find next non-delimiter
		}
		sort(tokens.begin(), tokens.end());
		for (map<string, vector<string> >::iterator it = cmd.begin(); it != cmd.end(); it++) {
			if (!binary_search(tokens.begin(),tokens.end(),it->first)) {
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
	void cmdCheckOptExists(map<string ,vector<string> >& cmd, const string option) {
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
	void cmdCheckOptSize(map<string ,vector<string> >& cmd, const string option, const unsigned int size = 1) {
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
	void cmdCheckOptRange(map<string ,vector<string> >& cmd, string option, unsigned int min = 0, unsigned int max = 1) {
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
	vector<string> * cmdGetOpt(map<string ,vector<string> >& cmd, const string option) {
		map<string, vector<string> >::iterator it = cmd.find(option);
		return (it != cmd.end()) ? &(it->second) : 0;
	}

	/*
	 * Returns number of parameters in an option
	 *
	 * cmd: commandline representation
	 * option: name of the option
	 */
	unsigned int cmdSizePars(map<string ,vector<string> >& cmd, const string option) {
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
	int cmdGetParInt(map<string ,vector<string> >& cmd, string option, unsigned int param = 0) {
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
	float cmdGetParFloat(map<string ,vector<string> >& cmd, const string option, const unsigned int param = 0) {
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
	string
	cmdGetPar(map<string ,vector<string> >& cmd, const string option, const unsigned int param = 0) {
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
	class Timing {
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
		Timing(long seconds, bool eraseMode) {
			updateInterval = seconds;
			progress = 1;
			total = 100;
			eraseCount = 0;
			erase = eraseMode;
			init();
		}

		/*
		 * Destructor
		 */
		~Timing() {
		}

		/*
		 * Initializes timing variables
		 */
		void init(void) {
			start = boost::posix_time::microsec_clock::universal_time();
			lastPrint = start - boost::posix_time::seconds(updateInterval);
		}

		/*
		 * Clears printing (for erase option only)
		 */
		void clear(void) {
			string erase(eraseCount, '\r');
			erase.append(eraseCount, ' ');
			erase.append(eraseCount, '\r');
			printf("%s", erase.c_str());
			eraseCount = 0;
		}

		/*
		 * Updates current time and returns true, if output should be printed
		 */
		bool update(void) {
			current = boost::posix_time::microsec_clock::universal_time();
			return ((current - lastPrint
					> boost::posix_time::seconds(updateInterval))
					|| (progress == total));
		}

		/*
		 * Prints timing object to STDOUT
		 */
		void print(void) {
			lastPrint = current;
			float percent = 100.f * progress / total;
			boost::posix_time::time_duration passed = (current - start);
			boost::posix_time::time_duration togo = passed * (total - progress)
					/ max(1, progress);
			if (erase) {
				string erase(eraseCount, '\r');
				printf("%s", erase.c_str());
				int newEraseCount =
						(progress != total) ?
								printf(
										"Progress ... %3.2f%% (%i/%i Total %i:%02i:%02i.%03i Remaining ca. %i:%02i:%02i.%03i)",
										percent, progress, total,
										passed.hours(), passed.minutes(),
										passed.seconds(),
										(int) (passed.total_milliseconds()
												% 1000), togo.hours(),
										togo.minutes(), togo.seconds(),
										(int) (togo.total_milliseconds() % 1000)) :
								printf(
										"Progress ... %3.2f%% (%i/%i Total %i:%02i:%02i.%03d)",
										percent, progress, total,
										passed.hours(), passed.minutes(),
										passed.seconds(),
										(int) (passed.total_milliseconds()
												% 1000));
				if (newEraseCount < eraseCount) {
					string erase(newEraseCount - eraseCount, ' ');
					erase.append(newEraseCount - eraseCount, '\r');
					printf("%s", erase.c_str());
				}
				eraseCount = newEraseCount;
			} else {
				eraseCount =
						(progress != total) ?
								printf(
										"Progress ... %3.2f%% (%i/%i Total %i:%02i:%02i.%03i Remaining ca. %i:%02i:%02i.%03i)\n",
										percent, progress, total,
										passed.hours(), passed.minutes(),
										passed.seconds(),
										(int) (passed.total_milliseconds()
												% 1000), togo.hours(),
										togo.minutes(), togo.seconds(),
										(int) (togo.total_milliseconds() % 1000)) :
								printf(
										"Progress ... %3.2f%% (%i/%i Total %i:%02i:%02i.%03d)\n",
										percent, progress, total,
										passed.hours(), passed.minutes(),
										passed.seconds(),
										(int) (passed.total_milliseconds()
												% 1000));
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
	string
patternSubstrRegex(string& pattern, size_t pos, size_t n) {
	string result;
	for (size_t i=pos, e=pos+n; i < e; i++ ) {
		char c = pattern[i];
		if ( c == '\\' || c == '.' || c == '+' || c == '[' || c == '{' || c == '|' || c == '(' || c == ')' || c == '^' || c == '$' || c == '}' || c == ']') {
			result.append(1,'\\');
			result.append(1,c);
		}
		else if (c == '*') {
			result.append("([^/\\\\]*)");
		}
		else if (c == '?') {
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
void patternToFiles(string& pattern, vector<string>& files, const size_t& pos, const string& path) {
	size_t first_unknown = pattern.find_first_of("*?",pos); // find unknown * in pattern
	if (first_unknown != string::npos) {
		size_t last_dirpath = pattern.find_last_of("/\\",first_unknown);
		size_t next_dirpath = pattern.find_first_of("/\\",first_unknown);
		if (next_dirpath != string::npos) {
			boost::regex expr((last_dirpath != string::npos && last_dirpath > pos) ? patternSubstrRegex(pattern,last_dirpath+1,next_dirpath-last_dirpath-1) : patternSubstrRegex(pattern,pos,next_dirpath-pos));
			boost::filesystem::directory_iterator end_itr; // default construction yields past-the-end
			try {
				for ( boost::filesystem::directory_iterator itr( ((path.length() > 0) ? path + pattern[pos-1] : (last_dirpath != string::npos && last_dirpath > pos) ? "" : "./") + ((last_dirpath != string::npos && last_dirpath > pos) ? pattern.substr(pos,last_dirpath-pos) : "")); itr != end_itr; ++itr )
				{
					if (boost::filesystem::is_directory(itr->path())) {
						boost::filesystem::path p = itr->path().filename();
						string s = p.string();
						if (boost::regex_match(s.c_str(), expr)) {
							patternToFiles(pattern,files,(int)(next_dirpath+1),((path.length() > 0) ? path + pattern[pos-1] : "") + ((last_dirpath != string::npos && last_dirpath > pos) ? pattern.substr(pos,last_dirpath-pos) + pattern[last_dirpath] : "") + s);
						}
					}
				}
			}
			catch (boost::filesystem::filesystem_error &e) {}
		}
		else {
			boost::regex expr((last_dirpath != string::npos && last_dirpath > pos) ? patternSubstrRegex(pattern,last_dirpath+1,pattern.length()-last_dirpath-1) : patternSubstrRegex(pattern,pos,pattern.length()-pos));
			boost::filesystem::directory_iterator end_itr; // default construction yields past-the-end
			try {
				for ( boost::filesystem::directory_iterator itr(((path.length() > 0) ? path + pattern[pos-1] : (last_dirpath != string::npos && last_dirpath > pos) ? "" : "./") + ((last_dirpath != string::npos && last_dirpath > pos) ? pattern.substr(pos,last_dirpath-pos) : "")); itr != end_itr; ++itr )
				{
					boost::filesystem::path p = itr->path().filename();
					string s = p.string();
					if (boost::regex_match(s.c_str(), expr)) {
						files.push_back(((path.length() > 0) ? path + pattern[pos-1] : "") + ((last_dirpath != string::npos && last_dirpath > pos) ? pattern.substr(pos,last_dirpath-pos) + pattern[last_dirpath] : "") + s);
					}
				}
			}
			catch (boost::filesystem::filesystem_error &e) {}
		}
	}
	else { // no unknown symbols
		boost::filesystem::path file(((path.length() > 0) ? path + "/" : "") + pattern.substr(pos,pattern.length()-pos));
		if (boost::filesystem::exists(file)) {
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
void patternToFiles(string& pattern, vector<string>& files) {
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
void patternFileRename(string& pattern, const string& renamePattern, const string& infile, string& outfile, const char par = '?') {
	size_t first_unknown = renamePattern.find_first_of(par,0); // find unknown ? in renamePattern
	if (first_unknown != string::npos) {
		string formatOut = "";
		for (size_t i=0, e=renamePattern.length(); i < e; i++ ) {
			char c = renamePattern[i];
			if ( c == par && i+1 < e) {
				c = renamePattern[i+1];
				if (c > '0' && c <= '9') {
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
		if (mode == MODE_MAIN) {
			// validate command line
			cmdCheckOpts(cmd,"-i|-o|-q|-t");
			cmdCheckOptExists(cmd,"-i");
			cmdCheckOptSize(cmd,"-i",2);
			string infilesSmpl = cmdGetPar(cmd,"-i",0);
			string infilesRef = cmdGetPar(cmd,"-i",1);
			string outfile;
			if (cmdGetOpt(cmd,"-o") != 0) {
				cmdCheckOptSize(cmd,"-o",1);
				outfile = cmdGetPar(cmd,"-o");
			}
			bool quiet = false;
			if (cmdGetOpt(cmd,"-q") != 0) {
				cmdCheckOptSize(cmd,"-q",0);
				quiet = true;
			}
			bool time = false;
			if (cmdGetOpt(cmd,"-t") != 0) {
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
			if (!outfile.empty()) {
				if (!quiet) printf("Opening result file '%s' ...\n", outfile.c_str());;
				cfile.open(outfile.c_str(),ios::out | ios::trunc);
				if (!(cfile.is_open())) {
					CV_Error(CV_StsError,"Could not open result file '" + outfile + "'");
				}
			}
			for (vector<string>::iterator infileSmpl = filesSmpl.begin(); infileSmpl != filesSmpl.end(); ++infileSmpl) {
				for (vector<string>::iterator infileRef = filesRef.begin(); infileRef != filesRef.end(); ++infileRef, timing.progress++) {
					double score = siftCompare((*infileSmpl).c_str(),(*infileRef).c_str());
					if (!quiet) 
					        printf("dist(%s,%s) = %f\n",(*infileSmpl).c_str(), (*infileRef).c_str(), score);
					if (!outfile.empty() && cfile.is_open()) {
						cfile << *infileSmpl << " " << *infileRef << " " << score << endl;
					}
					if (time && timing.update()) timing.print();
				}
			}
			if (time && quiet) timing.clear();
			if (!outfile.empty() && cfile.is_open()) {
				cfile.close();
			}
		}
		else if (mode == MODE_HELP) {
			// validate command line
			cmdCheckOpts(cmd,"-h");
			if (cmdGetOpt(cmd,"-h") != 0) cmdCheckOptSize(cmd,"-h",0);
			// starting routine
			printUsage();
		}
	}
	catch (...) {
		printf("Exit with errors.\n");
		exit(EXIT_FAILURE);
	}
	return EXIT_SUCCESS;
}
