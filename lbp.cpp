/*
 * lbp.cpp
 *
 * Author: F.Struck (florian.struck@cased.de), E. Pschernig (epschern@cosy.sbg.ac.at), P. Wild (pwild@cosy.sbg.ac.at)
 *
 * Generates an iris code from iris texture using the uniform LBP algorithm
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

using namespace std;
using namespace cv;

#define FILTER_SIZE 9
#define CELL_WIDTH 16
#define CELL_HEIGHT 16
#define SCALING_WIDTH 512
#define SCALING_HEIGHT 64
#define DELTA_SHIFT 8

/** no globbing in win32 mode **/
int _CRT_glob = 0;

/** Program modes **/
static const int MODE_MAIN = 1, MODE_HELP = 2;

/*
 * Print command line usage for this program
 */
void printUsage() {
    printf("+-----------------------------------------------------------------------------+\n");
    printf("| LBP - Iris-code generation (feature extraction) using the uniform LBP       |\n");
    printf("| algorithm with a 9x9 filter                                                 |\n");
    printf("|                                                                             |\n");
    printf("| MODES                                                                       |\n");
    printf("|                                                                             |\n");
    printf("| (# 1) Uniform LBP iris code extraction from iris textures                   |\n");
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
    printf("| Florian Struck (florian.struck@cased.de)                                    |\n");
    printf("| Elias Pschernig (epschern@cosy.sbg.ac.at)                                   |\n");
    printf("| Peter Wild (pwild@cosy.sbg.ac.at)                                           |\n");
    printf("|                                                                             |\n");
    printf("| COPYRIGHT                                                                   |\n");
    printf("|                                                                             |\n");
    printf("| (C) 2016 All rights reserved. Do not distribute without written permission. |\n");
    printf("+-----------------------------------------------------------------------------+\n");
}

/** ------------------------------- image processing functions ------------------------------- **/

struct Cell {
    int startX, startY;
    int endX, endY;
    int width, height;
};

/**
 * calculates the new index of a shifted pixel
 * @param index         the old index
 * @param indexWidth    the row length of this pixel
 * @param shift         the shift
 * @return              the new index
 */
int shiftIndex(const int index, const int indexWidth, const int shift) {
    int indexInRow = index % indexWidth;
    if (indexInRow + shift >= indexWidth) return index + shift - indexWidth;
    else if (indexInRow + shift < 0) return index + shift + indexWidth;
    else return index + shift;
}

uchar getAverageValue(const int centerX, const int centerY, const Cell& cell, const int filterSize, const Mat& texture) {
    int averageValue = 0;
    int mod_x, mod_y;
    const int halfCellSize = filterSize / 2;

    for (int y = centerY - halfCellSize; y <= centerY + halfCellSize; y++) {

        mod_y = y;
        if (mod_y < cell.startY) mod_y += cell.height;
        else if (mod_y >= cell.endY) mod_y -= cell.height;

        for (int x = centerX - halfCellSize; x <= centerX + halfCellSize; x++) {

            mod_x = x;
            if (mod_x < cell.startX) mod_x += cell.width;
            else if (mod_x >= cell.endX) mod_x -= cell.width;

            averageValue += *texture.ptr<uchar>(mod_y, mod_x);
        }
    }

    return (uchar) (averageValue / pow(filterSize, 2));
}

/**
 * Calculate the LBP-Value clockwise of a pixel
 * @param x             the x-coordinate of the pixel in the texture
 * @param y             the y-coordinate of the pixel in the texture
 * @param filterSize    the filter-size of the MB-LBP algorithm (has to be a multiple of 3)
 * @param texture       the iris texture
 * @return              the clockwise LBP-Value of this pixel
 */
uchar calculateLBPValue(const int centerY, const int centerX, const Cell& cellSize, const int filterSize, const Mat& texture) {
    const int filterCellSize = filterSize / 3;
    const uchar centerValue = getAverageValue(centerX, centerY, cellSize, filterCellSize, texture);

    uchar LBPValue = 0x00;
    const uchar bitMask = 0x01;

    //----------------------Top-Neighbours--------------------------------------    
    for (int x = centerX - filterCellSize; x <= centerX + filterCellSize; x += filterCellSize) {
        if (getAverageValue(x, centerY - filterCellSize, cellSize, filterCellSize, texture) >= centerValue) LBPValue |= bitMask;
        LBPValue = LBPValue << 1; //*2
    }
    //----------------------Right-Neighbour------------------------------------
    if (getAverageValue(centerX + filterCellSize, centerY, cellSize, filterCellSize, texture) >= centerValue) LBPValue |= bitMask;
    LBPValue = LBPValue << 1; //*2
    //----------------------Bottom-Neighbours----------------------------------
    for (int x = centerX + filterCellSize; x >= centerX - filterCellSize; x -= filterCellSize) {
        if (getAverageValue(x, centerY + filterCellSize, cellSize, filterCellSize, texture) >= centerValue) LBPValue |= bitMask;
        LBPValue = LBPValue << 1; //*2
    }
    //-----------------------Left-Neighbours-----------------------------------
    if (getAverageValue(centerX - filterCellSize, centerY, cellSize, filterCellSize, texture) >= centerValue) LBPValue |= bitMask;

    return LBPValue;
}

/**
 * Extracts the features of the complete iris texture with the lbp algorithm
 * @param extractedData     the destination for the extracted data
 * @param cell              the start- and end-positions of this cell in the texture
 * @param texture           the source iris-texture
 */
void extractTexture(Mat& extractedData, const int filterSize, const Mat& texture) {
    Cell cell;
    cell.startX = 0;
    cell.startY = 0;
    cell.width = texture.cols;
    cell.height = texture.rows;
    cell.endX = cell.startX + cell.width;
    cell.endY = cell.startY + cell.height;

    for (int row = 0; row < texture.rows; row++) {
        for (int col = 0; col < texture.cols; col++) {
            *extractedData.ptr<uchar>(row, col) = calculateLBPValue(row, col, cell, filterSize, texture);
        }
    }
}

/**
 * Tests if a lbp-value is uniform
 * @param lbp_value the lbp-value wich should be tested
 * @return true if uniform und false if the value is not uniform
 */
bool isUniform(const unsigned char lbpValue) {
    unsigned char transmissions = 0;
    unsigned char bitmask = 64;
    unsigned char shifter = 6;
    unsigned char lastValue = (lbpValue & 128) >> 7; //initialisation of the first bit    

    for (char index = 1; index < 8; index++) { //analyse every bit
        unsigned char value = (lbpValue & bitmask) >> shifter;
        if (lastValue != value) {
            lastValue = value;
            transmissions++;
        }
        shifter--;
        bitmask = bitmask >> 1;
    }
    return (transmissions <= 2);
}

/**
 * Translate a histogram to an uniformed histogram
 * @param uniformHistogram  a pointer of a uniformed histogram
 * @param extractedData     a pointer of a filled ununiformed histogram
 * @param size              the size of the ununiformed histogram
 */
void toUniformHistogram(uint16_t *uniformHistogram, const uint16_t *extractedData, const int size) {
    int uniformIndex = 0;
    uniformHistogram[58] = 0; //reset the last field

    for (int index = 0; index < size; index++) {
        if (isUniform(index)) {
            uniformHistogram[uniformIndex] = extractedData[index];
            uniformIndex++;
        }
        else {
            uniformHistogram[58] += extractedData[index];
        }
    }
}

/**
 * Transforms a part of a texture to a histogram
 * @param histogram     the destination of the histogram
 * @param cell          the part of the texture
 * @param sourceTexture the source texture
 * @param shift         the alignment of pixels for shifting
 */
void toHistogram(uint16_t *histogram, const Cell& cell, const Mat& sourceTexture, const int shift) {
    for (int row = cell.startY; row < cell.endY; row++) {
        for (int col = cell.startX; col < cell.endX; col++) {
            histogram[*sourceTexture.ptr<uchar>(row, shiftIndex(col, sourceTexture.cols, shift))]++;
        }
    }
}

/*
 * The LBP feature extraction algorithm
 *
 * code: Code matrix
 * texture: texture matrix
 */
/**
 * Creates normalized histograms of the extracted features
 * @param code          the destination for the feature data
 * @param extractedData the extracted feature of the iris
 * @param shift         the alignment of pixels for shifting
 */
void extractHistograms(Mat& code, const Mat& extractedData, const int shift) {
    const int verticalCellNumber = extractedData.cols / CELL_WIDTH;
    const int horizontalCellNumber = extractedData.rows / CELL_HEIGHT;
    const int histogramLength = 256;

    uint16_t *histogram = new uint16_t[histogramLength];
    uint16_t *uniformHistogram = new uint16_t[59]; //58 values of 255 are uniform plus rest

    Cell cell;
    cell.width = CELL_WIDTH;
    cell.height = CELL_HEIGHT;
    
    int cellIndex = 0;
    for (int cellRow = 0; cellRow < horizontalCellNumber; cellRow++) {
        for (int cellCol = 0; cellCol < verticalCellNumber; cellCol++, cellIndex++) {

            cell.startX = cellCol * cell.width;
            cell.startY = cellRow * cell.height;
            cell.endX = cell.startX + cell.width;
            cell.endY = cell.startY + cell.height;

            for (int i = 0; i < histogramLength; i++) histogram[i] = 0; //set all values to 0
            toHistogram(histogram, cell, extractedData, shift); //calculate Histogramm
            toUniformHistogram(uniformHistogram, histogram, histogramLength); //calculate uniform Histogramm

            int colIndex = 0;
            for (int arrayIndex = 0; arrayIndex < 59; arrayIndex++, colIndex += 2) { //translate 16-Bit integer array to 8-Bit-Image              
                code.at<uchar>(cellIndex, colIndex) = (uniformHistogram[arrayIndex] & 0xFF00) >> 8;
                code.at<uchar>(cellIndex, colIndex + 1) = (uniformHistogram[arrayIndex] & 0x00FF);
            }
        }
    }

    delete[] histogram;
    delete[] uniformHistogram;
}

/**
 * Extracts the iris texture with 3 diffrent alignments and saves these in one big code map
 * @param code          the big code mat for the extraction data
 * @param texture       the iris texture which should be extracted
 * @param deltaShift    the maximum alignment for shifting
 */
void multipleExtract(Mat& code, const Mat& texture, const int deltaShift) {
    const int verticalCellNumber = texture.cols / CELL_WIDTH;
    const int horizontalCellNumber = texture.rows / CELL_HEIGHT;
    const int histogrammLength = 59; //58 uniform histograms + rest
    
    Mat extractedData(texture.rows, texture.cols, CV_8UC1);
    extractTexture(extractedData, FILTER_SIZE, texture);     //extract all features of this texture 
    
    Mat singleCode(verticalCellNumber*horizontalCellNumber, 2 * histogrammLength, CV_8UC1);

    int counter = 0;
    for (int shift = -deltaShift; shift <= deltaShift; shift += deltaShift, counter++) {
        singleCode.setTo(0);
        extractHistograms(singleCode, extractedData, shift);
        singleCode.copyTo(code(Rect(0, counter*singleCode.rows, singleCode.cols, singleCode.rows)));
    }
}

/**
 * Scales a texuture
 * @param scaledTexture
 * @param orginalTexture
 */
void scaleImage(Mat& scaledTexture, const Mat& orginalTexture){  
    resize(orginalTexture, scaledTexture, Size(SCALING_WIDTH,SCALING_HEIGHT), 0, 0);
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
void cmdRead(map<string, vector<string> >& cmd, int argc, char *argv[]) {
    for (int i = 1; i < argc; i++) {
        char * argument = argv[i];
        if (strlen(argument) > 1 && argument[0] == '-' && (argument[1] < '0' || argument[1] > '9')) {
            cmd[argument]; // insert
            char * argument2;
            while (i + 1 < argc && (strlen(argument2 = argv[i + 1]) <= 1 || argument2[0] != '-' || (argument2[1] >= '0' && argument2[1] <= '9'))) {
                cmd[argument].push_back(argument2);
                i++;
            }
        }
        else {
            CV_Error(CV_StsBadArg, "Invalid command line format");
        }
    }
}

/**
 * Checks, if each command line option is valid, i.e. exists in the options array
 *
 * cmd: commandline representation
 * validOptions: list of valid options separated by pipe (i.e. |) character
 */
void cmdCheckOpts(map<string, vector<string> >& cmd, const string validOptions) {
    vector<string> tokens;
    const string delimiters = "|";
    string::size_type lastPos = validOptions.find_first_not_of(delimiters, 0); // skip delimiters at beginning
    string::size_type pos = validOptions.find_first_of(delimiters, lastPos); // find first non-delimiter
    while (string::npos != pos || string::npos != lastPos) {
        tokens.push_back(validOptions.substr(lastPos, pos - lastPos)); // add found token to vector
        lastPos = validOptions.find_first_not_of(delimiters, pos); // skip delimiters
        pos = validOptions.find_first_of(delimiters, lastPos); // find next non-delimiter
    }
    sort(tokens.begin(), tokens.end());
    for (map<string, vector<string> >::iterator it = cmd.begin(); it != cmd.end(); it++) {
        if (!binary_search(tokens.begin(), tokens.end(), it->first)) {
            CV_Error(CV_StsBadArg, "Command line parameter '" + it->first + "' not allowed.");
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
void cmdCheckOptExists(map<string, vector<string> >& cmd, const string option) {
    map<string, vector<string> >::iterator it = cmd.find(option);
    if (it == cmd.end()) CV_Error(CV_StsBadArg, "Command line parameter '" + option + "' is required, but does not exist.");
}

/*
 * Checks, if a specific option has the appropriate number of parameters
 *
 * cmd: commandline representation
 * option: option name
 * size: appropriate number of parameters for the option
 */
void cmdCheckOptSize(map<string, vector<string> >& cmd, const string option, const unsigned int size = 1) {
    map<string, vector<string> >::iterator it = cmd.find(option);
    if (it->second.size() != size) CV_Error(CV_StsBadArg, "Command line parameter '" + option + "' has unexpected size.");
}

/*
 * Checks, if a specific option has the appropriate number of parameters
 *
 * cmd: commandline representation
 * option: option name
 * min: minimum appropriate number of parameters for the option
 * max: maximum appropriate number of parameters for the option
 */
void cmdCheckOptRange(map<string, vector<string> >& cmd, string option, unsigned int min = 0, unsigned int max = 1) {
    map<string, vector<string> >::iterator it = cmd.find(option);
    unsigned int size = it->second.size();
    if (size < min || size > max) CV_Error(CV_StsBadArg, "Command line parameter '" + option + "' is out of range.");
}

/*
 * Returns the list of parameters for a given option
 *
 * cmd: commandline representation
 * option: name of the option
 */
vector<string> * cmdGetOpt(map<string, vector<string> >& cmd, const string option) {
    map<string, vector<string> >::iterator it = cmd.find(option);
    return (it != cmd.end()) ? &(it->second) : 0;
}

/*
 * Returns number of parameters in an option
 *
 * cmd: commandline representation
 * option: name of the option
 */
unsigned int cmdSizePars(map<string, vector<string> >& cmd, const string option) {
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
int cmdGetParInt(map<string, vector<string> >& cmd, string option, unsigned int param = 0) {
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
float cmdGetParFloat(map<string, vector<string> >& cmd, const string option, const unsigned int param = 0) {
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
string cmdGetPar(map<string, vector<string> >& cmd, const string option, const unsigned int param = 0) {
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
        return ((current - lastPrint > boost::posix_time::seconds(updateInterval)) || (progress == total));
    }

    /*
     * Prints timing object to STDOUT
     */
    void print(void) {
        lastPrint = current;
        float percent = 100.f * progress / total;
        boost::posix_time::time_duration passed = (current - start);
        boost::posix_time::time_duration togo = passed * (total - progress) / max(1, progress);
        if (erase) {
            string erase(eraseCount, '\r');
            printf("%s", erase.c_str());
            int newEraseCount = (progress != total) ? printf("Progress ... %3.2f%% (%i/%i Total %i:%02i:%02i.%03i Remaining ca. %i:%02i:%02i.%03i)", percent, progress, total, passed.hours(), passed.minutes(), passed.seconds(), (int) (passed.total_milliseconds() % 1000), togo.hours(), togo.minutes(), togo.seconds(), (int) (togo.total_milliseconds() % 1000)) : printf("Progress ... %3.2f%% (%i/%i Total %i:%02i:%02i.%03d)", percent, progress, total, passed.hours(), passed.minutes(), passed.seconds(), (int) (passed.total_milliseconds() % 1000));
            if (newEraseCount < eraseCount) {
                string erase(newEraseCount - eraseCount, ' ');
                erase.append(newEraseCount - eraseCount, '\r');
                printf("%s", erase.c_str());
            }
            eraseCount = newEraseCount;
        }
        else {
            eraseCount = (progress != total) ? printf("Progress ... %3.2f%% (%i/%i Total %i:%02i:%02i.%03i Remaining ca. %i:%02i:%02i.%03i)\n", percent, progress, total, passed.hours(), passed.minutes(), passed.seconds(), (int) (passed.total_milliseconds() % 1000), togo.hours(), togo.minutes(), togo.seconds(), (int) (togo.total_milliseconds() % 1000)) : printf("Progress ... %3.2f%% (%i/%i Total %i:%02i:%02i.%03d)\n", percent, progress, total, passed.hours(), passed.minutes(), passed.seconds(), (int) (passed.total_milliseconds() % 1000));
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
string patternSubstrRegex(string& pattern, size_t pos, size_t n) {
    string result;
    for (size_t i = pos, e = pos + n; i < e; i++) {
        char c = pattern[i];
        if (c == '\\' || c == '.' || c == '+' || c == '[' || c == '{' || c == '|' || c == '(' || c == ')' || c == '^' || c == '$' || c == '}' || c == ']') {
            result.append(1, '\\');
            result.append(1, c);
        }
        else if (c == '*') {
            result.append("([^/\\\\]*)");
        }
        else if (c == '?') {
            result.append("([^/\\\\])");
        }
        else {
            result.append(1, c);
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
    size_t first_unknown = pattern.find_first_of("*?", pos); // find unknown * in pattern
    if (first_unknown != string::npos) {
        size_t last_dirpath = pattern.find_last_of("/\\", first_unknown);
        size_t next_dirpath = pattern.find_first_of("/\\", first_unknown);
        if (next_dirpath != string::npos) {
            boost::regex expr((last_dirpath != string::npos && last_dirpath > pos) ? patternSubstrRegex(pattern, last_dirpath + 1, next_dirpath - last_dirpath - 1) : patternSubstrRegex(pattern, pos, next_dirpath - pos));
            boost::filesystem::directory_iterator end_itr; // default construction yields past-the-end
            try {
                for (boost::filesystem::directory_iterator itr(((path.length() > 0) ? path + pattern[pos - 1] : (last_dirpath != string::npos && last_dirpath > pos) ? "" : "./") + ((last_dirpath != string::npos && last_dirpath > pos) ? pattern.substr(pos, last_dirpath - pos) : "")); itr != end_itr; ++itr) {
                    if (boost::filesystem::is_directory(itr->path())) {
                        boost::filesystem::path p = itr->path().filename();
                        string s = p.string();
                        if (boost::regex_match(s.c_str(), expr)) {
                            patternToFiles(pattern, files, (int) (next_dirpath + 1), ((path.length() > 0) ? path + pattern[pos - 1] : "") + ((last_dirpath != string::npos && last_dirpath > pos) ? pattern.substr(pos, last_dirpath - pos) + pattern[last_dirpath] : "") + s);
                        }
                    }
                }
            } catch (boost::filesystem::filesystem_error &e) {
            }
        }
        else {
            boost::regex expr((last_dirpath != string::npos && last_dirpath > pos) ? patternSubstrRegex(pattern, last_dirpath + 1, pattern.length() - last_dirpath - 1) : patternSubstrRegex(pattern, pos, pattern.length() - pos));
            boost::filesystem::directory_iterator end_itr; // default construction yields past-the-end
            try {
                for (boost::filesystem::directory_iterator itr(((path.length() > 0) ? path + pattern[pos - 1] : (last_dirpath != string::npos && last_dirpath > pos) ? "" : "./") + ((last_dirpath != string::npos && last_dirpath > pos) ? pattern.substr(pos, last_dirpath - pos) : "")); itr != end_itr; ++itr) {
                    boost::filesystem::path p = itr->path().filename();
                    string s = p.string();
                    if (boost::regex_match(s.c_str(), expr)) {
                        files.push_back(((path.length() > 0) ? path + pattern[pos - 1] : "") + ((last_dirpath != string::npos && last_dirpath > pos) ? pattern.substr(pos, last_dirpath - pos) + pattern[last_dirpath] : "") + s);
                    }
                }
            } catch (boost::filesystem::filesystem_error &e) {
            }
        }
    }
    else { // no unknown symbols
        boost::filesystem::path file(((path.length() > 0) ? path + "/" : "") + pattern.substr(pos, pattern.length() - pos));
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
    patternToFiles(pattern, files, 0, "");
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
    size_t first_unknown = renamePattern.find_first_of(par, 0); // find unknown ? in renamePattern
    if (first_unknown != string::npos) {
        string formatOut = "";
        for (size_t i = 0, e = renamePattern.length(); i < e; i++) {
            char c = renamePattern[i];
            if (c == par && i + 1 < e) {
                c = renamePattern[i + 1];
                if (c > '0' && c <= '9') {
                    formatOut.append(1, '$');
                    formatOut.append(1, c);
                }
                else {
                    formatOut.append(1, par);
                    formatOut.append(1, c);
                }
                i++;
            }
            else {
                formatOut.append(1, c);
            }
        }
        boost::regex patternOut(patternSubstrRegex(pattern, 0, pattern.length()));
        outfile = boost::regex_replace(infile, patternOut, formatOut, boost::match_default | boost::format_perl);
    }
    else {
        outfile = renamePattern;
    }
}

/** ------------------------------- Program ------------------------------- **/

/*
 * Main program
 */
int main(int argc, char *argv[]) {
    int mode = MODE_HELP;
    map<string, vector<string> > cmd;
    try {
        cmdRead(cmd, argc, argv);
        if (cmd.size() == 0 || cmdGetOpt(cmd, "-h") != 0) mode = MODE_HELP;
        else mode = MODE_MAIN;
        if (mode == MODE_MAIN) {
            // validate command line
            cmdCheckOpts(cmd, "-i|-o|-m|-q|-t");

            cmdCheckOptExists(cmd, "-i");
            cmdCheckOptSize(cmd, "-i", 1);
            string inFiles = cmdGetPar(cmd, "-i");

            cmdCheckOptExists(cmd, "-o");
            cmdCheckOptSize(cmd, "-o", 1);
            string outFiles = cmdGetPar(cmd, "-o");

            string imaskFiles, omaskFiles;

            if (cmdGetOpt(cmd, "-m") != 0) {
                cmdCheckOptSize(cmd, "-m", 2);
                imaskFiles = cmdGetPar(cmd, "-m", 0);
                omaskFiles = cmdGetPar(cmd, "-m", 1);
            }
            bool quiet = false;
            if (cmdGetOpt(cmd, "-q") != 0) {
                cmdCheckOptSize(cmd, "-q", 0);
                quiet = true;
            }
            bool time = false;
            if (cmdGetOpt(cmd, "-t") != 0) {
                cmdCheckOptSize(cmd, "-t", 0);
                time = true;
            }
            // starting routine
            Timing timing(1, quiet);
            vector<string> files;
            patternToFiles(inFiles, files);
            CV_Assert(files.size() > 0);
            timing.total = files.size();
            for (vector<string>::iterator inFile = files.begin(); inFile != files.end(); ++inFile, timing.progress++) {
                if (!quiet) printf("Loading texture '%s' ...\n", (*inFile).c_str());

                Mat imgOrginal = imread(*inFile, CV_LOAD_IMAGE_GRAYSCALE);
                CV_Assert(imgOrginal.data != 0);
                Mat out;
                if (imgOrginal.rows != 64 || imgOrginal.cols != 512) {
                    printf("Input texture has to be of size 512 x 64.\n");
                    exit(EXIT_FAILURE);
                }
                
                Mat img;
                scaleImage(img, imgOrginal);

                int verticalCellNumber = img.cols / CELL_WIDTH; //width
                int horizontalCellNumber = img.rows / CELL_HEIGHT; //height

                if (!quiet) printf("Creating %d x %d iris-code ...\n", 118, verticalCellNumber * horizontalCellNumber*3);
                Mat code(verticalCellNumber*horizontalCellNumber*3, 118, CV_8UC1);
                code.setTo(0);
                multipleExtract(code, img, DELTA_SHIFT);
                out = code;
                string outfile;
                patternFileRename(inFiles, outFiles, *inFile, outfile);
                if (!quiet) printf("Storing code '%s' ...\n", outfile.c_str());
                if (!imwrite(outfile, out)) CV_Error(CV_StsError, "Could not save image '" + outfile + "'");
                if (time && timing.update()) timing.print();
            }
            if (time && quiet) timing.clear();
        }
        else if (mode == MODE_HELP) {
            // validate command line
            cmdCheckOpts(cmd, "-h");
            if (cmdGetOpt(cmd, "-h") != 0) cmdCheckOptSize(cmd, "-h", 0);
            // starting routine
            printUsage();
        }
    } catch (...) {
        printf("Exit with errors.\n");
        exit(EXIT_FAILURE);
    }
    return EXIT_SUCCESS;
}