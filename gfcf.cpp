/*
 * gfcf.cpp
 *
 * Author: P. Wild (pwild@cosy.sbg.ac.at)
 *
 * Face and eye segmentation tool: performs cascaded classifying
 * The software implements the following technique:
 *
 * Gaussian Face and Face-part Classifier Fusion
 *
 * see:
 *
 * A. Uhl and P. Wild. Combining Face with Face-Part Detectors under Gaussian Assumption. In A. Campilho and M.~Kamel, editors,
 * Proceedings of the 9th International Conference on Image Analysis and Recognition (ICIAR'12)}, volume 7325 of LNCS,
 * pages 80--89, Aveiro, Portugal, June 25--27, 2012.
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
#include <opencv2/objdetect/objdetect.hpp>
#include <boost/regex.hpp>
#include <boost/filesystem.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>


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
	printf("| gfcf - Gaussian Face and Face-part Classifier Fusion                        |\n");
	printf("|                                                                             |\n");
	printf("| A. Uhl and P. Wild. Combining Face with Face-Part Detectors under Gaussian  |\n");
	printf("| Assumption. In A. Campilho and M. Kamel, editors, Proc. of the 9th Int'l    |\n");
	printf("| Conf. on Image Analysis and Recognition (ICIAR’12), volume 7325 of LNCS,    |\n");
	printf("| pages 80–89, Aveiro, Portugal, June 25–27, 2012. Springer.                  |\n");
	printf("| doi: 10.1007/978-3-642-31298-4 10                                           |\n");
	printf("|                                                                             |\n");
	printf("| MODES                                                                       |\n");
	printf("|                                                                             |\n");
    printf("| (# 1) face and eye detection                                                |\n");
    printf("| (# 2) usage                                                                 |\n");
    printf("|                                                                             |\n");
    printf("| ARGUMENTS                                                                   |\n");
    printf("|                                                                             |\n");
    printf("+------+------------+---+---+-------------------------------------------------+\n");
    printf("| Name | Parameters | # | ? | Description                                     |\n");
    printf("+------+------------+---+---+-------------------------------------------------+\n");
    printf("| -i   | infile     | 1 | N | input image (use * as wildcard, all other       |\n");
    printf("|      |            |   |   | file parameters may refer to n-th * with ?n)    |\n");
    printf("| -o   | facefile   | 1 | Y | output image of the most dominant face and the  |\n");
    printf("|      | leyefile   |   |   | two correspending left and right full-eye images|\n");
    printf("|      | reyefile   |   |   |                                                 |\n");
    printf("| -c   | coordfile  | 1 | Y | output coordinates (one file, no wildcards)     |\n");
/* unused
    printf("| -s   | facesize   | 1 | Y | minimum output face and eye size, if available  |\n");
    printf("|      | eyesize    |   |   | (0=maximum, i.e. original size)                 |\n");
*/
    printf("| -q   |            | 1 | Y | quiet mode on (off)                             |\n");
    printf("| -t   |            | 1 | Y | time progress on (off)                          |\n");
    printf("| -dr  | detectfile | 1 | Y | write indiv classifier detection result (off)   |\n");
    printf("| -tr  | trkresfile | 1 | Y | write tracking result (off)                     |\n");
    printf("| -lt  | thickness  | 1 | Y | thickness for lines to be drawn (1)             |\n");
    printf("| -h   |            | 2 | N | prints usage                                    |\n");
    printf("+------+------------+---+---+-------------------------------------------------+\n");
    printf("|                                                                             |\n");
    printf("| EXAMPLE USAGE                                                               |\n");
    printf("|                                                                             |\n");
    printf("| -i *.tiff -o ?1_face.png ?1_eyeleft.png ?1_eyeright.png -q -t               |\n");
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
 * Type for the location of a face model
 *
 * x: reference x-coordinate (center between eyes) in PDS (part detection scale) coordinates
 * y: reference y-coordinate (center between eyes) in PDS (part detection scale) coordinates
 * w: reference inter-eye width (between eye centers) in PDS (part detection scale) pixels
 */
struct ModelLocation{
	float x;
	float sigmax;
	float y;
	float sigmay;
	float width;
	float sigmawidth;

	ModelLocation(){
		x=0;
		y=0;
		width=0;
		sigmax=0;
		sigmay=0;
		sigmawidth=0;
	}

	ModelLocation(float _x, float _y, float _width, float _sigmax, float _sigmay, float _sigmawidth){
		x = _x;
		y = _y;
		width = _width;
		sigmax = _sigmax;
		sigmay = _sigmay;
		sigmawidth = _sigmawidth;
	}
};

/**
 * Retrieves the estimated left model location given a single eye classifier output (based on trained model)
 * eye: classifier output
 * model: where to add estimated left-eye model (assuming eye represents the left eye)
 */
void addModelLocationFromLeftEye(const Rect eye, vector<ModelLocation>& model){
	ModelLocation left;
	left.y = (eye.y + eye.height/2.f); // center of eye rectangle
	left.width = eye.width / 0.55f;
	left.x = (eye.x + eye.width/2.f) + (left.width/2.f);
	left.sigmax = 0.063026 * left.width; // eyeXVar * leftwidth
	//left.sigmay = 0.022274 * left.width; // eyeYVar * leftwidth
	left.sigmay = 3 * 0.022274 * left.width; // eyeYVar * leftwidth

	left.sigmawidth = 0.122256 * left.width; // eyeWidthVar * leftwidth

	model.push_back(left);
}

/**
 * Retrieves the estimated right model location given a single eye classifier output (based on trained model)
 * eye: classifier output
 * model: where to add estimated right-eye model (assuming eye represents the left eye)
 */
void addModelLocationFromRightEye(const Rect eye, vector<ModelLocation>& model){
	ModelLocation right;
	right.y = (eye.y + eye.height/2.f); // center of eye rectangle
	right.width = eye.width / 0.55f;
	right.x = (eye.x + eye.width/2.f) - (right.width/2.f);
	right.sigmax = 0.063026 * right.width;
	//right.sigmay = 0.022274 * right.width;
	right.sigmay = 3 * 0.022274 * right.width;

	right.sigmawidth = 0.122256 * right.width;
	model.push_back(right);
}

/**
 * Retrieves the estimated model location given a single eyepair classifier output (based on trained model)
 * eyepair: classifier output
 * model: where to add estimated eyepair model
 */
void addModelLocationFromEyepair(const Rect eyepair, vector<ModelLocation>& model){
	ModelLocation mloc;
	mloc.x = (eyepair.x + eyepair.width/2.f);
	mloc.y = (eyepair.y + eyepair.height/2.f);
	mloc.width = eyepair.width / 1.69f;
	mloc.sigmax = 0.014679 * mloc.width;
	mloc.sigmay = 0.007536 * mloc.width;
	mloc.sigmawidth = 0.062195 * mloc.width;
	model.push_back(mloc);
}

/**
 * Retrieves the estimated model location given a single nose classifier output (based on trained model)
 * nose: classifier output
 * model: where to add estimated nose model
 */
void addModelLocationFromNose(const Rect nose, vector<ModelLocation>& model){
	ModelLocation mloc;
	mloc.x = (nose.x + nose.width/2.f);
	mloc.width = nose.width / 0.61f;
	mloc.y = (nose.y + nose.height/2.f) - 0.58*mloc.width;
	mloc.sigmax = 0.056888 * mloc.width;
	mloc.sigmay = 0.093718 * mloc.width;
	mloc.sigmawidth = 0.104735 * mloc.width;
	model.push_back(mloc);
}

/**
 * Retrieves the estimated model location given a single face classifier output (based on trained model)
 * face: classifier output
 * model: where to add estimated face model
 */
void addModelLocationFromFace(const Rect face, vector<ModelLocation>& model){
	ModelLocation mloc;
	mloc.x = (face.x + face.width/2.f);
	mloc.width = face.width / 2.45f;
	mloc.y = (face.y + face.height/2.f) - 0.28*mloc.width;
	mloc.sigmax = 0.040364 * mloc.width;
	mloc.sigmay = 0.030532 * mloc.width;
	mloc.sigmawidth = 0.117894 * mloc.width;
	model.push_back(mloc);
}

/**
 * Computes the average model
 */
void averageModel(const vector<ModelLocation>& models, const int * m, const int cnt, ModelLocation& average){
	double invsigmaxsum = 0, invsigmaysum = 0, invsigmawidthsum = 0;
	for (int i = 0; i < cnt; i++){
		invsigmaxsum += 1 / models[m[i]].sigmax;
		invsigmaysum += 1 / models[m[i]].sigmay;
		invsigmawidthsum += 1 / models[m[i]].sigmawidth;
	}
	average.x = 0; average.y = 0, average.width = 0;
	for (int i = 0; i < cnt; i++){
		average.x += models[m[i]].x * (1 / models[m[i]].sigmax) / invsigmaxsum;
		average.y += models[m[i]].y * (1 / models[m[i]].sigmay) / invsigmaysum;
		average.width += models[m[i]].width * (1 / models[m[i]].sigmawidth) / invsigmawidthsum;
	}
}

/**
 * Evaluates for a set of selected models the likeliness of that model
 *
 * model: list of models
 * m: array of selected models (we consider m[0]..m[cnt-1] only
 * cnt: number of models in m
 * best: updated best model
 * bestVal: best distance (less is better)
 */
void evaluateModel(const vector<ModelLocation>& models, int * m, int cnt, int * bestm, int& bestcnt, float& bestVal){
	// calculate average model
	ModelLocation average;
	averageModel(models,m,cnt,average);

	// now estimate probability of estimated model
	double support = 1;
	for (int i=0; i < cnt; i++) {
		// support zwischen 0 und 2: ok sonst eher unwahrscheinlich dass average model passt
		support += max(
				max((abs(models[m[i]].x -average.x) / (models[m[i]].sigmax)), (abs(models[m[i]].y -average.y) / (models[m[i]].sigmay))),
				(abs(models[m[i]].width -average.width) / (models[m[i]].sigmawidth)));
	}
	support = sqrt(support) / cnt;
	if (support < bestVal){
		bestVal = support;
		for (int i=0; i < cnt; i++) bestm[i] = m[i];
		bestcnt = cnt;
	}
	//for (int i=0; i < cnt; i++) printf("%i, ", m[i]);
	//printf(" support: %f",support);
	//printf("\n");
}

/**
 * Evaluates all possible combinations of single classifiers and finds the best-fitting average model
 *
 * models: all ModelLocations for each fcassifier (trained average location based on classifier outcome)
 * subModelSep: separators (indices) of individual submodel types within models (eye, eyepair, nose, etc)
 * bestm: array of the best model indices selected
 * bestcnt: number of valid items in bestm
 * bestVal: best corresponding result value (the smaller, the better)
 * best: best average model location
 */
void evaluateAllModels(const vector<ModelLocation>& models, int * subModelSep, int * bestm, int& bestcnt, float& bestVal){
	int m[5]; // selected submodels
	const int faceStart = subModelSep[0], eyepairStart = subModelSep[1], noseStart =  subModelSep[2], leftEyeStart = subModelSep[3], rightEyeStart = subModelSep[4], modelStop = subModelSep[5];
	for (unsigned int sel = 1; sel < (1 << 5); sel++){
		if ((sel & (1<<0)) != 0){ // face is selected
			if (eyepairStart == faceStart) continue; // no face exists
			m[0] = faceStart;
			if ((sel & (1<<1)) != 0){ // eyepair is selected
				for (int ep = eyepairStart; ep < noseStart; ep++){
					m[1] = ep;
					if ((sel & (1<<2)) != 0){ // nose is selected
						for (int ns = noseStart; ns < leftEyeStart; ns++){
							m[2] = ns;
							if ((sel & (1<<3)) != 0){ // left eye is selected
								for (int le = leftEyeStart; le < rightEyeStart; le++){
									m[3] = le;
									if ((sel & (1<<4)) != 0){ // right eye is selected
										for (int re = rightEyeStart; re < modelStop; re++){
											if (re-rightEyeStart == le - leftEyeStart) continue; // same eyes
											m[4] = re;
											// auswertung m0..4 (face, eyepair, nose, lefteye, righteye)
											evaluateModel(models, m, 5, bestm, bestcnt, bestVal);
										}
									}
									else {
										// auswertung m0..3 (face, eyepair, nose, lefteye)
										evaluateModel(models, m, 4, bestm, bestcnt, bestVal);
									}
								}
							}
							else {
								if ((sel & (1<<4)) != 0){ // right eye is selected
									for (int re = rightEyeStart; re < modelStop; re++){
										m[3] = re;
										// auswertung m0..3 (face, eyepair, nose, righteye)
										evaluateModel(models, m, 4, bestm, bestcnt, bestVal);
									}
								}
								else {
									// auswertung m0..2 (face, eyepair, nose)
									evaluateModel(models, m, 3, bestm, bestcnt, bestVal);
								}
							}
						}
					}
					else {
						if ((sel & (1<<3)) != 0){ // left eye is selected
							for (int le = leftEyeStart; le < rightEyeStart; le++){
								m[2] = le;
								if ((sel & (1<<4)) != 0){ // right eye is selected
									for (int re = rightEyeStart; re < modelStop; re++){
										if (re-rightEyeStart == le - leftEyeStart) continue; // same eyes
										m[3] = re;
										// auswertung m0..3 (face, eyepair, lefteye, righteye)
										evaluateModel(models, m, 4, bestm, bestcnt, bestVal);
									}
								}
								else {
									// auswertung m0..2 (face, eyepair, lefteye)
									evaluateModel(models, m, 3, bestm, bestcnt, bestVal);
								}
							}
						}
						else {
							if ((sel & (1<<4)) != 0){ // right eye is selected
								for (int re = rightEyeStart; re < modelStop; re++){
									m[2] = re;
									// auswertung m0..2 (face, eyepair, righteye)
									evaluateModel(models, m, 3, bestm, bestcnt, bestVal);
								}
							}
							else {
								// auswertung m0..1 (face, eyepair)
								evaluateModel(models, m, 2, bestm, bestcnt, bestVal);
							}
						}
					}
				}
			}
			else { // no eyepair
				if ((sel & (1<<2)) != 0){ // nose is selected
					for (int ns = noseStart; ns < leftEyeStart; ns++){
						m[1] = ns;
						if ((sel & (1<<3)) != 0){ // left eye is selected
							for (int le = leftEyeStart; le < rightEyeStart; le++){
								m[2] = le;
								if ((sel & (1<<4)) != 0){ // right eye is selected
									for (int re = rightEyeStart; re < modelStop; re++){
										if (re-rightEyeStart == le - leftEyeStart) continue; // same eyes
										m[3] = re;
										// auswertung m0..3 (face, nose, lefteye, righteye)
										evaluateModel(models, m, 4, bestm, bestcnt, bestVal);
									}
								}
								else {
									// auswertung m0..2 (face, nose, lefteye)
									evaluateModel(models, m, 3, bestm, bestcnt, bestVal);
								}
							}
						}
						else {
							if ((sel & (1<<4)) != 0){ // right eye is selected
								for (int re = rightEyeStart; re < modelStop; re++){
									m[2] = re;
									// auswertung m0..2 (face, nose, righteye)
									evaluateModel(models, m, 3, bestm, bestcnt, bestVal);
								}
							}
							else {
								// auswertung m0..1 (face, nose)
								evaluateModel(models, m, 2, bestm, bestcnt, bestVal);
							}
						}
					}
				}
				else {
					if ((sel & (1<<3)) != 0){ // left eye is selected
						for (int le = leftEyeStart; le < rightEyeStart; le++){
							m[1] = le;
							if ((sel & (1<<4)) != 0){ // right eye is selected
								for (int re = rightEyeStart; re < modelStop; re++){
									if (re-rightEyeStart == le - leftEyeStart) continue; // same eyes
									m[2] = re;
									// auswertung m0..2 (face, lefteye, righteye)
									evaluateModel(models, m, 3, bestm, bestcnt, bestVal);
								}
							}
							else {
								// auswertung m0..1 (face, lefteye)
								evaluateModel(models, m, 2, bestm, bestcnt, bestVal);
							}
						}
					}
					else {
						if ((sel & (1<<4)) != 0){ // right eye is selected
							for (int re = rightEyeStart; re < modelStop; re++){
								m[1] = re;
								// auswertung m0..1 (face, righteye)
								evaluateModel(models, m, 2, bestm, bestcnt, bestVal);
							}
						}
						else {
							// auswertung m0 (face)
							evaluateModel(models, m, 1, bestm, bestcnt, bestVal);
						}
					}
				}
			}
		}
		else {
			if ((sel & (1<<1)) != 0){ // eyepair is selected
				for (int ep = eyepairStart; ep < noseStart; ep++){
					m[0] = ep;
					if ((sel & (1<<2)) != 0){ // nose is selected
						for (int ns = noseStart; ns < leftEyeStart; ns++){
							m[1] = ns;
							if ((sel & (1<<3)) != 0){ // left eye is selected
								for (int le = leftEyeStart; le < rightEyeStart; le++){
									m[2] = le;
									if ((sel & (1<<4)) != 0){ // right eye is selected
										for (int re = rightEyeStart; re < modelStop; re++){
											if (re-rightEyeStart == le - leftEyeStart) continue; // same eyes
											m[3] = re;
											// auswertung m0..3 (eyepair, nose, lefteye, righteye)
											evaluateModel(models, m, 4, bestm, bestcnt, bestVal);
										}
									}
									else {
										// auswertung m0..2 (eyepair, nose, lefteye)
										evaluateModel(models, m, 3, bestm, bestcnt, bestVal);
									}
								}
							}
							else {
								if ((sel & (1<<4)) != 0){ // right eye is selected
									for (int re = rightEyeStart; re < modelStop; re++){
										m[2] = re;
										// auswertung m0..2 (eyepair, nose, righteye)
										evaluateModel(models, m, 3, bestm, bestcnt, bestVal);
									}
								}
								else {
									// auswertung m0..1 (eyepair, nose)
									evaluateModel(models, m, 2, bestm, bestcnt, bestVal);
								}
							}
						}
					}
					else {
						if ((sel & (1<<3)) != 0){ // left eye is selected
							for (int le = leftEyeStart; le < rightEyeStart; le++){
								m[1] = le;
								if ((sel & (1<<4)) != 0){ // right eye is selected
									for (int re = rightEyeStart; re < modelStop; re++){
										if (re-rightEyeStart == le - leftEyeStart) continue; // same eyes
										m[2] = re;
										// auswertung m0..2 (eyepair, lefteye, righteye)
										evaluateModel(models, m, 3, bestm, bestcnt, bestVal);
									}
								}
								else {
									// auswertung m0..1 (eyepair, lefteye)
									evaluateModel(models, m, 2, bestm, bestcnt, bestVal);
								}
							}
						}
						else {
							if ((sel & (1<<4)) != 0){ // right eye is selected
								for (int re = rightEyeStart; re < modelStop; re++){
									m[1] = re;
									// auswertung m0..1 (eyepair, righteye)
									evaluateModel(models, m, 2, bestm, bestcnt, bestVal);
								}
							}
							else {
								// auswertung m0 (eyepair)
								evaluateModel(models, m, 1, bestm, bestcnt, bestVal);
							}
						}
					}
				}
			}
			else { // no eyepair
				if ((sel & (1<<2)) != 0){ // nose is selected
					for (int ns = noseStart; ns < leftEyeStart; ns++){
						m[0] = ns;
						if ((sel & (1<<3)) != 0){ // left eye is selected
							for (int le = leftEyeStart; le < rightEyeStart; le++){
								m[1] = le;
								if ((sel & (1<<4)) != 0){ // right eye is selected
									for (int re = rightEyeStart; re < modelStop; re++){
										if (re-rightEyeStart == le - leftEyeStart) continue; // same eyes
										m[2] = re;
										// auswertung m0..2 (nose, lefteye, righteye)
										evaluateModel(models, m, 3, bestm, bestcnt, bestVal);
									}
								}
								else {
									// auswertung m0..1 (nose, lefteye)
									evaluateModel(models, m, 2, bestm, bestcnt, bestVal);
								}
							}
						}
						else {
							if ((sel & (1<<4)) != 0){ // right eye is selected
								for (int re = rightEyeStart; re < modelStop; re++){
									m[1] = re;
									// auswertung m0..1 (nose, righteye)
									evaluateModel(models, m, 2, bestm, bestcnt, bestVal);
								}
							}
							else {
								// auswertung m0 (nose)
								evaluateModel(models, m, 1, bestm, bestcnt, bestVal);
							}
						}
					}
				}
				else {
					if ((sel & (1<<3)) != 0){ // left eye is selected
						for (int le = leftEyeStart; le < rightEyeStart; le++){
							m[0] = le;
							if ((sel & (1<<4)) != 0){ // right eye is selected
								for (int re = rightEyeStart; re < modelStop; re++){
									if (re-rightEyeStart == le - leftEyeStart) continue; // same eyes
									m[1] = re;
									// auswertung m0..1 (lefteye, righteye)
									evaluateModel(models, m, 2, bestm, bestcnt, bestVal);
								}
							}
							else {
								// auswertung m0 (lefteye)
								evaluateModel(models, m, 1, bestm, bestcnt, bestVal);
							}
						}
					}
					else {
						if ((sel & (1<<4)) != 0){ // right eye is selected
							for (int re = rightEyeStart; re < modelStop; re++){
								m[0] = re;
								// auswertung m0 (righteye)
								evaluateModel(models, m, 1, bestm, bestcnt, bestVal);
							}
						}
						else {
							// no model
						}
					}
				}
			}
		}
		//printf("selected: %i (%i %i %i %i %i)", sel, (sel & (1<<0)),(sel & (1<<1)),(sel & (1<<2)),(sel & (1<<3)),(sel & (1<<4)));
	}
}

/**
 * This method computes the position of face parts combining the best-fitting average model and corresponding selected sub-models
 *
 * allParts: rectangles of each classifier (classifier outcome)
 * separators: separators (indices) of individual submodel types within models (eye, eyepair, nose, etc)
 * bestm: array of the best model indices selected
 * bestcnt: number of valid items in bestm
 * best: best average model location
 * parts: (result) computed parts based on model and classifiers
 */
void partsFromModels(const vector<Rect>& allParts, int * separators, int * bestm, const int& bestcnt, const ModelLocation& best, Rect * parts){
	//printf("width: %f\n",best.width);
	int bestidx = 0;
	if (bestidx < bestcnt && bestm[bestidx] < separators[1]){ // face in bestm
		parts[0] = allParts[bestm[bestidx++]];
	}
	else { // generate from average model
		parts[0].width = best.width * 2.45f;
		parts[0].height = parts[0].width;
		parts[0].x = best.x - (parts[0].width * 0.5f);
		parts[0].y = best.y - (parts[0].height * 0.5f) + (best.width * 0.28f);

	}
	//printf("0.x: %i\n",parts[0].x);
	if (bestidx < bestcnt && bestm[bestidx] < separators[2]){ // eyepair in bestm
		parts[1] = allParts[bestm[bestidx++]];
	}
	else { // generate from average model
		parts[1].width = best.width * 1.69f;
		parts[1].height = parts[1].width * (11.f / 45.f);
		parts[1].x = best.x - (parts[1].width * 0.5f);
		parts[1].y = best.y - (parts[1].height * 0.5f);
	}
	if (bestidx < bestcnt && bestm[bestidx] < separators[3]){ // noses in bestm
		parts[2] = allParts[bestm[bestidx++]];
	}
	else { // generate from average model
		parts[2].width = best.width * 0.61f;
		parts[2].height = parts[2].width * (15.f / 18.f);
		parts[2].x = best.x - (parts[2].width * 0.5f);
		parts[2].y = best.y - (parts[2].height * 0.5f) + 0.58f * best.width;
	}
	if (bestidx < bestcnt && bestm[bestidx] < separators[4]){ // left eye in bestm
		parts[3] = allParts[bestm[bestidx++]];
	}
	else { // generate from average model
		parts[3].width = best.width * 0.55f;
		parts[3].height = parts[3].width;
		parts[3].x = best.x - (parts[3].width * 0.5f) - (best.width * 0.5f);
		parts[3].y = best.y - (parts[3].height * 0.5f);
	}
	if (bestidx < bestcnt && bestm[bestidx] < separators[5]){ // right eye in bestm
		parts[4] = allParts[bestm[bestidx++]];
	}
	else { // generate from average model
		parts[4].width = best.width * 0.55f;
		parts[4].height = parts[4].width;
		parts[4].x = best.x - (parts[4].width * 0.5f) + (best.width * 0.5f);
		parts[4].y = best.y - (parts[4].height * 0.5f);
	}
}

/**
 * enhancedPDS: matrix region of interest representing the face
 * allParts: resulting detected Parts of detection mode
 */
void detectPartsInFace(const Mat& enhancedPDS, vector<Rect>& allParts, vector<ModelLocation>& models, int * subModelSep, CascadeClassifier& eyeCascade, CascadeClassifier& eyepairCascade, CascadeClassifier& noseCascade, bool isFace = true){
	vector<Rect> eyepairsPDS;
	eyepairCascade.detectMultiScale( enhancedPDS, eyepairsPDS, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, Size(45, 11) );
	vector<Rect> eyesPDS;
	eyeCascade.detectMultiScale( enhancedPDS, eyesPDS, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(20, 20) );
	vector<Rect> nosesPDS;
	noseCascade.detectMultiScale( enhancedPDS, nosesPDS, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(20, 20) );
	// build a model location for each classifier result
	subModelSep[0] = models.size();
	if (isFace) {
		addModelLocationFromFace(Rect(0,0,enhancedPDS.cols,enhancedPDS.rows),models);
		allParts.push_back(Rect(0,0,enhancedPDS.cols,enhancedPDS.rows));
	}
	subModelSep[1] = models.size();
	if (eyepairsPDS.size() > 0) {
		for (vector<Rect>::iterator it = eyepairsPDS.begin(), eit = eyepairsPDS.end(); it < eit; it++){
			addModelLocationFromEyepair(*it,models);
			allParts.push_back(*it);
		}
	}
	subModelSep[2] = models.size();
	if (nosesPDS.size() > 0) {
		for (vector<Rect>::iterator it = nosesPDS.begin(), eit = nosesPDS.end(); it < eit; it++){
			addModelLocationFromNose(*it,models);
			allParts.push_back(*it);
		}
	}
	subModelSep[3] = models.size();
	if (eyesPDS.size() > 0) {
		for (vector<Rect>::iterator it = eyesPDS.begin(), eit = eyesPDS.end(); it < eit; it++){
			addModelLocationFromLeftEye(*it,models);
			allParts.push_back(*it);
		}
	}
	subModelSep[4] = models.size();
	if (eyesPDS.size() > 0) {
		for (vector<Rect>::iterator it = eyesPDS.begin(), eit = eyesPDS.end(); it < eit; it++){
			addModelLocationFromRightEye(*it,models);
			allParts.push_back(*it);
		}
	}
	subModelSep[5] = models.size();
}

/**
 * Computes the integer logarithm dualis (log2) of the input value
 * value: integer x
 * returning ld(x)
 */
inline unsigned int ilog2(unsigned int value)
{
    unsigned int f=0, s=32;
    while(s) {
        s>>=1;
        if( value >= 1u<<(f+s) ) f+=s;
    }
    return f;
}

void scaleRect(Rect& dst, const Rect& src, int scaleDiff){
	if (scaleDiff >= 0){
		dst.x = src.x>>scaleDiff;
		dst.y = src.y>>scaleDiff;
		dst.width = src.width>>scaleDiff;
		dst.height = src.height>>scaleDiff;
	}
	else if (scaleDiff < 0){
		dst.x = src.x<<(-scaleDiff);
		dst.y = src.y<<(-scaleDiff);
		dst.width = src.width<<(-scaleDiff);
		dst.height = src.height<<(-scaleDiff);
	}
}

/**
 * How many times width and height may be downscaled (positive result) or upscaled (negative result) by factor two,
 * such that the size is greater than or equal to dstWidth and dstHeight
 */
int countScale(const unsigned int srcWidth, const unsigned int srcHeight, const unsigned int dstWidth, const unsigned int dstHeight){
	int result = 0;
	const unsigned int tWidth = 2*dstWidth;
	const unsigned int tHeight = 2*dstHeight;
	unsigned int sWidth = srcWidth;
	unsigned int sHeight = srcHeight;
	while (sWidth >= tWidth && sHeight >= tHeight){
		sWidth >>= 1;
		sHeight >>= 1;
		result++;
	}
	while (sWidth < dstWidth || sHeight < dstHeight){
		sWidth <<= 1;
		sHeight <<= 1;
		result--;
	}
	return result;
}

void assertPyrDepth(vector<Mat>& pyr, const unsigned int depth){
	while (pyr.size() < depth + 1){
		Mat tmp;
		pyrDown(pyr[pyr.size()-1],tmp);
		pyr.push_back(tmp);
	}
}

/**
 * Constructs a gaussian pyramid by iteratively applying pyrDown to an input image as long as the resulting image size is above a minimum
 * src: input image
 * pyr: image pyramid
 * minSize: minimum size
 */
void makePyr(const Mat& src, vector<Mat>& pyr, int minSize = 256){
	Mat tmp = src;
	pyr.push_back(src);
	int size = minSize*2;
	while (tmp.rows >= size || tmp.cols >= size){
		// first generate intermediate resolutions
		Mat tmp2;
		pyrDown(tmp,tmp2);
		tmp = tmp2;
		pyr.push_back(tmp);
	}
}

/**
 * Creates an image of the face part (i.e. the rectangle describing the location of the face part)
 * pyr: the image pyramid
 * dst: the destination image (is created)
 * part: the rectangle describing the extractable part
 * PDS: the rectangle describing the extraction space
 * partDetectionScale: the detection scale for the PDS rectangle
 * tWidth: target minimum width of the part
 * tHeight: the target minimum height of the part
 */
void part2Mat(vector<Mat>& pyr, Mat& dst, const Rect& part, const Rect& PDS, const int partDetectionScale, const int tWidth, const int tHeight){
	Mat outMat, outCopyFrom, outCopyTo;
	Rect outRect, scaledRect, toRect;
	int outScale;
	//printf("dscale: %i\n",partDetectionScale);
	//printf("PDS (x,y,w,h): %i %i %i %i\n",PDS.x, PDS.y,PDS.width,PDS.height);

	//printf("part (x,y,w,h): %i %i %i %i\n",part.x, part.y,part.width,part.height);
	outRect.x = (PDS.x + part.x)<<partDetectionScale;
	outRect.y = (PDS.y +part.y)<<partDetectionScale;
	outRect.width = part.width<<partDetectionScale;
	outRect.height = part.height<<partDetectionScale;
	outScale = std::max(0,countScale(outRect.width,outRect.height,tWidth,tHeight));

	//
	scaledRect.x = outRect.x >> outScale;
	scaledRect.y = outRect.y >> outScale;
	scaledRect.width = 	outRect.width >> outScale;
	scaledRect.height = outRect.height >> outScale;

	assertPyrDepth(pyr,outScale);
	Rect from = scaledRect & Rect(0,0,pyr[outScale].cols,pyr[outScale].rows);
	//printf("from (x,y,w,h): %i %i %i %i\n",from.x, from.y,from.width,from.height);
	Rect to = Rect(from.x - scaledRect.x,from.y - scaledRect.y,from.width,from.height);
	//printf("to (x,y,w,h): %i %i %i %i\n",to.x, to.y,to.width,to.height);
	/*
	printf("outrect (x,y,w,h): %i %i %i %i\n",outRect.x, outRect.y,outRect.width,outRect.height);

	printf("outscale: %i\n",outScale);
	scaledRect.x = min(max(0,outRect.x >> outScale),pyr[outScale].cols);
	scaledRect.y = min(max(0,outRect.y >> outScale),pyr[outScale].rows);
	scaledRect.width = 	min(max(0,outRect.width >> outScale),pyr[outScale].cols-scaledRect.x);
	scaledRect.height = min(max(0,outRect.height >> outScale),pyr[outScale].rows-scaledRect.y);
	printf("from (x,y,w,h): %i %i %i %i\n",scaledRect.x, scaledRect.y,scaledRect.width,scaledRect.height);

	printf("should be %i\n", (outRect.x >> outScale));
	toRect.x = scaledRect.x - (outRect.x >> outScale);
	toRect.y = scaledRect.y - (outRect.y >> outScale);
	toRect.width = scaledRect.width;
	toRect.height = scaledRect.height;
	printf("to (x,y,w,h): %i %i %i %i\n",toRect.x, toRect.y,toRect.width,toRect.height);

	dst.create((outRect.height >> outScale),(outRect.width >> outScale),CV_8UC1);
	*/
	dst.create(scaledRect.height, scaledRect.width,CV_8UC1);
	//printf("total (w,h): %i %i",dst.cols,dst.rows);
	dst.setTo(0);
	Mat fromROI(pyr[outScale],from);
	Mat toROI(dst,to);
	fromROI.copyTo(toROI);
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
			cmdCheckOpts(cmd,"-i|-o|-c|-s|-q|-t|-dr|-tr|-lt");
			cmdCheckOptExists(cmd,"-i");
			cmdCheckOptSize(cmd,"-i",1);
			string inFiles = cmdGetPar(cmd,"-i");
			string faceFiles, leyeFiles, reyeFiles;
			if (cmdGetOpt(cmd,"-o") != 0){
				cmdCheckOptSize(cmd,"-o",3);
				faceFiles = cmdGetPar(cmd,"-o",0);
				leyeFiles = cmdGetPar(cmd,"-o",1);
				reyeFiles = cmdGetPar(cmd,"-o",2);
			}
			string coordFile;
			bool saveCoords = false;
			if (cmdGetOpt(cmd,"-c") != 0){
				cmdCheckOptSize(cmd,"-c",1);
				coordFile = cmdGetPar(cmd,"-c",0);
				saveCoords = true;
			}
            /*unused
			int faceSize = 0, eyeSize = 0;
			if (cmdGetOpt(cmd,"-s") != 0){
				cmdCheckOptSize(cmd,"-s",2);
				faceSize = cmdGetParInt(cmd,"-s",0);
				eyeSize = cmdGetParInt(cmd,"-s",1);
			}
            */
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
			string detectFiles;
			if (cmdGetOpt(cmd,"-dr") != 0){
				cmdCheckOptSize(cmd,"-dr",1);
				detectFiles = cmdGetPar(cmd,"-dr");
			}
			string trkresFiles;
			if (cmdGetOpt(cmd,"-tr") != 0){
				cmdCheckOptSize(cmd,"-tr",1);
				trkresFiles = cmdGetPar(cmd,"-tr");
			}
			int lt = 1;
			if (cmdGetOpt(cmd,"-lt") != 0){
				cmdCheckOptSize(cmd,"-lt",1);
				lt = cmdGetParInt(cmd,"-lt");
			}

			boost::filesystem::path progPath(boost::filesystem::initial_path<boost::filesystem::path>());
			progPath = boost::filesystem::system_complete(boost::filesystem::path(argv[0])).parent_path();
			string faceCascadeFile = (progPath / "haarcascade_frontalface_default.xml").string();
			string eyeCascadeFile = (progPath / "haarcascade_eye_tree_eyeglasses.xml").string();
			string eyepairCascadeFile = (progPath / "haarcascade_mcs_eyepair_big.xml").string();
			string noseCascadeFile = (progPath / "haarcascade_mcs_nose.xml").string();

			// starting routine
			Timing timing(1,quiet);
			CascadeClassifier faceCascade, eyeCascade, eyepairCascade, noseCascade;
			if (!quiet) printf("Loading face cascade '%s' ...\n", faceCascadeFile.c_str());;
			if (!faceCascade.load(faceCascadeFile))  CV_Error(CV_StsError,"Could not load face classifier cascade");
			if (!quiet) printf("Loading eye cascade '%s' ...\n", eyeCascadeFile.c_str());;
			if (!eyeCascade.load(eyeCascadeFile))  CV_Error(CV_StsError,"Could not load eye classifier cascade");
			if (!quiet) printf("Loading eyepair cascade '%s' ...\n", eyepairCascadeFile.c_str());;
			if (!eyepairCascade.load(eyepairCascadeFile))  CV_Error(CV_StsError,"Could not load eyepair classifier cascade");
			if (!quiet) printf("Loading nose cascade '%s' ...\n", noseCascadeFile.c_str());;
			if (!noseCascade.load(noseCascadeFile))  CV_Error(CV_StsError,"Could not load eyepair classifier cascade");

			vector<string> files;
			patternToFiles(inFiles,files);
			CV_Assert(files.size() > 0);
			timing.total = files.size();
			vector<float> leyex, leyey, leyew, reyex, reyey, reyew, ediffx, ediffy, ediffw;

			ofstream coords;
			if (saveCoords){
				coords.open(coordFile.c_str(),ios::out | ios::trunc);
				if (!coords.is_open()) CV_Error(CV_StsError,"Could not open coordinate output file '" + coordFile + "'");
			}
			int detectSize = 256;
			for (vector<string>::iterator inFile = files.begin(); inFile != files.end(); ++inFile, timing.progress++){
				if (!quiet) printf("Loading image '%s' ...\n", (*inFile).c_str());
				vector<Mat> pyr;
				pyr.push_back(imread(*inFile, CV_LOAD_IMAGE_GRAYSCALE));
				CV_Assert(pyr[0].data != 0);
				if (!quiet) printf("Resizing and enhancing image ...\n");
				// scales: original scale, scale at which faces are detected, scale at which face parts are detected
				int faceDetectionScale, partDetectionScale; // referring to pyr index
				Rect faceFS(0,0,0,0), leyeES(0,0,0,0), reyeES(0,0,0,0);
				faceDetectionScale = max(0,countScale(pyr[0].cols,pyr[0].rows,detectSize,detectSize));
				assertPyrDepth(pyr,faceDetectionScale);
				Mat enhancedFDS;
				equalizeHist(pyr[faceDetectionScale],enhancedFDS);
				if (!quiet) printf("Detecting faces ...\n");
				vector<Rect> facesFDS;
				faceCascade.detectMultiScale( enhancedFDS, facesFDS, 1.1, 2, 0|CV_HAAR_FIND_BIGGEST_OBJECT|CV_HAAR_SCALE_IMAGE, Size(20, 20) );
				if (!quiet) printf("Found %li face(s).\n",facesFDS.size());
				// work with image patch, if necessary, else use whole image
				Mat enhancedPDS;
				Rect PDS;
				vector<ModelLocation> models;
				vector<Rect> allParts;
				int subModelSep[6];
				int bestm[5];
				int bestcnt;
				float bestVal = INT_MAX;
				ModelLocation average;
				Rect parts[5];
				if (facesFDS.size() > 0){
					if (!quiet) printf("Detecting face parts in face ...\n");
					int scaleDiff = std::min(faceDetectionScale,-countScale(facesFDS[0].width,facesFDS[0].height,256,256));
					partDetectionScale = faceDetectionScale - scaleDiff;
					PDS.x = facesFDS[0].x<<scaleDiff;
					PDS.y = facesFDS[0].y << scaleDiff;
					PDS.width = facesFDS[0].width << scaleDiff;
					PDS.height = facesFDS[0].height << scaleDiff;
					Mat roi(pyr[partDetectionScale],PDS);
					equalizeHist(roi,enhancedPDS);
					detectPartsInFace(enhancedPDS,allParts,models,subModelSep,eyeCascade,eyepairCascade,noseCascade,true);
					if (!quiet) printf("Found %i eyepair(s).\n",subModelSep[2] - subModelSep[1]);
					if (!quiet) printf("Found %i nose(s).\n",subModelSep[3] - subModelSep[2]);
					if (!quiet) printf("Found %i eye(s).\n",subModelSep[4] - subModelSep[3]);
					// find best model location
					// test all combinations of models
					evaluateAllModels(models, subModelSep, bestm, bestcnt, bestVal);
					averageModel(models,bestm,bestcnt,average);
					partsFromModels(allParts, subModelSep, bestm, bestcnt, average, parts);
					if (!quiet) {
						printf("Best combination: ");
						for (int i=0; i < bestcnt; i++) printf("%i, ", bestm[i]);
						printf(" (Score: %f)\n",bestVal);
					}
				}
				if (bestVal >= 1){ // lets assume the entire image contains and represents one face
					models.clear(); allParts.clear(); bestVal = INT_MAX;
					if (!quiet) printf("Detecting face parts in entire image ...\n");
					partDetectionScale = faceDetectionScale;
					enhancedPDS = enhancedFDS;
					PDS.x = 0;
					PDS.y = 0;
					PDS.width = enhancedPDS.cols;
					PDS.height = enhancedPDS.rows;
					detectPartsInFace(enhancedPDS,allParts,models,subModelSep,eyeCascade,eyepairCascade,noseCascade,true);
					if (!quiet) printf("Found %i eyepair(s).\n",subModelSep[2] - subModelSep[1]);
					if (!quiet) printf("Found %i nose(s).\n",subModelSep[3] - subModelSep[2]);
					if (!quiet) printf("Found %i eye(s).\n",subModelSep[4] - subModelSep[3]);
					// find best model location
					// test all combinations of models
					evaluateAllModels(models, subModelSep, bestm, bestcnt, bestVal);
					averageModel(models,bestm,bestcnt,average);
					partsFromModels(allParts, subModelSep, bestm, bestcnt, average, parts);
					if (!quiet) {
						printf("Best combination: ");
						for (int i=0; i < bestcnt; i++) printf("%i, ", bestm[i]);
						printf(" (Score: %f)\n",bestVal);
					}
				}
				if (saveCoords){
					int leftEyeX = ((PDS.x + parts[3].x)<<partDetectionScale) + ((parts[3].width)<<partDetectionScale)/2 ;
					int leftEyeY = ((PDS.y + parts[3].y)<<partDetectionScale) + ((parts[3].height)<<partDetectionScale)/2 ;
					int rightEyeX = ((PDS.x + parts[4].x)<<partDetectionScale) + ((parts[4].width)<<partDetectionScale)/2 ;
					int rightEyeY = ((PDS.y + parts[4].y)<<partDetectionScale) + ((parts[4].height)<<partDetectionScale)/2 ;
					int noseX = ((PDS.x + parts[2].x)<<partDetectionScale) + ((parts[2].width)<<partDetectionScale)/2 ;
					int noseY = ((PDS.y + parts[2].y)<<partDetectionScale) + ((parts[2].height)<<partDetectionScale)/2 ;
					coords << (*inFile) << ": " << leftEyeX << "," <<  leftEyeY << ","<< rightEyeX << "," <<  rightEyeY << ","<< noseX << "," << noseY << "\n";
				}
				// WRITE FACE
				// face in original coordinates
				if (!faceFiles.empty()){
					Mat outMat;
					part2Mat(pyr,outMat, parts[0], PDS, partDetectionScale, 256, 256);
					string faceFile;
					patternFileRename(inFiles,faceFiles,*inFile,faceFile);
					if (!quiet) printf("Storing face image '%s' ...\n", faceFile.c_str());
					if (!imwrite(faceFile,outMat)) CV_Error(CV_StsError,"Could not save image '" + faceFile + "'");
					part2Mat(pyr,outMat, parts[3], PDS, partDetectionScale, 512, 512);
					string leyeFile;
					patternFileRename(inFiles,leyeFiles,*inFile,leyeFile);
					if (!quiet) printf("Storing left eye image '%s' ...\n", leyeFile.c_str());
					if (!imwrite(leyeFile,outMat)) CV_Error(CV_StsError,"Could not save image '" + leyeFile + "'");
					part2Mat(pyr,outMat, parts[4], PDS, partDetectionScale, 512, 512);
					string reyeFile;
					patternFileRename(inFiles,reyeFiles,*inFile,reyeFile);
					if (!quiet) printf("Storing right eye image '%s' ...\n", reyeFile.c_str());
					if (!imwrite(reyeFile,outMat)) CV_Error(CV_StsError,"Could not save image '" + reyeFile + "'");
				}

				//printf("face resolution %i (%i %i)",countScale(outRect.width,outRect.height,256,256), outRect.width,outRect.height);
//				Rect((PDS.x + parts[0].x)<<partDetectionScale,(PDS.y +parts[0].y)<<partDetectionScale,parts[0].width<<partDetectionScale,parts[0].height<<partDetectionScale);

				if (!detectFiles.empty()){
					Mat visual;
					cvtColor(pyr[0],visual,CV_GRAY2BGR);
					vector<Rect>::iterator it = allParts.begin();
					if (subModelSep[1] > subModelSep[0]){
						for(vector<Rect>::iterator itEnd = it + (subModelSep[1] - subModelSep[0]); it < itEnd; it++){
							rectangle(visual,Rect((PDS.x + (*it).x)<<partDetectionScale,(PDS.y +(*it).y)<<partDetectionScale,(*it).width<<partDetectionScale,(*it).height<<partDetectionScale),Scalar(0,255,255,0),lt);
						}
					}
					if (subModelSep[2] > subModelSep[1]){
						for(vector<Rect>::iterator itEnd = it + (subModelSep[2] - subModelSep[1]); it < itEnd; it++){
							rectangle(visual,Rect((PDS.x + (*it).x)<<partDetectionScale,(PDS.y +(*it).y)<<partDetectionScale,(*it).width<<partDetectionScale,(*it).height<<partDetectionScale),Scalar(0,0,255,0),lt);
						}
					}
					if (subModelSep[3] > subModelSep[2]){
						for(vector<Rect>::iterator itEnd = it + (subModelSep[3] - subModelSep[2]); it < itEnd; it++){
							rectangle(visual,Rect((PDS.x + (*it).x)<<partDetectionScale,(PDS.y +(*it).y)<<partDetectionScale,(*it).width<<partDetectionScale,(*it).height<<partDetectionScale),Scalar(255,0,0,0),lt);
						}
					}
					if (subModelSep[4] > subModelSep[3]){
						for(vector<Rect>::iterator itEnd = it + (subModelSep[4] - subModelSep[3]); it < itEnd; it++){
							rectangle(visual,Rect((PDS.x + (*it).x)<<partDetectionScale,(PDS.y +(*it).y)<<partDetectionScale,(*it).width<<partDetectionScale,(*it).height<<partDetectionScale),Scalar(0,255,0,0),lt);
						}
					}
					string detectFile;
					patternFileRename(inFiles,detectFiles,*inFile,detectFile);
					if (!quiet) printf("Storing detection result image '%s' ...\n", detectFile.c_str());
					if (!imwrite(detectFile,visual)) CV_Error(CV_StsError,"Could not save image '" + detectFile + "'");
				}
				if (!trkresFiles.empty()){
					Mat visual;
					cvtColor(pyr[0],visual,CV_GRAY2BGR);

					rectangle(visual,Rect((PDS.x + parts[0].x)<<partDetectionScale,(PDS.y +parts[0].y)<<partDetectionScale,parts[0].width<<partDetectionScale,parts[0].height<<partDetectionScale),Scalar(0,255,255,0),lt);
					rectangle(visual,Rect((PDS.x + parts[1].x)<<partDetectionScale,(PDS.y +parts[1].y)<<partDetectionScale,parts[1].width<<partDetectionScale,parts[1].height<<partDetectionScale),Scalar(0,0,255,0),lt);
					rectangle(visual,Rect((PDS.x + parts[2].x)<<partDetectionScale,(PDS.y +parts[2].y)<<partDetectionScale,parts[2].width<<partDetectionScale,parts[2].height<<partDetectionScale),Scalar(255,0,0,0),lt);
					rectangle(visual,Rect((PDS.x + parts[3].x)<<partDetectionScale,(PDS.y +parts[3].y)<<partDetectionScale,parts[3].width<<partDetectionScale,parts[3].height<<partDetectionScale),Scalar(0,255,0,0),lt);
					rectangle(visual,Rect((PDS.x + parts[4].x)<<partDetectionScale,(PDS.y +parts[4].y)<<partDetectionScale,parts[4].width<<partDetectionScale,parts[4].height<<partDetectionScale),Scalar(0,255,0,0),lt);
					string trkresFile;
					patternFileRename(inFiles,trkresFiles,*inFile,trkresFile);
					if (!quiet) printf("Storing segmentation image '%s' ...\n", trkresFile.c_str());
					if (!imwrite(trkresFile,visual)) CV_Error(CV_StsError,"Could not save image '" + trkresFile + "'");
				}
				if (time && timing.update()) timing.print();
			}
			if (saveCoords){
				coords.close();
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
