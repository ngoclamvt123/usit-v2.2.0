#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>

using namespace std;

int main(int argc, char* argv[])
{
	ifstream file(argv[1]); // pass file name as argument
	string linebuffer;

	float minValue = 1.0; // return min value from file
	string fileResult; //return file that has min value

	while (file && getline(file, linebuffer)){
		if (linebuffer.length() == 0)
			continue;

		size_t pos = linebuffer.find("=");  // position of "=" in str
		string strvalue = linebuffer.substr(pos+2);

		float result = atof(strvalue.c_str());

		if(result < minValue) {
			minValue = result;

			unsigned first = linebuffer.find("(");
			unsigned last  = linebuffer.find_last_of(")");

			string strnew = linebuffer.substr(first,last-first);
			size_t found = strnew.find_last_of("/\\");
			fileResult = strnew.substr(found+1, 5);
		}
	}
	
	cout <<"Minimum Hamming-Distance: " << minValue << endl;

	if(minValue < 0.4)	{
		cout <<"The best group matching: " << fileResult << endl;
		if(fileResult.compare(argv[2]) == 0)
			cout <<"Past Test" << endl;
		else
			cout <<"Fail Test" << endl;
	}
	else
		cout <<"No file matching" << endl;

	file.close();
	return 0;
}
