CXX=g++

# Linux
#CXXFLAGS=-O3 -Wall -fmessage-length=0  -s
CXXFLAGS=-g -gdwarf-3 -Wall -std=c++11 -Wformat
LINKFLAGS=-L/opt/local/lib \
		  -lopencv_core \
		  -lopencv_highgui \
		  -lopencv_imgproc \
		  -lopencv_objdetect \
		  -lopencv_contrib \
		  -lopencv_nonfree \
		  -lopencv_features2d \
		  -lboost_filesystem \
		  -lboost_system \
		  -lboost_regex \
		  -lopencv_photo \

ALLTARGETS=lbp lbpc sift siftc surf surfc cg caht wahet gfcf lg hd hdverify qsw ko koc cb cbc cr dct dctc maskcmp ifpp manuseg

%:%.cpp version.h
	$(CXX) -o $@ $(CXXFLAGS) $< $(LINKFLAGS)

all: ${ALLTARGETS}
clean:
	rm ${ALLTARGETS}
