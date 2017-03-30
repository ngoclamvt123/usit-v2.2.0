CXX=g++

# Linux
CXXFLAGS=-O3 -std=c++11 -Wall -fmessage-length=0  -s
CXXFLAGS=-g -gdwarf-3 -Wall -std=c++11 -Wformat
LINKFLAGS=-L/opt/local/lib \
		  -lopencv_core \
		  -lopencv_highgui \
		  -lopencv_imgproc \
		  -lopencv_objdetect \
		  -lopencv_contrib \
		  -lopencv_legacy \
		  -lopencv_nonfree \
		  -lopencv_features2d \
		  -lboost_filesystem \
		  -lboost_system \
		  -lboost_regex \
		  -lopencv_photo \
		  -lmatio \


ALLTARGETS=bsif  bsifc

%:%.cpp
	$(CXX) -o $@ $(CXXFLAGS) $< $(LINKFLAGS)

all: ${ALLTARGETS}
clean:
	rm ${ALLTARGETS}
