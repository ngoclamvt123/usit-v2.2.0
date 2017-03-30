CXX=g++

# Linux
#CXXFLAGS=-O3 -Wall -fmessage-length=0 
CXXFLAGS=-g -std=c++11 -Wall -Ofast -march=native -flto -fwhole-program -fmodulo-sched
LINKFLAGS=-s -L/opt/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_contrib -lboost_filesystem -lboost_system -lboost_regex -lopencv_photo

TARGET=TripleA
	
TripleA: TripleA.cpp
	$(CXX) -o TripleA $(CXXFLAGS) TripleA.cpp $(LINKFLAGS)
	
all: ${TARGET}
clean:
	rm ${TARGET}
