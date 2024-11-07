# src/Makefile

CXX = g++
CXXFLAGS = -std=c++17 -Wall -D_GLIBCXX_USE_CXX11_ABI=1 `pkg-config --cflags opencv4` -I./libtorch/include -I./libtorch/include/torch/csrc/api/include
LDFLAGS = `pkg-config --libs opencv4` -L./libtorch/lib -Wl,-rpath=./libtorch/lib -ltorch_cpu -ltorch -lc10 -lstdc++fs

LIBRARY_OBJS = octree.o

all: $(LIBRARY_OBJS)

octree.o: octree.cpp octree.h
	$(CXX) $(CXXFLAGS) -c octree.cpp -o octree.o

clean:
	rm -f $(LIBRARY_OBJS)
