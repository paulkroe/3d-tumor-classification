# test/Makefile

CXX = g++
CXXFLAGS = -std=c++17 -Wall -D_GLIBCXX_USE_CXX11_ABI=1 `pkg-config --cflags opencv4` -I../src -I../src/libtorch/include -I../src/libtorch/include/torch/csrc/api/include
LDFLAGS = `pkg-config --libs opencv4` -L../src/libtorch/lib -Wl,-rpath=../src/libtorch/lib -ltorch_cpu -ltorch -lc10 -lstdc++fs

TARGETS = main octree_test
OBJS_MAIN = main.o ../src/octree.o
OBJS_OCTREE_TEST = octree_test.o ../src/octree.o

all: $(TARGETS)

main: $(OBJS_MAIN)
	$(CXX) $(CXXFLAGS) $(OBJS_MAIN) -o main $(LDFLAGS)

octree_test: $(OBJS_OCTREE_TEST)
	$(CXX) $(CXXFLAGS) $(OBJS_OCTREE_TEST) -o octree_test $(LDFLAGS)

main.o: main.cpp ../src/octree.h
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

octree_test.o: octree_test.cpp ../src/octree.h
	$(CXX) $(CXXFLAGS) -c octree_test.cpp -o octree_test.o

clean:
	rm -f $(TARGETS) $(OBJS_MAIN) $(OBJS_OCTREE_TEST)

