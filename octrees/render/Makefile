CXX = g++
CXXFLAGS = -std=c++17 -Wall -D_GLIBCXX_USE_CXX11_ABI=1 `pkg-config --cflags opencv4` -I../src
LDFLAGS = `pkg-config --libs opencv4` -lstdc++fs -lsqlite3

TARGET = renderer
SHARED_LIB = librenderer.so
OBJS = renderer.o ../src/octree.o

all: $(TARGET) $(SHARED_LIB)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS)

$(SHARED_LIB): renderer.o ../src/octree.o
	$(CXX) $(CXXFLAGS) -shared -o $(SHARED_LIB) renderer.o ../src/octree.o $(LDFLAGS)

renderer.o: renderer.cpp ../src/octree.h
	$(CXX) $(CXXFLAGS) -fPIC -c renderer.cpp -o renderer.o

../src/octree.o: ../src/octree.cpp ../src/octree.h
	$(CXX) $(CXXFLAGS) -fPIC -c ../src/octree.cpp -o ../src/octree.o

clean:
	rm -f $(TARGET) $(SHARED_LIB) $(OBJS)
