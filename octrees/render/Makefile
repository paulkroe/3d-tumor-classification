# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -D_GLIBCXX_USE_CXX11_ABI=1 `pkg-config --cflags opencv4` -I../src -I../src/libtorch/include -I../src/libtorch/include/torch/csrc/api/include
LDFLAGS = `pkg-config --libs opencv4` -L../src/libtorch/lib -Wl,-rpath=../src/libtorch/lib -ltorch_cpu -ltorch -lc10 -lstdc++fs

# Targets and object files
TARGET = renderer
OBJS = renderer.o ../src/octree.o

# Default target
all: $(TARGET)

# Link object files to create the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS)

# Compile renderer.cpp into an object file
renderer.o: renderer.cpp ../src/octree.h
	$(CXX) $(CXXFLAGS) -c renderer.cpp -o renderer.o

# Compile octree.cpp into an object file
../src/octree.o: ../src/octree.cpp ../src/octree.h
	$(CXX) $(CXXFLAGS) -c ../src/octree.cpp -o ../src/octree.o

# Clean up build files
clean:
	rm -f $(TARGET) $(OBJS)
