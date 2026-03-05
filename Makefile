NVCC := nvcc
CXXFLAGS := -std=c++14 -O2
INCLUDE := -I reduction/include
SRCS := reduction/reduction.cu reduction/main.cpp
TARGET := reduction_out

.PHONY: all build run clean

all: build

build:
	$(NVCC) $(CXXFLAGS) $(SRCS) $(INCLUDE) -o $(TARGET)

run: build
	./$(TARGET)

clean:
	rm -f $(TARGET)

