NVCC     := nvcc
CXXFLAGS := -std=c++14 -O2
INCLUDE  := -I reduction/include
SRCS     := reduction/reduction.cu reduction/main.cpp
TARGET      := reduction_out
TEST_TARGET := test_reduction_out
BENCH_TARGET := bench_reduction_out

.PHONY: all build run test bench clean

all: build

build:
	$(NVCC) $(CXXFLAGS) $(SRCS) $(INCLUDE) -o $(TARGET)

run: build
	./$(TARGET)

test:
	$(NVCC) $(CXXFLAGS) reduction/reduction.cu reduction/test_reduction.cu $(INCLUDE) -o $(TEST_TARGET)
	./$(TEST_TARGET)

bench:
	$(NVCC) $(CXXFLAGS) reduction/reduction.cu reduction/bench_reduction.cu $(INCLUDE) -o $(BENCH_TARGET)
	./$(BENCH_TARGET)

clean:
	rm -f $(TARGET) $(TEST_TARGET) $(BENCH_TARGET)

