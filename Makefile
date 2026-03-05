NVCC     := nvcc
CXXFLAGS := -std=c++14 -O2
INCLUDE  := -I reduction/include
SRCS     := reduction/reduction.cu reduction/main.cpp
TARGET       := reduction_out
TEST_TARGET  := test_reduction_out
BENCH_TARGET := bench_reduction_out
CUB_TARGET   := bench_cub_out

.PHONY: all build run test bench bench-cub clean

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

bench-cub:
	$(NVCC) $(CXXFLAGS) reduction/bench_cub.cu $(INCLUDE) -o $(CUB_TARGET)
	./$(CUB_TARGET)

clean:
	rm -f $(TARGET) $(TEST_TARGET) $(BENCH_TARGET) $(CUB_TARGET)

