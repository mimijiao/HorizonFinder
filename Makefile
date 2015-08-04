CUDA_PATH       ?= /usr/local/cuda-6.0
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin

# CUDA code generation flags
GENCODE_FLAGS   := -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35

# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc

TARGETS = houghTransform

all:
	$(NVCC) $(GENCODE_FLAGS) houghTransform.cu cannyEdge.cu -o houghTransform