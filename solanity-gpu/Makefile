# Makefile for solanity-gpu

# CUDA compiler
NVCC := nvcc

# Flags
NVCC_FLAGS := -O3 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_86,code=sm_86

# Target
TARGET := solanity

# Source
# main.cu includes other files directly (Unity Build)
SRC := main.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -o $@ $< $(NVCC_FLAGS)

clean:
	rm -f $(TARGET)
