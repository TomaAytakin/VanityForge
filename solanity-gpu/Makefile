# Makefile for solanity-gpu

# CUDA compiler
NVCC := nvcc

# Flags
NVCC_FLAGS := -O3 --generate-code arch=compute_89,code=sm_89 --generate-code arch=compute_75,code=sm_75 --generate-code arch=compute_80,code=sm_80

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
