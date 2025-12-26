# Makefile for solanity-gpu

# CUDA compiler
NVCC := nvcc

# Flags
# Added --use_fast_math and --ptxas-options=-v for performance tuning
# -O3 is already level 3 optimization
# Target sm_89 (Ada Lovelace / L4) [CRITICAL: sm_89 must be present to avoid JIT]
NVCC_FLAGS := -O3 --use_fast_math --ptxas-options=-v -Xptxas -O3 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_86,code=sm_86

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
