# Makefile for solanity-gpu

# CUDA compiler
NVCC := nvcc

# Flags
NVCC_FLAGS := -O3 -arch=sm_75

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
