CC=nvcc
CFLAGS=-arch=sm_89 -O3 --cudart static
TARGET=gpu_grinder

all: $(TARGET)

$(TARGET): gpu_grinder.cu
	$(CC) $(CFLAGS) -o $(TARGET) gpu_grinder.cu

clean:
	rm -f $(TARGET)
