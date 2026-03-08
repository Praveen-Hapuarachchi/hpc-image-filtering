
CC      = gcc
CFLAGS  = -O2 -Wall -fopenmp
LDFLAGS = -lm

TARGET  = openmp_filter

# Relative paths from openmp/
SRC     = src/main.c
COMMON  = ../../common/image_io.c

# Default number of threads for `make run`
THREADS ?= 4
INPUT   ?= ../../data/input/test_image.jpg
OUT_G   ?= ../../data/output/openmp_gaussian.jpg
OUT_S   ?= ../../data/output/openmp_sobel.jpg

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC) $(COMMON)
	$(CC) $(CFLAGS) -I../../common -o $(TARGET) $(SRC) $(COMMON) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) $(INPUT) $(OUT_G) $(OUT_S) $(THREADS)

clean:
	rm -f $(TARGET)
