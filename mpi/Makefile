CC = mpicc
CFLAGS = -O3 -Wall -I../common
LIBS = -lm

SRCS = src/main.c ../common/image_io.c
TARGET = image_filter_mpi

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)