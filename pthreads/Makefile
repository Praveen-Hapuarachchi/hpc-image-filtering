CC = gcc
CFLAGS = -O3 -Wall -I../common
LIBS = -lpthread -lm

# Include both the main file and the image utility file
SRCS = src/main.c ../common/image_io.c
TARGET = image_filter_pthreads

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)