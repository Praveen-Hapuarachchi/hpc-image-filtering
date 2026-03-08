CC = gcc
CFLAGS = -O3 -Wall
LIBS = -lm
INCLUDES = -I../common

# Target name
TARGET = image_filter

# Source files
SRCS = src/main.c ../common/image_io.c

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)