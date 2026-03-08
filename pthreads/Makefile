CC = gcc
CFLAGS = -O3 -Wall
LIBS = -lpthread -lm

all: image_filter_pthreads

image_filter_pthreads: src/main.c
	$(CC) $(CFLAGS) src/main.c -o image_filter_pthreads $(LIBS)

clean:
	rm -f image_filter_pthreads