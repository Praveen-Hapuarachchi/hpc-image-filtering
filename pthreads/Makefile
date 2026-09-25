# pthreads/Makefile
# ─────────────────────────────────────────────────────────────────────────────
# Pthreads Image Filter
# EC7207 High Performance Computing | Group 02
# ─────────────────────────────────────────────────────────────────────────────

CC       = gcc
CFLAGS   = -O2 -Wall -I../common
LDFLAGS  = -lpthread -lm
TARGET   = pthreads_filter

SRC      = src/main.c ../common/image_io.c

# Default run parameters (override on command line)
THREADS  = 4
INPUT    = ../data/input/test.jpg
OUT_GAUSS= ../data/output/pthreads_gaussian.jpg
OUT_SOBEL= ../data/output/pthreads_sobel.jpg

# ── Build ──────────────────────────────────────────────────────────────────
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

# ── Run ───────────────────────────────────────────────────────────────────
# Example: make run THREADS=4
run: $(TARGET)
	./$(TARGET) $(INPUT) $(OUT_GAUSS) $(OUT_SOBEL) $(THREADS)

# ── Clean ─────────────────────────────────────────────────────────────────
clean:
	rm -f $(TARGET)

.PHONY: all run clean
