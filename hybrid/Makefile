# hybrid/Makefile
# ─────────────────────────────────────────────────────────────────────────────
# Hybrid MPI + OpenMP Image Filter
# EC7207 High Performance Computing | Group 02
# ─────────────────────────────────────────────────────────────────────────────

CC       = mpicc
CFLAGS   = -fopenmp -O2 -Wall -I../common
LDFLAGS  = -lm
TARGET   = hybrid_filter

SRC      = src/main.c ../common/image_io.c

# Default values (override on command line)
NP       = 4
THREADS  = 2
INPUT    = ../data/input/test.jpg
OUT_GAUSS= ../data/output/hybrid_gaussian.jpg
OUT_SOBEL= ../data/output/hybrid_sobel.jpg

# ── Build ──────────────────────────────────────────────────────────────────
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

# ── Run ───────────────────────────────────────────────────────────────────
# Example: make run NP=4 THREADS=2
run: $(TARGET)
	mpirun -np $(NP) ./$(TARGET) $(INPUT) $(OUT_GAUSS) $(OUT_SOBEL) $(THREADS)

# ── Clean ─────────────────────────────────────────────────────────────────
clean:
	rm -f $(TARGET)

.PHONY: all run clean