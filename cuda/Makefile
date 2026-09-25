# cuda/Makefile
# ─────────────────────────────────────────────────────────────────────────────
# CUDA Image Filter
# EC7207 High Performance Computing | Group 02
#
# Adjust -arch flag to match your GPU:
#   GTX 900 series       → sm_52
#   GTX 10xx / Titan X   → sm_61
#   RTX 20xx             → sm_75
#   RTX 30xx             → sm_86
#   RTX 40xx             → sm_89
#
# Check yours: nvcc --run -code=- -arch=native (or nvidia-smi)
# ─────────────────────────────────────────────────────────────────────────────

NVCC     = nvcc
ARCH     = sm_75          # ← change to your GPU's compute capability
CFLAGS   = -O2 -arch=$(ARCH) -I../common
LDFLAGS  = -lm
TARGET   = cuda_filter

SRC_CU   = src/main.cu
SRC_C    = ../common/image_io.c

INPUT    = ../data/input/test.jpg
OUT_GAUSS= ../data/output/cuda_gaussian.jpg
OUT_SOBEL= ../data/output/cuda_sobel.jpg

# ── Build ──────────────────────────────────────────────────────────────────
all: $(TARGET)

$(TARGET): $(SRC_CU) $(SRC_C)
	$(NVCC) $(CFLAGS) $(SRC_CU) $(SRC_C) -o $(TARGET) $(LDFLAGS)

# ── Run ───────────────────────────────────────────────────────────────────
run: $(TARGET)
	./$(TARGET) $(INPUT) $(OUT_GAUSS) $(OUT_SOBEL)

# ── Show GPU info ─────────────────────────────────────────────────────────
gpuinfo:
	nvidia-smi

# ── Clean ─────────────────────────────────────────────────────────────────
clean:
	rm -f $(TARGET)

.PHONY: all run gpuinfo clean
