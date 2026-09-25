/*
 * cuda/src/main.cu
 * ─────────────────────────────────────────────────────────────────────────────
 * Multi-Paradigm Parallel Image Filtering — CUDA GPU Acceleration
 * EC7207 High Performance Computing | Group 02
 *
 * Strategy:
 *   • Each CUDA thread processes ONE output pixel.
 *   • A 2D grid of 2D thread blocks maps threads to image pixels.
 *   • Convolution kernels are stored in CUDA __constant__ memory
 *     (cached, broadcast-friendly for uniform kernel access).
 *   • Shared memory tiling is used to reduce global memory bandwidth:
 *     each block loads a (TILE + 2) × (TILE + 2) halo patch into
 *     shared memory, then all threads in the block read from it.
 *
 * Thread/Block layout:
 *   ┌──────────────────────────────────────┐
 *   │           Image (W × H)             │
 *   │  ┌────────┬────────┬────────┐        │
 *   │  │Block   │Block   │Block   │  ...   │  ← gridDim.x  = ceil(W/TILE)
 *   │  │(0,0)   │(1,0)   │(2,0)   │        │
 *   │  ├────────┼────────┼────────┤        │
 *   │  │Block   │Block   │  ...   │        │  ← gridDim.y  = ceil(H/TILE)
 *   │  │(0,1)   │(1,1)   │        │        │
 *   │  └────────┴────────┴────────┘        │
 *   │  Each block: TILE×TILE threads       │  ← blockDim = (TILE, TILE)
 *   └──────────────────────────────────────┘
 *
 * Compilation:
 *   nvcc -O2 -arch=sm_50 src/main.cu ../common/image_io.c \
 *        -I../common -lm -o cuda_filter
 *   (replace sm_50 with your GPU's compute capability, e.g. sm_75, sm_86)
 *
 * Usage:
 *   ./cuda_filter <input> <out_gaussian> <out_sobel>
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

extern "C" {
#include "../../common/image_io.h"
#include "../../common/timer.h"
}

/* ── Tile size: each thread block covers TILE×TILE output pixels ─────────── */
#define TILE 16

/* ─────────────────────────────────────────────────────────────────────────────
 * CUDA Error-Check Macro
 * Wraps every CUDA call; prints file/line and aborts on failure.
 * ───────────────────────────────────────────────────────────────────────────*/
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                 \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

/* ─────────────────────────────────────────────────────────────────────────────
 * Constant Memory — Gaussian kernel (3×3, row-major)
 *
 * __constant__ memory resides in a read-only cache optimised for the case
 * where all threads in a warp read the same address (uniform access).
 * Perfect for small, fixed convolution kernels.
 * ───────────────────────────────────────────────────────────────────────────*/
__constant__ float d_gaussian_kernel[9];   /* 3×3 = 9 floats */

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL 1 — Gaussian Blur (CUDA, shared-memory tiled)
 *
 * Grid  : (ceil(W/TILE), ceil(H/TILE))  2-D blocks
 * Block : (TILE, TILE)  threads
 *
 * Shared memory layout (per block):
 *   smem[(TILE+2)][(TILE+2)] — one halo pixel on each side
 *   Threads cooperatively load the interior AND the border halo.
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__ void gaussian_blur_kernel(
        const unsigned char * __restrict__ input,
        unsigned char       * __restrict__ output,
        int width, int height, int channels)
{
    /* Shared memory: (TILE+2) × (TILE+2) × channels
     * We allocate for max 3 channels; unused channels waste nothing at runtime
     * because the channel loop simply doesn't execute. */
    __shared__ unsigned char smem[TILE + 2][TILE + 2][3];

    /* Global pixel this thread is responsible for */
    int gx = blockIdx.x * TILE + threadIdx.x;  /* column */
    int gy = blockIdx.y * TILE + threadIdx.y;  /* row    */

    /* Shared memory coordinates (offset by 1 for halo) */
    int sx = threadIdx.x + 1;
    int sy = threadIdx.y + 1;

    /* ── Load centre pixels into shared memory ── */
    for (int c = 0; c < channels; c++) {
        if (gx < width && gy < height)
            smem[sy][sx][c] = input[(gy * width + gx) * channels + c];
        else
            smem[sy][sx][c] = 0;
    }

    /* ── Load halo pixels (border threads only) ── */
    /* Left halo */
    if (threadIdx.x == 0) {
        int hx = gx - 1;
        for (int c = 0; c < channels; c++)
            smem[sy][0][c] = (hx >= 0 && gy < height)
                             ? input[(gy * width + hx) * channels + c] : 0;
    }
    /* Right halo */
    if (threadIdx.x == TILE - 1) {
        int hx = gx + 1;
        for (int c = 0; c < channels; c++)
            smem[sy][TILE + 1][c] = (hx < width && gy < height)
                                    ? input[(gy * width + hx) * channels + c] : 0;
    }
    /* Top halo */
    if (threadIdx.y == 0) {
        int hy = gy - 1;
        for (int c = 0; c < channels; c++)
            smem[0][sx][c] = (hy >= 0 && gx < width)
                             ? input[(hy * width + gx) * channels + c] : 0;
    }
    /* Bottom halo */
    if (threadIdx.y == TILE - 1) {
        int hy = gy + 1;
        for (int c = 0; c < channels; c++)
            smem[TILE + 1][sx][c] = (hy < height && gx < width)
                                    ? input[(hy * width + gx) * channels + c] : 0;
    }

    /* Corner halos (diagonal neighbours) */
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int hx = gx - 1, hy = gy - 1;
        for (int c = 0; c < channels; c++)
            smem[0][0][c] = (hx >= 0 && hy >= 0)
                            ? input[(hy * width + hx) * channels + c] : 0;
    }
    if (threadIdx.x == TILE-1 && threadIdx.y == 0) {
        int hx = gx + 1, hy = gy - 1;
        for (int c = 0; c < channels; c++)
            smem[0][TILE+1][c] = (hx < width && hy >= 0)
                                 ? input[(hy * width + hx) * channels + c] : 0;
    }
    if (threadIdx.x == 0 && threadIdx.y == TILE-1) {
        int hx = gx - 1, hy = gy + 1;
        for (int c = 0; c < channels; c++)
            smem[TILE+1][0][c] = (hx >= 0 && hy < height)
                                 ? input[(hy * width + hx) * channels + c] : 0;
    }
    if (threadIdx.x == TILE-1 && threadIdx.y == TILE-1) {
        int hx = gx + 1, hy = gy + 1;
        for (int c = 0; c < channels; c++)
            smem[TILE+1][TILE+1][c] = (hx < width && hy < height)
                                      ? input[(hy * width + hx) * channels + c] : 0;
    }

    /* ── Synchronise: all shared memory loads must complete before compute ── */
    __syncthreads();

    /* ── Skip image border pixels and out-of-bounds threads ── */
    if (gx <= 0 || gx >= width - 1 || gy <= 0 || gy >= height - 1) return;

    /* ── Apply 3×3 Gaussian convolution from shared memory ── */
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        int ki = 0;
        for (int ky = 0; ky < 3; ky++) {
            for (int kx = 0; kx < 3; kx++) {
                sum += smem[sy + ky - 1][sx + kx - 1][c]
                       * d_gaussian_kernel[ki++];
            }
        }
        output[(gy * width + gx) * channels + c] = (unsigned char)sum;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL 2 — Sobel Edge Detection (CUDA, shared-memory tiled)
 *
 * Grayscale conversion is performed on-the-fly (same formula as serial/OMP/MPI).
 * Output magnitude is written to ALL channels (consistent with other versions).
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__ void sobel_edge_kernel(
        const unsigned char * __restrict__ input,
        unsigned char       * __restrict__ output,
        int width, int height, int channels)
{
    /* Sobel kernels — stored in registers (small, per-thread constants) */
    const int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int Gy[3][3] = {{-1,-2,-1}, { 0, 0, 0}, { 1, 2, 1}};

    /* Shared memory: grayscale patch (TILE+2) × (TILE+2) */
    __shared__ float smem_gray[TILE + 2][TILE + 2];

    int gx = blockIdx.x * TILE + threadIdx.x;
    int gy = blockIdx.y * TILE + threadIdx.y;

    int sx = threadIdx.x + 1;
    int sy = threadIdx.y + 1;

    /* ── Inline grayscale helper (lambda-style macro) ── */
    #define TO_GRAY(base_idx) \
        ((channels >= 3) \
         ? (0.299f * input[(base_idx)]     \
          + 0.587f * input[(base_idx) + 1] \
          + 0.114f * input[(base_idx) + 2]) \
         : (float)input[(base_idx)])

    /* ── Load centre grayscale values ── */
    smem_gray[sy][sx] = (gx < width && gy < height)
                        ? TO_GRAY((gy * width + gx) * channels)
                        : 0.0f;

    /* ── Halo loads (grayscale) ── */
    if (threadIdx.x == 0) {
        int hx = gx - 1;
        smem_gray[sy][0] = (hx >= 0 && gy < height)
                           ? TO_GRAY((gy * width + hx) * channels) : 0.0f;
    }
    if (threadIdx.x == TILE - 1) {
        int hx = gx + 1;
        smem_gray[sy][TILE+1] = (hx < width && gy < height)
                                ? TO_GRAY((gy * width + hx) * channels) : 0.0f;
    }
    if (threadIdx.y == 0) {
        int hy = gy - 1;
        smem_gray[0][sx] = (hy >= 0 && gx < width)
                           ? TO_GRAY((hy * width + gx) * channels) : 0.0f;
    }
    if (threadIdx.y == TILE - 1) {
        int hy = gy + 1;
        smem_gray[TILE+1][sx] = (hy < height && gx < width)
                                ? TO_GRAY((hy * width + gx) * channels) : 0.0f;
    }

    /* Corner halos */
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int hx = gx-1, hy = gy-1;
        smem_gray[0][0] = (hx>=0 && hy>=0)
                          ? TO_GRAY((hy*width+hx)*channels) : 0.0f;
    }
    if (threadIdx.x == TILE-1 && threadIdx.y == 0) {
        int hx = gx+1, hy = gy-1;
        smem_gray[0][TILE+1] = (hx<width && hy>=0)
                               ? TO_GRAY((hy*width+hx)*channels) : 0.0f;
    }
    if (threadIdx.x == 0 && threadIdx.y == TILE-1) {
        int hx = gx-1, hy = gy+1;
        smem_gray[TILE+1][0] = (hx>=0 && hy<height)
                               ? TO_GRAY((hy*width+hx)*channels) : 0.0f;
    }
    if (threadIdx.x == TILE-1 && threadIdx.y == TILE-1) {
        int hx = gx+1, hy = gy+1;
        smem_gray[TILE+1][TILE+1] = (hx<width && hy<height)
                                    ? TO_GRAY((hy*width+hx)*channels) : 0.0f;
    }

    #undef TO_GRAY

    __syncthreads();

    if (gx <= 0 || gx >= width-1 || gy <= 0 || gy >= height-1) return;

    /* ── Sobel convolution from shared grayscale memory ── */
    float sumX = 0.0f, sumY = 0.0f;
    for (int ky = 0; ky < 3; ky++) {
        for (int kx = 0; kx < 3; kx++) {
            float g = smem_gray[sy + ky - 1][sx + kx - 1];
            sumX += g * Gx[ky][kx];
            sumY += g * Gy[ky][kx];
        }
    }

    unsigned char magnitude =
        (unsigned char)fminf(sqrtf(sumX*sumX + sumY*sumY), 255.0f);

    int out_base = (gy * width + gx) * channels;
    for (int c = 0; c < channels; c++)
        output[out_base + c] = magnitude;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv)
{
    if (argc < 4) {
        printf("Usage: %s <input> <out_gaussian> <out_sobel>\n", argv[0]);
        return 1;
    }

    /* ── Print GPU info ── */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Running CUDA filter on: %s (Compute %d.%d)\n",
           prop.name, prop.major, prop.minor);

    /* ── Load input image (CPU) ── */
    Image imgIn = load_image(argv[1]);
    printf("Image loaded: %d x %d, %d channel(s)\n",
           imgIn.width, imgIn.height, imgIn.channels);

    int W = imgIn.width;
    int H = imgIn.height;
    int C = imgIn.channels;
    size_t imgBytes = (size_t)W * H * C;

    /* ── Allocate output buffers (CPU) ── */
    Image imgGaussian = create_image(W, H, C);
    Image imgSobel    = create_image(W, H, C);

    /* ── Allocate GPU memory ── */
    unsigned char *d_input, *d_gaussian, *d_sobel;
    CUDA_CHECK(cudaMalloc(&d_input,    imgBytes));
    CUDA_CHECK(cudaMalloc(&d_gaussian, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_sobel,    imgBytes));

    /* ── Copy input image to GPU ── */
    CUDA_CHECK(cudaMemcpy(d_input, imgIn.data, imgBytes, cudaMemcpyHostToDevice));

    /* ── Upload Gaussian kernel to __constant__ memory ── */
    const float h_gaussian_kernel[9] = {
        1/16.0f, 2/16.0f, 1/16.0f,
        2/16.0f, 4/16.0f, 2/16.0f,
        1/16.0f, 2/16.0f, 1/16.0f
    };
    CUDA_CHECK(cudaMemcpyToSymbol(d_gaussian_kernel, h_gaussian_kernel,
                                  9 * sizeof(float)));

    /* ── Configure 2D grid and block dimensions ── */
    dim3 blockDim(TILE, TILE);
    dim3 gridDim((W + TILE - 1) / TILE,
                 (H + TILE - 1) / TILE);

    /* ── CUDA Events for precise GPU timing ── */
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    /* ════════════════════════════════════
     * Gaussian Blur
     * ════════════════════════════════════ */
    CUDA_CHECK(cudaEventRecord(ev_start));
    gaussian_blur_kernel<<<gridDim, blockDim>>>(d_input, d_gaussian, W, H, C);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    /* Check for kernel launch errors */
    CUDA_CHECK(cudaGetLastError());

    float gauss_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gauss_ms, ev_start, ev_stop));
    printf("Gaussian Blur  - CUDA Execution Time : %.6f seconds\n",
           gauss_ms / 1000.0f);

    /* ════════════════════════════════════
     * Sobel Edge Detection
     * ════════════════════════════════════ */
    CUDA_CHECK(cudaEventRecord(ev_start));
    sobel_edge_kernel<<<gridDim, blockDim>>>(d_input, d_sobel, W, H, C);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    CUDA_CHECK(cudaGetLastError());

    float sobel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&sobel_ms, ev_start, ev_stop));
    printf("Sobel Edge Det - CUDA Execution Time : %.6f seconds\n",
           sobel_ms / 1000.0f);

    /* ── Copy results back to CPU ── */
    CUDA_CHECK(cudaMemcpy(imgGaussian.data, d_gaussian, imgBytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(imgSobel.data,    d_sobel,    imgBytes,
                          cudaMemcpyDeviceToHost));

    /* ── Save output images ── */
    save_image(argv[2], imgGaussian);
    save_image(argv[3], imgSobel);
    printf("Results saved to: %s and %s\n", argv[2], argv[3]);

    /* ── Clean up ── */
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_gaussian));
    CUDA_CHECK(cudaFree(d_sobel));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    free_image(imgIn);
    free(imgGaussian.data);
    free(imgSobel.data);

    return 0;
}
