/*
 * pthreads/src/main.c
 * ─────────────────────────────────────────────────────────────────────────────
 * Multi-Paradigm Parallel Image Filtering — POSIX Threads (Pthreads)
 * EC7207 High Performance Computing | Group 02
 *
 * Strategy:
 *   • Row-wise domain decomposition — image rows divided among N threads.
 *   • Each thread independently processes its assigned rows for convolution.
 *   • Threads share the input and output Image buffers (shared memory model).
 *   • No synchronisation needed: each output pixel is written by exactly
 *     one thread → zero race conditions.
 *
 * Decomposition layout (N threads, H rows):
 *   Thread 0 : rows  0          ..  (H/N) - 1
 *   Thread 1 : rows  H/N        ..  (2H/N) - 1
 *   ...
 *   Thread N-1: rows (N-1)*H/N  ..  H - 1   ← gets any remainder rows too
 *
 * Compilation:
 *   gcc src/main.c ../common/image_io.c -I../common -lpthread -lm -o pthreads_filter
 *
 * Usage:
 *   ./pthreads_filter <input> <out_gaussian> <out_sobel> <num_threads>
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>

#include "../../common/image_io.h"
#include "../../common/timer.h"

/* ─────────────────────────────────────────────────────────────────────────────
 * ThreadData — passed to every worker thread via void* argument.
 *
 * Both gaussian_thread and sobel_thread use the same struct layout so
 * a single typedef covers both filter functions cleanly.
 * ───────────────────────────────────────────────────────────────────────────*/
typedef struct {
    Image in;        /* full input image  (read-only inside thread)  */
    Image out;       /* full output image (thread writes its slice)  */
    int   start_row; /* first row this thread owns (inclusive)       */
    int   end_row;   /* last  row this thread owns (exclusive)       */
} ThreadData;

/* ═══════════════════════════════════════════════════════════════════════════
 * 1.  GAUSSIAN BLUR THREAD WORKER
 *
 *  Applies the 3×3 Gaussian kernel to rows [start_row, end_row).
 *  Rows 0 and H-1 are global image borders — skipped (no full neighbourhood).
 *  No mutex needed: each pixel address (y * w + x) belongs to one thread only.
 * ═══════════════════════════════════════════════════════════════════════════ */
static void *gaussian_thread(void *arg)
{
    ThreadData *d = (ThreadData *)arg;

    /* Kernel declared const inside the thread — no shared mutable state */
    const float kernel[3][3] = {
        {1/16.0f, 2/16.0f, 1/16.0f},
        {2/16.0f, 4/16.0f, 2/16.0f},
        {1/16.0f, 2/16.0f, 1/16.0f}
    };

    int w  = d->in.width;
    int ch = d->in.channels;

    for (int y = d->start_row; y < d->end_row; y++) {

        /* Skip image top/bottom border rows — 3×3 kernel needs one neighbour */
        if (y <= 0 || y >= d->in.height - 1) continue;

        for (int x = 1; x < w - 1; x++) {
            for (int c = 0; c < ch; c++) {

                float sum = 0.0f;
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int idx = ((y + ky) * w + (x + kx)) * ch + c;
                        sum += d->in.data[idx] * kernel[ky + 1][kx + 1];
                    }
                }
                d->out.data[(y * w + x) * ch + c] = (unsigned char)sum;
            }
        }
    }

    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2.  SOBEL EDGE DETECTION THREAD WORKER
 *
 *  Applies Gx/Gy Sobel kernels to rows [start_row, end_row).
 *  Grayscale conversion is performed on-the-fly (same formula as all other
 *  implementations in this project).
 *  Output magnitude is written to ALL channels for consistency.
 * ═══════════════════════════════════════════════════════════════════════════ */
static void *sobel_thread(void *arg)
{
    ThreadData *d = (ThreadData *)arg;

    const int Gx[3][3] = {
        {-1,  0,  1},
        {-2,  0,  2},
        {-1,  0,  1}
    };
    const int Gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    int w  = d->in.width;
    int ch = d->in.channels;

    for (int y = d->start_row; y < d->end_row; y++) {

        if (y <= 0 || y >= d->in.height - 1) continue;

        for (int x = 1; x < w - 1; x++) {

            float sumX = 0.0f, sumY = 0.0f;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int base = ((y + ky) * w + (x + kx)) * ch;

                    /* Grayscale conversion — identical to serial/OMP/MPI/CUDA */
                    float gray = (ch >= 3)
                        ? (0.299f * d->in.data[base    ]
                         + 0.587f * d->in.data[base + 1]
                         + 0.114f * d->in.data[base + 2])
                        : (float)d->in.data[base];

                    sumX += gray * Gx[ky + 1][kx + 1];
                    sumY += gray * Gy[ky + 1][kx + 1];
                }
            }

            unsigned char magnitude =
                (unsigned char)fminf(sqrtf(sumX * sumX + sumY * sumY), 255.0f);

            /* Write magnitude to all channels (grayscale-equivalent output) */
            int out_base = (y * w + x) * ch;
            for (int c = 0; c < ch; c++)
                d->out.data[out_base + c] = magnitude;
        }
    }

    return NULL;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * launch_threads — helper that creates N threads running func(thread_data[i]),
 * waits for all to finish, and returns wall-clock elapsed time in seconds.
 *
 * Parameters:
 *   func         : thread worker function (gaussian_thread or sobel_thread)
 *   thread_data  : array of ThreadData, one per thread (already populated)
 *   threads      : pre-allocated pthread_t array of length num_threads
 *   num_threads  : number of threads to launch
 *
 * Returns: elapsed wall-clock time in seconds
 * ───────────────────────────────────────────────────────────────────────────*/
static double launch_threads(void *(*func)(void *),
                             ThreadData *thread_data,
                             pthread_t  *threads,
                             int         num_threads)
{
    double t_start = get_time();

    for (int i = 0; i < num_threads; i++)
        pthread_create(&threads[i], NULL, func, &thread_data[i]);

    for (int i = 0; i < num_threads; i++)
        pthread_join(threads[i], NULL);

    return get_time() - t_start;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3.  MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv)
{
    if (argc < 5) {
        printf("Usage: %s <input> <out_gaussian> <out_sobel> <num_threads>\n",
               argv[0]);
        return 1;
    }

    /* ── Parse thread count ── */
    int num_threads = atoi(argv[4]);
    if (num_threads <= 0) {
        fprintf(stderr, "Error: num_threads must be a positive integer.\n");
        return 1;
    }

    printf("Running Pthreads filter with %d thread(s)\n", num_threads);

    /* ── Load input image ── */
    Image imgIn       = load_image(argv[1]);
    printf("Image loaded: %d x %d, %d channel(s)\n",
           imgIn.width, imgIn.height, imgIn.channels);

    /* ── Allocate output buffers ── */
    Image imgGaussian = create_image(imgIn.width, imgIn.height, imgIn.channels);
    Image imgSobel    = create_image(imgIn.width, imgIn.height, imgIn.channels);

    /* ── Allocate thread handles and per-thread data arrays ── */
    pthread_t  *threads     = (pthread_t  *)malloc(num_threads * sizeof(pthread_t));
    ThreadData *thread_data = (ThreadData *)malloc(num_threads * sizeof(ThreadData));

    if (!threads || !thread_data) {
        fprintf(stderr, "Error: failed to allocate thread structures.\n");
        return 1;
    }

    /* ── Row decomposition
     *
     *  base_rows : rows every thread gets
     *  remainder : leftover rows (height not divisible by num_threads)
     *
     *  Distribution: first `remainder` threads get (base_rows + 1) rows,
     *  the rest get base_rows rows. Maximum imbalance = 1 row.
     * ── */
    int H         = imgIn.height;
    int base_rows = H / num_threads;
    int remainder = H % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start = i * base_rows + (i < remainder ? i : remainder);
        int end   = start + base_rows + (i < remainder ? 1 : 0);

        thread_data[i].in        = imgIn;
        thread_data[i].start_row = start;
        thread_data[i].end_row   = end;
    }

    /* ── Gaussian Blur ── */
    for (int i = 0; i < num_threads; i++)
        thread_data[i].out = imgGaussian;

    double gauss_time = launch_threads(gaussian_thread,
                                       thread_data, threads, num_threads);
    printf("Gaussian Blur  - Pthreads Execution Time : %.6f seconds\n",
           gauss_time);

    /* ── Sobel Edge Detection ── */
    for (int i = 0; i < num_threads; i++)
        thread_data[i].out = imgSobel;

    double sobel_time = launch_threads(sobel_thread,
                                       thread_data, threads, num_threads);
    printf("Sobel Edge Det - Pthreads Execution Time : %.6f seconds\n",
           sobel_time);

    /* ── Save results ── */
    save_image(argv[2], imgGaussian);
    save_image(argv[3], imgSobel);
    printf("Results saved to: %s and %s\n", argv[2], argv[3]);

    /* ── Clean up ── */
    free(threads);
    free(thread_data);
    free_image(imgIn);
    free(imgGaussian.data);
    free(imgSobel.data);

    return 0;
}