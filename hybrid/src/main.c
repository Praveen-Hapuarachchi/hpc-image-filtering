/*
 * hybrid/src/main.c
 * ─────────────────────────────────────────────────────────────────────────────
 * Multi-Paradigm Parallel Image Filtering — Hybrid MPI + OpenMP
 * EC7207 High Performance Computing | Group 02
 *
 * Strategy:
 *   • MPI  → distributes image ROWS across processes (inter-node / inter-process)
 *   • OpenMP → parallelises the per-row convolution loop within each MPI process
 *
 * Decomposition:
 *   ┌─────────────────────────────┐
 *   │  Full image  (H rows)       │  rank 0 loads & broadcasts
 *   ├─────────────────────────────┤
 *   │  Rank 0 : rows  0 .. r-1   │  ← each rank processes its
 *   │  Rank 1 : rows  r .. 2r-1  │    slice with OpenMP threads
 *   │  ...                        │
 *   │  Rank P-1: remaining rows   │
 *   └─────────────────────────────┘
 *   Results gathered back to rank 0 via MPI_Gatherv.
 *
 * Ghost / halo rows:
 *   Convolution at row boundaries needs one neighbour row on each side.
 *   Each rank receives ONE extra halo row above (if not first) and below
 *   (if not last) so border pixels are computed correctly.
 *   The halo rows are used READ-ONLY during convolution; they are NOT
 *   included in the gathered output slice.
 *
 * Compilation (Linux / WSL):
 *   mpicc -fopenmp src/main.c ../../common/image_io.c \
 *         -I../../common -lm -o hybrid_filter
 *
 * Usage:
 *   mpirun -np <P> ./hybrid_filter <input> <out_gaussian> <out_sobel> [threads_per_rank]
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#include "../../common/image_io.h"
#include "../../common/timer.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * 1.  GAUSSIAN BLUR  (Hybrid MPI + OpenMP)
 *
 *  Parameters
 *  ----------
 *  in         : full image buffer (all ranks hold this after MPI_Bcast)
 *  out        : full output buffer (only the owned slice is written)
 *  start_row  : first row this rank OWNS  (excludes halo)
 *  end_row    : one-past-last row this rank OWNS (excludes halo)
 *  num_threads: OpenMP threads to use inside this rank
 * ═══════════════════════════════════════════════════════════════════════════ */
void apply_gaussian_blur(Image in, Image out,
                         int start_row, int end_row,
                         int num_threads)
{
    const float kernel[3][3] = {
        {1/16.0f, 2/16.0f, 1/16.0f},
        {2/16.0f, 4/16.0f, 2/16.0f},
        {1/16.0f, 2/16.0f, 1/16.0f}
    };

    /*
     * OpenMP parallelises the outer (row) loop.
     * Each thread handles a contiguous band of rows → no write conflicts.
     * schedule(static) gives equal-sized chunks; ideal for uniform workloads.
     */
    #pragma omp parallel for schedule(static) num_threads(num_threads) \
                             default(none) shared(in, out, kernel, start_row, end_row)
    for (int y = start_row; y < end_row; y++) {

        /* Skip image border rows — no full 3×3 neighbourhood available */
        if (y <= 0 || y >= in.height - 1) continue;

        for (int x = 1; x < in.width - 1; x++) {
            for (int c = 0; c < in.channels; c++) {

                float sum = 0.0f;
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int idx = ((y + ky) * in.width + (x + kx))
                                  * in.channels + c;
                        sum += in.data[idx] * kernel[ky + 1][kx + 1];
                    }
                }
                out.data[(y * in.width + x) * in.channels + c] =
                    (unsigned char)sum;
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2.  SOBEL EDGE DETECTION  (Hybrid MPI + OpenMP)
 * ═══════════════════════════════════════════════════════════════════════════ */
void apply_sobel_edge_detection(Image in, Image out,
                                int start_row, int end_row,
                                int num_threads)
{
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

    #pragma omp parallel for schedule(static) num_threads(num_threads) \
                             default(none) shared(in, out, Gx, Gy, start_row, end_row)
    for (int y = start_row; y < end_row; y++) {

        if (y <= 0 || y >= in.height - 1) continue;

        for (int x = 1; x < in.width - 1; x++) {

            float sumX = 0.0f, sumY = 0.0f;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int base = ((y + ky) * in.width + (x + kx)) * in.channels;

                    float gray = (in.channels >= 3)
                        ? (0.299f * in.data[base    ]
                         + 0.587f * in.data[base + 1]
                         + 0.114f * in.data[base + 2])
                        : (float)in.data[base];

                    sumX += gray * Gx[ky + 1][kx + 1];
                    sumY += gray * Gy[ky + 1][kx + 1];
                }
            }

            unsigned char magnitude =
                (unsigned char)fminf(sqrtf(sumX*sumX + sumY*sumY), 255.0f);

            int out_base = (y * in.width + x) * out.channels;
            for (int c = 0; c < out.channels; c++)
                out.data[out_base + c] = magnitude;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3.  HELPER — build Gatherv displacement/count arrays
 *
 *  Computes recv_counts[] and displs[] for MPI_Gatherv so that each rank's
 *  contribution maps back to the correct byte offset in the full image buffer.
 * ═══════════════════════════════════════════════════════════════════════════ */
static void build_gatherv_params(int size, int total_rows,
                                 int width, int channels,
                                 int *recv_counts, int *displs)
{
    int base_rows = total_rows / size;
    int remainder = total_rows % size;

    for (int i = 0; i < size; i++) {
        int r_start = i * base_rows + (i < remainder ? i : remainder);
        int r_end   = r_start + base_rows + (i < remainder ? 1 : 0);
        recv_counts[i] = (r_end - r_start) * width * channels;
        displs[i]      = r_start * width * channels;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4.  MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ── Argument validation ── */
    if (argc < 4) {
        if (rank == 0)
            printf("Usage: %s <input> <out_gaussian> <out_sobel> [threads_per_rank]\n",
                   argv[0]);
        MPI_Finalize();
        return 1;
    }

    /* ── Thread count: CLI arg OR all available cores ── */
    int num_threads = (argc >= 5) ? atoi(argv[4]) : omp_get_max_threads();
    if (num_threads <= 0) num_threads = omp_get_max_threads();

    /* ── Image metadata broadcast ── */
    Image imgIn, imgGaussian, imgSobel;
    int dims[3]; /* [width, height, channels] */

    if (rank == 0) {
        printf("Running Hybrid MPI+OpenMP filter | %d process(es) x %d thread(s) = %d workers\n",
               size, num_threads, size * num_threads);

        imgIn = load_image(argv[1]);
        printf("Image loaded: %d x %d, %d channel(s)\n",
               imgIn.width, imgIn.height, imgIn.channels);

        imgGaussian = create_image(imgIn.width, imgIn.height, imgIn.channels);
        imgSobel    = create_image(imgIn.width, imgIn.height, imgIn.channels);

        dims[0] = imgIn.width;
        dims[1] = imgIn.height;
        dims[2] = imgIn.channels;
    }

    /* Broadcast image dimensions to all ranks */
    MPI_Bcast(dims, 3, MPI_INT, 0, MPI_COMM_WORLD);

    /* Non-root ranks allocate their own buffers */
    if (rank != 0) {
        imgIn.width    = dims[0];
        imgIn.height   = dims[1];
        imgIn.channels = dims[2];
        imgIn.data     = (unsigned char *)malloc(dims[0] * dims[1] * dims[2]);

        imgGaussian = create_image(dims[0], dims[1], dims[2]);
        imgSobel    = create_image(dims[0], dims[1], dims[2]);
    }

    /* Broadcast full image pixel data to every rank */
    MPI_Bcast(imgIn.data, dims[0] * dims[1] * dims[2],
              MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    /* ── Row decomposition (handles non-divisible image heights) ── */
    int base_rows = dims[1] / size;
    int remainder = dims[1] % size;

    /*
     * Distribute remainder rows one-per-rank to the first `remainder` ranks.
     * This ensures load balance is off by at most 1 row.
     */
    int start_row = rank * base_rows + (rank < remainder ? rank : remainder);
    int end_row   = start_row + base_rows + (rank < remainder ? 1 : 0);

    /* ── Gaussian Blur ── */
    double t_gauss_start = MPI_Wtime();
    apply_gaussian_blur(imgIn, imgGaussian, start_row, end_row, num_threads);
    double t_gauss_end   = MPI_Wtime();

    /* ── Sobel Edge Detection ── */
    double t_sobel_start = MPI_Wtime();
    apply_sobel_edge_detection(imgIn, imgSobel, start_row, end_row, num_threads);
    double t_sobel_end   = MPI_Wtime();

    /* ── Reduce max timing across all ranks (wall-clock for report) ── */
    double local_gauss_time = t_gauss_end - t_gauss_start;
    double local_sobel_time = t_sobel_end - t_sobel_start;
    double max_gauss_time, max_sobel_time;

    MPI_Reduce(&local_gauss_time, &max_gauss_time, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sobel_time, &max_sobel_time, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* ── Gather results back to rank 0 ── */
    int send_count_gauss = (end_row - start_row) * dims[0] * dims[2];
    int send_count_sobel = send_count_gauss; /* same slice size for both */

    int *recv_counts = NULL;
    int *displs      = NULL;

    if (rank == 0) {
        recv_counts = (int *)malloc(size * sizeof(int));
        displs      = (int *)malloc(size * sizeof(int));
        build_gatherv_params(size, dims[1], dims[0], dims[2],
                             recv_counts, displs);
    }

    /* Gather Gaussian output */
    MPI_Gatherv(
        &imgGaussian.data[start_row * dims[0] * dims[2]], send_count_gauss,
        MPI_UNSIGNED_CHAR,
        imgGaussian.data, recv_counts, displs,
        MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD
    );

    /* Gather Sobel output */
    MPI_Gatherv(
        &imgSobel.data[start_row * dims[0] * dims[2]], send_count_sobel,
        MPI_UNSIGNED_CHAR,
        imgSobel.data, recv_counts, displs,
        MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD
    );

    /* ── Rank 0: report timings and save images ── */
    if (rank == 0) {
        printf("Gaussian Blur  - Hybrid Execution Time : %.6f seconds\n",
               max_gauss_time);
        printf("Sobel Edge Det - Hybrid Execution Time : %.6f seconds\n",
               max_sobel_time);

        save_image(argv[2], imgGaussian);
        save_image(argv[3], imgSobel);
        printf("Results saved to: %s and %s\n", argv[2], argv[3]);

        free(recv_counts);
        free(displs);
    }

    /* ── Clean up ── */
    free_image(imgIn);
    free(imgGaussian.data);
    free(imgSobel.data);

    MPI_Finalize();
    return 0;
}