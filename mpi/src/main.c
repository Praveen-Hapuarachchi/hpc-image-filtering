#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "../../common/image_io.h"
#include "../../common/timer.h"

/* 1. GAUSSIAN BLUR (MPI) */
void apply_gaussian_blur(Image in, Image out, int start_row, int end_row) {
    const float kernel[3][3] = {
        {1/16.0f, 2/16.0f, 1/16.0f},
        {2/16.0f, 4/16.0f, 2/16.0f},
        {1/16.0f, 2/16.0f, 1/16.0f}
    };

    for (int y = start_row; y < end_row; y++) {
        if (y <= 0 || y >= in.height - 1) continue;
        for (int x = 1; x < in.width - 1; x++) {
            for (int c = 0; c < in.channels; c++) {
                float sum = 0.0f;
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int idx = ((y + ky) * in.width + (x + kx)) * in.channels + c;
                        sum += in.data[idx] * kernel[ky + 1][kx + 1];
                    }
                }
                out.data[(y * in.width + x) * in.channels + c] = (unsigned char)sum;
            }
        }
    }
}

/* 2. SOBEL EDGE DETECTION (MPI) */
void apply_sobel_edge_detection(Image in, Image out, int start_row, int end_row) {
    const int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (int y = start_row; y < end_row; y++) {
        if (y <= 0 || y >= in.height - 1) continue;
        for (int x = 1; x < in.width - 1; x++) {
            float sumX = 0.0f, sumY = 0.0f;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int base = ((y + ky) * in.width + (x + kx)) * in.channels;
                    float gray = (in.channels >= 3) ? 
                                 (0.299f * in.data[base] + 0.587f * in.data[base+1] + 0.114f * in.data[base+2]) : 
                                 (float)in.data[base];
                    sumX += gray * Gx[ky + 1][kx + 1];
                    sumY += gray * Gy[ky + 1][kx + 1];
                }
            }
            unsigned char magnitude = (unsigned char)fminf(sqrtf(sumX*sumX + sumY*sumY), 255.0f);
            int out_base = (y * in.width + x) * out.channels;
            for (int c = 0; c < out.channels; c++) {
                out.data[out_base + c] = magnitude;
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == 0) printf("Usage: %s <input> <out_gaussian> <out_sobel>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    Image imgIn, imgGaussian, imgSobel;
    int dims[3];

    if (rank == 0) {
        printf("Running MPI filter with %d process(es)\n", size);
        imgIn = load_image(argv[1]);
        printf("Image loaded: %d x %d, %d channel(s)\n", imgIn.width, imgIn.height, imgIn.channels);
        
        imgGaussian = create_image(imgIn.width, imgIn.height, imgIn.channels);
        imgSobel = create_image(imgIn.width, imgIn.height, imgIn.channels);
        dims[0] = imgIn.width; dims[1] = imgIn.height; dims[2] = imgIn.channels;
    }

    MPI_Bcast(dims, 3, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        imgIn.width = dims[0]; imgIn.height = dims[1]; imgIn.channels = dims[2];
        imgIn.data = malloc(dims[0] * dims[1] * dims[2]);
        imgGaussian = create_image(dims[0], dims[1], dims[2]);
        imgSobel = create_image(dims[0], dims[1], dims[2]);
    }

    MPI_Bcast(imgIn.data, dims[0] * dims[1] * dims[2], MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    int rows_per_proc = dims[1] / size;
    int start_row = rank * rows_per_proc;
    int end_row = (rank == size - 1) ? dims[1] : (rank + 1) * rows_per_proc;

    // ── GAUSSIAN BLUR ──
    double t0 = MPI_Wtime();
    apply_gaussian_blur(imgIn, imgGaussian, start_row, end_row);
    double t1 = MPI_Wtime();

    // ── SOBEL EDGE DETECTION ──
    double t2 = MPI_Wtime();
    apply_sobel_edge_detection(imgIn, imgSobel, start_row, end_row);
    double t3 = MPI_Wtime();

    // Gather results for Gaussian
    int send_count = (end_row - start_row) * dims[0] * dims[2];
    int* recv_counts = (rank == 0) ? malloc(size * sizeof(int)) : NULL;
    int* displs = (rank == 0) ? malloc(size * sizeof(int)) : NULL;

    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            int r_start = i * rows_per_proc;
            int r_end = (i == size - 1) ? dims[1] : (i + 1) * rows_per_proc;
            recv_counts[i] = (r_end - r_start) * dims[0] * dims[2];
            displs[i] = r_start * dims[0] * dims[2];
        }
    }

    MPI_Gatherv(&imgGaussian.data[start_row * dims[0] * dims[2]], send_count, MPI_UNSIGNED_CHAR,
                imgGaussian.data, recv_counts, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    MPI_Gatherv(&imgSobel.data[start_row * dims[0] * dims[2]], send_count, MPI_UNSIGNED_CHAR,
                imgSobel.data, recv_counts, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Gaussian Blur  - MPI Execution Time : %.6f seconds\n", t1 - t0);
        printf("Sobel Edge Det - MPI Execution Time : %.6f seconds\n", t3 - t2);
        save_image(argv[2], imgGaussian);
        save_image(argv[3], imgSobel);
        printf("Results saved to: %s and %s\n", argv[2], argv[3]);
        free(recv_counts); free(displs);
    }

    free(imgIn.data);
    free(imgGaussian.data);
    free(imgSobel.data);
    MPI_Finalize();
    return 0;
}