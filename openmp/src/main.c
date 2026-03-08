#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "../../common/image_io.h"
#include "../../common/timer.h"

/*  1. GAUSSIAN BLUR  (3×3 kernel, OpenMP) */

void apply_gaussian_blur(Image in, Image out, int num_threads) {

    const float kernel[3][3] = {
        {1/16.0f, 2/16.0f, 1/16.0f},
        {2/16.0f, 4/16.0f, 2/16.0f},
        {1/16.0f, 2/16.0f, 1/16.0f}
    };

    #pragma omp parallel for schedule(static) num_threads(num_threads) \
                             default(none) shared(in, out, kernel)
    for (int y = 1; y < in.height - 1; y++) {
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

/* 2. SOBEL EDGE DETECTION  (OpenMP) */
void apply_sobel_edge_detection(Image in, Image out, int num_threads) {

    /* Sobel kernels */
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
                             default(none) shared(in, out, Gx, Gy)
    for (int y = 1; y < in.height - 1; y++) {
        for (int x = 1; x < in.width - 1; x++) {

            /* Accumulate gradient using the first channel */
            float sumX = 0.0f, sumY = 0.0f;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    /* Convert to grayscale on-the-fly if multi-channel */
                    int base = ((y + ky) * in.width + (x + kx)) * in.channels;
                    float gray;
                    if (in.channels >= 3) {
                        gray = 0.299f  * in.data[base    ]
                             + 0.587f  * in.data[base + 1]
                             + 0.114f  * in.data[base + 2];
                    } else {
                        gray = (float)in.data[base];
                    }
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

/*  3. MAIN Usage: ./openmp_filter <input> <output_gaussian> <output_sobel> [num_threads] */

int main(int argc, char **argv) {

    if (argc < 4) {
        printf("Usage: %s <input> <output_gaussian> <output_sobel> [num_threads]\n", argv[0]);
        return 1;
    }

    /* Optional thread count argument; default = all available cores */
    int num_threads = (argc >= 5) ? atoi(argv[4]) : omp_get_max_threads();
    if (num_threads <= 0) num_threads = omp_get_max_threads();

    printf("Running OpenMP filter with %d thread(s)\n", num_threads);

    /* ── Load input image ── */
    Image imgIn = load_image(argv[1]);
    printf("Image loaded: %d x %d, %d channel(s)\n",
           imgIn.width, imgIn.height, imgIn.channels);

    /* ── Allocate output buffers ── */
    Image imgGaussian = create_image(imgIn.width, imgIn.height, imgIn.channels);
    Image imgSobel    = create_image(imgIn.width, imgIn.height, imgIn.channels);

    /* ── Gaussian Blur ── */
    double t0 = get_time();
    apply_gaussian_blur(imgIn, imgGaussian, num_threads);
    double t1 = get_time();
    printf("Gaussian Blur  - OpenMP Execution Time : %.6f seconds\n", t1 - t0);

    /* ── Sobel Edge Detection ── */
    double t2 = get_time();
    apply_sobel_edge_detection(imgIn, imgSobel, num_threads);
    double t3 = get_time();
    printf("Sobel Edge Det - OpenMP Execution Time : %.6f seconds\n", t3 - t2);

    /* ── Save results ── */
    save_image(argv[2], imgGaussian);
    save_image(argv[3], imgSobel);
    printf("Results saved to: %s  and  %s\n", argv[2], argv[3]);

    /* ── Clean up ── */
    free_image(imgIn);
    free(imgGaussian.data);
    free(imgSobel.data);

    return 0;
}
