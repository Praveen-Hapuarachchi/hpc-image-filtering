#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include "../../common/image_io.h"
#include "../../common/timer.h"

typedef struct {
    Image in;
    Image out;
    int start_y;
    int end_y;
} ThreadData;

/* 1. GAUSSIAN BLUR WORKER */
void* gaussian_worker(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    const float kernel[3][3] = {
        {1/16.0f, 2/16.0f, 1/16.0f},
        {2/16.0f, 4/16.0f, 2/16.0f},
        {1/16.0f, 2/16.0f, 1/16.0f}
    };

    for (int y = data->start_y; y < data->end_y; y++) {
        if (y <= 0 || y >= data->in.height - 1) continue;
        for (int x = 1; x < data->in.width - 1; x++) {
            for (int c = 0; c < data->in.channels; c++) {
                float sum = 0.0f;
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int idx = ((y + ky) * data->in.width + (x + kx)) * data->in.channels + c;
                        sum += data->in.data[idx] * kernel[ky + 1][kx + 1];
                    }
                }
                data->out.data[(y * data->in.width + x) * data->in.channels + c] = (unsigned char)sum;
            }
        }
    }
    return NULL;
}

/* 2. SOBEL EDGE DETECTION WORKER */
void* sobel_worker(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    const int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (int y = data->start_y; y < data->end_y; y++) {
        if (y <= 0 || y >= data->in.height - 1) continue;
        for (int x = 1; x < data->in.width - 1; x++) {
            float sumX = 0.0f, sumY = 0.0f;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int base = ((y + ky) * data->in.width + (x + kx)) * data->in.channels;
                    float gray = (data->in.channels >= 3) ? 
                                 (0.299f * data->in.data[base] + 0.587f * data->in.data[base+1] + 0.114f * data->in.data[base+2]) : 
                                 (float)data->in.data[base];
                    sumX += gray * Gx[ky + 1][kx + 1];
                    sumY += gray * Gy[ky + 1][kx + 1];
                }
            }
            unsigned char magnitude = (unsigned char)fminf(sqrtf(sumX*sumX + sumY*sumY), 255.0f);
            int out_base = (y * data->in.width + x) * data->out.channels;
            for (int c = 0; c < data->out.channels; c++) {
                data->out.data[out_base + c] = magnitude;
            }
        }
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        printf("Usage: %s <input> <out_gaussian> <out_sobel> <num_threads>\n", argv[0]);
        return 1;
    }

    int num_threads = atoi(argv[4]);
    if (num_threads <= 0) {
        printf("Error: num_threads must be >= 1\n");
        return 1;
    }
    Image imgIn = load_image(argv[1]);
    Image imgGaussian = create_image(imgIn.width, imgIn.height, imgIn.channels);
    Image imgSobel = create_image(imgIn.width, imgIn.height, imgIn.channels);

    pthread_t threads[num_threads];
    ThreadData args[num_threads];
    int rows_per_thread = imgIn.height / num_threads;

    

    /* Execute Gaussian Blur */
    double t0 = get_time();
    for (int i = 0; i < num_threads; i++) {
        args[i].in = imgIn; args[i].out = imgGaussian;
        args[i].start_y = i * rows_per_thread;
        args[i].end_y = (i == num_threads - 1) ? imgIn.height : (i + 1) * rows_per_thread;
        pthread_create(&threads[i], NULL, gaussian_worker, &args[i]);
    }
    for (int i = 0; i < num_threads; i++) pthread_join(threads[i], NULL);
    double t1 = get_time();
    printf("Gaussian Blur  - Pthreads (%d threads): %.6f seconds\n", num_threads, t1 - t0);

    /* Execute Sobel Edge Detection */
    double t2 = get_time();
    for (int i = 0; i < num_threads; i++) {
        args[i].in = imgIn; args[i].out = imgSobel;
        pthread_create(&threads[i], NULL, sobel_worker, &args[i]);
    }
    for (int i = 0; i < num_threads; i++) pthread_join(threads[i], NULL);
    double t3 = get_time();
    printf("Sobel Edge Det - Pthreads (%d threads): %.6f seconds\n", num_threads, t3 - t2);

    save_image(argv[2], imgGaussian);
    save_image(argv[3], imgSobel);

    free_image(imgIn);
    free_image(imgGaussian);
    free_image(imgSobel);
    return 0;
}