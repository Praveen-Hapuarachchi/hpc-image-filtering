#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include "../../common/image_io.h"
#include "../../common/timer.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../../common/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../common/stb_image_write.h"

typedef struct {
    Image in;
    Image out;
    int start_y;
    int end_y;
} ThreadData;

// Gaussian Kernel 3x3
float kernel[3][3] = {
    {1/16.0, 2/16.0, 1/16.0},
    {2/16.0, 4/16.0, 2/16.0},
    {1/16.0, 2/16.0, 1/16.0}
};

void* gaussian_thread(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int w = data->in.width;
    int ch = data->in.channels;

    for (int y = data->start_y; y < data->end_y; y++) {
        // Stay within bounds for 3x3 kernel
        if (y == 0 || y >= data->in.height - 1) continue; 

        for (int x = 1; x < w - 1; x++) {
            for (int c = 0; c < ch; c++) {
                float sum = 0.0;
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        sum += data->in.data[((y + ky) * w + (x + kx)) * ch + c] * kernel[ky + 1][kx + 1];
                    }
                }
                data->out.data[(y * w + x) * ch + c] = (unsigned char)sum;
            }
        }
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("Usage: %s <input> <output> <num_threads>\n", argv[0]);
        return 1;
    }

    int num_threads = atoi(argv[3]);
    int w, h, ch;
    unsigned char *pixels = stbi_load(argv[1], &w, &h, &ch, 0);
    Image imgIn = {w, h, ch, pixels};
    Image imgOut = {w, h, ch, malloc(w * h * ch)};

    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];

    double start = get_time();

    int rows_per_thread = h / num_threads;
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].in = imgIn;
        thread_data[i].out = imgOut;
        thread_data[i].start_y = i * rows_per_thread;
        thread_data[i].end_y = (i == num_threads - 1) ? h : (i + 1) * rows_per_thread;
        
        pthread_create(&threads[i], NULL, gaussian_thread, &thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    double end = get_time();
    printf("Pthreads (%d threads) Time: %f seconds\n", num_threads, end - start);

    stbi_write_jpg(argv[2], w, h, ch, imgOut.data, 100);

    free(imgOut.data);
    stbi_image_free(pixels);
    return 0;
}