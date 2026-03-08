#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../../common/image_io.h"
#include "../../common/timer.h"

void apply_gaussian_blur(Image in, Image out) {
    float kernel[3][3] = {
        {1/16.0, 2/16.0, 1/16.0},
        {2/16.0, 4/16.0, 2/16.0},
        {1/16.0, 2/16.0, 1/16.0}
    };

    for (int y = 1; y < in.height - 1; y++) {
        for (int x = 1; x < in.width - 1; x++) {
            for (int c = 0; c < in.channels; c++) {
                float sum = 0.0;
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        sum += in.data[((y + ky) * in.width + (x + kx)) * in.channels + c] * kernel[ky + 1][kx + 1];
                    }
                }
                out.data[(y * in.width + x) * in.channels + c] = (unsigned char)sum;
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 3) { 
        printf("Usage: %s <input> <output>\n", argv[0]); 
        return 1; 
    }

    // Use the wrapper functions from image_io.h
    Image imgIn = load_image(argv[1]);
    Image imgOut = create_image(imgIn.width, imgIn.height, imgIn.channels);

    double start = get_time();
    apply_gaussian_blur(imgIn, imgOut);
    double end = get_time();

    printf("Serial Execution Time: %f seconds\n", end - start);

    // Use the wrapper function to save
    save_image(argv[2], imgOut);

    // Clean up using our helpers
    free_image(imgIn);
    free(imgOut.data);

    return 0;
}