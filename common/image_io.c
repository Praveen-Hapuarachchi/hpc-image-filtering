#include <stdio.h>
#include <stdlib.h>
#include "image_io.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

Image load_image(const char *filename) {
    Image img;
    img.data = stbi_load(filename, &img.width, &img.height, &img.channels, 0);

    if (!img.data) {
        printf("Error loading image %s\n", filename);
        exit(1);
    }

    return img;
}

void save_image(const char *filename, Image img) {
    stbi_write_jpg(filename, img.width, img.height, img.channels, img.data, 100);
}

void free_image(Image img) {
    stbi_image_free(img.data);
}

Image create_image(int width, int height, int channels) {
    Image img;
    img.width = width;
    img.height = height;
    img.channels = channels;
    img.data = malloc(width * height * channels);
    return img;
}