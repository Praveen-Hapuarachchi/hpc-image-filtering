#ifndef IMAGE_IO_H
#define IMAGE_IO_H

typedef struct {
    int width;
    int height;
    int channels;
    unsigned char *data;
} Image;

Image load_image(const char *filename);
void save_image(const char *filename, Image img);
void free_image(Image img);
Image create_image(int width, int height, int channels);

#endif