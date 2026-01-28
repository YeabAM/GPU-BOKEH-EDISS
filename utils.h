#ifndef UTILS_H
#define UTILS_H

#include <string>

// Load an image from disk.
// If force_grayscale is true, the image is loaded as a single-channel mask.
unsigned char* load_image(const std::string& path,
                          int& width,
                          int& height,
                          int& channels,
                          bool force_grayscale = false);

// Save image data to disk as a PNG file.
void save_image(const std::string& path,
                unsigned char* data,
                int width,
                int height,
                int channels);

#endif // UTILS_H
