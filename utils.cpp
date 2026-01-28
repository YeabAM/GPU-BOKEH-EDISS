// Enable stb image implementations (should appear in ONE source file only)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "utils.h"
#include <iostream>

// Load an image from disk using stb_image
// If force_grayscale is true, the image is loaded as 1 channel (mask)
unsigned char* load_image(const std::string& path,
                          int& width,
                          int& height,
                          int& channels,
                          bool force_grayscale)
{
    // Request either RGB (3) or grayscale (1)
    int desired_channels = force_grayscale ? 1 : 3;

    // Load image data
    unsigned char* img = stbi_load(path.c_str(),
                                   &width,
                                   &height,
                                   &channels,
                                   desired_channels);

    // Error handling
    if (!img) {
        std::cerr << "Error loading image: " << path << std::endl;
    }

    // Ensure reported channel count matches requested format
    channels = desired_channels;
    return img;
}

// Save image data to disk as PNG
void save_image(const std::string& path,
                unsigned char* data,
                int width,
                int height,
                int channels)
{
    // Write image with row stride = width * channels
    stbi_write_png(path.c_str(),
                   width,
                   height,
                   channels,
                   data,
                   width * channels);
}
