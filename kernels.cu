#include <cuda_runtime.h>
#define TILE_W (16 + 2*RADIUS)
#define TILE_H (16 + 2*RADIUS)
#define RADIUS 15
#define KSIZE (2*RADIUS + 1)
#define SEP_TILE_W 128
#define SEP_TILE_H 16

// Naive blur kernel - straightforward approach where each thread computes
// the average of all pixels in its neighborhood. Simple but not optimized.
__global__ void blur_naive_31(
    unsigned char* input,
    unsigned char* output,
    int w, int h, int c)
{
    // Calculate the global position of this thread in the image
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Bounds check - early exit if out of image bounds
    if (x >= w || y >= h) return;

    float r = 0, g = 0, b = 0;
    int count = 0;

    // Iterate through the neighborhood defined by RADIUS
    for (int dy = -RADIUS; dy <= RADIUS; dy++) {
        for (int dx = -RADIUS; dx <= RADIUS; dx++) {

            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                int idx = (ny * w + nx) * c;
                r += input[idx];
                g += input[idx + 1];
                b += input[idx + 2];
                count++;
            }
        }
    }

    // Write the averaged result back to the output image
    int out = (y * w + x) * c;
    output[out]     = (unsigned char)(r / count);
    output[out + 1] = (unsigned char)(g / count);
    output[out + 2] = (unsigned char)(b / count);
}

// Merge kernel - combines original and blurred images based on a mask
// This creates the bokeh effect: sharp foreground, blurred background
__global__ void merge_mask(
    unsigned char* orig,
    unsigned char* blur,
    unsigned char* mask,
    unsigned char* out,
    int w, int h, int c,
    unsigned char threshold)
{
    // Get the pixel position this thread is responsible for
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    int id = (y * w + x) * c;
    int mid = (y * w + x);

    // Decision: use original if mask value is above threshold (foreground),
    // otherwise use blurred version (background)
    if (mask[mid] > threshold) {
        out[id]     = orig[id];
        out[id + 1] = orig[id + 1];
        out[id + 2] = orig[id + 2];
    } else {
        out[id]     = blur[id];
        out[id + 1] = blur[id + 1];
        out[id + 2] = blur[id + 2];
    }
}