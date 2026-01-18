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


__global__ void blur_shared_31(
    unsigned char* input,
    unsigned char* output,
    int w, int h, int c)
{
    // Shared memory tile including halo for 31x31 blur
    __shared__ unsigned char tile[TILE_W * TILE_H * 3];

    // Block origin in global image coordinates
    int bx = blockIdx.x * 16;
    int by = blockIdx.y * 16;

    // Thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Load input image region (including halo) into shared memory
    for (int yy = ty; yy < TILE_H; yy += blockDim.y) {
        for (int xx = tx; xx < TILE_W; xx += blockDim.x) {

            int gx = bx + xx - RADIUS;
            int gy = by + yy - RADIUS;

            // Clamp global coordinates at image boundaries
            if (gx < 0) gx = 0;
            if (gy < 0) gy = 0;
            if (gx >= w) gx = w - 1;
            if (gy >= h) gy = h - 1;

            int in_idx = (gy * w + gx) * 3;
            int t_idx  = (yy * TILE_W + xx) * 3;

            // Copy RGB values into shared memory
            tile[t_idx]     = input[in_idx];
            tile[t_idx + 1] = input[in_idx + 1];
            tile[t_idx + 2] = input[in_idx + 2];
        }
    }

    // Ensure all threads have loaded shared memory
    __syncthreads();

    // Compute output pixel coordinates
    int x = bx + tx;
    int y = by + ty;

    if (x >= w || y >= h) return;

    float r = 0, g = 0, b = 0;

    // Apply 31x31 blur using shared memory
    for (int dy = -RADIUS; dy <= RADIUS; dy++) {
        for (int dx = -RADIUS; dx <= RADIUS; dx++) {

            int nx = tx + dx + RADIUS;
            int ny = ty + dy + RADIUS;

            int t_idx = (ny * TILE_W + nx) * 3;

            r += tile[t_idx];
            g += tile[t_idx + 1];
            b += tile[t_idx + 2];
        }
    }

    // Write blurred RGB values to output image
    int out_idx = (y * w + x) * 3;
    output[out_idx]     = (unsigned char)(r / (KSIZE * KSIZE));  // R
    output[out_idx + 1] = (unsigned char)(g / (KSIZE * KSIZE));  // G
    output[out_idx + 2] = (unsigned char)(b / (KSIZE * KSIZE));  // B
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

// Horizontal pass: blur along X axis, output to temp buffer
__global__ void blur_separable_h(
    unsigned char* input,
    float* temp,
    int w, int h, int c)
{
    // Shared memory: row tile + left/right halo
    __shared__ unsigned char tile[(SEP_TILE_W + 2 * RADIUS) * 3];

    int x = blockIdx.x * SEP_TILE_W + threadIdx.x;
    int y = blockIdx.y;  // One row per block.y

    if (y >= h) return;

    int tile_start = blockIdx.x * SEP_TILE_W - RADIUS;

    //all threads load tile + halo with boundary clamping
    for (int i = threadIdx.x; i < SEP_TILE_W + 2 * RADIUS; i += blockDim.x) {
        int gx = tile_start + i;
        if (gx < 0) gx = 0;
        if (gx >= w) gx = w - 1;

        int in_idx = (y * w + gx) * 3;
        int t_idx = i * 3;
        tile[t_idx]     = input[in_idx];
        tile[t_idx + 1] = input[in_idx + 1];
        tile[t_idx + 2] = input[in_idx + 2];
    }

    __syncthreads();  // Wait for all threads to finish loading

    if (x >= w) return;

    // Sum 31 horizontal neighbors from shared memory
    float r = 0, g = 0, b = 0;
    for (int dx = -RADIUS; dx <= RADIUS; dx++) {
        int t_idx = (threadIdx.x + RADIUS + dx) * 3;
        r += tile[t_idx];
        g += tile[t_idx + 1];
        b += tile[t_idx + 2];
    }

    // Store to temp as float (preserves precision for vertical pass)
    int out_idx = (y * w + x) * 3;
    temp[out_idx]     = r / KSIZE;
    temp[out_idx + 1] = g / KSIZE;
    temp[out_idx + 2] = b / KSIZE;
}

// Vertical pass: blur along Y axis, output final result
__global__ void blur_separable_v(
    float* temp,
    unsigned char* output,
    int w, int h, int c)
{
    // Shared memory: column tile + top/bottom halo
    __shared__ float tile[(SEP_TILE_H + 2 * RADIUS) * 3];

    int x = blockIdx.x;
    int y = blockIdx.y * SEP_TILE_H + threadIdx.x;

    if (x >= w) return;

    int tile_start = blockIdx.y * SEP_TILE_H - RADIUS;

    //all threads load tile + halo with boundary clamping
    for (int i = threadIdx.x; i < SEP_TILE_H + 2 * RADIUS; i += blockDim.x) {
        int gy = tile_start + i;
        if (gy < 0) gy = 0;
        if (gy >= h) gy = h - 1;

        int in_idx = (gy * w + x) * 3;
        int t_idx = i * 3;
        tile[t_idx]     = temp[in_idx];
        tile[t_idx + 1] = temp[in_idx + 1];
        tile[t_idx + 2] = temp[in_idx + 2];
    }

    __syncthreads();

    if (y >= h) return;

    // Sum 31 vertical neighbors from shared memory
    float r = 0, g = 0, b = 0;
    for (int dy = -RADIUS; dy <= RADIUS; dy++) {
        int t_idx = (threadIdx.x + RADIUS + dy) * 3;
        r += tile[t_idx];
        g += tile[t_idx + 1];
        b += tile[t_idx + 2];
    }

    // Store final result as unsigned char
    int out_idx = (y * w + x) * 3;
    output[out_idx]     = (unsigned char)(r / KSIZE);
    output[out_idx + 1] = (unsigned char)(g / KSIZE);
    output[out_idx + 2] = (unsigned char)(b / KSIZE);
}