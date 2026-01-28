# GPU-Based Depth-Aware Background Blur (CUDA)

This project implements a **DSLR-style portrait background blur** using CUDA.
The pipeline processes a sequence of video frames, blurs the background using a
**31×31 box blur**, and preserves the subject using a binary mask.

Three blur kernels are provided:

- **Naive Kernel** (`blur_naive_31`)
  - Direct global-memory sampling
  - 31×31 = 961 samples per pixel
  - Baseline for comparison

- **Shared Memory Kernel** (`blur_shared_31`)
  - Uses shared-memory tiling (16×16 blocks + halo)
  - 31×31 = 961 samples per pixel
  - Faster memory access than naive

- **Separable + Shared Memory Kernel** (`blur_separable_h` + `blur_separable_v`)
  - Two-pass approach: horizontal blur → vertical blur
  - 31 + 31 = 62 samples per pixel (~15× fewer than 2D)
  - Both passes use shared memory
  - Fastest implementation

The project can process a full batch of frames and measures **per-frame GPU time**.

---

## Dataset & Results

Dataset and output videos available here:
🔗 [Google Drive - Dataset & Results](https://drive.google.com/drive/folders/19qtjntF4VsYbGNFjB8XgPcqoTAl_U4TI?usp=sharing)

---

## Project Structure
```
├── main.cu                 # Main pipeline: load → blur → merge → save
├── kernels.cu              # Naive, shared-memory, and separable blur kernels
├── utils.cpp               # Image loading/writing (stb_image)
├── utils.h
├── stb_image.h
├── stb_image_write.h
├── frames/                 # Input frames (added by user)
├── masks/                  # Binary masks (added by user)
├── output_frames/          # Output frames written here
└── bokeh                   # Compiled binary
```

---

## Dependencies

No external libraries besides:
- stb_image.h (already included)
- stb_image_write.h (already included)
- FFmpeg *(for stitching video)*

Everything required is in this repository (except FFmpeg).

---

## How to Compile

From inside the project folder:
```bash
module load cuda
nvcc -O3 -arch=sm_80 main.cu utils.cpp kernels.cu -o bokeh
```

This produces:
```
./bokeh
```

---

## How to Run

Default run:
```bash
./bokeh
```

It will:
1. Load frames from `frames/`
2. Load masks from `masks/`
3. Blur background
4. Merge subject + blurred background
5. Save results into `output_frames/`
6. Print timing per frame

Example output:
```
Frame 1 GPU time: 2.34 ms
Frame 2 GPU time: 2.31 ms
...
Average GPU time per frame: 2.32 ms
```

---

## Switching Between Kernels

Inside `main.cu` there is a mode variable:
```cpp
int mode = 2;  // 0 = naive, 1 = shared, 2 = separable
```

| Mode | Kernel | Description |
|------|--------|-------------|
| 0 | `blur_naive_31` | Global memory, 961 samples |
| 1 | `blur_shared_31` | Shared memory, 961 samples |
| 2 | `blur_separable_h` + `blur_separable_v` | Separable + shared memory, 62 samples |

Change the mode value, recompile, and run to compare performance.

---

## GPU Timing

The project measures:
- Blur kernel time
- Merge kernel time
- Full GPU pipeline per frame

Timing uses CUDA events:
```cpp
cudaEventRecord(start);
// blur + merge kernels
cudaEventRecord(stop);
cudaEventElapsedTime(&ms, start, stop);
```

---
