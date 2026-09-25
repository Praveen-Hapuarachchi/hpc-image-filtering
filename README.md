# Multi-Paradigm Parallel Image Filtering

**EC7207 — High Performance Computing | Group 02**

| Member | Registration No. |
|---|---|
| DILSHAN P.S. | EG/2020/3898 |
| HAPUARACHCHI H.P.L. | EG/2020/3953 |
| JAYALATH K.M.S.M. | EG/2020/3979 |

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Algorithms Implemented](#2-algorithms-implemented)
3. [Parallel Paradigms](#3-parallel-paradigms)
4. [Project Structure](#4-project-structure)
5. [Prerequisites](#5-prerequisites)
6. [Build and Run — Step by Step](#6-build-and-run--step-by-step)
7. [Performance Metrics](#7-performance-metrics)
8. [Benchmark Results](#8-benchmark-results)
9. [Key Findings](#9-key-findings)
10. [Viva Questions and Answers](#10-viva-questions-and-answers)

---

## 1. Project Overview

Image filtering operations such as **Gaussian Blur** and **Sobel Edge Detection**
are convolution-based algorithms that demand significant computation, especially
at high resolutions. Processing these serially is a bottleneck for real-time or
large-scale applications.

This project implements both filters across **six parallel programming approaches**
and performs a rigorous comparison of execution time, speedup, efficiency, and
scalability. All parallel outputs are verified for numerical correctness against
the serial baseline using RMSE (Root Mean Square Error).

```
Serial ──► Pthreads ──► OpenMP ──► MPI ──► Hybrid (MPI+OpenMP) ──► CUDA
  CPU          CPU         CPU      CPU          CPU+CPU              GPU
(1 core)   (N threads) (N threads) (P procs)   (P×T workers)    (thousands)
```

---

## 2. Algorithms Implemented

### 2.1 Gaussian Blur

Reduces image noise by applying a weighted average over each pixel's 3×3
neighbourhood using the following kernel:

```
          1   2   1
K  =  1/16 × 2   4   2
          1   2   1
```

Each output pixel is the sum of neighbouring pixels weighted by K.
Border pixels (row 0, row H-1, col 0, col W-1) are skipped — no full
3×3 neighbourhood is available there.

### 2.2 Sobel Edge Detection

Detects edges by computing the image gradient magnitude using two directional
kernels:

```
       -1   0  +1          -1  -2  -1
Gx =  -2   0  +2     Gy =   0   0   0
       -1   0  +1          +1  +2  +1

Magnitude = min( sqrt(Gx² + Gy²) , 255 )
```

For colour (RGB) images, each pixel is first converted to grayscale
using the luminance formula:

```
Gray = 0.299 × R  +  0.587 × G  +  0.114 × B
```

The magnitude is written to all output channels, producing a
grayscale-equivalent edge map stored as a 3-channel image — consistent
across all six implementations.

---

## 3. Parallel Paradigms

| # | Paradigm | Technology | Memory Model | Decomposition Strategy |
|---|---|---|---|---|
| 1 | Serial baseline | C (gcc) | — | None — single thread |
| 2 | Shared memory | **Pthreads** | Shared | Row-wise, manual `pthread_create/join` |
| 3 | Shared memory | **OpenMP** | Shared | `#pragma omp parallel for` on rows |
| 4 | Distributed memory | **MPI** | Distributed | Row partition + `MPI_Bcast` + `MPI_Gatherv` |
| 5 | Hybrid | **MPI + OpenMP** | Both | MPI rows × OpenMP threads per rank |
| 6 | GPU acceleration | **CUDA** | GPU global/shared | 2D thread grid + shared memory tiling |

### How Parallelism is Applied — Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                   Input Image  (W × H pixels)                │
└──────────────────────────────────────────────────────────────┘
          │
          ├─── PTHREADS / OPENMP / MPI / HYBRID
          │
          │    Divide image rows among N workers:
          │
          │    Worker 0  │ rows   0   →  H/N - 1    │
          │    Worker 1  │ rows  H/N  →  2H/N - 1   │  each worker runs
          │    Worker 2  │ rows 2H/N  →  3H/N - 1   │  convolution on
          │    ...       │ ...                       │  its slice only
          │    Worker N-1│ rows       →  H - 1       │
          │
          │    Recombine all slices → full output image
          │
          └─── CUDA
               One thread per output pixel:

               Grid  = ceil(W/16) × ceil(H/16) blocks
               Block = 16 × 16 threads  (256 threads/block)

               Each block loads an 18×18 shared memory tile
               (16×16 output + 1-pixel halo on every side)
               then all 256 threads compute from fast shared memory.
```

---

## 4. Project Structure

```
hpc-image-filtering/
│
├── common/                       # Shared utilities used by ALL implementations
│   ├── image_io.h                # Image struct: { width, height, channels, *data }
│   ├── image_io.c                # load_image, save_image, create_image, free_image
│   ├── timer.h                   # get_time() — gettimeofday(), returns double seconds
│   ├── stb_image.h               # Single-header image loader  (JPEG / PNG / BMP)
│   └── stb_image_write.h         # Single-header image writer  (JPEG output)
│
├── serial/
│   ├── src/main.c                # Sequential baseline — reference for correctness & timing
│   └── Makefile
│
├── pthreads/
│   ├── src/main.c                # POSIX threads — row-wise decomposition
│   │                             # gaussian_thread() + sobel_thread() workers
│   │                             # launch_threads() helper for clean create/join
│   └── Makefile
│
├── openmp/
│   ├── src/main.c                # OpenMP — #pragma omp parallel for schedule(static)
│   │                             # num_threads() clause, default(none) data sharing
│   └── Makefile
│
├── mpi/
│   ├── src/main.c                # MPI — MPI_Bcast full image → row partition →
│   │                             # MPI_Gatherv results back to rank 0
│   └── Makefile
│
├── hybrid/
│   ├── src/main.c                # MPI (inter-process) + OpenMP (intra-process)
│   │                             # MPI_Reduce(MPI_MAX) for correct wall-clock timing
│   │                             # Fair remainder row distribution across ranks
│   └── Makefile
│
├── cuda/
│   ├── src/main.cu               # CUDA — 2D thread grid, __shared__ memory tiling
│   │                             # __constant__ memory for Gaussian kernel
│   │                             # cudaEvent timing for accurate GPU measurement
│   └── Makefile
│
├── data/
│   ├── input/
│   │   └── test.jpg              # Test image: 1024 × 810, 3-channel RGB JPEG
│   └── output/                   # All output images written here
│       ├── gaussian_serial.jpg
│       ├── sobel_serial.jpg
│       ├── gaussian_pthreads.jpg
│       ├── sobel_pthreads.jpg
│       ├── gaussian_omp.jpg
│       ├── sobel_omp.jpg
│       ├── mpi_gaussian.jpg
│       ├── mpi_sobel.jpg
│       ├── hybrid_gaussian.jpg
│       ├── hybrid_sobel.jpg
│       ├── cuda_gaussian.jpg
│       └── cuda_sobel.jpg
│
├── scripts/                      # Benchmark automation scripts
├── report/                       # Analysis report and diagrams
└── README.md
```

---

## 5. Prerequisites

All implementations require **Linux or WSL (Windows Subsystem for Linux)**.

```bash
# Update package list
sudo apt update

# GCC compiler (Serial, Pthreads, OpenMP)
sudo apt install gcc

# OpenMP — bundled with GCC, no separate install needed
# Verify:  gcc -fopenmp --version

# Pthreads — bundled with glibc, no separate install needed
# Link with: -lpthread

# MPI — for MPI and Hybrid implementations
sudo apt install libopenmpi-dev openmpi-bin

# CUDA — requires an NVIDIA GPU
# Download installer from: https://developer.nvidia.com/cuda-downloads
# After install, verify:
nvcc --version
nvidia-smi
```

### Verify everything is ready

```bash
gcc       --version          # C compiler
mpicc     --version          # MPI C compiler wrapper
mpirun    --version          # MPI launcher
nvcc      --version          # CUDA compiler (if GPU available)
nvidia-smi                   # GPU status and compute capability
```

---

## 6. Build and Run — Step by Step

> All commands are executed in a **WSL / Linux terminal**.
> Run implementations in the order shown — Serial must run first
> to produce the baseline timing for speedup calculations.

---

### Step 0 — Navigate to Project Root

```bash
cd "/mnt/c/Users/hapup/OneDrive/Desktop/7th sem/EC7207 High Performance Computing/Project/hpc-image-filtering"
```

---

### Step 1 — Serial Baseline

The reference implementation. Single CPU core, no parallelism.
Run this first and record the timing — all speedup values are
calculated relative to these numbers.

```bash
cd serial

# Compile
gcc src/main.c ../common/image_io.c -I../common -lm -o serial_filter

# Run
./serial_filter \
    ../data/input/test.jpg \
    ../data/output/gaussian_serial.jpg \
    ../data/output/sobel_serial.jpg
```

Expected output:
```
Running Serial filter (1 thread)
Image loaded: 1024 x 810, 3 channel(s)
Gaussian Blur  - Serial Execution Time : 0.XXXXXX seconds
Sobel Edge Det - Serial Execution Time : 0.XXXXXX seconds
Results saved to: ../data/output/gaussian_serial.jpg and ../data/output/sobel_serial.jpg
```

---

### Step 2 — Pthreads

POSIX threads with manual row-wise decomposition.
Both Gaussian and Sobel are implemented with separate thread worker functions.

```bash
cd ../pthreads

# Compile
gcc src/main.c ../common/image_io.c -I../common -lpthread -lm -o pthreads_filter

# Run — 2 threads
./pthreads_filter \
    ../data/input/test.jpg \
    ../data/output/gaussian_pthreads.jpg \
    ../data/output/sobel_pthreads.jpg 2

# Run — 4 threads
./pthreads_filter \
    ../data/input/test.jpg \
    ../data/output/gaussian_pthreads.jpg \
    ../data/output/sobel_pthreads.jpg 4

# Run — 8 threads
./pthreads_filter \
    ../data/input/test.jpg \
    ../data/output/gaussian_pthreads.jpg \
    ../data/output/sobel_pthreads.jpg 8

# Via Makefile
make
make run THREADS=4
```

Expected output:
```
Running Pthreads filter with 4 thread(s)
Image loaded: 1024 x 810, 3 channel(s)
Gaussian Blur  - Pthreads Execution Time : 0.XXXXXX seconds
Sobel Edge Det - Pthreads Execution Time : 0.XXXXXX seconds
Results saved to: ../data/output/gaussian_pthreads.jpg and ../data/output/sobel_pthreads.jpg
```

---

### Step 3 — OpenMP

Compiler-directive parallelism. The `#pragma omp parallel for` directive
distributes the row loop across threads automatically.

```bash
cd ../openmp

# Compile
make
# (manual: gcc -fopenmp src/main.c ../common/image_io.c -I../common -lm -o openmp_filter)

# Run — 2 threads
./openmp_filter \
    ../data/input/test.jpg \
    ../data/output/gaussian_omp.jpg \
    ../data/output/sobel_omp.jpg 2

# Run — 4 threads
./openmp_filter \
    ../data/input/test.jpg \
    ../data/output/gaussian_omp.jpg \
    ../data/output/sobel_omp.jpg 4

# Run — 6 threads
make run THREADS=6

# Run — 8 threads
make run THREADS=8
```

Expected output:
```
Running OpenMP filter with 6 thread(s)
Image loaded: 1024 x 810, 3 channel(s)
Gaussian Blur  - OpenMP Execution Time : 0.044493 seconds
Sobel Edge Det - OpenMP Execution Time : 0.031705 seconds
Results saved to: ../data/output/gaussian_omp.jpg and ../data/output/sobel_omp.jpg
```

---

### Step 4 — MPI

Distributed memory model. Each process receives the full image via
`MPI_Bcast`, processes its assigned row slice, and results are collected
back to rank 0 via `MPI_Gatherv`.

```bash
cd ../mpi

# Compile
mpicc src/main.c ../common/image_io.c -I../common -lm -o image_filter_mpi
# (or: make)

# Run — 2 processes
mpirun -np 2 ./image_filter_mpi \
    ../data/input/test.jpg \
    ../data/output/mpi_gaussian.jpg \
    ../data/output/mpi_sobel.jpg

# Run — 4 processes
mpirun -np 4 ./image_filter_mpi \
    ../data/input/test.jpg \
    ../data/output/mpi_gaussian.jpg \
    ../data/output/mpi_sobel.jpg

# Run — 8 processes
mpirun -np 8 ./image_filter_mpi \
    ../data/input/test.jpg \
    ../data/output/mpi_gaussian.jpg \
    ../data/output/mpi_sobel.jpg
```

Expected output:
```
Running MPI filter with 4 process(es)
Image loaded: 1024 x 810, 3 channel(s)
Gaussian Blur  - MPI Execution Time : 0.037992 seconds
Sobel Edge Det - MPI Execution Time : 0.034692 seconds
Results saved to: ../data/output/mpi_gaussian.jpg and ../data/output/mpi_sobel.jpg
```

---

### Step 5 — Hybrid MPI + OpenMP

Combines MPI inter-process distribution with OpenMP intra-process threading.
Total workers = MPI processes × OpenMP threads per process.

```bash
cd ../hybrid

# Compile
mpicc -fopenmp src/main.c ../common/image_io.c -I../common -lm -o hybrid_filter
# (or: make)

# Run — 2 processes × 2 threads = 4 workers
mpirun -np 2 ./hybrid_filter \
    ../data/input/test.jpg \
    ../data/output/hybrid_gaussian.jpg \
    ../data/output/hybrid_sobel.jpg 2

# Run — 4 processes × 2 threads = 8 workers
mpirun -np 4 ./hybrid_filter \
    ../data/input/test.jpg \
    ../data/output/hybrid_gaussian.jpg \
    ../data/output/hybrid_sobel.jpg 2

# Run — 2 processes × 4 threads = 8 workers
mpirun -np 2 ./hybrid_filter \
    ../data/input/test.jpg \
    ../data/output/hybrid_gaussian.jpg \
    ../data/output/hybrid_sobel.jpg 4

# Via Makefile
make run NP=4 THREADS=2
```

Expected output:
```
Running Hybrid MPI+OpenMP filter | 4 process(es) x 2 thread(s) = 8 workers
Image loaded: 1024 x 810, 3 channel(s)
Gaussian Blur  - Hybrid Execution Time : 0.094477 seconds
Sobel Edge Det - Hybrid Execution Time : 0.065175 seconds
Results saved to: ../data/output/hybrid_gaussian.jpg and ../data/output/hybrid_sobel.jpg
```

---

### Step 6 — CUDA

GPU-accelerated implementation. Each GPU thread processes one output pixel.
Shared memory tiling reduces global memory bandwidth usage by ~88%.

```bash
cd ../cuda

# Check your GPU compute capability
nvidia-smi
# Look for "Compute Capability" e.g. 7.5 → use sm_75

# Edit Makefile: set ARCH = sm_XX to match your GPU
# RTX 40xx → sm_89  |  RTX 30xx → sm_86
# RTX 20xx → sm_75  |  GTX 10xx → sm_61
nano Makefile

# Compile
nvcc -O2 -arch=sm_75 src/main.cu ../common/image_io.c \
     -I../common -lm -o cuda_filter
# (or: make  — after setting ARCH in Makefile)

# Run
./cuda_filter \
    ../data/input/test.jpg \
    ../data/output/cuda_gaussian.jpg \
    ../data/output/cuda_sobel.jpg

# Via Makefile
make run
```

Expected output:
```
Running CUDA filter on: NVIDIA GeForce RTX XXXX (Compute X.X)
Image loaded: 1024 x 810, 3 channel(s)
Gaussian Blur  - CUDA Execution Time : 0.XXXXXX seconds
Sobel Edge Det - CUDA Execution Time : 0.XXXXXX seconds
Results saved to: ../data/output/cuda_gaussian.jpg and ../data/output/cuda_sobel.jpg
```

---

## 7. Performance Metrics

### Speedup

How many times faster the parallel version is compared to serial:

```
S = T_serial / T_parallel
```

### Efficiency

How well each worker is utilised (ideal = 100%):

```
E = S / N × 100%       where N = number of threads or processes
```

### RMSE — Numerical Accuracy

Root Mean Square Error between parallel output and serial baseline.
A value of 0.00 means bit-perfect agreement:

```
RMSE = sqrt(  (1 / (W × H × C))  ×  Σ (parallel[i] − serial[i])²  )
```

---

## 8. Benchmark Results

> Test image: **1024 × 810 pixels, 3-channel RGB JPEG**
> Machine: Single node (WSL on Windows 10)

### Gaussian Blur — Timing and Speedup

| Implementation | Config | Time (s) | Speedup | Efficiency |
|---|---|---|---|---|
| Serial | 1 core | ~0.101 | 1.00× | 100% |
| Pthreads | 2 threads | TBD | TBD | TBD |
| Pthreads | 4 threads | TBD | TBD | TBD |
| Pthreads | 8 threads | TBD | TBD | TBD |
| OpenMP | 2 threads | TBD | TBD | TBD |
| OpenMP | 4 threads | TBD | TBD | TBD |
| OpenMP | 6 threads | ~0.044 | ~2.27× | ~38% |
| OpenMP | 8 threads | TBD | TBD | TBD |
| MPI | 2 processes | TBD | TBD | TBD |
| MPI | 4 processes | ~0.038 | ~2.66× | ~66% |
| MPI | 8 processes | TBD | TBD | TBD |
| Hybrid | 4P × 2T | ~0.094 | ~1.07× | ~13% |
| CUDA | GPU | TBD | TBD | — |

### Sobel Edge Detection — Timing and Speedup

| Implementation | Config | Time (s) | Speedup | Efficiency |
|---|---|---|---|---|
| Serial | 1 core | TBD | 1.00× | 100% |
| Pthreads | 4 threads | TBD | TBD | TBD |
| OpenMP | 6 threads | ~0.032 | TBD | TBD |
| MPI | 4 processes | ~0.035 | TBD | TBD |
| Hybrid | 4P × 2T | ~0.065 | TBD | TBD |
| CUDA | GPU | TBD | TBD | — |

> Fill in TBD values after running each implementation with the commands
> in Section 6. Use the formula S = T_serial / T_parallel for each row.

---

## 9. Key Findings

### Finding 1 — MPI outperforms OpenMP at equivalent worker counts

MPI with 4 processes (~0.038s) beat OpenMP with 6 threads (~0.044s) despite
using fewer workers. This is because MPI processes each have their own memory
space — no false sharing or cache-line contention. OpenMP threads share the
same cache and can interfere with each other's memory access patterns.

### Finding 2 — OpenMP has an optimal thread count

OpenMP with 6 threads (~0.044s) outperformed 7 threads (~0.059s). Beyond the
optimal count, thread management overhead (spawning, scheduling, joining)
exceeds the benefit of parallelising fewer remaining rows. The optimal thread
count is hardware-dependent — typically equals the number of physical cores.

### Finding 3 — Hybrid is slower than MPI on a single node

Hybrid MPI+OpenMP (4×2 = 8 workers, ~0.094s) ran slower than plain MPI
(4 processes, ~0.038s) on a single machine. All MPI processes share the
same physical CPU cores, so adding OpenMP threads causes oversubscription,
cache thrashing, and combined MPI+OpenMP startup overhead. On a true
multi-node HPC cluster, Hybrid would be the fastest CPU implementation.

### Finding 4 — CUDA expected to give largest speedup

CUDA launches thousands of threads simultaneously — one per pixel — and
uses shared memory tiling to minimise global memory access. For a 1024×810
image, this means 829,440 threads launching concurrently. Expected speedup
of 10×–50× over serial depending on GPU model.

---

## 10. Viva Questions and Answers

---

### Q1: What is the fundamental difference between shared memory and distributed memory parallelism? How does your project demonstrate both?

**Answer:**

In **shared memory parallelism** (Pthreads, OpenMP), all threads run within
the same process and directly access a single memory space. No data needs
to be copied between workers — every thread reads from the same `imgIn.data`
buffer and writes to its own non-overlapping slice of `imgOut.data`.
Communication is implicit through memory.

In **distributed memory parallelism** (MPI), each process has its own
completely separate address space. Processes cannot see each other's
variables at all. Data must be explicitly sent using message-passing calls.
In our MPI implementation, rank 0 loads the image and calls `MPI_Bcast` to
send the entire pixel buffer to every other process. After each process
computes its row slice, `MPI_Gatherv` collects all results back to rank 0.

Our project demonstrates all three models:
- **Pthreads / OpenMP** → shared memory: all threads read from the same
  `imgIn` buffer with no copying
- **MPI** → distributed memory: explicit `MPI_Bcast` and `MPI_Gatherv`
  calls for all data movement
- **Hybrid** → both simultaneously: MPI handles inter-process communication,
  OpenMP handles intra-process thread parallelism within each MPI rank

---

### Q2: Explain exactly how Pthreads parallelism works in your implementation. How do you avoid race conditions?

**Answer:**

In our Pthreads implementation, the image height is divided into N equal
row bands — one per thread. The decomposition is:

```
base_rows = H / N
Thread i  →  rows [ i × base_rows  to  (i+1) × base_rows - 1 ]
Last thread gets any remainder rows
```

Each thread is given a `ThreadData` struct containing the full input image,
the full output image buffer, and its `start_row` / `end_row` boundaries.
`pthread_create` launches all N threads simultaneously, and `pthread_join`
waits for all to finish before timing stops.

Race conditions are completely avoided because:

1. **Input is read-only** — all threads read from `imgIn.data` but never
   write to it. Multiple simultaneous reads are always safe.

2. **Output writes are non-overlapping** — thread i only writes to rows
   `[start_row, end_row)` of `imgOut.data`. Since row ranges are
   non-overlapping across threads, no two threads ever write to the same
   memory address. No mutex or lock is needed.

3. **The kernel array is `const` and stack-allocated** inside each thread
   function — it is thread-local, not shared mutable state.

---

### Q3: Why does your Hybrid MPI+OpenMP implementation run slower than plain MPI on your test machine? When would Hybrid be faster?

**Answer:**

On our single machine (WSL/local PC), all 4 MPI processes share the same
physical CPU cores. Adding 2 OpenMP threads per process creates 8 software
workers competing for the same hardware. This causes:

1. **Oversubscription** — more software threads than physical cores forces
   the OS to context-switch, which adds scheduling overhead
2. **Cache contention** — 8 workers accessing the same image data thrash
   each other out of the shared L2/L3 cache
3. **MPI overhead is not eliminated** — `MPI_Bcast` still sends the full
   image to every process even on a single node, consuming memory bandwidth
4. **Combined startup cost** — MPI process creation plus OpenMP thread
   spawn inside each process adds latency that exceeds the benefit of
   extra parallelism for a small image

Hybrid would outperform all CPU implementations on a **true multi-node
HPC cluster**:
- One MPI process per physical node → no oversubscription
- OpenMP threads fill all cores within each node → full shared-memory
  parallelism with no network overhead
- MPI only carries the inter-node communication over the network
- This gives the best of both: fast on-node shared memory + scalable
  cross-node message passing

---

### Q4: Explain how your CUDA implementation uses shared memory. Why is this faster than accessing global memory directly?

**Answer:**

In a naive GPU implementation, each thread would independently read all 9
pixels of its 3×3 neighbourhood from **global memory** — the main GPU DRAM
with approximately 400–800 cycle latency per access.

The problem is that neighbouring threads access **overlapping regions**.
For a 16×16 thread block, the 256 threads collectively need an 18×18 patch
(the 16×16 output area plus a 1-pixel halo on each side). A naive approach
causes up to 9×256 = 2,304 global memory reads per block with heavy
redundancy — adjacent threads repeatedly fetch the same pixel from DRAM.

Our shared memory tiling works in three stages:

1. **Cooperative load** — every thread in a block loads one or more pixels
   from global memory into a `__shared__ unsigned char smem[18][18]` array,
   including the border halo pixels. This touches global memory only once
   per unique pixel needed.

2. **Synchronise** — `__syncthreads()` acts as a barrier, ensuring every
   thread has finished loading before any thread starts computing. Without
   this, a thread might read from shared memory before its neighbour has
   written to it.

3. **Compute from shared memory** — each thread reads its 9 kernel values
   from `smem` instead of global memory. Shared memory is on-chip SRAM
   with ~4 cycle latency — approximately 100× faster than global DRAM.

The Gaussian kernel itself is stored in `__constant__` memory — a read-only
cache optimised for the case where all threads in a warp read the same
address simultaneously (uniform access pattern), which triggers a single
broadcast instead of 32 separate memory transactions.

---

### Q5: How did you verify that your parallel implementations produce correct results? What could cause numerical differences?

**Answer:**

We verify correctness using **RMSE (Root Mean Square Error)** — comparing
every pixel of each parallel output image against the serial baseline:

```
RMSE = sqrt( (1 / (W × H × C)) × Σ (parallel[i] − serial[i])² )
```

A result of RMSE = 0.00 means bit-perfect, pixel-identical output.

Potential sources of differences:

1. **Race conditions (Pthreads)** — if two threads wrote to overlapping
   memory regions, output would be non-deterministic. Our non-overlapping
   row decomposition completely eliminates this risk.

2. **Boundary / halo errors (MPI)** — if the row slice boundary pixels
   cannot access the neighbouring process's halo row, they produce wrong
   results. Our MPI implementation broadcasts the complete image to every
   process, so every process always has full access to all neighbour pixels
   — at the cost of higher memory usage.

3. **Floating-point reordering** — different thread execution orders can
   change the sequence of floating-point additions, which due to rounding
   may produce slightly different values. In our implementation each pixel
   is computed independently with the same fixed 3×3 kernel multiplication
   order, so this does not occur.

4. **CUDA precision** — GPU hardware performs the same `float` arithmetic
   as CPU but may use fused multiply-add (FMA) instructions that differ
   by the last floating-point bit. Any such difference is ≤ 1 intensity
   level after `(unsigned char)` truncation.

5. **Consistent `(unsigned char)` truncation** — all six implementations
   use the same `(unsigned char)sum` and `(unsigned char)magnitude` casts,
   so truncation errors are identical across all versions.

---

### Q6: What is Amdahl's Law and how do your results demonstrate it?

**Answer:**

**Amdahl's Law** defines the theoretical maximum speedup achievable by
parallelising a program that contains both serial and parallel sections:

```
S(N) =        1
         ─────────────────────────
         f  +  (1 − f) / N

Where:
  N  =  number of parallel workers
  f  =  fraction of execution time that is inherently serial
  (1-f) =  parallelisable fraction
```

As N → ∞, the maximum speedup approaches 1/f. Even if f = 5% (serial), the
maximum possible speedup is 20× regardless of how many cores are used.

In our project the serial fractions include:
- `stbi_load` — image loading, single thread only
- `stbi_write_jpg` — image saving, single thread only
- `MPI_Bcast` — broadcasting full image to all processes (grows with P)
- `MPI_Gatherv` — gathering results back to rank 0
- Thread/process creation and join overhead

Our results confirm Amdahl's predictions:

| Implementation | Workers | Speedup | Theoretical max (f=5%) |
|---|---|---|---|
| OpenMP | 6 threads | ~2.27× | up to 20× |
| MPI | 4 processes | ~2.66× | up to 20× |
| Hybrid | 8 workers | ~1.07× | limited by oversubscription |

The low efficiency (~38% for OpenMP, ~66% for MPI) compared to ideal (100%)
shows that the serial fraction and communication overheads are consuming a
significant proportion of execution time — exactly as Amdahl's Law predicts.

The OpenMP regression from 6 to 7 threads is a practical demonstration of
Amdahl's overhead component: once the parallelisable work per thread becomes
too small, the fixed cost of thread management dominates.

---

### Q7: Compare Pthreads and OpenMP. What are the advantages and disadvantages of each for this application?

**Answer:**

| Aspect | Pthreads | OpenMP |
|---|---|---|
| Control level | Manual — full control over thread creation, data assignment, and joining | Automatic — compiler handles thread creation and work distribution |
| Code complexity | Higher — explicit `pthread_create`, `ThreadData` structs, `pthread_join` | Lower — single `#pragma omp parallel for` directive |
| Portability | POSIX standard — works on any Unix/Linux system | Compiler-dependent — requires `-fopenmp` flag |
| Flexibility | Can implement any decomposition strategy, custom scheduling | Limited to loop-based parallelism and OpenMP constructs |
| Race condition risk | Higher — programmer must reason carefully about data sharing | Lower — OpenMP `default(none)` forces explicit declaration of sharing |
| Performance | Comparable — both create OS-level threads on the same cores | Comparable — `schedule(static)` gives equal-sized chunks like Pthreads |
| Debugging | Harder — data races are silent and non-deterministic | Easier — OpenMP race detector tools available |

For our image filtering application, **OpenMP is the better practical choice**
because the workload is uniform (every row takes the same time), the loop
structure is simple, and `#pragma omp parallel for schedule(static)` perfectly
matches the row-wise decomposition. Pthreads gives equivalent performance
but requires significantly more code for no additional benefit in this case.

Pthreads would be preferable if we needed irregular decomposition, dynamic
load balancing, or fine-grained synchronisation between specific threads.

---

*End of README — EC7207 High Performance Computing | Group 02*