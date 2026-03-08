# Multi-Paradigm Parallel Image Filtering

A comparative study of parallel programming models (Pthreads, OpenMP, MPI, CUDA) 
for Gaussian Blur and Sobel Edge Detection.

## Project Contributors (Group 02)
- DILSHAN P.S. (EG/2020/3898)
- HAPUARACHCHI H.P.L. (EG/2020/3953)
- JAYALATH K.M.S.M. (EG/2020/3979)

## Overview
This project implements image filtering algorithms across five parallel paradigms 
to analyze speedup, efficiency, and scalability on HPC clusters. The two primary 
algorithms implemented are:
1. **Gaussian Blur**: Used for noise reduction using a 3x3 convolution kernel.
2. **Sobel Edge Detection**: Used to find image gradients and highlights edges.

## Project Structure
- `serial/`: Sequential baseline implementation.
- `pthreads/`: Shared memory parallelism using POSIX threads.
- `openmp/`: Directive-based shared memory parallelism.
- `mpi/`: Distributed memory using message passing.
- `hybrid/`: Combined MPI and OpenMP approach.
- `cuda/`: GPU-accelerated implementation.
- `common/`: Shared utilities for image I/O and timing.
- `data/`: Input images and generated output results.

---

## 1. Serial Implementation (Baseline)
The serial version is the reference for all performance measurements. It processes the image sequentially on a single CPU core.

### Compilation
Navigate to the `serial/` directory and run:
```bash
gcc src/main.c ../common/image_io.c -o image_filter -lm

```

### Execution

The serial program requires three arguments: input path, Gaussian output path, and Sobel output path.

```bash
./image_filter_serial ../data/input/test.jpg ../data/output/gaussian_serial.jpg ../data/output/sobel_serial.jpg

```

---

## 2. Shared Memory Parallelism - OpenMP

Uses `#pragma omp parallel for` directives to distribute the image processing workload across multiple CPU cores.

### Compilation

Navigate to the `openmp/` directory and use the provided Makefile:

```bash
make

```

*Manual command:* `gcc -fopenmp src/main.c ../common/image_io.c -o openmp_filter -I../common -lm`

### Execution

You can specify the number of threads as the fourth argument:

```bash
./openmp_filter ../data/input/test.jpg ../data/output/gaussian_omp.jpg ../data/output/sobel_omp.jpg 4

```

Alternatively, use the Makefile run command:

```bash
make run THREADS=6

```

---

## Performance Metrics

To calculate the performance gains for your report, use the following formulas:

1. **Speedup (S)**: $S = T_{serial} / T_{parallel}$
2. **Efficiency (E)**: $E = S / N$ (where $N$ is the number of threads)

### Benchmarking Example

Based on current tests with 1024x810 images:

* **Serial Gaussian**: ~0.101s
* **OpenMP Gaussian (6 threads)**: ~0.047s
* **Achieved Speedup**: ~2.14x

```

### Key Improvements in this MD file:
1. **Directory Context**: Added specific navigation instructions (`cd` commands) to ensure paths like `../../common` work correctly.
2. **Compilation Flags**: Included the `-I` flag for header inclusion and `-lm` for the math library to prevent linker errors.
3. **Makefile Integration**: Included the `make run` syntax which is helpful for quick testing with different thread counts.
4. **Logic Explanation**: Added a brief overview of why we use Gaussian and Sobel to help anyone reading the project understand the goal.

**Quick Check**: Your OpenMP test showed that with **6 threads**, you got **0.047s**, but with **7 threads**, it was **0.059s**. This is a great observation for your report—it shows that adding more threads doesn't always make it faster because of thread management overhead!

```