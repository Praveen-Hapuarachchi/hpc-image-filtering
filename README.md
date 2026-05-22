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

### Sample Output (1024x810, 3-channel image, 6 threads)

```
Running OpenMP filter with 6 thread(s)
Image loaded: 1024 x 810, 3 channel(s)
Gaussian Blur  - OpenMP Execution Time : 0.044493 seconds
Sobel Edge Det - OpenMP Execution Time : 0.031705 seconds
Results saved to: ../data/output/gaussian.jpg  and  ../data/output/sobel.jpg
```

---

## 3. Distributed Memory Parallelism - MPI

Uses MPI (Message Passing Interface) to distribute image rows across multiple processes. Each process independently applies the filters on its assigned partition, and results are gathered back to the root process.

### Compilation

Navigate to the `mpi/` directory and compile using `mpicc`:

```bash
mpicc src/main.c ../common/image_io.c -o image_filter_mpi -I../common -lm
```

### Execution

Use `mpirun` to launch with the desired number of processes:

```bash
mpirun -np 4 ./image_filter_mpi ../data/input/test.jpg ../data/output/mpi_gaussian.jpg ../data/output/mpi_sobel.jpg
```

### Sample Output (1024x810, 3-channel image, 4 processes)

```
Running MPI filter with 4 process(es)
Image loaded: 1024 x 810, 3 channel(s)
Gaussian Blur  - MPI Execution Time : 0.037992 seconds
Sobel Edge Det - MPI Execution Time : 0.034692 seconds
Results saved to: ../data/output/mpi_gaussian.jpg and ../data/output/mpi_sobel.jpg
```

---

## Performance Metrics

To calculate the performance gains for your report, use the following formulas:

1. **Speedup (S)**: $S = T_{serial} / T_{parallel}$
2. **Efficiency (E)**: $E = S / N$ (where $N$ is the number of processes/threads)

### Benchmarking Summary

Based on current tests with **1024x810** images:

| Implementation       | Filter        | Time (s)  | Speedup vs Serial |
|----------------------|---------------|-----------|-------------------|
| Serial               | Gaussian Blur | ~0.101s   | 1.00x (baseline)  |
| OpenMP (6 threads)   | Gaussian Blur | ~0.044s   | ~2.27x            |
| OpenMP (6 threads)   | Sobel Edge    | ~0.032s   | —                 |
| MPI (4 processes)    | Gaussian Blur | ~0.038s   | ~2.66x            |
| MPI (4 processes)    | Sobel Edge    | ~0.035s   | —                 |

> **Note:** OpenMP tests showed that 6 threads (~0.047s) outperformed 7 threads (~0.059s), demonstrating that thread management overhead can negate parallelism benefits beyond an optimal thread count. This is a key finding for scalability analysis in the report.