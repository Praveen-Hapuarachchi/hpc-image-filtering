# Multi-Paradigm Parallel Image Filtering

A comparative study of parallel programming models (Pthreads, OpenMP, MPI, CUDA) 
for Gaussian Blur and Sobel Edge Detection.

## Project Contributors (Group 02)
- DILSHAN P.S. (EG/2020/3898)
- HAPUARACHCHI H.P.L. (EG/2020/3953)
- JAYALATH K.M.S.M. (EG/2020/3979)

## Overview
This project implements image filtering algorithms across five parallel paradigms 
to analyze speedup, efficiency, and scalability on HPC clusters.

## Project Structure
- `serial/`: Sequential baseline.
- `pthreads/`: Shared memory using POSIX threads.
- `openmp/`: Directive-based shared memory parallelism.
- `mpi/`: Distributed memory using message passing.
- `hybrid/`: Combined MPI and OpenMP approach.
- `cuda/`: GPU-accelerated implementation.