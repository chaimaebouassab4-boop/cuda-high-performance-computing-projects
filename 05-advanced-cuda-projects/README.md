# Advanced CUDA Projects

This directory contains **advanced CUDA programming examples** focused on
**performance optimization techniques** commonly used in High Performance Computing (HPC).

Each program illustrates a key CUDA concept such as **shared memory**, **parallel reduction**,
and **constant memory**, and can be used for **study, experimentation, and revision**.

---

## 📌 Project Overview

| File | Concept | Key Optimization |
|----|----|----|
| `shared_memory_matmul.cu` | Matrix Multiplication | Shared memory tiling |
| `reduction_sum.cu` | Parallel Reduction | Tree-based reduction |
| `convolution_2d.cu` | 2D Convolution | Constant memory |

---

## 🧮 shared_memory_matmul.cu

**Purpose:**  
Optimized matrix multiplication using **shared memory tiling** to reduce global memory access.

**Key Features:**
- Uses shared memory to load matrix tiles
- Reduces redundant global memory accesses
- Improves performance over naive matrix multiplication

**Configuration:**
- Matrix size: `N = 1024`
- Input matrices initialized with all ones (easy correctness check)

**Learning Goals:**
- Understand shared memory usage
- Learn tiling strategies for matrix operations
- Compare naive vs optimized CUDA kernels

---

## ➕ reduction_sum.cu

**Purpose:**  
Efficiently computes the **sum of a large array** using parallel reduction.

**Key Features:**
- Tree-based reduction in shared memory
- Multi-block support
- Final accumulation performed on the host

**Configuration:**
- Array size: up to `1,000,000` elements
- Works for any large input size

**Learning Goals:**
- Understand parallel reduction patterns
- Learn synchronization with `__syncthreads()`
- Reduce warp divergence and memory traffic

---

## 🖼️ convolution_2d.cu

**Purpose:**  
Performs **2D convolution** using a small kernel stored in **constant memory**.

**Key Features:**
- Constant memory for convolution kernel
- Example uses Sobel filter on a uniform image
- Efficient access for read-only kernel data

**Extensions (Optional):**
- Add padding to handle image borders
- Extend to real images (grayscale / RGB)
- Experiment with different filters

**Learning Goals:**
- Understand constant memory usage
- Learn convolution operations in CUDA
- Apply CUDA to image processing problems

---

## ⚙️ Compilation & Execution

Compile each program using:

```bash
nvcc -arch=sm_75 file.cu -o file
