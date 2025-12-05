# cuda-high-performance-computing-projects
High-Performance Computing repository containing CUDA implementations of vector/matrix operations, GPU-accelerated algorithms, HPC optimization techniques, and cybersecurity/big data applications. Includes benchmarks, visual explanations, and GPU vs CPU performance analytics.

# 🚀 GPU & High-Performance Computing Projects

<div align="center">

### CUDA • Parallel Computing • Cybersecurity • Big Data

*A curated collection of high-performance CUDA implementations, HPC algorithms, GPU-accelerated cybersecurity techniques, and benchmarking experiments.*

[![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)](https://isocpp.org/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

</div>

---

## 📁 Repository Structure

```
📦 cuda-hpc-projects
│
├── 01-vector-multiplication/
│   ├── vector_mul.cu
│   ├── README.md
│   └── results.png
│
├── 02-matrix-multiplication-basic/
│   ├── matmul_2x2.cu
│   └── README.md
│
├── 03-matrix-multiplication-generic/
│   ├── matmul_dynamic.cu
│   └── README.md
│
├── 04-performance-comparison-cpu-vs-gpu/
│   ├── cpu_version.c
│   ├── gpu_version.cu
│   ├── benchmarks.md
│   └── charts.png
│
├── 05-advanced-cuda-projects/
│   ├── shared_memory_matmul.cu
│   ├── reduction_sum.cu
│   ├── convolution_2d.cu
│   └── README.md
│
├── 06-gpu-for-big-data/
│   ├── gpu_sorting.cu
│   ├── gpu_histogram.cu
│   └── README.md
│
└── README.md
```

---

## 🎯 Project Goal

This repository showcases **practical and advanced GPU programming skills** using NVIDIA CUDA, with applications in:

- 🖥️ **High-Performance Computing (HPC)**
- 🔐 **Cybersecurity** (GPU-based cracking simulation, parallel port scans, anomaly detection)
- 📊 **Big Data Processing** (parallel sorting, histograms, clustering)
- 🔬 **Scientific & Numerical Computing**
- ⚡ **Algorithm Optimization** (GPU vs CPU)

It is designed to demonstrate **strong engineering ability**, **optimization skills**, and **parallel computing expertise** for recruiters and engineering teams.

---

## 🧠 Key Topics Covered

<table>
<tr>
<td width="33%" valign="top">

### 🟩 GPU Fundamentals
- CUDA threads, blocks, grids
- Memory hierarchy (global, shared, registers)
- Synchronization
- Memory coalescing
- Kernel optimization

</td>
<td width="33%" valign="top">

### 🟦 Performance Engineering
- CPU vs GPU benchmarking
- Profiling techniques
- Warp behavior
- Shared-memory tiling
- Occupancy optimization

</td>
<td width="34%" valign="top">

### 🟨 Applied Projects
- GPU-accelerated algorithms
- Cybersecurity simulations
- Big-data processing tasks
- Scientific computation

</td>
</tr>
</table>

---

## 📌 Highlighted Projects

### 1️⃣ Vector Multiplication (Intro to CUDA)

Simple kernel computing `Y[i] = X[i] * Y[i]`.

**Demonstrates:**
- Thread indexing
- Memory transfer
- Basic parallelism

---

### 2️⃣ Matrix Multiplication (2×2 and NxN)

Basic and dynamic versions.

**Demonstrates:**
- 2D thread grids
- Row/column mapping
- Memory layout

---

### 3️⃣ CPU vs GPU Performance Benchmark

A comparison of speed between serial CPU implementation and parallel CUDA kernel.

**Includes:**
- Execution time table
- GPU acceleration factor
- Visualization charts

---

### 4️⃣ Advanced CUDA Algorithms

Advanced HPC kernels including:

- **Shared-memory tiled matrix multiplication**
- **Parallel reduction** (sum / min / max)
- **2D convolution** (image filter)
- **Prefix sum** (scan)

These demonstrate **real GPU optimization techniques**.

---

### 5️⃣ GPU for Big Data & Cybersecurity

Practical applications linking HPC to specialized fields:

#### 🔐 Cybersecurity
- GPU password-cracking simulator (SHA-256 hashing)
- Parallel port scanner
- Log anomaly detection using CUDA

#### 📊 Big Data Processing
- Parallel sorting (bitonic / radix)
- Histogram computation
- K-means acceleration
- Data analytics workloads

These make the repo uniquely valuable.

---

## 🛠️ Installation & Compilation

### Requirements

```
✓ NVIDIA GPU compatible with CUDA
✓ CUDA Toolkit installed
✓ GCC or Clang compiler
✓ Linux/Windows
```

### Compile a CUDA Program

```bash
nvcc program_name.cu -o program_name
./program_name
```

### Compile a CPU C Program

```bash
gcc program_name.c -o program_name
./program_name
```

---

## 📊 Benchmarks

| Algorithm        | CPU Time | GPU Time | Acceleration   |
|------------------|----------|----------|----------------|
| Vector Mult      | 4.1 ms   | 0.12 ms  | **34× faster** |
| MatMul 512×512   | 1.8 s    | 0.07 s   | **25× faster** |
| SHA-256 cracking | 220k H/s | 6.1M H/s | **27× faster** |

*Your real results will be added here.*

---

## 🖼️ Visuals & Architecture Diagrams

Include diagrams such as:

- GPU memory hierarchy
- Block/grid layout
- Warp scheduling
- Thread indexing formula

Place them in `/assets/`.

---

## 🏆 Why This Repository Stands Out

<div align="center">

| Feature | Description |
|---------|-------------|
| 💪 **Hands-on GPU Programming** | Real-world CUDA implementations |
| 🔗 **Cross-domain Applications** | HPC + Cybersecurity + Big Data |
| 📈 **Performance Focus** | Benchmarks and optimizations |
| 📂 **Clean Structure** | Modular, well-organized projects |
| 📝 **Documentation** | Recruiter-friendly explanations |
| 🎓 **Technical Maturity** | Advanced concepts and best practices |

</div>

---

## 🤝 Contributions

Feel free to open issues or PRs for discussion, improvement, or new CUDA optimizations.

---

## 👨‍💻 About the Author

Master's student in **Cybersecurity & Big Data**, passionate about:

- High-Performance Computing
- GPU Programming
- Applied Machine Learning
- Security Engineering
- Distributed Systems

---


### ⭐ If you find this useful, star the repository to support the work!

**Made with 💻 and ☕**

</div>
