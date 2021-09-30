#include <random>
#include <iostream>
#include "matrix.hpp"

__global__
void multiply_matrices_kernel(const float* A, const float* B, float* C, 
  size_t m, size_t k, size_t p) {

   // block size
  size_t N = blockDim.x;

  // column and row of current submatrix (NxN block)
  size_t bx = blockIdx.x;
  size_t by = blockIdx.y;

  // column and row of current element inside submatrix
  size_t tx = threadIdx.x;
  size_t ty = threadIdx.y;
  
  // absolute position in matrix C
  size_t row = by * N + ty;
  size_t col = bx * N + tx;

  extern __shared__ float blocks[]; // two NxN blocks
  float sum = 0;

  if (col < p && row < m) {
    // Loop over all submatrices necessary to compute
    // corresponding submatrix of C
    for (size_t i = 0; i < k / N; ++i) {
      blocks[tx + N * ty] = A[i * N + tx + k * (ty + by * N)];
      blocks[tx + N * ty + N * N] = B[bx * N + tx + p * (ty + i * N)];

      // Synchronize threads to make sure blocks are loaded
      __syncthreads();

      for (size_t i = 0; i < N; ++i) {
        sum += blocks[ty * N + i] * blocks[tx + N * i + N * N];
      }

      // Synchronize threads to make sure that computation is complete
      __syncthreads();
    }
    C[row * p + col] = sum;
  }
}

__global__
void multiply_matrices_kernel_naive(const float* A, const float* B, float* C, 
  size_t m, size_t k, size_t p) {

  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0;
  if (col < p && row < m) {
    for (size_t i = 0; i < k; ++i) {
      sum += A[row * k + i] * B[col + p * i];
    }
    C[row * p + col] = sum;
  }
}

//
// Matrix multiplication on GPU
//
CPU_Matrix multiply_matrices(const GPU_Matrix& A, const GPU_Matrix& B, 
  size_t block_size, bool use_shared_memory = true) {

  size_t m = A.height(), k = A.width(), p = B.width();
  GPU_Matrix C(m, p);
  dim3 dim_grid(m / block_size, p / block_size); // blocks per grid
  dim3 dim_block(block_size, block_size); // threads per block
  if (use_shared_memory) {
    size_t shared_size = 2 * sizeof(float) * block_size * block_size;
    multiply_matrices_kernel<<<dim_grid, dim_block, shared_size>>>(
        A.data(), B.data(), 
        C.data(), m, k, p);
  } else {
    multiply_matrices_kernel_naive<<<dim_grid, dim_block>>>(
        A.data(), B.data(), 
        C.data(), m, k, p);
  }
  return C; // implicit conversion to CPU_Matrix
}

//
// Matrix multiplication on CPU
//
CPU_Matrix multiply_matrices(const CPU_Matrix& A, const CPU_Matrix& B) {
  CPU_Matrix C(A.height(), B.width());
  for (size_t row = 0; row < C.height(); ++row) {
    for (size_t col = 0; col < C.width(); ++col) {
      C(row, col) = 0;
      for (size_t i = 0; i < A.width(); ++i) {
        C(row, col) += A(row, i) * B(i, col);
      }
    }
  }
  return C;
}

int main() {
  // block size N x N
  constexpr size_t N = 16;

  // Matrices dimensions (assumed to be divisible by block size N)
  //
  //                 | -  -  -  -  - |
  //   | -  -  - |   |               |      | -  -  -  -  - |
  // m |    A    | x |       B       | k  = |       C       | m
  //   | -  -  - |   |               |      | -  -  -  -  - |
  //        k        | -  -  -  -  - |              p
  //                         p
  constexpr size_t m = 800 * N;
  constexpr size_t k = 1000 * N;
  constexpr size_t p = 900 * N;

  constexpr size_t seed = 0; // for replication purposes
  std::mt19937 gen{seed};
  std::uniform_real_distribution<float> distr(0, 10);
  auto generator = [&gen, &distr]() { return distr(gen); };
  CPU_Matrix h_A(m, k, generator);
  CPU_Matrix h_B(k, p, generator);
  GPU_Matrix d_A = h_A;
  GPU_Matrix d_B = h_B;

  float time_shared = 0.0, time_naive = 0.0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Multiply matrices on GPU without shared memory
  cudaEventRecord(start);
  auto C_gpu_naive = multiply_matrices(d_A, d_B, N, false);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_naive, start, stop);

  // Multiply matrices on GPU wit shared memory
  cudaEventRecord(start);
  auto C_gpu = multiply_matrices(d_A, d_B, N, true);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_shared, start, stop);

  std::cout << "Are matrices equal? " << std::boolalpha
            << (C_gpu_naive == C_gpu) << std::endl;
  std::cout << "Time of computation without shared memory: " 
            << time_naive << "ms" << std::endl;
  std::cout << "Time of computation with shared memory: " 
            << time_shared << "ms" << std::endl;
  return 0;
}