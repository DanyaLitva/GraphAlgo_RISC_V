#pragma once
#include "matrix.h"
#include <vector>
#include <chrono>
#include <iostream>
#include <type_traits>

const size_t REPEAT_COUNT_FLOAT = 1;

template <typename T>
using mspgemmAlgorithm = void(*)(bool, bool, const sparseMtx<T>&, const sparseMtx<T>&, const sparseMtx<T>&, sparseMtx<T>&);

int* triangle_counting_vertex(const sparseMtx<int> &A, mspgemmAlgorithm<int> matrixMult, bool isParallel, bool isVectorization);
int64_t triangle_counting_masked_lu(const sparseMtx<int> &A, mspgemmAlgorithm<int> matrixMult, bool isParallel, bool isVectorization);
int64_t triangle_counting(const sparseMtx<int> &A, mspgemmAlgorithm<int> matrixMult, bool isParallel, bool isVectorization);

sparseMtx<int> k_truss(const sparseMtx<int> &A, int k, mspgemmAlgorithm<int> matrixMult, bool isParallel, bool isVectorization);

std::vector<float> brandes_batch(bool isParallel, bool isVectorization, const sparseMtx<int> &A, size_t batch_size);

template <typename FloatType>
void floating_mask_mult(const sparseMtx<FloatType> &A, size_t mask_density, mspgemmAlgorithm<FloatType> matrixMult, bool isParallel, bool isVectorization) {
    size_t total_elements = static_cast<size_t>(A.m) * A.n;
    size_t m_nz = (total_elements + mask_density - 1) / mask_density;
    
    sparseMtx<FloatType> M(A.m, A.n, m_nz);
    std::vector<int> row_counts(A.m, 0);
    
    for (size_t k = 0; k < m_nz; ++k) {
        size_t global_idx = k * mask_density;
        size_t row = global_idx / A.n;
        size_t col = global_idx % A.n;
        
        row_counts[row]++;
        M.Col[k] = col;
        M.Val[k] = FloatType(1);
    }
    
    M.Rst[0] = 0;
    for (size_t i = 0; i < A.m; ++i) {
        M.Rst[i+1] = M.Rst[i] + row_counts[i];
    }

    sparseMtx<FloatType> result(A.m, A.n);

    auto start = std::chrono::steady_clock::now();
    // printf("Count repeat: %zu, Iteration done: ", REPEAT_COUNT_FLOAT);
    for(size_t i = 0; i < REPEAT_COUNT_FLOAT; ++i) {
        result = sparseMtx<FloatType>(A.m, A.n);
        matrixMult(isParallel, isVectorization, A, A, M, result);
        // printf("%zu ", i + 1);
    }
    auto finish = std::chrono::steady_clock::now();
    
    std::cout << "\nTime:       " << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << " ms\n";
}

template <typename FloatType>
void check_correctness_float_mult(const sparseMtx<FloatType>& A, size_t mask_density, mspgemmAlgorithm<FloatType> matrixMult, bool isParallel) {
  size_t total_elements = static_cast<size_t>(A.m) * A.n;
  size_t m_nz = (total_elements + mask_density - 1) / mask_density;

  sparseMtx<FloatType> M(A.m, A.n, m_nz);
  std::vector<int> row_counts(A.m, 0);

  for (size_t k = 0; k < m_nz; ++k) {
    size_t global_idx = k * mask_density;
    size_t row = global_idx / A.n;
    size_t col = global_idx % A.n;

    row_counts[row]++;
    M.Col[k] = col;
    M.Val[k] = FloatType(1);
  }

  M.Rst[0] = 0;
  for (size_t i = 0; i < A.m; ++i) {
    M.Rst[i + 1] = M.Rst[i] + row_counts[i];
  }

  sparseMtx<FloatType> result_scal(A.m, A.n);
  sparseMtx<FloatType> result_vec(A.m, A.n);

  std::cout << "Running scalar version...\n";
  matrixMult(isParallel, false, A, A, M, result_scal);

  std::cout << "Running vectorized version...\n";
  matrixMult(isParallel, true, A, A, M, result_vec);

  std::cout << "Comparing results...\n";

  if (result_scal.nz != result_vec.nz) {
    std::cout << "FAILED! Mismatch in non-zero count: scal=" << result_scal.nz << ", vec=" << result_vec.nz << "\n";
    return;
  }

  for (size_t i = 0; i <= A.m; ++i) {
    if (result_scal.Rst[i] != result_vec.Rst[i]) {
      std::cout << "FAILED! Mismatch in Rst array at row " << i << ": scal=" << result_scal.Rst[i] << ", vec=" << result_vec.Rst[i] << "\n";
      return;
    }
  }

  FloatType tol = std::is_same<FloatType, float>::value ? 1e-7 : 1e-13;
  FloatType max_diff = 0;

  for (size_t i = 0; i < result_scal.nz; ++i) {
    if (result_scal.Col[i] != result_vec.Col[i]) {
      std::cout << "FAILED! Mismatch in Col array at nz-index " << i << ": scal=" << result_scal.Col[i] << ", vec=" << result_vec.Col[i] << "\n";
      return;
    }

    FloatType diff = std::abs(result_scal.Val[i] - result_vec.Val[i]);
    if (diff > max_diff) max_diff = diff;

    if (diff > tol) {
      std::cout << "FAILED! Mismatch in Val array at nz-index " << i << "\n";
      std::cout << "        scal=" << result_scal.Val[i] << ", vec=" << result_vec.Val[i] << "\n";
      std::cout << "        (diff=" << diff << " > tol=" << tol << ")\n";
      return;
    }
  }

  std::cout << "Succes!\n";
  std::cout << "Max difference: " << max_diff << " (tolerance=" << tol << ")\n";
}