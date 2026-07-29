#pragma once
#include "matrix.h"
#include <vector>
#include <chrono>

const size_t REPEAT_COUNT_FLOAT = 10;

template <typename T>
using mspgemmAlgorithm = void(*)(bool, bool, const sparseMtx<T>&, const sparseMtx<T>&, const sparseMtx<T>&, sparseMtx<T>&);

int* triangle_counting_vertex(const sparseMtx<int> &A, mspgemmAlgorithm<int> matrixMult, bool isParallel, bool isVectorization);
int64_t triangle_counting_masked_lu(const sparseMtx<int> &A, mspgemmAlgorithm<int> matrixMult, bool isParallel, bool isVectorization);
int64_t triangle_counting(const sparseMtx<int> &A, mspgemmAlgorithm<int> matrixMult, bool isParallel, bool isVectorization);

sparseMtx<int> k_truss(const sparseMtx<int> &A, int k, mspgemmAlgorithm<int> matrixMult, bool isParallel, bool isVectorization);

std::vector<float> brandes_batch(bool isParallel, bool isVectorization, const sparseMtx<int> &A, size_t batch_size);

template <typename T>
sparseMtx<T> floating_mask_mult(const sparseMtx<T> &A, size_t mask_density, mspgemmAlgorithm<T> matrixMult, bool isParallel, bool isVectorization);







//defenition of the template function floating_mask_mult
template <typename FloatType>
void floating_mask_mult(const sparseMtx<FloatType> &A, size_t mask_density, mspgemmAlgorithm<FloatType> matrixMult, bool isParallel, bool isVectorization) {
    size_t total_elements = static_cast<size_t>(A.m) * A.n;
    size_t m_nz = (total_elements + mask_density - 1) / mask_density;
    
    sparseMtx<bool> M(A.m, A.n, m_nz);
    std::vector<int> row_counts(A.m, 0);
    
    for (size_t k = 0; k < m_nz; ++k) {
        size_t global_idx = k * mask_density;
        size_t row = global_idx / A.n;
        size_t col = global_idx % A.n;
        
        row_counts[row]++;
        M.Col[k] = col;
        M.Val[k] = true;
    }
    
    M.Rst[0] = 0;
    for (size_t i = 0; i < A.m; ++i) {
        M.Rst[i+1] = M.Rst[i] + row_counts[i];
    }

    sparseMtx<FloatType> result(A.m, A.n);

    auto start = std::chrono::steady_clock::now();
    printf("Count repeat: %zu, Iteration done: ", REPEAT_COUNT_FLOAT);
    for(size_t i = 0; i < REPEAT_COUNT_FLOAT; ++i) {
        result = sparseMtx<FloatType>(A.m, A.n);
        matrixMult(isParallel, isVectorization, A, A, M, result);
        printf("%zu ", i + 1);
    }
    auto finish = std::chrono::steady_clock::now();
    
    std::cout << "Time:       " << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << " ms\n";
}