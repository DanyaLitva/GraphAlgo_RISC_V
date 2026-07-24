#pragma once
#include "matrix.h"
#include <vector>

template <typename T>
using mspgemmAlgorithm = void(*)(bool, bool, const sparseMtx<T>&, const sparseMtx<T>&, const sparseMtx<T>&, sparseMtx<T>&);

int* triangle_counting_vertex(const sparseMtx<int> &A, mspgemmAlgorithm<int> matrixMult, bool isParallel, bool isVectorization);
int64_t triangle_counting_masked_lu(const sparseMtx<int> &A, mspgemmAlgorithm<int> matrixMult, bool isParallel, bool isVectorization);
int64_t triangle_counting(const sparseMtx<int> &A, mspgemmAlgorithm<int> matrixMult, bool isParallel, bool isVectorization);

sparseMtx<int> k_truss(const sparseMtx<int> &A, int k, mspgemmAlgorithm<int> matrixMult, bool isParallel, bool isVectorization);

std::vector<float> brandes_batch(bool isParallel, bool isVectorization, const sparseMtx<int> &A, size_t batch_size);

template <typename T>
sparseMtx<T> floating_mask_mult(const sparseMtx<T> &A, double mask_density, mspgemmAlgorithm<T> matrixMult, bool isParallel, bool isVectorization);







//defenition of the template function floating_mask_mult
template <typename T>
sparseMtx<T> floating_mask_mult(const sparseMtx<T> &A, double mask_density, mspgemmAlgorithm<T> matrixMult, bool isParallel, bool isVectorization){
    return sparseMtx<T>();
}