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

sparseMtx<int> k_truss_test(const sparseMtx<int> &A, int k, mspgemmAlgorithm<int> matrixMult, bool isParallel, bool isVectorization);

int64_t triangle_counting_test(const sparseMtx<int> &A, mspgemmAlgorithm<int> matrixMult, bool isParallel, bool isVectorization);
