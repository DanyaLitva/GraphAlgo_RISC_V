//matrix algorithm by Alexander Ustinov from https://github.com/UAlex322/GraphAlgo/tree/latest
#pragma once

#include "matrix.h"
#include <queue>
#include <vector>
#include <utility>
#include <typeinfo>
#include <iostream>
#include <cstring>

// declarations

template <typename T>
struct MSA {
  static enum { UNALLOWED = 0, ALLOWED, SET } msa_states;
  char* state;
  T* value;
  size_t  len;

  MSA(size_t n) {
    value = new T[n]();
    state = new char[n]();
    len = n;
  }

  ~MSA() {
    delete[] value;
    delete[] state;
  }
};

template <typename T>
sparseMtx<T> transpose(const sparseMtx<T>& A);

// MSA cmask
template<typename T, typename U>
void _mspgemm_msa_cmask_parallel_scalar(const sparseMtx<T>& A,
  const sparseMtx<T>& B,
  const sparseMtx<U>& M,
  sparseMtx<T>& C);

template<typename T, typename U>
void _mspgemm_msa_cmask_parallel_vectorized(const sparseMtx<T>& A,
  const sparseMtx<T>& B,
  const sparseMtx<U>& M,
  sparseMtx<T>& C);

template<typename T, typename U>
void mspgemm_msa_cmask(bool isVectorization,
  const sparseMtx<T>& A,
  const sparseMtx<T>& B,
  const sparseMtx<U>& M,
  sparseMtx<T>& C);

// definitions

template <typename T>
sparseMtx<T> transpose(const sparseMtx<T>& A) {
  sparseMtx<T> AT(A.n, A.m, A.nz);

  // filling the column indices array and current column positions array
  for (size_t i = 0; i < A.nz; ++i)
    ++AT.Rst[A.Col[i] + 1];
  for (size_t i = 0; i < AT.m; ++i)
    AT.Rst[i + 1] += AT.Rst[i];

  // transposing
  for (size_t i = 0; i < A.m; ++i) {
    for (int j = A.Rst[i]; j < A.Rst[i + 1]; ++j) {
      AT.Val[AT.Rst[A.Col[j]]] = std::move(A.Val[j]);
      AT.Col[AT.Rst[A.Col[j]]++] = i;
    }
  }
  // set Rst indices to normal state
  // AT.Rst[AT.m] already has the correct value
  for (int i = AT.m - 1; i > 0; --i)
    AT.Rst[i] = AT.Rst[i - 1];
  AT.Rst[0] = 0;

  return AT;
}

// MSA cmask parallel scalar
template<typename T, typename U>
void _mspgemm_msa_cmask_parallel_scalar(const sparseMtx<T>& A, const sparseMtx<T>& B, const sparseMtx<U>& M, sparseMtx<T>& C) {
  //std::cerr << "Scalar\n";
#pragma omp parallel
  {
    MSA<T> accum(B.n);
    std::vector<int> changed_states;
    changed_states.reserve(B.n);

#pragma omp for schedule(dynamic, 64)
    for (size_t i = 0; i < A.m; ++i) {
      int m_begin = M.Rst[i];
      int m_end = M.Rst[i + 1];
      int row_nz = 0;

      for (int t = A.Rst[i]; t < A.Rst[i + 1]; ++t) {
        int k = A.Col[t];
        int b_begin = B.Rst[k];
        int b_end = B.Rst[k + 1];

        for (int j = b_begin; j < b_end; ++j) {
          int col = B.Col[j];
          if (accum.state[col] == MSA<T>::UNALLOWED) {
            accum.state[col] = MSA<T>::ALLOWED;
            changed_states.push_back(col);
            ++row_nz;
          }
        }
      }
      for (int j = m_begin; j < m_end; ++j) {
        // OPTIMIZATION 1: GET RID OF IF STATEMENT
        row_nz -= accum.state[M.Col[j]];
        // if (accum.state[M.Col[j]] == MSA<T>::ALLOWED)
        //     --row_nz;
      }
      C.Rst[i + 1] = row_nz;

      for (int col_idx : changed_states)
        accum.state[col_idx] = MSA<T>::UNALLOWED;
      changed_states.clear();
    }
#pragma omp single
    {
      C.Rst[0] = 0;
      for (int i = 1; i < A.m; ++i)
        C.Rst[i + 1] += C.Rst[i];
      if (C.Rst[A.m] > C.nz)
        C.resize_vals(C.Rst[A.m]);
      C.nz = C.Rst[A.m];
    }

    constexpr T zero = T(0);
    for (size_t i = 0; i < accum.len; ++i)
      accum.state[i] = MSA<T>::ALLOWED;

#pragma omp for schedule(dynamic, 256)
    for (size_t i = 0; i < A.m; ++i) {
      int m_begin = M.Rst[i];
      int m_end = M.Rst[i + 1];

      for (size_t j = m_begin; j < m_end; ++j)
        accum.state[M.Col[j]] = MSA<T>::UNALLOWED;

      for (int t = A.Rst[i]; t < A.Rst[i + 1]; ++t) {
        int k = A.Col[t];
        int b_begin = B.Rst[k];
        int b_end = B.Rst[k + 1];
        T   a_val = A.Val[t];

#pragma omp simd
        for (int j = b_begin; j < b_end; ++j) {
          int col = B.Col[j];
          // if (accum.state[col] == MSA<T>::ALLOWED) {
          //     accum.state[col] = MSA<T>::SET;
          //     changed_states.push_back(col);
          // }

          accum.state[col] = MSA<T>::SET;

          accum.value[col] += a_val * B.Val[j];
        }
      }
      for (size_t j = m_begin; j < m_end; ++j) {
        accum.state[M.Col[j]] = MSA<T>::ALLOWED;
        accum.value[M.Col[j]] = zero;
      }

      int c_pos = C.Rst[i];
      for (int i = 0; i < accum.len; ++i) {
        if (accum.state[i] == MSA<T>::SET) {
          C.Col[c_pos] = i;
          C.Val[c_pos++] = accum.value[i];
          accum.state[i] = MSA<T>::ALLOWED;
          accum.value[i] = zero;
        }
      }

      // sort(changed_states.begin(), changed_states.end());
      // for (int col_idx : changed_states) {
      //     C.Col[c_pos] = col_idx;
      //     C.Val[c_pos++] = accum.value[col_idx];
      //     accum.state[col_idx] = MSA<T>::ALLOWED;
      //     accum.value[col_idx] = zero;
      // }
      // changed_states.clear();
    }
  }
}

// MSA cmask parallel vectorized (generic)
template<typename T, typename U>
void _mspgemm_msa_cmask_parallel_vectorized(const sparseMtx<T>& A, const sparseMtx<T>& B, const sparseMtx<U>& M, sparseMtx<T>& C) {
  //std::cerr << "Vectorization no spec\n";
  _mspgemm_msa_cmask_parallel_scalar(A, B, M, C);
}

// MSA cmask parallel vectorized specialization for int
template<typename U>
inline void _mspgemm_msa_cmask_parallel_vectorized(const sparseMtx<int>& A, const sparseMtx<int>& B, const sparseMtx<U>& M, sparseMtx<int>& C) {
#ifdef USE_RVV
  //std::cerr << "Vectorization spec int\n";
#else
  //std::cerr << "No RVV build\n";
#endif
  _mspgemm_msa_cmask_parallel_scalar(A, B, M, C);
}

// MSA cmask dispatcher
template<typename T, typename U>
void mspgemm_msa_cmask(bool isVectorization, const sparseMtx<T>& A, const sparseMtx<T>& B, const sparseMtx<U>& M, sparseMtx<T>& C) {
  C.resize_rows(M.m);
  C.n = M.n;

  if (isVectorization)
    _mspgemm_msa_cmask_parallel_vectorized(A, B, M, C);
  else
    _mspgemm_msa_cmask_parallel_scalar(A, B, M, C);
}