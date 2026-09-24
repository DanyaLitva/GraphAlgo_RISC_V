#pragma once

#include "matrix.h"
#include "matrix_la_scalar.h"
#include <queue>
#include <vector>
#include <utility>
#include <typeinfo>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <algorithm>

#ifdef USE_RVV
#include <riscv_vector.h>
#endif

// declarations

template <typename T>
struct MCA {
    T      *values;
    size_t  len;

    MCA(size_t n) {
        values = new T[n]();
        len = n;
    }

    ~MCA() {
        delete[] values;
    }
};

// MCA
template<typename T, typename U>
void _mspgemm_mca_sequential(const sparseMtx<T> &A,
                          const sparseMtx<T> &B,
                          const sparseMtx<U> &M,
                          sparseMtx<T> &C);

template<typename T, typename U>
void _mspgemm_mca_parallel_scalar(const sparseMtx<T> &A,
                               const sparseMtx<T> &B,
                               const sparseMtx<U> &M,
                               sparseMtx<T> &C);

template<typename T, typename U>
void _mspgemm_mca_parallel_vectorized(const sparseMtx<T> &A,
                                    const sparseMtx<T> &B,
                                    const sparseMtx<U> &M,
                                    sparseMtx<T> &C);

template<typename T, typename U>
sparseMtx<T> mspgemm_mca(bool isParallel,
                  bool isVectorization,
                  const sparseMtx<T> &A,
                  const sparseMtx<T> &B,
                  const sparseMtx<U> &M);

template<typename T, typename U>
void mspgemm_mca(bool isParallel,
              bool isVectorization,
              const sparseMtx<T> &A,
              const sparseMtx<T> &B,
              const sparseMtx<U> &M,
              sparseMtx<T> &C);

// MSA
template<typename T, typename U>
void _mspgemm_msa_sequential(const sparseMtx<T> &A,
                          const sparseMtx<T> &B,
                          const sparseMtx<U> &M,
                          sparseMtx<T> &C);

template<typename T, typename U>
void _mspgemm_msa_parallel_scalar(const sparseMtx<T> &A,
                               const sparseMtx<T> &B,
                               const sparseMtx<U> &M,
                               sparseMtx<T> &C);

template<typename T, typename U>
void _mspgemm_msa_parallel_vectorized(const sparseMtx<T> &A,
                                    const sparseMtx<T> &B,
                                    const sparseMtx<U> &M,
                                    sparseMtx<T> &C);

template<typename T, typename U>
void mspgemm_msa(bool isParallel,
              bool isVectorization,
              const sparseMtx<T> &A,
              const sparseMtx<T> &B,
              const sparseMtx<U> &M,
              sparseMtx<T> &C);


// Heap
template<typename T, typename U>
void _mspgemm_heap_sequential(const sparseMtx<T> &A,
                           const sparseMtx<T> &B,
                           const sparseMtx<U> &M,
                           sparseMtx<T> &C);

template<typename T, typename U>
void _mspgemm_heap_parallel_scalar(const sparseMtx<T> &A,
                                const sparseMtx<T> &B,
                                const sparseMtx<U> &M,
                                sparseMtx<T> &C);

template<typename T, typename U>
void _mspgemm_heap_parallel_vectorized(const sparseMtx<T> &A,
                                     const sparseMtx<T> &B,
                                     const sparseMtx<U> &M,
                                     sparseMtx<T> &C);

template<typename T, typename U>
sparseMtx<T> mspgemm_heap(bool isParallel,
                   bool isVectorization,
                   const sparseMtx<T> &A,
                   const sparseMtx<T> &B,
                   const sparseMtx<U> &M);

template<typename T, typename U>
void mspgemm_heap(bool isParallel,
               bool isVectorization,
               const sparseMtx<T> &A,
               const sparseMtx<T> &B,
               const sparseMtx<U> &M,
               sparseMtx<T> &C);


// definitions

// MCA sequential
template<typename T, typename U>
void _mspgemm_mca_sequential(const sparseMtx<T> &A, const sparseMtx<T> &B, const sparseMtx<U> &M, sparseMtx<T> &C) {
    int mca_len = 0;
    for (size_t i = 0; i < A.m; ++i)
        if (M.Rst[i+1] - M.Rst[i] > mca_len)
            mca_len = M.Rst[i+1] - M.Rst[i];

    MCA<T> accum(mca_len);

    for (size_t i = 0; i < A.m; ++i) {
        int m_row_len = M.Rst[i+1] - M.Rst[i];
        int m_pos;

        for (int t = A.Rst[i]; t < A.Rst[i+1]; ++t) {
            int k = A.Col[t];
            int b_pos = B.Rst[k];
            int b_max = B.Rst[k+1];
            T   a_val = A.Val[t];
            m_pos = M.Rst[i];
            for (int j = 0; j < m_row_len; ++j, ++m_pos) {
                while (b_pos < b_max && B.Col[b_pos] < M.Col[m_pos])
                    ++b_pos;
                if (b_pos < b_max && B.Col[b_pos] == M.Col[m_pos])
                    accum.values[j] += a_val * B.Val[b_pos];
            }
        }
        if (m_row_len > 0)
            memcpy(C.Val + C.Rst[i], accum.values, m_row_len*sizeof(T));
        memset(accum.values, 0, m_row_len * sizeof(T));
    }
}

// MCA parallel scalar
template<typename T, typename U>
void _mspgemm_mca_parallel_scalar(const sparseMtx<T> &A, const sparseMtx<T> &B, const sparseMtx<U> &M, sparseMtx<T> &C) {
    int mca_len = 0;
#pragma omp parallel for reduction(max:mca_len)
    for (size_t i = 0; i < A.m; ++i) {
        int len = M.Rst[i + 1] - M.Rst[i];
        if (len > mca_len) mca_len = len;
    }

#pragma omp parallel
    {
        MCA<T> accum(mca_len);

#pragma omp for schedule(dynamic, 32)
        for (size_t i = 0; i < A.m; ++i) {
            int m_row_len = M.Rst[i+1] - M.Rst[i];
            int m_pos;

            for (int t = A.Rst[i]; t < A.Rst[i+1]; ++t) {
                int k = A.Col[t];
                int b_pos = B.Rst[k];
                int b_max = B.Rst[k+1];
                T   a_val = A.Val[t];
                m_pos = M.Rst[i];
                for (int j = 0; j < m_row_len; ++j, ++m_pos) {
                    while (b_pos < b_max && B.Col[b_pos] < M.Col[m_pos])
                        ++b_pos;
                    if (b_pos < b_max && B.Col[b_pos] == M.Col[m_pos])
                        accum.values[j] += a_val * B.Val[b_pos];
                }
            }

            if (m_row_len > 0)
            memcpy(C.Val + C.Rst[i], accum.values, m_row_len*sizeof(T));
            memset(accum.values, 0, m_row_len * sizeof(T));
        }
    }
}

// MCA parallel vectorized (generic)
template<typename T, typename U>
void _mspgemm_mca_parallel_vectorized(const sparseMtx<T>& A, const sparseMtx<T>& B, const sparseMtx<U>& M, sparseMtx<T>& C) {
#ifdef USE_RVV
  const size_t A_m = A.m;
  const int* A_Rst = A.Rst;
  const int* A_Col = A.Col;
  const T* A_Val = A.Val;
  const int* B_Rst = B.Rst;
  const int* B_Col = B.Col;
  const T* B_Val = B.Val;
  const int* M_Rst = M.Rst;
  const int* M_Col = M.Col;
  const int* C_Rst = C.Rst;
  T* C_Val = C.Val;

  int mca_len = 0;
#pragma omp parallel for reduction(max:mca_len)
  for (size_t i = 0; i < A_m; ++i) {
    int len = M_Rst[i + 1] - M_Rst[i];
    if (len > mca_len) mca_len = len;
  }

#pragma omp parallel
  {
    MCA<T> accum(mca_len);
    T* accum_ptr = accum.values;
#pragma omp for schedule(dynamic, 32)
    for (size_t i = 0; i < A_m; ++i) {
      int m_start = M_Rst[i];
      int m_max = M_Rst[i + 1];
      int m_row_len = m_max - m_start;

      int A_Rst_end = A_Rst[i + 1];
      for (int t = A_Rst[i]; t < A_Rst_end; ++t) {
        int m_pos = m_start;
        int k = A_Col[t];
        int b_pos = B_Rst[k];
        int b_max = B_Rst[k + 1];
        T a_val = A_Val[t];

        //Vectorize the longer row
        if ((b_max - b_pos) <= m_row_len) {   //vectorize M row
          while (b_pos < b_max && m_pos < m_max) {
#if defined(MCA_LMUL1)
            size_t vl = __riscv_vsetvl_e32m1(m_max - m_pos);
            vint32m1_t v_m_cols = __riscv_vle32_v_i32m1(&M_Col[m_pos], vl);
            vbool32_t v_match = __riscv_vmseq_vx_i32m1_b32(v_m_cols, B_Col[b_pos], vl);
            long match_idx = __riscv_vfirst_m_b32(v_match, vl);
#elif defined(MCA_LMUL2)
            size_t vl = __riscv_vsetvl_e32m2(m_max - m_pos);
            vint32m2_t v_m_cols = __riscv_vle32_v_i32m2(&M_Col[m_pos], vl);
            vbool16_t v_match = __riscv_vmseq_vx_i32m2_b16(v_m_cols, B_Col[b_pos], vl);
            long match_idx = __riscv_vfirst_m_b16(v_match, vl);
#elif defined(MCA_LMUL4)
            size_t vl = __riscv_vsetvl_e32m4(m_max - m_pos);
            vint32m4_t v_m_cols = __riscv_vle32_v_i32m4(&M_Col[m_pos], vl);
            vbool8_t v_match = __riscv_vmseq_vx_i32m4_b8(v_m_cols, B_Col[b_pos], vl);
            long match_idx = __riscv_vfirst_m_b8(v_match, vl);
#else
#error "MCA_LMUL1, MCA_LMUL2 or MCA_LMUL4 must be defined"
#endif
            if (match_idx >= 0) {
              accum_ptr[(m_pos - m_start) + match_idx] += a_val * B_Val[b_pos];
              b_pos++;
              m_pos += match_idx + 1;
            }
            else {
              if (B_Col[b_pos] > M_Col[m_pos + vl - 1])
                m_pos += vl;
              else
                b_pos++;
            }
          }
        }
        else {   //vectorize B row
          while (b_pos < b_max && m_pos < m_max) {
#if defined(MCA_LMUL1)
            size_t vl = __riscv_vsetvl_e32m1(b_max - b_pos);
            vint32m1_t v_b_cols = __riscv_vle32_v_i32m1(&B_Col[b_pos], vl);
            vbool32_t v_match = __riscv_vmseq_vx_i32m1_b32(v_b_cols, M_Col[m_pos], vl);
            long match_idx = __riscv_vfirst_m_b32(v_match, vl);
#elif defined(MCA_LMUL2)
            size_t vl = __riscv_vsetvl_e32m2(b_max - b_pos);
            vint32m2_t v_b_cols = __riscv_vle32_v_i32m2(&B_Col[b_pos], vl);
            vbool16_t v_match = __riscv_vmseq_vx_i32m2_b16(v_b_cols, M_Col[m_pos], vl);
            long match_idx = __riscv_vfirst_m_b16(v_match, vl);
#elif defined(MCA_LMUL4)
            size_t vl = __riscv_vsetvl_e32m4(b_max - b_pos);
            vint32m4_t v_b_cols = __riscv_vle32_v_i32m4(&B_Col[b_pos], vl);
            vbool8_t v_match = __riscv_vmseq_vx_i32m4_b8(v_b_cols, M_Col[m_pos], vl);
            long match_idx = __riscv_vfirst_m_b8(v_match, vl);
#else
#error "MCA_LMUL1, MCA_LMUL2 or MCA_LMUL4 must be defined"
#endif
            if (match_idx >= 0) {
              accum_ptr[m_pos - m_start] += a_val * B_Val[b_pos + match_idx];
              b_pos += match_idx + 1;
              m_pos++;
            }
            else {
              if (M_Col[m_pos] > B_Col[b_pos + vl - 1])
                b_pos += vl;
              else 
                m_pos++;              
            }
          }
        }
      }

      if (m_row_len > 0)
        memcpy(C_Val + C_Rst[i], accum_ptr, m_row_len * sizeof(T));
      memset(accum_ptr, 0, m_row_len * sizeof(T));
    }
  }
#else
//   std::cerr << "No RVV build for vectorization!\n";
  _mspgemm_mca_parallel_scalar(A, B, M, C);
#endif
}

// MCA dispatchers
template<typename T, typename U>
sparseMtx<T> mspgemm_mca(bool isParallel, bool isVectorization, const sparseMtx<T> &A, const sparseMtx<T> &B, const sparseMtx<U> &M) {
    if (A.n != B.m || M.m != A.m || M.n != B.n)
        throw std::invalid_argument("invalid dimensions for masked sparse matrix multiplication");
    sparseMtx<T> C(A.m, B.n, M.nz);
    if (M.nz > 0)
        memcpy(C.Col, M.Col, M.nz * sizeof(int));
    memcpy(C.Rst, M.Rst, (M.m + 1) * sizeof(int));

    if (!isParallel)
        _mspgemm_mca_sequential(A, B, M, C);
    else if (isVectorization)
        _mspgemm_mca_parallel_vectorized(A, B, M, C);
    else
        _mspgemm_mca_parallel_scalar(A, B, M, C);

    return C;
}

template<typename T, typename U>
void mspgemm_mca(bool isParallel, bool isVectorization, const sparseMtx<T> &A, const sparseMtx<T> &B, const sparseMtx<U> &M, sparseMtx<T> &C) {
    if (A.n != B.m || M.m != A.m || M.n != B.n)
        throw std::invalid_argument("invalid dimensions for masked sparse matrix multiplication");
    C.resize_rows(M.m);
    if (C.Rst == nullptr)
        C.Rst = new int[M.m + 1]();
    C.resize_vals(M.nz);
    C.n = M.n;
    if (C.nz > 0)
        memcpy(C.Col, M.Col, C.nz * sizeof(int));
    memcpy(C.Rst, M.Rst, (C.m + 1) * sizeof(int));

    if (!isParallel)
        _mspgemm_mca_sequential(A, B, M, C);
    else if (isVectorization)
        _mspgemm_mca_parallel_vectorized(A, B, M, C);
    else
        _mspgemm_mca_parallel_scalar(A, B, M, C);
}

// MSA sequential
template<typename T, typename U>
void _mspgemm_msa_sequential(const sparseMtx<T> &A, const sparseMtx<T> &B, const sparseMtx<U> &M, sparseMtx<T> &C) {
    MSA<T> accum(B.n);

    for (size_t i = 0; i < A.m; ++i) {
        int m_min = M.Rst[i];
        int m_max = M.Rst[i+1];

        for (int j = m_min; j < m_max; ++j) {
            accum.state[M.Col[j]] = MSA<T>::ALLOWED;
            accum.value[M.Col[j]] = T(0);
        }

        for (int t = A.Rst[i]; t < A.Rst[i+1]; ++t) {
            int k = A.Col[t];
            int b_pos = B.Rst[k];
            int b_max = B.Rst[k+1];
            T   a_val = A.Val[t];

            for (int j = b_pos; j < b_max; ++j) {
                int b_col = B.Col[j];
                if (accum.state[b_col] == MSA<T>::ALLOWED) {
                    accum.state[b_col] = MSA<T>::SET;
                    accum.value[b_col] = a_val * B.Val[j];
                } else if (accum.state[b_col] == MSA<T>::SET)
                    accum.value[b_col] += a_val * B.Val[j];
            }
        }

        for (int j = m_min; j < m_max; ++j) {
            C.Val[j] = accum.value[M.Col[j]];
            accum.state[M.Col[j]] = MSA<T>::UNALLOWED;
        }
    }
}

// MSA parallel scalar
template<typename T, typename U>
void _mspgemm_msa_parallel_scalar(const sparseMtx<T> &A, const sparseMtx<T> &B, const sparseMtx<U> &M, sparseMtx<T> &C) {
    //std::cerr << "Scalar\n";
#pragma omp parallel
    {
        MSA<T> accum(B.n);
        const T zero = (T)0;

#pragma omp for schedule(dynamic, 32)
        for (size_t i = 0; i < A.m; ++i) {
            int m_min = M.Rst[i];
            int m_max = M.Rst[i+1];

            for (int j = m_min; j < m_max; ++j)
                accum.value[M.Col[j]] = zero;

            for (int t = A.Rst[i]; t < A.Rst[i+1]; ++t) {
                int k = A.Col[t];
                int b_pos = B.Rst[k];
                int b_max = B.Rst[k+1];
                T   a_val = A.Val[t];

                for (int j = b_pos; j < b_max; ++j)
                    accum.value[B.Col[j]] += a_val * B.Val[j];
            }

            for (int j = m_min; j < m_max; ++j) {
                C.Val[j] = accum.value[M.Col[j]];
            }
        }
    }
}

// MSA parallel vectorized (generic)
template<typename T, typename U>
void _mspgemm_msa_parallel_vectorized(const sparseMtx<T> &A, const sparseMtx<T> &B, const sparseMtx<U> &M, sparseMtx<T> &C) {
    //std::cerr << "Vectorization no spec\n";
    _mspgemm_msa_parallel_scalar(A, B, M, C);
}

// MSA parallel vectorized specialization for double
template<typename U>
inline void _mspgemm_msa_parallel_vectorized(const sparseMtx<double> &A, const sparseMtx<double> &B, const sparseMtx<U> &M, sparseMtx<double> &C) {
#ifdef USE_RVV
#pragma omp parallel
    {
        MSA<double> accum(B.n);

#pragma omp for schedule(dynamic, 32)
        for (size_t i = 0; i < A.m; ++i) {
            int m_min = M.Rst[i];
            int m_max = M.Rst[i + 1];

            int j_init = m_min;
            int remain_init = m_max - m_min;
            while (remain_init > 0) {
                size_t vl = __riscv_vsetvl_e64m2(remain_init);

                vuint32m1_t vm_col = __riscv_vle32_v_u32m1(reinterpret_cast<const uint32_t*>(&M.Col[j_init]), vl);

                vuint32m1_t v_byte_offsets = __riscv_vsll_vx_u32m1(vm_col, 3, vl);

                vfloat64m2_t v_zero = __riscv_vfmv_v_f_f64m2(0.0, vl);

                __riscv_vsuxei32_v_f64m2(accum.value, v_byte_offsets, v_zero, vl);

                j_init += vl;
                remain_init -= vl;
            }
            //for (int j = m_min; j < m_max; ++j)
                //accum.value[M.Col[j]] = zero;

            for (int t = A.Rst[i]; t < A.Rst[i + 1]; ++t) {
                int k = A.Col[t];
                int b_pos = B.Rst[k];
                int b_max = B.Rst[k + 1];
                double   a_val = A.Val[t];

                int j_calc = b_pos;
                int remain_calc = b_max - b_pos;
                while (remain_calc > 0) {
                    size_t vl = __riscv_vsetvl_e64m2(remain_calc);

                    vuint32m1_t vb_col = __riscv_vle32_v_u32m1(reinterpret_cast<const uint32_t*>(&B.Col[j_calc]), vl);

                    vuint32m1_t v_byte_offsets = __riscv_vsll_vx_u32m1(vb_col, 3, vl);

                    vfloat64m2_t vb_val = __riscv_vle64_v_f64m2(&B.Val[j_calc], vl);

                    vfloat64m2_t v_acc = __riscv_vluxei32_v_f64m2(accum.value, v_byte_offsets, vl);

                    v_acc = __riscv_vfmacc_vf_f64m2(v_acc, a_val, vb_val, vl);

                    __riscv_vsuxei32_v_f64m2(accum.value, v_byte_offsets, v_acc, vl);

                    j_calc += vl;
                    remain_calc -= vl;
                }
                //for (int j = b_pos; j < b_max; ++j)
                    //accum.value[B.Col[j]] += a_val * B.Val[j];
            }

            int j_store = m_min;
            int remain_store = m_max - m_min;
            while (remain_store > 0) {
                size_t vl = __riscv_vsetvl_e64m2(remain_store);

                vuint32m1_t vm_col = __riscv_vle32_v_u32m1(reinterpret_cast<const uint32_t*>(&M.Col[j_store]), vl);

                vuint32m1_t v_byte_offsets = __riscv_vsll_vx_u32m1(vm_col, 3, vl);

                vfloat64m2_t v_acc_res = __riscv_vluxei32_v_f64m2(accum.value, v_byte_offsets, vl);

                __riscv_vse64_v_f64m2(&C.Val[j_store], v_acc_res, vl);

                j_store += vl;
                remain_store -= vl;
            }
            //for (int j = m_min; j < m_max; ++j) {
            //    C.Val[j] = accum.value[M.Col[j]];
            //}
        }
    }
#else
    _mspgemm_msa_parallel_scalar(A, B, M, C);
#endif
}

// MSA parallel vectorized specialization for float
template<typename U>
inline void _mspgemm_msa_parallel_vectorized(const sparseMtx<float>& A, const sparseMtx<float>& B, const sparseMtx<U>& M, sparseMtx<float>& C) {
#ifdef USE_RVV
#pragma omp parallel
    {
        MSA<float> accum(B.n);

#pragma omp for schedule(dynamic, 32)
        for (size_t i = 0; i < A.m; ++i) {
            int m_min = M.Rst[i];
            int m_max = M.Rst[i + 1];

            int j_init = m_min;
            int remain_init = m_max - m_min;
            while (remain_init > 0) {
                size_t vl = __riscv_vsetvl_e32m1(remain_init);

                vuint32m1_t vm_col = __riscv_vle32_v_u32m1(reinterpret_cast<const uint32_t*>(&M.Col[j_init]), vl);

                vuint32m1_t v_byte_offsets = __riscv_vsll_vx_u32m1(vm_col, 2, vl);

                vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0, vl);

                __riscv_vsuxei32_v_f32m1(accum.value, v_byte_offsets, v_zero, vl);

                j_init += vl;
                remain_init -= vl;
            }
            //for (int j = m_min; j < m_max; ++j)
                //accum.value[M.Col[j]] = zero;

            for (int t = A.Rst[i]; t < A.Rst[i + 1]; ++t) {
                int k = A.Col[t];
                int b_pos = B.Rst[k];
                int b_max = B.Rst[k + 1];
                double   a_val = A.Val[t];

                int j_calc = b_pos;
                int remain_calc = b_max - b_pos;
                while (remain_calc > 0) {
                    size_t vl = __riscv_vsetvl_e32m1(remain_calc);

                    vuint32m1_t vb_col = __riscv_vle32_v_u32m1(reinterpret_cast<const uint32_t*>(&B.Col[j_calc]), vl);

                    vuint32m1_t v_byte_offsets = __riscv_vsll_vx_u32m1(vb_col, 2, vl);

                    vfloat32m1_t vb_val = __riscv_vle32_v_f32m1(&B.Val[j_calc], vl);

                    vfloat32m1_t v_acc = __riscv_vluxei32_v_f32m1(accum.value, v_byte_offsets, vl);

                    v_acc = __riscv_vfmacc_vf_f32m1(v_acc, a_val, vb_val, vl);

                    __riscv_vsuxei32_v_f32m1(accum.value, v_byte_offsets, v_acc, vl);

                    j_calc += vl;
                    remain_calc -= vl;
                }
                //for (int j = b_pos; j < b_max; ++j)
                    //accum.value[B.Col[j]] += a_val * B.Val[j];
            }

            int j_store = m_min;
            int remain_store = m_max - m_min;
            while (remain_store > 0) {
                size_t vl = __riscv_vsetvl_e32m1(remain_store);

                vuint32m1_t vm_col = __riscv_vle32_v_u32m1(reinterpret_cast<const uint32_t*>(&M.Col[j_store]), vl);

                vuint32m1_t v_byte_offsets = __riscv_vsll_vx_u32m1(vm_col, 2, vl);

                vfloat32m1_t v_acc_res = __riscv_vluxei32_v_f32m1(accum.value, v_byte_offsets, vl);

                __riscv_vse32_v_f32m1(&C.Val[j_store], v_acc_res, vl);

                j_store += vl;
                remain_store -= vl;
            }
            //for (int j = m_min; j < m_max; ++j) {
            //    C.Val[j] = accum.value[M.Col[j]];
            //}
        }
    }
#else
    _mspgemm_msa_parallel_scalar(A, B, M, C);
#endif
}


// MSA parallel vectorized specialization for int
template<typename U>
inline void _mspgemm_msa_parallel_vectorized(const sparseMtx<int>& A, const sparseMtx<int>& B, const sparseMtx<U>& M, sparseMtx<int>& C) {
#ifdef USE_RVV
#pragma omp parallel
    {
        MSA<int> accum(B.n);

#pragma omp for schedule(dynamic, 32)
        for (size_t i = 0; i < A.m; ++i) {
            int m_min = M.Rst[i];
            int m_max = M.Rst[i + 1];

            int j_init = m_min;
            int remain_init = m_max - m_min;
            while (remain_init > 0) {
                size_t vl = __riscv_vsetvl_e32m1(remain_init);

                vuint32m1_t vm_col = __riscv_vle32_v_u32m1(reinterpret_cast<const uint32_t*>(&M.Col[j_init]), vl);

                vuint32m1_t v_byte_offsets = __riscv_vsll_vx_u32m1(vm_col, 2, vl);

                vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, vl);

                __riscv_vsuxei32_v_i32m1(accum.value, v_byte_offsets, v_zero, vl);

                j_init += vl;
                remain_init -= vl;
            }
            //for (int j = m_min; j < m_max; ++j)
                //accum.value[M.Col[j]] = 0;

            for (int t = A.Rst[i]; t < A.Rst[i + 1]; ++t) {
                int k = A.Col[t];
                int b_pos = B.Rst[k];
                int b_max = B.Rst[k + 1];
                int a_val = A.Val[t];

                int j_calc = b_pos;
                int remain_calc = b_max - b_pos;
                while (remain_calc > 0) {
                    size_t vl = __riscv_vsetvl_e32m1(remain_calc);

                    vuint32m1_t vb_col = __riscv_vle32_v_u32m1(reinterpret_cast<const uint32_t*>(&B.Col[j_calc]), vl);

                    vuint32m1_t v_byte_offsets = __riscv_vsll_vx_u32m1(vb_col, 2, vl);

                    vint32m1_t vb_val = __riscv_vle32_v_i32m1(&B.Val[j_calc], vl);

                    vint32m1_t v_acc = __riscv_vluxei32_v_i32m1(accum.value, v_byte_offsets, vl);

                    v_acc = __riscv_vmacc_vx_i32m1(v_acc, a_val, vb_val, vl);

                    __riscv_vsuxei32_v_i32m1(accum.value, v_byte_offsets, v_acc, vl);

                    j_calc += vl;
                    remain_calc -= vl;
                }
                //for (int j = b_pos; j < b_max; ++j)
                    //accum.value[B.Col[j]] += a_val * B.Val[j];
            }

            int j_store = m_min;
            int remain_store = m_max - m_min;
            while (remain_store > 0) {
                size_t vl = __riscv_vsetvl_e32m1(remain_store);

                vuint32m1_t vm_col = __riscv_vle32_v_u32m1(reinterpret_cast<const uint32_t*>(&M.Col[j_store]), vl);

                vuint32m1_t v_byte_offsets = __riscv_vsll_vx_u32m1(vm_col, 2, vl);

                vint32m1_t v_acc_res = __riscv_vluxei32_v_i32m1(accum.value, v_byte_offsets, vl);

                __riscv_vse32_v_i32m1(&C.Val[j_store], v_acc_res, vl);

                j_store += vl;
                remain_store -= vl;
            }
            //for (int j = m_min; j < m_max; ++j) {
            //    C.Val[j] = accum.value[M.Col[j]];
            //}
        }
    }
#else
    // std::cerr << "No RVV build for vectorization!\n";
    _mspgemm_msa_parallel_scalar(A, B, M, C);
#endif
}

// MSA dispatcher
template<typename T, typename U>
void mspgemm_msa(bool isParallel, bool isVectorization, const sparseMtx<T> &A, const sparseMtx<T> &B, const sparseMtx<U> &M, sparseMtx<T> &C) {
    if (A.n != B.m || M.m != A.m || M.n != B.n)
        throw std::invalid_argument("invalid dimensions for masked sparse matrix multiplication");
    C.resize_rows(M.m);
    if (C.Rst == nullptr)
        C.Rst = new int[M.m + 1]();
    C.resize_vals(M.nz);
    C.n = M.n;
    if (C.nz > 0)
        memcpy(C.Col, M.Col, C.nz * sizeof(int));
    memcpy(C.Rst, M.Rst, (C.m + 1) * sizeof(int));

    if (!isParallel)
        _mspgemm_msa_sequential(A, B, M, C);
    else if (isVectorization)
        _mspgemm_msa_parallel_vectorized(A, B, M, C);
    else
        _mspgemm_msa_parallel_scalar(A, B, M, C);
}






//TODO: HEAP


// Heap sequential
template<typename T, typename U>
void _mspgemm_heap_sequential(const sparseMtx<T> &A, const sparseMtx<T> &B,
                           const sparseMtx<U> &M, sparseMtx<T> &C) {
    int m_col;
    int m_pos;
    int m_max_pos;
    std::priority_queue<heap_iterator<T>> heap;
    heap_iterator<T> iter;

    for (size_t i = 0; i < A.m; ++i) {
        for (int j = A.Rst[i]; j < A.Rst[i+1]; ++j) {
            int k = B.Rst[A.Col[j]];
            heap.emplace(k, B.Rst[A.Col[j]+1], B.Col[k], A.Val[j]);
        }
        m_pos = M.Rst[i];
        m_col = M.Col[m_pos];
        m_max_pos = M.Rst[i+1];

        while (!heap.empty()) {
            iter = heap.top();
            heap.pop();

            while (m_col < iter.b_col && m_pos < m_max_pos)
                m_col = M.Col[++m_pos];
            if (m_pos == m_max_pos)
                break;

            if (m_col == iter.b_col && iter.b_pos < iter.b_max_pos)
                C.Val[m_pos] += iter.val * B.Val[iter.b_pos];

            iter.b_col = B.Col[++iter.b_pos];
            while (iter.b_pos < iter.b_max_pos && iter.b_col < m_col)
                iter.b_col = B.Col[++iter.b_pos];
            if (iter.b_pos < iter.b_max_pos)
                heap.push(iter);
        }
        heap = std::priority_queue<heap_iterator<T>>();
    }
}

// Heap parallel scalar
template<typename T, typename U>
void _mspgemm_heap_parallel_scalar(const sparseMtx<T> &A, const sparseMtx<T> &B,
                                const sparseMtx<U> &M, sparseMtx<T> &C) 
{
memset(C.Val, 0, C.nz * sizeof(T));
#pragma omp parallel
    {
        int m_pos;      
        int m_col;      
        int m_max_pos;  
        std::priority_queue<heap_iterator<T>> heap;
        heap_iterator<T> iter;

#pragma omp for schedule(dynamic, 32)
        for (size_t i = 0; i < A.m; ++i) {
            for (int j = A.Rst[i]; j < A.Rst[i+1]; ++j) {
                int k = B.Rst[A.Col[j]];
                heap.emplace(k, B.Rst[A.Col[j]+1], B.Col[k], A.Val[j]);
            }
            m_pos = M.Rst[i];
            m_col = M.Col[m_pos];
            m_max_pos = M.Rst[i+1];

            while (!heap.empty()) {
                iter = heap.top();
                heap.pop();

                while (m_pos < m_max_pos && m_col < iter.b_col)
                    m_col = M.Col[++m_pos];
                if (m_pos == m_max_pos)
                    break;
                if (m_col == iter.b_col && iter.b_pos < iter.b_max_pos)
                    C.Val[m_pos] += iter.val * B.Val[iter.b_pos];

                iter.b_col = B.Col[++iter.b_pos];
                while (iter.b_pos < iter.b_max_pos && iter.b_col < m_col)
                    iter.b_col = B.Col[++iter.b_pos];
                if (iter.b_pos < iter.b_max_pos)
                    heap.push(iter);
            }
            while (!heap.empty())
                heap.pop();
        }
    }
}

// Heap parallel vectorized (generic)
template<typename T, typename U>
void _mspgemm_heap_parallel_vectorized(const sparseMtx<T> &A, const sparseMtx<T> &B,
                                     const sparseMtx<U> &M, sparseMtx<T> &C) {
    _mspgemm_heap_parallel_scalar(A, B, M, C);
}

// Heap dispatchers
template<typename T, typename U>
sparseMtx<T> mspgemm_heap(bool isParallel, bool isVectorization, const sparseMtx<T> &A, const sparseMtx<T> &B, const sparseMtx<U> &M) {
    sparseMtx<T> C(A.m, B.n, M.nz);
    memcpy(C.Col, M.Col, M.nz * sizeof(int));
    memcpy(C.Rst, M.Rst, (M.m + 1) * sizeof(int));

    if (!isParallel)
        _mspgemm_heap_sequential(A, B, M, C);
    else if (isVectorization)
        _mspgemm_heap_parallel_vectorized(A, B, M, C);
    else
        _mspgemm_heap_parallel_scalar(A, B, M, C);

    return C;
}

template<typename T, typename U>
void mspgemm_heap(bool isParallel, bool isVectorization, const sparseMtx<T> &A, const sparseMtx<T> &B, const sparseMtx<U> &M, sparseMtx<T> &C) {
    C.resize_rows(M.m);
    C.resize_vals(M.nz);
    C.n = M.n;
    memcpy(C.Col, M.Col, C.nz * sizeof(int));
    memcpy(C.Rst, M.Rst, (C.m + 1) * sizeof(int));

    if (!isParallel)
        _mspgemm_heap_sequential(A, B, M, C);
    else if (isVectorization)
        _mspgemm_heap_parallel_vectorized(A, B, M, C);
    else
        _mspgemm_heap_parallel_scalar(A, B, M, C);
}