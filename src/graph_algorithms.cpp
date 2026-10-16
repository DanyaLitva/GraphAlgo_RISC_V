#include "graph_algorithms.h"
#include "matrix_la.h"
using namespace std;

int* triangle_counting_vertex(const sparseMtx<int> &A, mspgemmAlgorithm<int> matrixMult, bool isVectorization) {
    /* PREPARE DATA */
    int *nums_of_tr = new int[A.m];
    int num_of_tr;
    sparseMtx<int> SQ;

    auto start = chrono::steady_clock::now();

    /* TRIANGLE COUNTING ITSELF */
    matrixMult(isVectorization, A, A, A, SQ);
    for (size_t i = 0; i < A.m; ++i) {
        num_of_tr = 0;
        for (int j = SQ.Rst[i]; j < SQ.Rst[i+1]; ++j)
            num_of_tr += SQ.Val[j];
        nums_of_tr[i] = num_of_tr >>= 1;
    }
    /* TRIANGLE COUNTING ITSELF */

    auto finish = chrono::steady_clock::now();
    cout << "Time:       " << chrono::duration_cast<chrono::milliseconds>(finish - start).count() << '\n';

    return nums_of_tr;
}


int64_t triangle_counting_masked_lu(const sparseMtx<int> &A, mspgemmAlgorithm<int> matrixMult, bool isVectorization) {
    int64_t num_of_tr = 0;
    sparseMtx<int> L = extract_lower_triangle(A);
    sparseMtx<int> U = transpose(L);
    sparseMtx<int> C;

    auto start = chrono::steady_clock::now();

    /* TRIANGLE COUNTING ITSELF */
    matrixMult(isVectorization, L, U, A, C);

    // Count the total number of triangles
    for (int j = 0; j < C.Rst[C.m]; ++j)
        num_of_tr += C.Val[j];
    num_of_tr >>= 1;
    /* TRIANGLE COUNTING ITSELF */

    auto finish = chrono::steady_clock::now();
    cout << "Time:       " << chrono::duration_cast<chrono::milliseconds>(finish - start).count() << " ms\n";
    cout << "Triangles:  " << num_of_tr << '\n';

    return num_of_tr;
}


int64_t triangle_counting(const sparseMtx<int> &A, mspgemmAlgorithm<int> matrixMult, bool isVectorization) {
    int64_t num_of_tr = 0;
    sparseMtx<int> L = extract_lower_triangle(A);
    sparseMtx<int> C;

    auto start = chrono::steady_clock::now();

    /* TRIANGLE COUNTING ITSELF */
    matrixMult(isVectorization, L, L, L, C);
#pragma omp parallel for reduction(+:num_of_tr)
    for (int j = 0; j < C.Rst[C.m]; ++j)
        num_of_tr += C.Val[j];
    /* TRIANGLE COUNTING ITSELF */

    auto finish = chrono::steady_clock::now();
    cout << "Time:       " << chrono::duration_cast<chrono::milliseconds>(finish - start).count() << " ms\n";
    cout << "Triangles:  " << num_of_tr << '\n';

    return num_of_tr;
}

/* K-TRUSS */
sparseMtx<int> k_truss(const sparseMtx<int> &A, int k, mspgemmAlgorithm<int> matrixMult, bool isVectorization) {
    sparseMtx<int> C = A;  // a copy of adjacency matrix
    sparseMtx<int> Tmp;
    int n = A.m;
    int totalIterationNum = 0;
    int *tmp_Rst = new int[n+1];
    tmp_Rst[0] = 0;

    auto start = chrono::steady_clock::now();
    
    int t = 0;
    while (true) {
        // Tmp<C> = C*C
        matrixMult(isVectorization, C, C, C, Tmp);

        // remove all edges included in less than (k-2) triangles
        // and replace values of remaining entries with 1
        int new_curr_pos = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = Tmp.Rst[i]; j < Tmp.Rst[i+1]; ++j) {
                if (Tmp.Val[j] >= k-2) {
                    Tmp.Col[new_curr_pos]   = Tmp.Col[j];
                    Tmp.Val[new_curr_pos++] = 1;
                }
            }
            tmp_Rst[i+1] = new_curr_pos;
        }
        memcpy(Tmp.Rst, tmp_Rst, (n+1)*sizeof(int));
        Tmp.nz = Tmp.Rst[n];

        const bool samePattern =
            (Tmp.m == C.m) && (Tmp.n == C.n) && (Tmp.nz == C.nz) &&
            std::memcmp(Tmp.Rst, C.Rst, (n + 1) * sizeof(int)) == 0 &&
            (Tmp.nz == 0 || std::memcmp(Tmp.Col, C.Col, Tmp.nz * sizeof(int)) == 0);
        if (samePattern) {
            totalIterationNum = ++t;
            break;
        }

        // Assign 'Tmp' to 'C'
        std::swap(C, Tmp);
    }

    auto finish = chrono::steady_clock::now();
    cout << "Time:       " << chrono::duration_cast<chrono::milliseconds>(finish - start).count() << " ms\n";
    cout << "Iterations: " << totalIterationNum << '\n';

    if (C.nz < A.nz) {
        int *new_Adj = new int[C.nz];
        int *new_Wgt = new int[C.nz];
        std::memcpy(new_Adj, C.Col, C.nz * sizeof(int));
        std::memcpy(new_Wgt, C.Val, C.nz * sizeof(int));
        delete[] C.Col;
        delete[] C.Val;
        C.Col = new_Adj;
        C.Val = new_Wgt;
    }

    delete[] tmp_Rst;
    return C;
}

template <typename T>
void brandes_backward_step(const sparseMtx<T> &A,
                      sparseMtx<T> &Front,
                      const sparseMtx<T> &Next,
                      const denseMtx<T> &Nspinv,
                      const denseMtx<T> &Numsp,
                      denseMtx<T> &Bcu,
                      long long &ewisemult_time,
                      long long &mspgemm_time) {
    size_t m = Front.m;
    size_t n = Front.n;
    std::chrono::high_resolution_clock::time_point time_begin, time_end;

    // element-wise matrix multiplication
    time_begin = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 256)
    for (size_t i = 0; i < m; ++i) {
        T *nspinv_row = Nspinv.Val + i * n;
        T *bcu_row = Bcu.Val + i * n;
        for (size_t j = Front.Rst[i]; j < Front.Rst[i+1]; ++j) {
            size_t idx = Front.Col[j];
            Front.Val[j] = nspinv_row[idx] * bcu_row[idx];
        }
    }
    time_end = std::chrono::high_resolution_clock::now();
    ewisemult_time += (time_end - time_begin).count();
    
    // MSpGEMM + addition straight into dense matrix
    time_begin = std::chrono::high_resolution_clock::now();
#pragma omp parallel
    {
        MSA<T> accum(Front.n);
        const T zero = T(0);

    #pragma omp for schedule(dynamic, 256)
        for (size_t i = 0; i < Next.m; ++i) {
            int m_min = Next.Rst[i];
            int m_max = Next.Rst[i+1];

            for (int j = m_min; j < m_max; ++j)
                accum.value[Next.Col[j]] = zero;

            for (int t = A.Rst[i]; t < A.Rst[i+1]; ++t) {
                int k = A.Col[t];
                int b_pos = Front.Rst[k];
                int b_max = Front.Rst[k+1];
                T a_val = A.Val[t];

                for (int j = b_pos; j < b_max; ++j)
                    accum.value[Front.Col[j]] += a_val * Front.Val[j];
            }

            T *bcu_row = Bcu.Val + i * n;
            T *numsp_row = Numsp.Val + i * n;
            for (int j = m_min; j < m_max; ++j) {
                int idx = Next.Col[j];
                bcu_row[idx] += accum.value[idx] * numsp_row[idx];
            }
        }
    }
    time_end = std::chrono::high_resolution_clock::now();
    mspgemm_time += (time_end - time_begin).count();
}

// template <typename T>
void brandes_forward_step(const sparseMtx<int> &AT,
                     denseMtx<int> &Numsp,
                     sparseMtx<int> &Front,
                     sparseMtx<int> &FrontTmp,
                     long long &init_numsp_time,
                     long long &symbolic_time,
                     long long &numeric_time) {
    size_t m = Front.m;
    size_t n = Front.n;
    std::chrono::high_resolution_clock::time_point time_begin, time_end;

    // Add `Front` into `Numsp`
    time_begin = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (int i = 0; i < m; ++i)
        for (int j = Front.Rst[i]; j < Front.Rst[i+1]; ++j)
            Numsp.Val[i * n + Front.Col[j]] += Front.Val[j];
    time_end = std::chrono::high_resolution_clock::now();
    init_numsp_time += (time_end - time_begin).count();
    
    
#pragma omp parallel
    {
        MSA<int> accum(n);
        // Loop by rows
    #pragma omp single
        {
            time_begin = std::chrono::high_resolution_clock::now();
        }
    #pragma omp for schedule(dynamic, 256)
        for (size_t i = 0; i < m; ++i) {
            for (int t = AT.Rst[i]; t < AT.Rst[i+1]; ++t) {
                int k = AT.Col[t];
                for (int j = Front.Rst[k]; j < Front.Rst[k+1]; ++j)
                    accum.state[Front.Col[j]] = MSA<int>::ALLOWED;
            }
            int row_nz = 0;
            int *numsp_row = Numsp.Val + i * n;
            for (size_t j = 0; j < n; ++j) {
                if (numsp_row[j] == 0 && accum.state[j] == MSA<int>::ALLOWED)
                    ++row_nz;
                accum.state[j] = MSA<int>::UNALLOWED;
            }
            FrontTmp.Rst[i+1] = row_nz;
        }
    #pragma omp single
        {
            time_end = std::chrono::high_resolution_clock::now();
            symbolic_time += (time_end - time_begin).count();
        }
        
        
    #pragma omp single
        {
            FrontTmp.Rst[0] = 0;
            for (int i = 1; i < m; ++i)
                FrontTmp.Rst[i+1] += FrontTmp.Rst[i];
            FrontTmp.nz = FrontTmp.Rst[m];
            FrontTmp.resize_vals(FrontTmp.nz);
        }
        
    #pragma omp single
        {
            time_begin = std::chrono::high_resolution_clock::now();
        }
    #pragma omp for schedule(dynamic, 256)
        for (size_t i = 0; i < m; ++i) {
            for (int t = AT.Rst[i]; t < AT.Rst[i+1]; ++t) {
                int k = AT.Col[t];
                int a_val = AT.Val[t];
                for (int j = Front.Rst[k]; j < Front.Rst[k+1]; ++j)
                    accum.value[Front.Col[j]] += a_val * Front.Val[j];
            }

            int c_pos = FrontTmp.Rst[i];
            int *numsp_row = Numsp.Val + i * n;
            for (int j = 0; j < accum.len; ++j) {
                if (numsp_row[j] == 0 && accum.value[j] != 0) {
                    FrontTmp.Col[c_pos] = j;
                    FrontTmp.Val[c_pos++] = accum.value[j];
                }
                accum.value[j] = 0;
            }
        }
    #pragma omp single
        {
            time_end = std::chrono::high_resolution_clock::now();
            numeric_time += (time_end - time_begin).count();
        }
    }

    std::swap(Front, FrontTmp);
}

std::chrono::time_point<std::chrono::steady_clock> start_test, finish_test;
sparseMtx<int> k_truss_test(const sparseMtx<int> &A, int k, mspgemmAlgorithm<int> matrixMult, bool isVectorization) {
    sparseMtx<int> C = A;  // a copy of adjacency matrix
    sparseMtx<int> Tmp;
    int n = A.m;
    int totalIterationNum = 0;
    int *tmp_Rst = new int[n+1];
    tmp_Rst[0] = 0;

    start_test = chrono::steady_clock::now();
    
    int t = 0;
    while (true) {
        // Tmp<C> = C*C
        matrixMult(isVectorization, C, C, C, Tmp);

        // remove all edges included in less than (k-2) triangles
        // and replace values of remaining entries with 1
        int new_curr_pos = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = Tmp.Rst[i]; j < Tmp.Rst[i+1]; ++j) {
                if (Tmp.Val[j] >= k-2) {
                    Tmp.Col[new_curr_pos]   = Tmp.Col[j];
                    Tmp.Val[new_curr_pos++] = 1;
                }
            }
            tmp_Rst[i+1] = new_curr_pos;
        }
        memcpy(Tmp.Rst, tmp_Rst, (n+1)*sizeof(int));
        Tmp.nz = Tmp.Rst[n];

        const bool samePattern =
            (Tmp.m == C.m) && (Tmp.n == C.n) && (Tmp.nz == C.nz) &&
            std::memcmp(Tmp.Rst, C.Rst, (n + 1) * sizeof(int)) == 0 &&
            (Tmp.nz == 0 || std::memcmp(Tmp.Col, C.Col, Tmp.nz * sizeof(int)) == 0);
        if (samePattern) {
            totalIterationNum = ++t;
            break;
        }

        // Assign 'Tmp' to 'C'
        std::swap(C, Tmp);
    }

    finish_test = chrono::steady_clock::now();

    if (C.nz < A.nz) {
        int *new_Adj = new int[C.nz];
        int *new_Wgt = new int[C.nz];
        std::memcpy(new_Adj, C.Col, C.nz * sizeof(int));
        std::memcpy(new_Wgt, C.Val, C.nz * sizeof(int));
        delete[] C.Col;
        delete[] C.Val;
        C.Col = new_Adj;
        C.Val = new_Wgt;
    }

    delete[] tmp_Rst;
    return C;
}

int64_t triangle_counting_test(const sparseMtx<int> &A, mspgemmAlgorithm<int> matrixMult, bool isVectorization) {
    int64_t num_of_tr = 0;
    sparseMtx<int> L = extract_lower_triangle(A);
    sparseMtx<int> C;

    start_test = chrono::steady_clock::now();

    /* TRIANGLE COUNTING ITSELF */
    matrixMult(isVectorization, L, L, L, C);
#pragma omp parallel for reduction(+:num_of_tr)
    for (int j = 0; j < C.Rst[C.m]; ++j)
        num_of_tr += C.Val[j];
    /* TRIANGLE COUNTING ITSELF */

    finish_test = chrono::steady_clock::now();
    //cout << "Time:       " << chrono::duration_cast<chrono::milliseconds>(finish_test - start_test).count() << " ms\n";
    //cout << "Triangles:  " << num_of_tr << '\n';

    return num_of_tr;
}