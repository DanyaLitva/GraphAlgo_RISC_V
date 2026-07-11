#include "graph_algorithms.h"
#include "matrix_la.h"
using namespace std;

int* triangle_counting_vertex(const sparseMtx<int> &A, mspgemmAlgorithm<int> matrixMult, bool isParallel) {
    /* PREPARE DATA */
    int *nums_of_tr = new int[A.m];
    int num_of_tr;
    sparseMtx<int> SQ;

    auto start = chrono::steady_clock::now();

    /* TRIANGLE COUNTING ITSELF */
    matrixMult(isParallel, A, A, A, SQ);
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


int64_t triangle_counting_masked_lu(const sparseMtx<int> &A, mspgemmAlgorithm<int> matrixMult, bool isParallel) {
    int64_t num_of_tr = 0;
    sparseMtx<int> L = extract_lower_triangle(A);
    sparseMtx<int> U = transpose(L);
    sparseMtx<int> C;

    auto start = chrono::steady_clock::now();

    /* TRIANGLE COUNTING ITSELF */
    matrixMult(isParallel, L, U, A, C);

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


int64_t triangle_counting(const sparseMtx<int> &A, mspgemmAlgorithm<int> matrixMult, bool isParallel) {
    int64_t num_of_tr = 0;
    sparseMtx<int> L = extract_lower_triangle(A);
    sparseMtx<int> C;

    auto start = chrono::steady_clock::now();

    /* TRIANGLE COUNTING ITSELF */
    matrixMult(isParallel, L, L, L, C);
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
sparseMtx<int> k_truss(const sparseMtx<int> &A, int k, mspgemmAlgorithm<int> matrixMult, bool isParallel) {
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
        matrixMult(isParallel, C, C, C, Tmp);

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

        // check if the number of edges has changed
        if (Tmp.nz == C.nz) {
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

std::vector<float> brandes_batch(bool isParallel, const sparseMtx<int> &A, size_t batch_size) {
    if (A.m != A.n)
        throw "non-square matrix; BC is only for square matrices";
    
    std::chrono::high_resolution_clock::time_point time_begin, time_end;
    long long alloc_time = 0, forward_init_numsp_time = 0,
              forward_symbolic_time = 0, forward_numeric_time = 0,
              conversion_time = 0, fill_time = 0,
              backward_ewisemult_time = 0, backward_mspgemm_time = 0;
    
    time_begin = std::chrono::high_resolution_clock::now();

    size_t m = A.m;
    size_t n = std::min(A.n, batch_size);
    std::vector<float> bcv(m);
    float *bc = bcv.data();
    std::vector<sparseMtx<float>> Sigmas(m);
    // std::vector<sparseMtx<int>> Sigmas(m);
    sparseMtx<int> AT = transpose(A);
    sparseMtx<int> Front(m, n);
    sparseMtx<int> Fronttmp(m, n);
    // sparseMtx<int> Numsp(m, n);
    sparseMtx<int> Numspbuf(m, n);
    denseMtx<int> Numsp(m, n);

    sparseMtx<float> Af = convertType<float>(A);
    denseMtx<float> Numspd(m, n);
    denseMtx<float> Bcu(m, n);
    denseMtx<float> Nspinv(m, n);
    denseMtx<float> W;
    denseMtx<float> Wbuf;

    // long long eWiseAdd_time = 0,
    //     spMul_time = 0,
    //     eWiseMult_time = 0,
    //     dMul_time = 0,
    //     eWiseMultAdd_time = 0;

    size_t mxn = (size_t)m * n;
    // Numsp.resize_vals(n);
    // Numsp.n = n;
    // Numspbuf.n = n;
    // for (size_t j = 0; j < n; ++j) {
    //     Numsp.Rst[j] = j;
    //     Numsp.Col[j] = j;
    //     Numsp.Val[j] = 1;
    // }
    // for (size_t j = n; j <= m; ++j)
    //     Numsp.Rst[j] = n;
    for (size_t j = 0; j < n; ++j)
        Numsp.Val[j*(n+1)] = 1;
    Front = transpose(A.extractRows(0, n));
    
    time_end = std::chrono::high_resolution_clock::now();
    alloc_time = (time_end - time_begin).count();

    size_t d = 0;
    do {
        time_begin = std::chrono::high_resolution_clock::now();
        Sigmas[d] = convertType<float, int>(Front);
        time_end = std::chrono::high_resolution_clock::now();
        conversion_time += (time_end - time_begin).count();
        // Sigmas[d] = Front;

        // eWiseAdd_begin = std::chrono::high_resolution_clock::now();
        brandes_forward_step(AT, Numsp, Front, Fronttmp, forward_init_numsp_time,
                        forward_symbolic_time, forward_numeric_time);
        std::chrono::high_resolution_clock::time_point time_begin, time_end;
        // sparse_add_nointersect(Numsp, Front, Numsp, Numspbuf);
        // eWiseAdd_end = std::chrono::high_resolution_clock::now();
        // eWiseAdd_time += (eWiseAdd_end - eWiseAdd_begin).count();

        // spMul_begin = std::chrono::high_resolution_clock::now();
        // mspgemm_msa_cmask(isParallel, AT, Front, Numsp, Fronttmp);
        // spMul_end = std::chrono::high_resolution_clock::now();
        // spMul_time += (spMul_end - spMul_begin).count();

        // Front = Fronttmp;
        ++d;
    } while (Front.nz != 0);

    time_begin = std::chrono::high_resolution_clock::now();

    Nspinv.resize(m, n);
    Bcu.resize(m, n);
    // Numspd = Numsp;
#pragma omp parallel for
    for (size_t i = 0; i < mxn; ++i)
        Numspd.Val[i] = float(Numsp.Val[i]);

#pragma omp parallel for simd
    for (size_t i = 0; i < mxn; ++i)
        Bcu.Val[i] = 1.0f;
#pragma omp parallel for simd
    for (size_t i = 0; i < mxn; ++i)
        Nspinv.Val[i] = 1.0f / Numspd.Val[i];

    time_end = std::chrono::high_resolution_clock::now();
    fill_time += (time_end - time_begin).count();

    for (size_t k = d-1; k > 0; --k) {
        // eWiseMult_begin = std::chrono::high_resolution_clock::now();
        // masked_ewise_mult(Nspinv, Bcu, Sigmas[k], W);
        // eWiseMult_end = std::chrono::high_resolution_clock::now();
        // eWiseMult_time += (eWiseAdd_end - eWiseAdd_begin).count();

        // dMul_begin = std::chrono::high_resolution_clock::now();
        // masked_spmm(Af, W, Sigmas[k-1], W, Wbuf);
        // dMul_end = std::chrono::high_resolution_clock::now();
        // dMul_time += (dMul_end - dMul_begin).count();

        // dMul_begin = std::chrono::high_resolution_clock::now();
        brandes_backward_step(Af, Sigmas[k], Sigmas[k-1], Nspinv, Numspd, Bcu,
                         backward_ewisemult_time, backward_mspgemm_time);
        // fuse_mxmm_eWiseMultAdd(Af, W, Sigmas[k - 1], Numspd, Bcu);
        // dMul_end = std::chrono::high_resolution_clock::now();
        // dMul_time += (dMul_end - dMul_begin).count();

        // eWiseMultAdd_begin = std::chrono::high_resolution_clock::now();
        // ewise_mult_and_add(W, Numspd, Bcu);
        // eWiseMultAdd_end = std::chrono::high_resolution_clock::now();
        // eWiseMultAdd_time += (eWiseMultAdd_end - eWiseMultAdd_begin).count();
    }

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < m; ++i) {
        float *bcu_ptr = Bcu.Val + n*i;
        for (size_t j = 0; j < n; ++j) {
            bc[i] += bcu_ptr[j];
        }
        bc[i] -= (float)n;
    }

    // std::cerr << "Sparse addition time:            " << eWiseAdd_time    /1000000ll << "ms\n";
    // std::cerr << "Sparse matrix mult time:         " << spMul_time       /1000000ll << "ms\n";
    // std::cerr << "Dense elem-wise mult time:       " << eWiseMult_time   /1000000ll << "ms\n";
    // std::cerr << "Dense-sparse mult time:          " << dMul_time        /1000000ll << "ms\n";
    // std::cerr << "Dense elem-wise mult + add time: " << eWiseMultAdd_time/1000000ll << "ms\n";
    std::cout << "Allocation time:         "
              << (long long)(alloc_time / 1000000ll) << " ms\n";
    std::cout << "Conversion time:         "
              << (long long)(conversion_time / 1000000ll) << " ms\n";
    std::cout << "Forward InitNumsp time:  "
              << (long long)(forward_init_numsp_time / 1000000ll) << " ms\n";
    std::cout << "Forward symbolic time:   "
              << (long long)(forward_symbolic_time / 1000000ll) << " ms\n";
    std::cout << "Forward numeric time:    "
              << (long long)(forward_numeric_time / 1000000ll) << " ms\n";
    std::cout << "Filling time:            "
              << (long long)(fill_time / 1000000ll) << " ms\n";
    std::cout << "Backward EWiseMult time: "
              << (long long)(backward_ewisemult_time / 1000000ll) << " ms\n";
    std::cout << "Backward MSpGEMM time:   "
              << (long long)(backward_mspgemm_time / 1000000ll) << " ms\n";

    return bcv;
}