#include "triangle_counting.h"
#include "matrix_la.h"
#include <chrono>
using namespace std;

int* triangle_counting_vertex(const spMtx<int> &A, mxmOp<int> matrixMult, bool isParallel, bool isVectorization) {
    /* PREPARE DATA */
    int *nums_of_tr = new int[A.m];
    int num_of_tr;
    spMtx<int> SQ; // A^2 (A is adjacency matrix)

    auto start = chrono::steady_clock::now();

    /* TRIANGLE COUNTING ITSELF */
    matrixMult(isParallel, isVectorization, A, A, A, SQ);
    // for each vertex we count the number of triangles it belongs to
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


int64_t triangle_counting_masked_lu(const spMtx<int> &A, mxmOp<int> matrixMult, bool isParallel, bool isVectorization) {
    int64_t num_of_tr = 0;
    spMtx<int> L = extract_lower_triangle(A);
    spMtx<int> U = transpose(L);
    spMtx<int> C;

    auto start = chrono::steady_clock::now();

    /* TRIANGLE COUNTING ITSELF */
    matrixMult(isParallel, isVectorization, L, U, A, C);

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


int64_t triangle_counting_masked_sandia(const spMtx<int> &A, mxmOp<int> matrixMult, bool isParallel, bool isVectorization) {
    int64_t num_of_tr = 0;
    spMtx<int> L = extract_lower_triangle(A);
    spMtx<int> C;

    auto start = chrono::steady_clock::now();

    /* TRIANGLE COUNTING ITSELF */
    matrixMult(isParallel, isVectorization, isVectorization, L, L, L, C);

    // Count the total number of triangles
#pragma omp parallel for reduction(+:num_of_tr)
    for (int j = 0; j < C.Rst[C.m]; ++j)
        num_of_tr += C.Val[j];
    /* TRIANGLE COUNTING ITSELF */

    auto finish = chrono::steady_clock::now();
    cout << "Time:       " << chrono::duration_cast<chrono::milliseconds>(finish - start).count() << " ms\n";
    cout << "Triangles:  " << num_of_tr << '\n';

    return num_of_tr;
}
