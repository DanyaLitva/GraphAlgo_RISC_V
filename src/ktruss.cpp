#include "ktruss.h"
#include "matrix_la.h"
#include <chrono>
using namespace std;

/* K-TRUSS */
spMtx<int> k_truss(const spMtx<int> &A, int k, mxmOp<int> matrixMult, bool isParallel) {
    spMtx<int> C = A;  // a copy of adjacency matrix
    spMtx<int> Tmp;
    int n = A.m;
    int totalIterationNum = 0;
    int *tmp_Rst = new int[n+1];
    tmp_Rst[0] = 0;

    auto start = chrono::steady_clock::now();

    for (int t = 0; t < n; ++t) {
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
