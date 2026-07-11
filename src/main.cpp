#include "matrix.h"
#include "matrix_la.h"
#include "matrix_utils.h"
#include "graph_algorithms.h"
#include <omp.h>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <sstream>
#include <functional>
using namespace std;

struct GraphInfo {
    std::string graphName;
    std::string graphPath;
    std::string logPath;
    std::string format;
};


GraphInfo get_graph_info(int argc, const char *argv[]) {
    std::time_t current_time = chrono::system_clock::to_time_t(chrono::system_clock::now());
    string current_date_time(ctime(&current_time));
    string graphName;
    string graphPath(argv[1]);
    string logfile(argv[2]);
    string action(argv[3]);
    string format;

    int slashPos = graphPath.size() - 1;
    int dotPos   = graphPath.size() - 1;

    while (graphPath[slashPos] != '/')
        --slashPos;
    while (graphPath[dotPos] != '.')
        --dotPos;

    graphName = graphPath.substr(slashPos + 1, dotPos - slashPos - 1);
    format = graphPath.substr(dotPos + 1, graphPath.size() - dotPos);
    graphPath = graphPath.substr(0, slashPos + 1);

    if (action == "launch") {
        current_date_time.pop_back();
        replace(current_date_time.begin(), current_date_time.end(), ' ', '_');
        replace(current_date_time.begin(), current_date_time.end(), ':', '-');

        stringstream ss;
        ss << logfile << '/' << current_date_time << '_' << graphName;
        for (int i = 4; i < argc; ++i)
            ss << '_' << argv[i];
        ss << '_' << to_string(omp_get_max_threads()) << ".txt";
        logfile = ss.str();
    }

    GraphInfo info {graphName, graphPath, logfile, format};

    return info;
}

int launch_test(const sparseMtx<int> &gr, const GraphInfo &info, int argc, const char *argv[]) {
    string benchmarkAlgorithm(argv[4]),
           parOrSeq(argv[5]),
           batchSizeStr;
    bool isParallel = (parOrSeq == "par");
    size_t batch_size;
    mspgemmAlgorithm<int> mxm_algorithm;
    sparseMtx<int> MxmResult, TestMtx = gr;
    chrono::high_resolution_clock::time_point start, finish, default_time;
    stringstream alg_ss;

    if (benchmarkAlgorithm == "bc") {
        if (argc < 7 || (batch_size = atoll(argv[6])) == 0) {
            cerr << "incorrect input, 6-th argument: batch has to be positive integer\n";
            return -6;
        }
    }
    else {
        string multiplicationAlgorithm(argv[6]);
        if (multiplicationAlgorithm == "naive")
            mxm_algorithm = mspgemm_naive<int>;
        if (multiplicationAlgorithm == "msa")
            mxm_algorithm = mspgemm_msa<int>;
        else if (multiplicationAlgorithm == "mca")
            mxm_algorithm = mspgemm_mca<int>;
        else if (multiplicationAlgorithm == "heap")
            mxm_algorithm = mspgemm_heap<int>;
        else {
            cerr << "incorrect input, 6-th argument: has to be 'naive', 'msa', 'mca' or 'heap')\n";
            return -7;
        }
    }

    if (benchmarkAlgorithm != "bc")
        TestMtx = build_symm_from_lower(extract_lower_triangle(TestMtx));

    if (parOrSeq == "par") {
        alg_ss << "Parallel,   " << omp_get_max_threads() << " threads\n";
        // cerr << "PARALLEL " << omp_get_max_threads() << " THREADS\n";
    } else if (parOrSeq == "seq") {
        alg_ss << "Sequential\n";
        // cerr << "SEQUENTIAL \n\n";
    } else {
        cerr << "incorrect input, 5-th argument: has to be 'par' or 'seq'\n";
        return -5;
    }

    if (benchmarkAlgorithm == "mxm") {
        start = chrono::high_resolution_clock::now();
        mxm_algorithm(isParallel, TestMtx, TestMtx, TestMtx, MxmResult);
        finish = chrono::high_resolution_clock::now();
        alg_ss << "Algorithm:  matrix square\n";
    }
    else if (benchmarkAlgorithm == "k-truss") {
        if (argc < 8 || atoi(argv[7]) < 3) {
            cerr << "incorrect input, 7-th argument: has to be positive integer bigger than 2\n";
        }
        alg_ss << "Algorithm:  k-truss, k = " << argv[7] << '\n';
        k_truss(TestMtx, stoi(argv[7]), mxm_algorithm, isParallel);
    }
    else if (benchmarkAlgorithm == "triangle") {
        alg_ss << "Algorithm:  triangle counting\n";
        triangle_counting(TestMtx, mxm_algorithm, isParallel);
    }
    else if (benchmarkAlgorithm == "bc") {
        vector<float> bcVector;
        // size_t cache_fit_size = 1747626; // to size of float matrix to fit into 20 Mb cache 
        // size_t batch_size = (cache_fit_size/Adj.m > 0) ? cache_fit_size/Adj.m : 3;
        start = chrono::high_resolution_clock::now();
        bcVector = brandes_batch(isParallel, TestMtx, batch_size);
        finish = chrono::high_resolution_clock::now();
        // bcVector = betweenness_centrality(isParallel, TestMtx, 5);
        float sum = 0.0f;
        for (size_t i = 0; i < bcVector.size(); ++i)
            sum += bcVector[i];
        for (size_t i = 0; i < bcVector.size(); ++i)
            cout << bcVector[i] << '\n';
        
        // cout << '\n';
        alg_ss << "Algorithm:  betweenness centrality\n" << "Batch size: " << batch_size << '\n';
        alg_ss << "Checksum:   " << sum << '\n';
    }
    else {
        cerr << "incorrect input, 4-th argument: has to be 'triangle', 'k-truss', 'mxm' or 'bc')\n";
        return -4;
    }
    long long time = chrono::duration_cast<chrono::milliseconds>(finish - start).count();
    if (start != default_time)
        cout << "Time:       " << time << " ms\n";
    cout << "Graph name: " << info.graphName << '\n';
    cout << "Vertices:   " << TestMtx.m << '\n';
    cout << "Edges:      " << TestMtx.nz << '\n';
    cout << alg_ss.str() << '\n';
    return 0;
}

string get_graph_val_type(const char *filename, const GraphInfo &info) {
    ifstream istr(filename, (info.format == "bin") ? std::ios::in | std::ios::binary
                                                   : std::ios::in);
    string stype;
    if (info.format == "bin") {
        char type;
        istr >> type >> type >> type;
        if (type == 'R')
            stype = "real";
        else if (type == 'I' || type == 'P')
            stype = "integer";
    } else if (info.format == "mtx") {
        string type;
        istr >> type >> type >> type >> type;
        if (type == "complex")
            throw "Can't use complex numbers!";
        if (type == "real")
            stype = "real";
        else
            stype = "integer";
    } else if (info.format == "graph")
        stype = "integer";
    else if (info.format == "rmat")
        stype = "integer";
    else {
        istr.close();
        throw "Unknown format";
    }

    istr.close();
    return stype;
}

// argv[1] - ����
// argv[2] - ����� ��� ������ �����
// argv[3] - ����������� �������� (tobinary / launch)
// ��� 'launch':
//   argv[4] - ����������� �������� (triangle / k-truss / mxm / bc)
//   argv[5] - ��������������� ��� ����������� (seq / par)
//   ��� 'bc':
//     argv[6] - batch size (����� ������, ��� ������� ����������� ��������)
//   ��� 'triangle' / 'k-truss' / 'mxm':
//     argv[6] - ������������ �������� ��������� (naive / msa / mca / heap)
//     ��� k-truss:
//       argv[7] - �������� 'k' � k-truss

template <typename ValType>
int read_graph_and_launch_test(const GraphInfo &info, int argc, const char *argv[]) {
    string action(argv[3]);
    sparseMtx<ValType> gr(argv[1], info.format.c_str());
    cerr << "finished reading\n";
    if (action == "to_bin") {
        gr.write_crs_to_bin((info.graphPath + info.graphName + ".bin").c_str());
        cerr << "finished writing to BIN\n";
        return 0;
    }
    else if (action == "to_mtx") {
        gr.write_crs_to_mtx((info.graphPath + info.graphName + ".mtx").c_str());
        cerr << "finished writing to MTX\n";
        return 0;
    }
    else if (action == "launch") {
       return launch_test(build_adjacency_matrix(gr), info, argc, argv);
    } else {
        cerr << "incorrect input (3-nd argument has to be 'tobinary' or 'launch')\n";
        return -3;
    }
}

//#define TEST
#ifndef TEST

int main(int argc, const char* argv[]) {
    if (argc < 4) {
        cerr << "not enough arguments (have to be at least 3)\n";
        return -1;
    }

    GraphInfo info = get_graph_info(argc, argv);
    string grType = get_graph_val_type(argv[1], info);

    if (grType == "integer")
        return read_graph_and_launch_test<int>(info, argc, argv);
    else if (grType == "real") {
        return read_graph_and_launch_test<double>(info, argc, argv);
    } else {
        cerr << "unknown value type\n";
        return -2;
    }

    return 0;
}
#else
void test_add_nointersect() {
    for (int iter = 0; iter < 10; ++iter) {
        sparseMtx<int> A = generate_adjacency_matrix(500, 20, 50),
                   C = generate_adjacency_matrix(500, 20, 50),
                   B(A.m, A.n, A.nz + C.nz);
        int bi = 0;
        B.Rst[0] = 0;
        for (int i = 0; i < A.m; ++i) {
            int ai = A.Rst[i];
            for (int ci = C.Rst[i]; ci < C.Rst[i+1]; ++ci) {
                while (ai < A.Rst[i+1] && A.Col[ai] < C.Col[ci])
                    ++ai;
                if (A.Col[ai] != C.Col[ci]) {
                    B.Col[bi] = C.Col[ci];
                    B.Val[bi++] = C.Val[ci];
                }
            }
            B.Rst[i+1] = bi;
        }

        sparseMtx<int> Sum(A.m, A.n);
        sparseMtx<int> Sumbuf(A.m, A.n);
        add_nointersect(A, B, Sum, Sumbuf);

        denseMtx<int> denseA = A, denseB = B;
        denseMtx<int> denseSum = A;
        for (size_t i = 0; i < A.m * A.n; ++i)
            denseSum.Val[i] += denseB.Val[i];

        denseMtx<int> spToDenseSum = Sum;
        if (spToDenseSum == denseSum) {
            cout << iter << ": fine\n";
        }
        else {
            cout << iter << ": WRONG ANSWER!\n";
            cout << "A:\n";
            denseA.print();
            cout << "B:\n";
            denseB.print();
        }
    }
}

void test_spmm_cmask() {
    for (int iter = 0; iter < 500; ++iter) {
        sparseMtx<int> A = generate_adjacency_matrix(500, 50, 70),
                   B = generate_adjacency_matrix(500, 50, 70),
                   M = generate_adjacency_matrix(500, 50, 70),
                   C(A.m, A.n);
        mspgemm_msa_cmask(true, A, B, M, C);

        denseMtx<int> denseA = A,
            denseB = B,
            denseM = M,
            denseC = C;
        for (size_t i = 0; i < A.m * A.n; ++i)
            denseM.Val[i] = 1 - denseM.Val[i];
        dense_mtx_mult(denseA, denseB, denseC);
        for (size_t i = 0; i < C.m * C.n; ++i)
            denseC.Val[i] *= denseM.Val[i];

        denseMtx<int> spToDenseC = C;
        if (spToDenseC == denseC) {
            cout << iter << ": fine\n";
        }
        else {
            cout << iter << ": WRONG ANSWER!\n";
            cout << "A:\n";
            denseA.print();
            cout << "B:\n";
            denseB.print();
            cout << "M:\n";
            denseM.print();
        }
    }
}

int main(int argc, const char *argv[]) {
    test_add_nointersect();
    test_spmm_cmask();
}
#endif
