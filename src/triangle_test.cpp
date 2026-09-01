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
#include <climits>
#define COUNT_REPEAT 8
using namespace std;

// ./build/k_truss_test ./graphs/netherlands_osm.mtx log.txt 5
// srun -p k3 -t 100 ./build/k_truss_test ./graphs/web-Google.mtx log.txt 5
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

    // if (action == "launch") {
    //     current_date_time.pop_back();
    //     replace(current_date_time.begin(), current_date_time.end(), ' ', '_');
    //     replace(current_date_time.begin(), current_date_time.end(), ':', '-');

    //     stringstream ss;
    //     ss << logfile << '/' << current_date_time << '_' << graphName;
    //     for (int i = 4; i < argc; ++i)
    //         ss << '_' << argv[i];
    //     ss << '_' << to_string(omp_get_max_threads()) << ".txt";
    //     logfile = ss.str();
    // }

    GraphInfo info {graphName, graphPath, logfile, format};

    return info;
}

int rvv_test_lmul = 1;

extern std::chrono::time_point<std::chrono::steady_clock> start_test, finish_test;

int launch_test(const sparseMtx<int> &gr, const GraphInfo &info, int argc, const char *argv[]) {
    size_t batch_size;
    sparseMtx<int> MxmResult, TestMtx = gr;
    stringstream alg_ss;
    long long time;
    long long min_time;
    // int arg_k = stoi(argv[3]);

    cout << info.graphName << ",";
    //cout << "Vertices:   " << TestMtx.m << '\n';
    //cout << "Edges:      " << TestMtx.nz << '\n';
    // cout << "MCA k-truss test, k = " << arg_k<< ":\n";
    

    min_time = LLONG_MAX;
    rvv_test_lmul = 1;
    for(size_t i = 0; i < COUNT_REPEAT; ++i){
      triangle_counting_test(TestMtx, mspgemm_mca_m<int>, true, true);
      time = chrono::duration_cast<chrono::milliseconds>(finish_test - start_test).count();
      if(time<min_time) min_time = time;
    }    
    
    cout <<  min_time << ",";


    // min_time = LLONG_MAX;
    // rvv_test_lmul = 2;
    // for(size_t i = 0; i < COUNT_REPEAT; ++i){
    //   triangle_counting_test(TestMtx, mspgemm_mca_m<int>, true, true);
    //   time = chrono::duration_cast<chrono::milliseconds>(finish_test - start_test).count();
    //   if(time<min_time) min_time = time;
    // }    
    
    // cout << min_time << ",";



    // min_time = LLONG_MAX;
    // rvv_test_lmul = 4;
    // for(size_t i = 0; i < COUNT_REPEAT; ++i){
    //   triangle_counting_test(TestMtx, mspgemm_mca_m<int>, true, true);
    //   time = chrono::duration_cast<chrono::milliseconds>(finish_test - start_test).count();
    //   if(time<min_time) min_time = time;
    // }    
    
    // cout <<  min_time << ",";


    min_time = LLONG_MAX;
    for(size_t i = 0; i < COUNT_REPEAT; ++i){
      triangle_counting_test(TestMtx, mspgemm_mca<int>, true, false);
      time = chrono::duration_cast<chrono::milliseconds>(finish_test - start_test).count();
      if(time<min_time) min_time = time;
    }    
    
    cout <<  min_time << ",";

    min_time = LLONG_MAX;
    for(size_t i = 0; i < COUNT_REPEAT; ++i){
      triangle_counting_test(TestMtx, mspgemm_msa<int>, true, false);
      time = chrono::duration_cast<chrono::milliseconds>(finish_test - start_test).count();
      if(time<min_time) min_time = time;
    }    
    
    cout << min_time << endl;


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

template <typename ValType>
int read_graph_and_launch_test(const GraphInfo &info, int argc, const char *argv[]) {
    // string action(argv[3]);
    sparseMtx<ValType> gr(argv[1], info.format.c_str());
    //cerr << "finished reading\n";

    return launch_test(build_adjacency_matrix(gr), info, argc, argv);
}

int main(int argc, const char* argv[]) {
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
