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
#include <iostream>
#include <string>

// ./build/floatMult ./graphs/netherlands_osm.mtx log.txt double par vec mca 10000
using namespace std;

struct GraphInfo {
  std::string graphName;
  std::string graphPath;
  std::string logPath;
  std::string format;
};

GraphInfo get_graph_info(int argc, const char* argv[]) {
  string graphFullPath(argv[1]);
  string logfile(argv[2]);

  int slashPos = graphFullPath.size() - 1;
  int dotPos = graphFullPath.size() - 1;

  while (slashPos >= 0 && graphFullPath[slashPos] != '/')
    --slashPos;
  while (dotPos >= 0 && graphFullPath[dotPos] != '.')
    --dotPos;

  string graphName = graphFullPath.substr(slashPos + 1, dotPos - slashPos - 1);
  string format = graphFullPath.substr(dotPos + 1);
  string graphPath = graphFullPath.substr(0, slashPos + 1);

  return GraphInfo{ graphName, graphPath, logfile, format };
}

template <typename FloatType>
int launch_float_test(const sparseMtx<FloatType>& gr, const GraphInfo& info, int argc, const char* argv[]) {
  string parOrSeq(argv[4]);
  string vecOrScal(argv[5]);
  string multiplicationAlgorithm(argv[6]);
  size_t mask_density = stoull(argv[7]);

  bool isParallel = (parOrSeq == "par");
  bool useVectorization = (vecOrScal == "vec");

  if (vecOrScal != "vec" && vecOrScal != "scal") {
    cerr << "incorrect input, 5-th argument: must be 'vec' or 'scal'\n";
    return -6;
  }
  if (parOrSeq != "par" && parOrSeq != "seq") {
    cerr << "incorrect input, 4-th argument: must be 'par' or 'seq'\n";
    return -5;
  }

  mspgemmAlgorithm<FloatType> mxm_algorithm;

  if (multiplicationAlgorithm == "naive")
    mxm_algorithm = mspgemm_naive<FloatType>;
  else if (multiplicationAlgorithm == "msa")
    mxm_algorithm = mspgemm_msa<FloatType>;
  else if (multiplicationAlgorithm == "mca")
    mxm_algorithm = mspgemm_mca<FloatType>;
  else if (multiplicationAlgorithm == "heap")
    mxm_algorithm = mspgemm_heap<FloatType>;
  else {
    cerr << "incorrect input, 6-th argument: has to be 'naive', 'msa', 'mca' or 'heap'\n";
    return -7;
  }

  cout << "Graph name: " << info.graphName << '\n';
  cout << "Vertices:   " << gr.m << '\n';
  cout << "Edges:      " << gr.nz << '\n';
  cout << "Type:       " << (typeid(FloatType) == typeid(float) ? "float" : "double") << '\n';
  cout << "Density:    " << mask_density << '\n';
  cout << "Algorithm:  " << multiplicationAlgorithm << '\n';
  cout << "Parallel:   " << (isParallel ? "yes" : "no") << '\n';
  cout << "Vectorized: " << (useVectorization ? "yes" : "no") << '\n';

  floating_mask_mult<FloatType>(gr, mask_density, mxm_algorithm, isParallel, useVectorization);

  return 0;
}

template <typename FloatType>
int read_graph_and_launch_test(const GraphInfo& info, int argc, const char* argv[]) {
  sparseMtx<FloatType> gr(argv[1], info.format.c_str());
  cerr << "finished reading\n";
  return launch_float_test<FloatType>(gr, info, argc, argv);
}

int main(int argc, const char* argv[]) {
  if (argc < 8) {
    cerr << "Usage: " << argv[0] << " <graph_path> <log_path> <float|double> <par|seq> <vec|scal> <naive|msa|mca|heap> <mask_density>\n";
    return -1;
  }

  GraphInfo info = get_graph_info(argc, argv);
  string valType(argv[3]);

  if (valType == "float") {
    return read_graph_and_launch_test<float>(info, argc, argv);
  }
  else if (valType == "double") {
    return read_graph_and_launch_test<double>(info, argc, argv);
  }
  else {
    cerr << "unknown value type (expected float or double)\n";
    return -2;
  }

  return 0;
}
