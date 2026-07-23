#pragma once
#include "matrix.h"

spMtx<int> k_truss(const spMtx<int> &A, int k, mxmOp<int> matrixMult, bool isParallel, bool isVectorization);
