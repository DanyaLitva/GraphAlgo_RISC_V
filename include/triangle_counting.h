#pragma once
#include "matrix.h"

int* triangle_counting_vertex(const spMtx<int> &A, mxmOp<int> matrixMult, bool isParallel);
int64_t triangle_counting_masked_lu(const spMtx<int> &A, mxmOp<int> matrixMult, bool isParallel);
int64_t triangle_counting_masked_sandia(const spMtx<int> &A, mxmOp<int> matrixMult, bool isParallel);
