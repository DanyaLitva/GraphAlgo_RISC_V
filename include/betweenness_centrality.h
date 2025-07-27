#pragma once
#include "matrix.h"

template <typename T>
void bc_backward_step(const spMtx<T> &A,
                      spMtx<T> &Front,
                      const spMtx<T> &Next,
                      const denseMtx<T> &Nspinv,
                      const denseMtx<T> &Numsp,
                      denseMtx<T> &Bcu,
                      long long &ewisemult_time,
                      long long &mspgemm_time);
// template <typename T>
void bc_forward_step(const spMtx<int> &AT,
                     denseMtx<int> &Numsp,
                     spMtx<int> &Front,
                     spMtx<int> &FrontTmp,
                     long long &init_numsp_time,
                     long long &symbolic_time,
                     long long &numeric_time);
std::vector<float> betweenness_centrality_batch(bool isParallel, const spMtx<int> &A, size_t batchSize);