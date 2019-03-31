/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_utils.h>
#include <limits>
#include <iostream>

#include "smo_sets.h"
#include "workingset.h"
#include "kernelcache.h"
#include "smoblocksolve.h"
#include "linalg/gemv.h"
#include "linalg/unary_op.h"
#include <cub/device/device_select.cuh>
#include "print_vec.h"
#include "linalg/cublas_wrappers.h"
#include "ws_util.h"

namespace ML {
namespace SVM {

using namespace MLCommon;



/** This kernel is called once after SVM is fitted, to collect results.
  * 
  */
template<typename math_t>
__global__ void CollectSupportVectors(const math_t *x, const int n_rows, const int n_cols, 
    const math_t *y, const math_t *alpha, int *idx, int n_coefs, math_t *dual_coefs,  
    math_t *x_support) {
  int tid =  blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < n_coefs) {
    int i = idx[tid]; 
    dual_coefs[tid] = alpha[i] * y[i];
    for (int j=0; j< n_cols; j++) {
        x_support[tid + j * n_coefs] = x[i + j * n_rows];
    }
  }
}

/** This kernel is called once after SVM is fitted, to calculate the bias for the decision function.
  * 
  */
template<typename math_t>
__global__ void CalcB(const math_t *cacheTile, int n_rows, int n_coefs, 
    const math_t *y, const math_t *alpha, const math_t *dual_coefs, int *idx, math_t C, math_t *b) {
  int n = threadIdx.x;
  __shared__ math_t bval[1024];
  bval[threadIdx.x] = INFINITY;
  // search for an unbound support vector (0 < alpha < C)
  while (n < n_coefs) {
    int i = idx[n]; 
    math_t a = alpha[i];
    if (a < C) {
      math_t s = 0;
      // Calculate s = \sum y_j \alpha_j K(x_j, x_i)
      for (int j=0; j< n_coefs; j++) {
        s += cacheTile[i + j * n_rows] * dual_coefs[j];
      }
      // The decision function for the support vector should give exactly y_i
      // f(x_i) =  s + b = y_i
      // therefore
      bval[threadIdx.x] = y[i] - s;
      break;
    }
    n += blockDim.x;
  }
  
  // return one of the b's (they should be equal for all non-bound support vector)
  __syncthreads();
  if (threadIdx.x==0) {
      for (int i=0; i<n_coefs && i<1024; i++) {
          if (bval[i] < INFINITY) {
              *b = bval[i];
              break;
          }
      }
  }
}

/**
* Implements SMO algorithm based on ThunderSVM and OHD-SVM. 
*/
template<typename math_t>
class SmoSolver {
public:
  int n_rows = 0;
  int n_ws = 0;
  int n_cols;
  // Buffers for the domain [n_rows]
  math_t *alpha = nullptr;       //< dual coordinates
  math_t *f = nullptr;           //< optimality indicator vector
  
  // Buffers for the working set [n_ws]
  math_t *delta_alpha = nullptr; // change for the working set
  
  // return some parameters from the kernel;
  math_t *return_buff = nullptr;
  math_t host_return_buff[2];  // used to return iteration numbef and convergence information from the kernel
  
  math_t C;
  math_t tol;

public:
  SmoSolver(math_t C = 1, math_t tol = 0.001) 
    : n_rows(n_rows), C(C), tol(tol)
  {
  }
  
  ~SmoSolver() {
      FreeBuffers();
  }
  

  // this needs to know n_ws, therefore it can be only called during the solve step
  void AllocateBuffers(int n_rows, int n_cols, int n_ws) {
    FreeBuffers();
    this->n_rows = n_rows;
    this->n_cols = n_cols;
    this->n_ws = n_ws;
    allocate(alpha, n_rows); 
    allocate(f, n_rows);  
    allocate(delta_alpha, n_ws);
    allocate(return_buff, 2);
  }    
  
  void FreeBuffers() {
    if(alpha) CUDA_CHECK(cudaFree(alpha));
    if(f) CUDA_CHECK(cudaFree(f));
    if(delta_alpha) CUDA_CHECK(cudaFree(delta_alpha));
    if(return_buff) CUDA_CHECK(cudaFree(return_buff));
    alpha = nullptr;
    f = nullptr;
    delta_alpha = nullptr;
    return_buff = nullptr;
  }
    
  ///
  /// Init the values of alpha, f, and helper buffers. 
  void Initialize(math_t* y) {
    // we initialize 
    // alpha_i = 0 and 
    // f_i = -y_i
    CUDA_CHECK(cudaMemset(alpha, 0, n_rows * sizeof(math_t)));
    LinAlg::unaryOp(f, y, n_rows, []__device__(math_t y){ return -y; });

  }

#define SMO_WS_SIZE 256
  
  void Solve(math_t *x, int n_rows, int n_cols, math_t *y, math_t **dual_coefs, int *n_coefs, 
             math_t **x_support, int **idx, math_t *b, cublasHandle_t cublas_handle,
             int max_outer_iter = -1, int max_inner_iter = 10000) {
    if (max_outer_iter == -1) {
        max_outer_iter = n_rows < std::numeric_limits<int>::max() / 100 ?  n_rows * 100 :
             std::numeric_limits<int>::max();
        max_outer_iter = max(100000, max_outer_iter);
    }
    WorkingSet<math_t> ws(n_rows, SMO_WS_SIZE);
    int n_ws = ws.GetSize();
    AllocateBuffers(n_rows, n_cols, n_ws);    
    Initialize(y);
    
    KernelCache<math_t> cache(x, n_rows, n_cols, n_ws, cublas_handle);
    
    int n_iter = 0;
    math_t diff = 10*tol;
    
    while (n_iter < max_outer_iter && diff >= tol) { 
      CUDA_CHECK(cudaMemset(delta_alpha, 0, n_ws * sizeof(math_t)));
      ws.Select(f, alpha, y, C);
      
      math_t * cacheTile = cache.GetTile(ws.idx); 

      SmoBlockSolve<math_t, SMO_WS_SIZE><<<1, n_ws>>>(y, n_rows, alpha, n_ws, delta_alpha, f, cacheTile,
                                  ws.idx, C, tol, return_buff, max_inner_iter);
      CUDA_CHECK(cudaPeekAtLastError());

      updateHost(host_return_buff, return_buff, 2);
        
      UpdateF(f, n_rows, delta_alpha, n_ws, cacheTile, cublas_handle);
      // check stopping condition
      diff = host_return_buff[0];
      n_iter++;
    }    
    
    std::cout<<"SMO solver finished after "<<n_iter<<" iterations with diff "<<diff<<"\n";
    GetResults(x, n_rows, n_cols, y, alpha, dual_coefs, n_coefs, idx, x_support, b, cublas_handle);  
    
    FreeBuffers(); 
  }
  
  void GetResults(const math_t *x,  int n_rows, int n_cols, const math_t *y, 
                  const math_t *alpha,
                  math_t **dual_coefs, int *n_coefs, int **idx, math_t **x_support, math_t *b,
                  cublasHandle_t cublas_handle
                 ) {
   
    int *f_idx, *f_idx_selected;
    int *d_num_selected;
    allocate(f_idx, n_rows);
    allocate(f_idx_selected, n_rows);
    allocate(d_num_selected, 1);
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    
    int TPB=256;
    init_f_idx<<<ceildiv(n_rows, TPB), TPB>>>(n_rows, f_idx);
    CUDA_CHECK(cudaPeekAtLastError());
    
    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, f_idx, f_idx_selected, d_num_selected, 
                          n_rows, [alpha]__device__ (int i) {return alpha[i] > 0;});
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run selection
    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, f_idx, f_idx_selected, d_num_selected, 
                          n_rows, [alpha]__device__ (int i) {return  alpha[i] > 0;});
    
    updateHost(n_coefs, d_num_selected, 1);
    if (*n_coefs > 0) {
      allocate(*idx, *n_coefs);
      copy(*idx, f_idx_selected, *n_coefs);
      allocate(*dual_coefs, *n_coefs);
      allocate(*x_support, (*n_coefs) * n_cols);
    
      CollectSupportVectors<<<ceildiv(*n_coefs,TPB),TPB>>>(x, n_rows, n_cols, y, alpha, *idx,
          *n_coefs, *dual_coefs, *x_support);
    
      CUDA_CHECK(cudaPeekAtLastError());

      // calculate b
      math_t *b_dev;
      allocate(b_dev,1);
      KernelCache<math_t> cache(x, n_rows, n_cols, *n_coefs, cublas_handle);
      math_t *cacheTile = cache.GetTile(*idx);
      CalcB<<<1,1024>>>(cacheTile,  n_rows, *n_coefs, y, alpha, *dual_coefs, *idx, C, b_dev);
      CUDA_CHECK(cudaPeekAtLastError());

      updateHost(b, b_dev, 1);
    
      CUDA_CHECK(cudaFree(b_dev));
    }
    CUDA_CHECK(cudaFree(f_idx));
    CUDA_CHECK(cudaFree(f_idx_selected));
    CUDA_CHECK(cudaFree(d_num_selected));
    CUDA_CHECK(cudaFree(d_temp_storage)); 
  }
  
  void UpdateF(math_t *f, int n_rows, const math_t *delta_alpha, int n_ws, const math_t *cacheTile, cublasHandle_t cublas_handle) {
    // check sign here too.
    math_t one = 1; // multipliers used in the equation : f = 1*cachtile * delta_alpha + 1*f
    CUBLAS_CHECK(LinAlg::cublasgemv(cublas_handle, CUBLAS_OP_N, n_rows, n_ws, &one, cacheTile,
        n_rows, delta_alpha, 1, &one, f, 1));
    //LinAlg::gemv(cacheTile, n_rows, n_ws, delta_alpha, f, false, math_t(1.0), math_t(1.0), cublas_handle);
  }
};

}; // end namespace SVM 
}; // end namespace ML

