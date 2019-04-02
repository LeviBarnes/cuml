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

#include "ml_utils.h"
#include <cuda_utils.h>
#include <linalg/gemm.h>

namespace ML {
namespace SVM {

using namespace MLCommon;

/**
 * @brief Collect rows of the training data into contiguous space
 *
 * The working set is a subset of all the training examples. Here we collect
 * all the training vectors that are in the working set.
 *
 * @param [in] x training data in column major format, size [n_rows x n_cols]
 * @param [in] n_rows
 * @param [in] n_cols
 * @param [out] x_ws training vectors in the working set in column major format, size [n_ws x n_cols]
 * @param [in] n_ws the number of elements in the working set
 * @param [in] ws_idx working set indices (row indices of x), size [n_ws]
 */
template <typename math_t>
__global__ void collect_rows(const math_t *x, int n_rows, int n_cols,
                              math_t *x_ws, int n_ws, const int *ws_idx)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int ws_row = tid % n_ws; // row idx
  if (ws_row < n_ws && tid < n_ws * n_cols) {
    int x_row = ws_idx[ws_row]; // row in the original matrix
    int col = tid / n_ws; // col idx
    if (x_row + col * n_rows < n_rows*n_cols) {
      x_ws[tid] = x[x_row + col * n_rows];
    }
  }
}

/**
* @brief Buffer to store a kernel tile
*
* We calculate the kernel matrix for the vectors in the working set.
* For every vector x_i in the working set, we alwas calculate a full row of the kernel matrix K(x_j, x_i), j=1..n_rows.
*
* A kernel tile stores all the kernel rows for the working set, i.e.  K(x_j, x_i) for all i in the working set, and j in 1..n_rows.
*
* Currently we just buffer the kernel values before we call the SmoBlockSolver.
* In the future we probably should can keep a larger cache and use that to avoid repeated kernel calculations.
*/
template<typename math_t>
class KernelCache {
private:
  const math_t *x;   //!< pointer to the training vectors
  math_t *x_ws;      //!< feature vectors in the current working set
  int *ws_idx_prev;
  int n_ws_prev = 0;
  int n_rows;
  int n_cols;
  int n_ws;
  int cache_size = 200; // MiB
  int n_cache; // number of rows cached

  math_t *tile = nullptr;
  math_t *cache = nullptr;
  cublasHandle_t cublas_handle;

  void (*kernelOp) (const math_t*, int, int, const math_t* , math_t*,
                    int, int, cublasOperation_t, cublasOperation_t,
                    math_t, math_t, cublasHandle_t) = nullptr;

  void AllocateAll() {
    allocate(x_ws, n_ws*n_cols);
    allocate(tile, n_rows*n_ws);
    allocate(ws_idx_prev, n_ws);
    n_cache = cache_size * 1024*1024 / sizeof(math_t) / n_cols;
    //allocate(cache, n_cache); //not yet used
  }
public:
  /**
   * Construct an object to manage kernel cache
   * 
   * @param x device array of training vectors in column major format, size [n_rows x n_cols]
   * @param n_rows number of training vectors
   * @param n_cols number of features
   * @param n_ws size of working set
   * @param cublas_handle
   * @param cache_size not used yet
   */
  KernelCache(const math_t *x, int n_rows, int n_cols, int n_ws, cublasHandle_t cublas_handle, int cache_size = 200)
    : x(x), n_rows(n_rows), n_cols(n_cols), n_ws(n_ws), cublas_handle(cublas_handle),
      cache_size(cache_size)
  {
    AllocateAll();
  };

  /**
   * Construct an object to manage kernel cache
   * 
   * @param x device array of training vectors in column major format, size [n_rows x n_cols]
   * @param n_rows number of training vectors
   * @param n_cols number of features
   * @param n_ws size of working set
   * @param cublas_handle
   * @param kernelop
   */
  KernelCache(math_t *x, int n_rows, int n_cols, int n_ws, cublasHandle_t cublas_handle,
       void (*kernelOp) (const math_t*, int, int, const math_t* , math_t*,
                         int, int, cublasOperation_t, cublasOperation_t,
                         math_t, math_t, cublasHandle_t)
                                                        ) : kernelOp(kernelOp), x(x), n_rows(n_rows),
                                       n_cols(n_cols), n_ws(n_ws), cublas_handle(cublas_handle)
  {
    AllocateAll();
  }

  ~KernelCache() {
    CUDA_CHECK(cudaFree(tile));
    CUDA_CHECK(cudaFree(x_ws));
    CUDA_CHECK(cudaFree(ws_idx_prev));
    //CUDA_CHECK(cudaFree(cache));
  };


  /**
   * @brief Calculate kernel function values for vectors x1 and x2
   * @param [in] x1 [n1 x n_cols] feature vectors
   * @param [in] x2 [n2 x n_cols] feature vectors
   * @param [out] K buffer for return values [n1xn2] (should be already allocated)
   */
  void calcKernel(const math_t *x1, int n1, const math_t *x2, int n2, math_t *K,
      int ld1=0, int ld2=0) {
    //calculate kernel function values for indices in ws_idx
    if (ld1<=0) {
      ld1 = n1;
    }
    if (ld2<=0) {
      ld2 = n2;
    }
    if (kernelOp) {
       (*kernelOp)(x1, n1, n_cols, x2, tile, n1, n2, CUBLAS_OP_N,
          CUBLAS_OP_T, math_t(1.0), math_t(0.0), cublas_handle) ;
    } else {
      math_t alpha = 1;
      math_t beta = 0;
      // LinAlg::gemm(x1, n1, n_cols, x2, tile, n1, n2, CUBLAS_OP_N,
       //   CUBLAS_OP_T, math_t(1.0), math_t(0.0), cublas_handle) ;
       CUBLAS_CHECK(LinAlg::cublasgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
         n1, n2, n_cols, &alpha, x1, ld1, x2, ld2, &beta, K, n1));
    }

  }
  /**
   * @brief Get all the kernel matrix rows for the working set.
   * @param ws_idx indices of the working set
   * @return pointer to the kernel tile [ n_rows x n_ws] K_j,i = K(x_j, x_q) where j=1..n_rows and q = ws_idx[i], j is the contiguous dimension
   * @note currently we just recalculate every value.
   * We have implemented linear kernels so far
   */
  math_t* GetTile(int *ws_idx) {
    // collect all the feature wectors in the working set
	if (n_ws > 0) {
      const int TPB=256;
      CUDA_CHECK(cudaPeekAtLastError());
      collect_rows<<<ceildiv(n_ws*n_cols,TPB), TPB>>>(x, n_rows, n_cols, x_ws, n_ws, ws_idx);
      CUDA_CHECK(cudaPeekAtLastError());

      calcKernel(x, n_rows, x_ws, n_ws, tile);
      
      n_ws_prev = n_ws;
      copy(ws_idx_prev, ws_idx, n_ws);
    }
    return tile;
  }
};

}; // end namespace SVM
}; // end namespace ML
