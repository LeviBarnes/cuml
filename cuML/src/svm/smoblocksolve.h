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
#include "selection/kselection.h"
#include "smo_sets.h"
#include <stdlib.h>
namespace ML {
namespace SVM {

/**
 * Solve the optimization problem for the actual working set.
 * 
 * TODO describe the math here.
 */
template<typename math_t, int WSIZE>
__global__ void SmoBlockSolve(math_t *y_array, int n_rows, math_t* alpha, int n_ws, 
      math_t *delta_alpha, math_t *f_array, math_t *kernel, int *ws_idx, 
      math_t C, math_t eps, math_t *return_buff, int max_iter = 10000)
  {
    typedef Selection::KVPair<math_t, int> Pair;
    typedef cub::BlockReduce<Pair, WSIZE> BlockReduce;
    typedef cub::BlockReduce<math_t, WSIZE> BlockReduceFloat;
    __shared__ union {
        typename BlockReduce::TempStorage pair; 
        typename BlockReduceFloat::TempStorage single;
    } temp_storage; 
    
    __shared__ math_t f_u;
    __shared__ int u;
    __shared__ int l;
    
    __shared__ math_t tmp_u, tmp_l;
    __shared__ math_t Kd[WSIZE]; // diagonal elements of the kernel matrix
    
    int tid = threadIdx.x; 
    int idx = ws_idx[tid];
    
    // store values in registers
    math_t y = y_array[idx];
    math_t f = f_array[idx];
    math_t a = alpha[idx];
    math_t a_save = a;
    __shared__ math_t diff_end;
    __shared__ math_t diff;
    
    Kd[tid] = kernel[tid*n_rows + idx];
    int n_iter = 0;
    
    for (; n_iter < max_iter; n_iter++) {
      // mask values outside of X_upper  
      math_t f_tmp = in_upper(a, y, C) ? f : INFINITY; 
      Pair pair{f_tmp, tid};
      Pair res = BlockReduce(temp_storage.pair).Reduce(pair, cub::Min(), n_ws);
      if (tid==0) {
        f_u = res.val;
        u = res.key;
      }
      // select f_max to check stopping condition
      f_tmp = in_lower(a, y, C) ? f : -INFINITY;
      __syncthreads();   // needed because we are reusing the shared memory buffer   
      math_t Kui = kernel[u * n_rows + idx];
      math_t f_max = BlockReduceFloat(temp_storage.single).Reduce(f_tmp, cub::Max(), n_ws);
      
      if (tid==0) {
        // f_max-f_u is used to check stopping condition.
        diff = f_max-f_u;
        if (n_iter==0) {
          return_buff[0] = diff;
          diff_end = max(eps, 0.1f*diff);
        }
      }
      __syncthreads();
      if (diff < diff_end ) {
        break;
      }
       
      if (f_u < f && in_lower(a, y, C)) {
        f_tmp = (f_u - f) * (f_u - f) / (Kd[tid] + Kd[u] - 2*Kui);
      } else {
        f_tmp = -INFINITY;     
      }
      pair = Pair{f_tmp, tid};
      res = BlockReduce(temp_storage.pair).Reduce(pair, cub::Max(), n_ws);
      if (tid==0) {
          l = res.key;
      }
      __syncthreads();
      math_t Kli = kernel[l * n_rows + idx];
      
      // Update alpha
      //
      // We know that 0 <= a <= C
      // We select q so that both delta alpha_u and delta alpha_l stay in this limit.
      if (threadIdx.x == u) 
            tmp_u = y > 0 ? C - a : a;
      if (threadIdx.x == l) {
            tmp_l = y > 0 ? a : C - a;
            tmp_l = min(tmp_l, (f - f_u) / (Kd[u] + Kd[l] - 2 * Kui)); // note: Kui == Kul for this thread
      }
      __syncthreads();
      math_t q = min(tmp_u, tmp_l);
      
      if (threadIdx.x == u) a += q * y;
      if (threadIdx.x == l) a -= q * y;
      f += q * (Kui - Kli);
    }
    // save results to global memory before exit
    alpha[idx] = a;
    delta_alpha[tid] = (a - a_save) * y; // it is actuall y * \Delta \alpha
    // f is recalculated in f_update, therefore we do not need to save that
    return_buff[1] = n_iter;
  }
}; // end namespace SVM
}; // end namespace ML
