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
#include <cub/cub.cuh>
//#include <cub/device/device_radix_sort.cuh>
#include <limits.h>
#include <linalg/unary_op.h>
#include <linalg/ternary_op.h>
#include <selection/kselection.h>

#include "smo_sets.h"

namespace ML {
namespace SVM {

using namespace MLCommon;

__global__ void init_f_idx(int n_rows, int *f_idx) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n_rows) { f_idx[tid] = tid; }
}

__global__ void set_available(bool *available, int n_rows, int *idx, int n_selected){ 
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n_rows) available[tid] = true;
    if (tid < n_selected) {
        available[idx[tid]] = false;
    }
}
/**
* Working set selection for the SMO algorithm.
*/
  
template<typename math_t>
class WorkingSet {

public:
  int *idx = nullptr; // indices for elements in the working set
  
  WorkingSet(int n_rows=0, int n_ws=0) {
      SetSize(n_rows, n_ws);
  }
  
  ~WorkingSet() {
      FreeBuffers();
  }
  
  /** Set the size of the working set and allocate buffers accordingly.
   * The default is to use min(n_rows, 1024).
   */
  void SetSize(int n_rows, int n_ws = 0) {
    if (n_ws == 0 || n_ws > n_rows) {
      n_ws = n_rows;
    } 
    n_ws = min(1024, n_ws); // should not be larger than the number of threads/block
    this->n_ws = n_ws;
    this->n_rows = n_rows;
    AllocateBuffers();
  }
  
  /** Return the size of the working set. */
  int GetSize() {
    return n_ws;
  }
  
//private:
  int n_rows = 0;
  int n_ws = 0;

  // Buffers for the domain [n_rows]
  int *f_idx = nullptr;          //< Arrays used for sorting f
  int *f_idx_sorted = nullptr;   //<
  int *f_idx_tmp = nullptr;  
  math_t *f_sorted = nullptr;
  math_t *f_masked = nullptr;
  int *d_num_selected = nullptr;
  bool *available = nullptr;
  
  // Buffers for the working set [n_ws]
  
  void *cub_temp_storage = NULL; // used by cub for reduction
  size_t cub_temp_storage_bytes = 0;
  
  void AllocateBuffers() {
    FreeBuffers();
    if (n_ws > 0) {
      allocate(f_idx, n_rows);     
      allocate(f_idx_sorted, n_rows);
      allocate(f_idx_tmp, n_rows);
      allocate(idx, n_ws);
      allocate(f_sorted, n_rows); 
      allocate(f_masked, n_rows); 
      allocate(available, n_rows); 
      allocate(d_num_selected, 1);
      // Determine temporary device storage requirements for cub
      cub_temp_storage = NULL;
      cub_temp_storage_bytes = 0;
      
      cub::DeviceRadixSort::SortPairs(cub_temp_storage, cub_temp_storage_bytes, f_idx, f_idx_sorted, f_masked, f_sorted, n_rows);
      size_t bytes;
      int tmp;
      cub::DeviceSelect::If(cub_temp_storage, bytes, f_idx, f_idx, &tmp, n_rows, []__device__(int idx){return true;});
      if (bytes>cub_temp_storage_bytes) cub_temp_storage_bytes = bytes;
      CUDA_CHECK(cudaMalloc(&cub_temp_storage, cub_temp_storage_bytes));
      //allocate((char*)cub_temp_storage, tmp);
      Initialize();
    }
  }    
  
  void FreeBuffers() {
    if (f_idx) CUDA_CHECK(cudaFree(f_idx));
    if (f_idx_sorted) CUDA_CHECK(cudaFree(f_idx_sorted));
    if (f_idx_tmp) CUDA_CHECK(cudaFree(f_idx_tmp));
    if (cub_temp_storage) (cudaFree(cub_temp_storage));
    if (idx) CUDA_CHECK(cudaFree(idx));
    if (f_masked) CUDA_CHECK(cudaFree(f_masked));
    if (f_sorted) CUDA_CHECK(cudaFree(f_sorted));
    if (available) CUDA_CHECK(cudaFree(available));
    if (d_num_selected) CUDA_CHECK(cudaFree(d_num_selected));
    f_idx = nullptr;
    f_idx_sorted = nullptr;
    cub_temp_storage = nullptr;
    idx = nullptr;
    f_masked = nullptr;
    f_sorted = nullptr;
    available = nullptr;
    d_num_selected = nullptr;
  }
  
  void Select(math_t *f, math_t *alpha, math_t *y, math_t C) {
    if (n_ws >= n_rows) {
        // we already initialized idx correctly in the beginning.
        return;
    }
    //Selection::warpTopKtemplate<false, false>(f_sorted, f_idx_sorted, f_masked, 512, n_rows, 1);    
    
    cub::DeviceRadixSort::SortPairs((void*) cub_temp_storage, cub_temp_storage_bytes, f, f_sorted, f_idx, f_idx_sorted, n_rows);
    
    int n_selected;
    cub::DeviceSelect::If(cub_temp_storage, cub_temp_storage_bytes, f_idx_sorted, f_idx_tmp, d_num_selected, n_rows, 
                          [alpha, y, C]__device__(int idx) { return in_upper(alpha[idx], y[idx], C); });
    
    
    updateHost(&n_selected, d_num_selected, 1);
    int n_copy1 = n_selected> n_ws/2 ? n_ws/2 : n_selected;
    copy(idx, f_idx_tmp, n_copy1);
    int TPB=256;
    bool *available = this->available;
    set_available<<<ceildiv(n_rows,TPB),TPB>>>(available, n_rows, idx, n_copy1);
    cub::DeviceSelect::If((void*)cub_temp_storage, cub_temp_storage_bytes, f_idx_sorted, f_idx_tmp, d_num_selected, n_rows, 
        [alpha, y, available, C]__device__(int idx) { 
            return available[idx] && in_lower(alpha[idx], y[idx], C); 
        }
    );
    updateHost(&n_selected, d_num_selected, 1);
    int n_copy2 = n_selected > n_ws-n_copy1 ? n_ws-n_copy1 : n_selected;
    copy(idx + n_copy1, f_idx_tmp+n_selected-n_copy2, n_copy2); 
    
    if (n_copy1 + n_copy2 < n_ws) {
       //std::cout<<"Warning: could not fill working set, found only "<<n_copy1 + n_copy2<< " elements.\n";
       //std::cout<<"Filling up with unused elements\n";
       n_copy1 += n_copy2;
       set_available<<<ceildiv(n_rows,TPB),TPB>>>(available, n_rows, idx, n_copy1);
       cub::DeviceSelect::Flagged((void*)cub_temp_storage, cub_temp_storage_bytes, f_idx, available, f_idx_tmp, d_num_selected, n_rows);
//     cub::DeviceSelect::If((void*)cub_temp_storage, cub_temp_storage_bytes, f_idx_sorted,
//       f_idx_tmp, # d_num_selected, n_rows, [available]__device__(int idx) {return available[idx];});
       updateHost(&n_selected, d_num_selected, 1);
       n_copy2 = n_selected > n_ws-n_copy1 ? n_ws-n_copy1 : n_selected;
       copy(idx + n_copy1, f_idx_tmp, n_copy2); 
    }
  }
 
  void Initialize() {
    int TPB = 256;
    init_f_idx<<<ceildiv(n_rows, TPB), TPB>>>(n_rows, f_idx);
    CUDA_CHECK(cudaPeekAtLastError());
    init_f_idx<<<ceildiv(n_ws, TPB), TPB>>>(n_rows, idx);
    CUDA_CHECK(cudaPeekAtLastError());
  }  
};

}; // end namespace SVM 
}; // end namespace ML
