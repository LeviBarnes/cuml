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
#include <limits.h>
#include "ws_util.h"
#include "smo_sets.h"

namespace ML {
namespace SVM {

using namespace MLCommon;

__device__ bool dummy_select_op (int idx) {
    return true;
}

/**
* Working set selection for the SMO algorithm.
* 
* The working set is a subset of the training vectors, by default it has 1024 elements.
* At every outer iteration in SmoSolver::Solve, we select a different working set, and
* optimize the dual coefficients for the working set.
* 
* The vectors are selected based on the f values, which is the difference between the
* target label and the decision function value
*/
template<typename math_t>
class WorkingSet {

public:
  int *idx = nullptr; //!< indices for elements in the working set
  
  /** Create a working set 
   * \param n_rows number of training vectors
   * \param n_ws number of elements in the working set (default 1024)
   */
  WorkingSet(int n_rows=0, int n_ws=0) {
      SetSize(n_rows, n_ws);
  }
  
  ~WorkingSet() {
      FreeBuffers();
  }
  
  /** 
   * Set the size of the working set and allocate buffers accordingly.
   * 
   * \param n_rows number of training vectors
   * \param n_ws working set size (default min(1024, n_rows))
   */
  void SetSize(int n_rows, int n_ws = 0) {
    if (n_ws == 0 || n_ws > n_rows) {
      n_ws = n_rows;
    } 
    n_ws = min(1024, n_ws);
    this->n_ws = n_ws;
    this->n_rows = n_rows;
    AllocateBuffers();
  }
  
  /** Return the size of the working set. */
  int GetSize() {
    return n_ws;
  }
  
  /**
   * Select a new working set.
   * 
   * Currently we follow the working set selection strategy of ThunderSVM.
   * Following (Joachims, 1998), we select n_ws traning instances as:
   *   - select n_ws/2 element of upper set, where f is largest
   *   - select n_ws/2 from lower set, wher f is smallest
   * 
   * @param f optimality indicator vector, size [n_rows]
   * @param alpha dual coefficients, size [n_rows]
   * @param y target labels (+/- 1)
   * @param C penalty parameter
   */
  void Select(math_t *f, math_t *alpha, math_t *y, math_t C) {
    if (n_ws >= n_rows) {
        // we have initialized idx to cover this case
        return;
    }
    // We are not using the topK kernel, because of the additional lower/upper constraint

    cub::DeviceRadixSort::SortPairs((void*) cub_temp_storage, cub_temp_storage_bytes, f, f_sorted, f_idx, f_idx_sorted, n_rows);
    int n_selected;
    
    // Select top n_ws/2 elements
    cub::DeviceSelect::If(cub_temp_storage, cub_temp_storage_bytes, f_idx_sorted, f_idx_tmp, d_num_selected, n_rows, 
                          [alpha, y, C]__device__(int idx) { return in_upper(alpha[idx], y[idx], C); });
    
    updateHost(&n_selected, d_num_selected, 1);
    int n_copy1 = n_selected> n_ws/2 ? n_ws/2 : n_selected;
    copy(idx, f_idx_tmp, n_copy1);
    
    // Select bottom n_ws/2 elements
    int TPB=256;
    bool *available = this->available;
    set_available<<<ceildiv(n_rows,TPB),TPB>>>(available, n_rows, idx, n_copy1);
    CUDA_CHECK(cudaPeekAtLastError());
    cub::DeviceSelect::If((void*)cub_temp_storage, cub_temp_storage_bytes, f_idx_sorted, f_idx_tmp, d_num_selected, n_rows, 
        [alpha, y, available, C]__device__(int idx) { 
            return available[idx] && in_lower(alpha[idx], y[idx], C); 
        }
    );
    updateHost(&n_selected, d_num_selected, 1);
    int n_copy2 = n_selected > n_ws-n_copy1 ? n_ws-n_copy1 : n_selected;
    copy(idx + n_copy1, f_idx_tmp+n_selected-n_copy2, n_copy2); 
    
    // In case we could not find enough elements, then we just fill using the still available elements.
    if (n_copy1 + n_copy2 < n_ws) {
       //std::cout<<"Warning: could not fill working set, found only "<<n_copy1 + n_copy2<< " elements.\n";
       //std::cout<<"Filling up with unused elements\n";
       n_copy1 += n_copy2;
       set_available<<<ceildiv(n_rows,TPB),TPB>>>(available, n_rows, idx, n_copy1);
       CUDA_CHECK(cudaPeekAtLastError());
       cub::DeviceSelect::Flagged((void*)cub_temp_storage, cub_temp_storage_bytes, f_idx, available, f_idx_tmp, d_num_selected, n_rows);
       updateHost(&n_selected, d_num_selected, 1);
       n_copy2 = n_selected > n_ws-n_copy1 ? n_ws-n_copy1 : n_selected;
       copy(idx + n_copy1, f_idx_tmp, n_copy2); 
    }
  }
  
private:
  int n_rows = 0;
  int n_ws = 0;

  // Buffers for the domain [n_rows]
  int *f_idx = nullptr;          //!< Arrays used for sorting for sorting
  int *f_idx_sorted = nullptr;   //!< Buffer for the sorted indices
  int *f_idx_tmp = nullptr;      //!< Temporary buffer for index manipulation
  math_t *f_sorted = nullptr;    //!< Sorted f values
  int *d_num_selected = nullptr;
  bool *available = nullptr;     //!< Flag vectors available for selection
  
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
      allocate(available, n_rows); 
      allocate(d_num_selected, 1);
      // Determine temporary device storage requirements for cub
      cub_temp_storage = NULL;
      cub_temp_storage_bytes = 0;
      
      cub::DeviceRadixSort::SortPairs(cub_temp_storage, cub_temp_storage_bytes, f_idx, f_idx_sorted, f_sorted, f_sorted, n_rows);
      size_t bytes;
      int tmp;
      cub::DeviceSelect::If(cub_temp_storage, bytes, f_idx, f_idx, &tmp, n_rows, dummy_select_op);
      if (bytes>cub_temp_storage_bytes) cub_temp_storage_bytes = bytes;
      CUDA_CHECK(cudaMalloc(&cub_temp_storage, cub_temp_storage_bytes));
      Initialize();
    }
  }    
  
  void FreeBuffers() {
    if (f_idx) CUDA_CHECK(cudaFree(f_idx));
    if (f_idx_sorted) CUDA_CHECK(cudaFree(f_idx_sorted));
    if (f_idx_tmp) CUDA_CHECK(cudaFree(f_idx_tmp));
    if (cub_temp_storage) (cudaFree(cub_temp_storage));
    if (idx) CUDA_CHECK(cudaFree(idx));
    if (f_sorted) CUDA_CHECK(cudaFree(f_sorted));
    if (available) CUDA_CHECK(cudaFree(available));
    if (d_num_selected) CUDA_CHECK(cudaFree(d_num_selected));
    f_idx = nullptr;
    f_idx_sorted = nullptr;
    cub_temp_storage = nullptr;
    idx = nullptr;
    f_sorted = nullptr;
    available = nullptr;
    d_num_selected = nullptr;
  }
 
  void Initialize() {
    int TPB = 256;
    range<<<ceildiv(n_rows, TPB), TPB>>>(f_idx, n_rows);
    CUDA_CHECK(cudaPeekAtLastError());
    range<<<ceildiv(n_ws, TPB), TPB>>>(idx, n_rows);
    CUDA_CHECK(cudaPeekAtLastError());
  }  
};

}; // end namespace SVM 
}; // end namespace ML
