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
#include <linalg/unary_op.h>

namespace ML {
namespace SVM {

using namespace MLCommon;

/**
 * Get unique class labels.
 * \param [in] y device array of labels, size [n]
 * \param [in] n number of labels
 * \param [out] y_unique device array of unique labels, size [n_unique]
 * \param [out] n_unique number of unique labels
 */
template<typename label_t> 
void get_unique_classes(label_t *y, int n, label_t **y_unique, int *n_unique) {
 
  void     *d_temp_storage = NULL;
  size_t   bytes = 0;
  label_t  *y2, *y3;
  allocate(y2, n);
  allocate(y3, n);
  cub::DeviceRadixSort::SortKeys(d_temp_storage, bytes, y, y2, n);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, bytes);
  // Run sorting operation
  cub::DeviceRadixSort::SortKeys(d_temp_storage, bytes, y, y2, n);
  
  int  *d_num_selected_out;
  allocate(d_num_selected_out, 1);
  
  size_t bytes2 = 0;
  cub::DeviceSelect::Unique(NULL, bytes2, y2, y3, d_num_selected_out, n);
  if (bytes2 > bytes) {
    CUDA_CHECK(cudaFree(d_temp_storage));
    cudaMalloc(&d_temp_storage, bytes2);
  }
  
  cub::DeviceSelect::Unique(d_temp_storage, bytes2, y2, y3, d_num_selected_out, n);
  updateHost(n_unique, d_num_selected_out, 1);
  allocate(*y_unique, *n_unique);
  copy(*y_unique, y3, *n_unique);

  CUDA_CHECK(cudaFree(y3));
  CUDA_CHECK(cudaFree(y2));
  CUDA_CHECK(cudaFree(d_num_selected_out));
  CUDA_CHECK(cudaFree(d_temp_storage));
}

/** Relabel y to +/-1.
 *  y_out will be +1 if y the same as y_unique[idx], otherwise -1
 * \param [in] y device array of labels, size [n]
 * \param [in] n number of labels
 * \param [in] y_unique device array of unique labels, size [n_unique]
 * \param [out] y_out device array of output labels, size [n]
 * \param [in] idx index of unique label that should be labeled as 1
 */
template<typename label_t, typename math_t>
__global__ void relabel(label_t *y, int n, label_t *y_unique, math_t *y_out, int idx) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    label_t selected = y_unique[idx];
    if (tid < n) {
      y_out[tid] = y[tid] == selected ? +1 : -1;
    }
}

/**
 * Assign one versus rest labels.
 * 
 * The output labels will have values +/-1:
 * y_out = (y == y_unique[idx]) ? +1 : -1;
 * 
 * The output type currently is set to math_t, but we are actually free to choose other type
 * for y_out (it should represent +/-1, and it is used in floating point arithmetics).
 * 
 * \param [in] y device array if input labels, size [n]
 * \param [in] n number of labels
 * \param [in] y_unique device array of unique labels, size [n_classes]
 * \param [in] n_classes number of unique labels
 * \param [out] y_out device array of output labels
 * \param [in] idx index of unique label that should be labeled as 1

 */
template<typename label_t, typename math_t>
void get_ovr_labels(label_t *y, int n, label_t *y_unique, int n_classes, math_t* y_out, int idx) {
  int TPB=256;
  ASSERT(idx < n_classes, "Parameter idx should not be larger than the number of classes");
  // unary op could be used if that would allow different input and output types
  relabel<<<ceildiv(n,TPB),TPB>>>(y, n, y_unique, y_out, idx);
  CUDA_CHECK(cudaPeekAtLastError());                           
}

}; // end namespace SVM
}; // end namespace ML
