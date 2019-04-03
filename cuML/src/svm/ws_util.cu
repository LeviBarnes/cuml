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

#include "ml_utils.h"
#include <cuda_utils.h>
#include <cub/cub.cuh>
#include <limits.h>
#include "smo_sets.h"


__global__ void range(int *f_idx, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) { f_idx[tid] = tid; }
}

__global__ void set_available(bool *available, int n_rows, int *idx, int n_selected){ 
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n_rows) available[tid] = true;
    if (tid < n_selected) {
        available[idx[tid]] = false;
    }
}
