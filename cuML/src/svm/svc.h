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
//#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace ML {
namespace SVM {

template<typename math_t, typename label_t>
class SVC {
public:
  int n_support = 0;              //< Number of non-zero dual coefficients
  int n_cols = 0;
  math_t *dual_coefs = nullptr; //< Non-zero dual coefficients (alpha)
  int *support_idx = nullptr;   //< Indices of the non-zero coefficients
  math_t *x_support = nullptr;  //< support vectors
  int n_classes;
  label_t *unique_labels = nullptr;
  math_t b;

  math_t C;
  math_t tol;
  cublasHandle_t cublas_handle;

public:

  SVC(math_t C, math_t tol);
  ~SVC();
  void fit(math_t *input, int n_rows, int n_cols, label_t *labels);
  void predict(math_t *input, int n_rows, int n_cols, label_t *preds);
};

/** @} */
}
;
}
;
// end namespace ML
