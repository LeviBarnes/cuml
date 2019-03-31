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
  int n_coefs = 0;              //< Number of non-zero dual coefficients
  math_t *dual_coefs = nullptr; //< Non-zero dual coefficients (alpha)
  int *support_idx = nullptr;   //< Indices of the non-zero coefficients
  math_t *x_support = nullptr;  //< support vectors
  math_t b;

  math_t C;
  math_t tol;
  cublasHandle_t cublas_handle;

public:

  SVC(math_t C, math_t tol);
  ~SVC();
  void fit(math_t *input, int n_rows, int n_cols, label_t *labels);
};


/*
template<typename math_t>
void svcPredict(const math_t *input, int n_rows, int n_cols, const math_t *coef,
		math_t intercept, math_t *preds, ML::loss_funct loss, cublasHandle_t cublas_handle) {

	ASSERT(n_cols > 1,
			"Parameter n_cols: number of columns cannot be less than two");
	ASSERT(n_rows > 1,
			"Parameter n_rows: number of rows cannot be less than two");

    std::cout<<"Hello SVM prediction World!\n";
}
*/
/** @} */
}
;
}
;
// end namespace ML
