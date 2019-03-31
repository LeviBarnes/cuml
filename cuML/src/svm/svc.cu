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

#include "svc.h"
#include "smosolver.h"
#include <iostream>
#include "classlabels.h"
#include "linalg/cublas_wrappers.h"

namespace ML {
namespace SVM {

using namespace MLCommon;


template<typename math_t, typename label_t>
SVC<math_t, label_t>::SVC(math_t C, math_t tol)
    :C(C), tol(tol)
{
	CUBLAS_CHECK(cublasCreate(&cublas_handle));
}

template<typename math_t, typename label_t>
SVC<math_t, label_t>::~SVC()
{
  if (dual_coefs) CUDA_CHECK(cudaFree(dual_coefs));
  if (support_idx) CUDA_CHECK(cudaFree(support_idx));
  if (x_support) CUDA_CHECK(cudaFree(x_support));
  CUBLAS_CHECK(cublasDestroy(cublas_handle));
}

template<typename math_t, typename label_t>
void SVC<math_t, label_t>::fit(math_t *input, int n_rows, int n_cols, label_t *labels) {
	ASSERT(n_cols > 0,
			"Parameter n_cols: number of columns cannot be less than one");
	ASSERT(n_rows > 0,
			"Parameter n_rows: number of rows cannot be less than one");

    label_t *unique_labels = nullptr;
    int n_classes;
    get_unique_classes(labels, n_rows,  &unique_labels, &n_classes);
    ASSERT(n_classes == 2,
           "We have only binary classification implemented at the moment");

    math_t *y;
    allocate(y, n_rows);
    get_ovr_labels(labels, n_rows, unique_labels, n_classes, y, 1);
    SmoSolver<math_t> smo(C, tol);
    smo.Solve(input, n_rows, n_cols, y, &dual_coefs, &n_coefs, &x_support, &support_idx, &b, cublas_handle);
    CUDA_CHECK(cudaFree(y));
    CUDA_CHECK(cudaFree(unique_labels));
  }

template class SVC<float,float>;
template class SVC<double,double>;
}
;
}
;
// end namespace ML
