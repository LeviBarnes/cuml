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
#include "kernelcache.h"
#include "linalg/unary_op.h"

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
  if (unique_labels) CUDA_CHECK(cudaFree(unique_labels));
  CUBLAS_CHECK(cublasDestroy(cublas_handle));
}

template<typename math_t, typename label_t>
void SVC<math_t, label_t>::fit(math_t *input, int n_rows, int n_cols, label_t *labels) {
  ASSERT(n_cols > 0,
		"Parameter n_cols: number of columns cannot be less than one");
  ASSERT(n_rows > 0,
		"Parameter n_rows: number of rows cannot be less than one");

  get_unique_classes(labels, n_rows,  &unique_labels, &n_classes);

  ASSERT(n_classes == 2,
         "Only binary classification is implemented at the moment");
  this->n_cols = n_cols;
  math_t *y;
  allocate(y, n_rows);
  get_ovr_labels(labels, n_rows, unique_labels, n_classes, y, 1);
  SmoSolver<math_t> smo(C, tol);
  smo.Solve(input, n_rows, n_cols, y, &dual_coefs, &n_support, &x_support, &support_idx, &b, cublas_handle);
  CUDA_CHECK(cudaFree(y));
}

template<typename math_t, typename label_t>
void SVC<math_t, label_t>::predict(math_t *input, int n_rows, int n_cols, label_t *preds) {
	ASSERT(n_cols == this->n_cols,
           "Parameter n_cols: shall be the same that was used for fitting");
#define N_PRED_BATCH 4096
  int n_batch = N_PRED_BATCH < n_rows ? N_PRED_BATCH : n_rows;
  math_t *K;
  math_t *y;
  allocate(K, n_batch*n_support);
  allocate(y, n_rows);
  KernelCache<math_t> kernel(x_support, n_support, n_cols, 1, cublas_handle);
  
  // We process the input data batchwise:
  //  - calculate the kernel values K[x_batch, x_support]
  //  - calculate y(x_batch) = K[x_batch, x_support] * dual_coeffs
  for (int i=0; i<n_rows; i+=n_batch) {
    if (i+n_batch >= n_rows) {
      n_batch = n_rows - i;
    }
    kernel.calcKernel(input + i, n_batch, x_support, n_support, K, n_rows);
    math_t one = 1;
    CUBLAS_CHECK(LinAlg::cublasgemv(cublas_handle, CUBLAS_OP_N, n_batch, n_support,
       &one, K, n_batch, dual_coefs, 1, &one, y + i, 1));
  }
  // Look up the label based on the value of the decision function: f(x) = sign(y(x) + b)
  label_t *labels = unique_labels;
  math_t b = this->b;
  LinAlg::unaryOp(preds, y, n_rows,
    [labels, b]__device__(math_t y) { return y+b<0 ? labels[0] : labels[1]; }
  );
  CUDA_CHECK(cudaFree(K));
  CUDA_CHECK(cudaFree(y));
}

// Instantiate templates for the shared library
template class SVC<float,float>;
template class SVC<double,double>;

}; // end namespace SVM
}; // end namespace ML
