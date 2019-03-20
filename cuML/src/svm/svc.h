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
#include <linalg/ternary_op.h>

#include "smosolver.h"

#include <linalg/gemv.h>
#include <stats/mean.h>
#include <stats/mean_center.h>
#include <linalg/add.h>
#include <linalg/subtract.h>
#include <linalg/norm.h>
#include <linalg/eltwise.h>
#include <linalg/unary_op.h>
#include <linalg/cublas_wrappers.h>

#include <linalg/map_then_reduce.h>

#include <iostream>

namespace ML {
namespace SVM {

using namespace MLCommon;



    
template<typename math_t>

void svcFit(math_t *input,
		    int n_rows,
		    int n_cols,
		    math_t *labels, // = y
		    math_t **coef,
            int *n_coefs,
            int **support_idx,
            math_t **x_support,
            math_t *b,
		    math_t C,
		    math_t tol,
		    cublasHandle_t cublas_handle) {

	ASSERT(n_cols > 0,
			"Parameter n_cols: number of columns cannot be less than one");
	ASSERT(n_rows > 0,
			"Parameter n_rows: number of rows cannot be less than one");

     // calculate the size of the working set
    int n_ws = min(1024, n_rows); // TODO: also check if we fit in memory (we will need n_ws*n_rows space for kernel cache)
    
    SmoSolver<math_t> smo(C, tol);
    
    smo.Solve(input, n_rows, n_cols, labels, coef, n_coefs, x_support, support_idx, b, cublas_handle);   
    
}


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
