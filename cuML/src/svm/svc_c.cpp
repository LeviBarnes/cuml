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

#include "svc_c.h"

//#include <linalg/cublas_wrappers.h>

namespace ML {
namespace SVM {

  SVC_py::SVC_py(float C, float tol) {
    cumlSvcCreate(&svc, C, tol);
  }
  SVC_py::~SVC_py() {
    cumlSvcDestroy(svc);
  }
  cumlError_t SVC_py::fit(float *input, int n_rows, int n_cols, float *labels) {
    cumlError_t e  = svcFit(svc, input, n_rows, n_cols, labels);
    if (e == CUML_SUCCESS) {
      e = cumlSvcGetRes(svc, &b, &n_coefs);
    }
    return e;
  }
}
;
}
;
// end namespace ML
