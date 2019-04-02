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


#ifdef __cplusplus
extern "C" {
#endif


enum cumlError_t { CUML_SUCCESS, CUML_ERROR_UNKOWN };

typedef struct
{
    void* ptr;
} cumlSvcHandle_t;

cumlError_t cumlSvcCreate( cumlSvcHandle_t* handle, float C, float tol);
cumlError_t cumlSvcDestroy( cumlSvcHandle_t handle );


cumlError_t svcFit(cumlSvcHandle_t handle,
            float *input,
	        int n_rows,
	        int n_cols,
	        float *labels);

cumlError_t cumlSvcGetRes( cumlSvcHandle_t handle, float *b, int *n_support);

#ifdef __cplusplus
}
#endif
