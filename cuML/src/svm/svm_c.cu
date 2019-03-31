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

#include <iostream>
#include "svm_c.h"
#include "svc.h"

extern "C" cumlError_t cumlSvcCreate( cumlSvcHandle_t* handle, float C, float tol )
{
    cumlError_t status = CUML_SUCCESS;
    try
    {
        handle->ptr = new ML::SVM::SVC<float, float>(C, tol);
    }
    catch (...)
    {
        status = CUML_ERROR_UNKOWN;
    }
    return status;
}

extern "C" cumlError_t cumlSvcDestroy( cumlSvcHandle_t handle )
{
    cumlError_t status = CUML_SUCCESS;
    try
    {
        delete reinterpret_cast<ML::SVM::SVC<float,float>*>(handle.ptr);

    }
    catch (std::exception & e)
    {
        std::cerr << "Exception in cuml: " << e.what() << std::endl;
        status = CUML_ERROR_UNKOWN;
    }
    catch (...)
    {
        status = CUML_ERROR_UNKOWN;
    }
    return status;
}

extern "C" cumlError_t svcFit(cumlSvcHandle_t handle, float *input, int n_rows, int n_cols, float *labels) {
    cumlError_t status = CUML_SUCCESS;
    try
    {
        reinterpret_cast<ML::SVM::SVC<float,float>*>(handle.ptr)->fit(input, n_rows, n_cols, labels);
    }
    catch (std::exception & e)
    {
        std::cerr << "Exception in cuml: " << e.what() << std::endl;
        status = CUML_ERROR_UNKOWN;
    }
    catch (...)
    {
        status = CUML_ERROR_UNKOWN;
    }
    return status;
}

extern "C" cumlError_t cumlSvcGetRes( cumlSvcHandle_t handle, float *b, int *n_coefs)
{
    cumlError_t status = CUML_SUCCESS;
    try
    {
      *b = (reinterpret_cast<ML::SVM::SVC<float,float>*>(handle.ptr))->b;
      *n_coefs = reinterpret_cast<ML::SVM::SVC<float,float>*>(handle.ptr)->n_coefs;
    }
    catch (...)
    {
        status = CUML_ERROR_UNKOWN;
    }
    return status;
}
