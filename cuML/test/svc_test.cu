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

//#include "svm/svc.h"
#include "svm/svm_c.h"
#include <gtest/gtest.h>
#include <cuda_utils.h>
#include <test_utils.h>

namespace ML {
namespace SVM {
using namespace MLCommon;


TEST(SvcSolverTest, SvcTest) {
  int n_rows = 6;
  int n_cols = 2;
  int n_ws = n_rows;
    
  float *x_dev;
  allocate(x_dev, n_rows*n_cols);
  float *y_dev;
  allocate(y_dev, n_rows);
  
  float x_host[] = {1, 2, 1, 2, 1, 2,   1, 1, 2, 2, 3, 3};
  updateDevice(x_dev, x_host, n_rows*n_cols);
  
  float y_host[] = {-1, -1, 1, -1, 1, 1};
  updateDevice(y_dev, y_host, n_rows);

  float epsilon = 0.001;
  
  float *dual_coefs;
  int n_coefs;
  int *idx;
  float *x_support;
  float b;
  int *support_idx;
  std::cout<<"Running LargeC test\n";
  
  svcFit(x_dev, n_rows, n_cols, y_dev, &dual_coefs, &n_coefs, &support_idx, &x_support, &b, 1.0f, 1e-3f);
  
  ASSERT_LE(n_coefs, 4);
  
  float dual_coefs_host[4]; 
  updateHost(dual_coefs_host, dual_coefs, n_coefs);  
  
  float dual_coefs_exp[] = { -2, 4, -2, 0, 0 };
  float ay = 0;
  for (int i=0; i<n_coefs; i++) {
    ay += dual_coefs_host[i];
  }
  // \sum \alpha_i y_i = 0
  EXPECT_LT(abs(ay), 1.0e-6f);

//   
   float x_support_host[8];
   updateHost(x_support_host, x_support, n_coefs * n_cols);
   float x_support_exp[] = { 1, 1, 2,  1, 2, 2, 0,0};
   for (int i=0; i<n_coefs*n_cols; i++) {
   //  EXPECT_FLOAT_EQ(x_support_host[i], x_support_exp[i]) << "dual coeff idx " << i;
   }
   
   
   float w[] = {0,0};
   
   for (int i=0; i<n_coefs; i++) {
       w[0] += x_support_host[i] * dual_coefs_host[i]; 
       w[1] += x_support_host[i + n_coefs] * dual_coefs_host[i];      
   }
   // for linear separable problems (large C) it should be unique
   // we should norm it and check the direction
   EXPECT_LT(abs(w[0] - (-0.4)), epsilon);
   EXPECT_LT(abs(w[1] - 1.2), epsilon);
  
   EXPECT_FLOAT_EQ(b, -1.8f);
 

  if (n_coefs > 0) {
    CUDA_CHECK(cudaFree(dual_coefs));
    CUDA_CHECK(cudaFree(support_idx));
    CUDA_CHECK(cudaFree(x_support));
  }
  CUDA_CHECK(cudaFree(x_dev));
  CUDA_CHECK(cudaFree(y_dev));
}

}; // end namespace SVM
}; // end namespace ML
