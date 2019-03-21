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

#include "svm/svc.h"
//#include "svm/svm_c.h"
#include "svm/classlabels.h"
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
  
  std::cout<<"Running LargeC test\n";
  
  SVC<float, float> svc(1.0f, epsilon);
  svc.fit(x_dev, n_rows, n_cols, y_dev);
  //, &dual_coefs, &n_coefs, &support_idx, &x_support, &b
  ASSERT_LE(svc.n_coefs, 4);
  
  float dual_coefs_host[4]; 
  updateHost(dual_coefs_host, svc.dual_coefs, svc.n_coefs);  
  
  float dual_coefs_exp[] = { -2, 4, -2, 0, 0 };
  float ay = 0;
  for (int i=0; i<svc.n_coefs; i++) {
    ay += dual_coefs_host[i];
  }
  // \sum \alpha_i y_i = 0
  EXPECT_LT(abs(ay), 1.0e-6f);

//   
   float x_support_host[8];
   updateHost(x_support_host, svc.x_support, svc.n_coefs * n_cols);
   float x_support_exp[] = { 1, 1, 2,  1, 2, 2, 0,0};
   for (int i=0; i<svc.n_coefs*n_cols; i++) {
   //  EXPECT_FLOAT_EQ(x_support_host[i], x_support_exp[i]) << "dual coeff idx " << i;
   }
   
   
   float w[] = {0,0};
   
   for (int i=0; i<svc.n_coefs; i++) {
       w[0] += x_support_host[i] * dual_coefs_host[i]; 
       w[1] += x_support_host[i + svc.n_coefs] * dual_coefs_host[i];      
   }
   // for linear separable problems (large C) it should be unique
   // we should norm it and check the direction
   EXPECT_LT(abs(w[0] - (-0.4)), epsilon);
   EXPECT_LT(abs(w[1] - 1.2), epsilon);
  
   EXPECT_FLOAT_EQ(svc.b, -1.8f);
 
  CUDA_CHECK(cudaFree(x_dev));
  CUDA_CHECK(cudaFree(y_dev));
}

TEST(SvcSolverTest, RelabelTest) {
  int n_rows = 6;
  float *y_d;
  allocate(y_d, n_rows);

  float y_h[] = {2, -1, 1, 2, 1, 1};
  updateDevice(y_d, y_h, n_rows);

  int n_classes;
  float *y_unique_d;
  get_unique_classes(y_d, n_rows, &y_unique_d, &n_classes);
 
  ASSERT_EQ(n_classes, 3);
  
  float y_unique_h[4]; 
  updateHost(y_unique_h, y_unique_d, 3);  
  
  float y_unique_exp[] = { -1, 1, 2 };
  
  for (int i=0; i<n_classes; i++) {
    EXPECT_EQ(y_unique_h[i], y_unique_exp[i]) << i;
  }
  
  float *y_relabeled_d;
  allocate(y_relabeled_d, n_rows);
  
  get_ovr_labels(y_d, n_rows, y_unique_d, n_classes, y_relabeled_d, 2);
  float y_relabeled_h[6];
  updateHost(y_relabeled_h, y_relabeled_d, 6);
  float y_relabeled_exp[] = {1, -1, -1, 1, -1, -1};
  for (int i=0; i<n_rows; i++) {
    EXPECT_EQ(y_relabeled_h[i], y_relabeled_exp[i]) <<i;
  }
  
  CUDA_CHECK(cudaFree(y_d));  
  CUDA_CHECK(cudaFree(y_unique_d));  
  CUDA_CHECK(cudaFree(y_relabeled_d));  
}

__global__ void init_training_vectors(float * x, int n_rows, int n_cols, float *y) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n_rows * n_cols) {
      int i = tid % n_rows;
      int k = tid / n_rows;
      x [tid] = tid;
      if (k==0) {
          y[i] = (i%2)*2 - 1;
      }
    }
    
}

TEST(SvcSolverTest, SvcTestLarge) {
  int n_rows = 1000;
  int n_cols = 780;
  int n_ws = n_rows;
    
  float *x_dev;
  allocate(x_dev, n_rows*n_cols);
  float *y_dev;
  allocate(y_dev, n_rows);

  int TPB = 256;
  init_training_vectors<<<ceildiv(n_rows*n_cols,TPB), TPB>>>(x_dev, n_rows, n_cols, y_dev);
  CUDA_CHECK(cudaPeekAtLastError());
  
  float epsilon = 0.001;
  
  SVC<float, float> svc(1.0f, epsilon);
  svc.fit(x_dev, n_rows, n_cols, y_dev);
  
  ASSERT_LE(svc.n_coefs, n_rows);
  
  float *dual_coefs_host = new float[n_rows]; 
  updateHost(dual_coefs_host, svc.dual_coefs, svc.n_coefs);  
  
  float ay = 0;
  for (int i=0; i<svc.n_coefs; i++) {
    ay += dual_coefs_host[i];
  }
  // \sum \alpha_i y_i = 0
  EXPECT_LT(abs(ay), 1.0e-6f);

 
  float *x_support_host = new float[n_rows * n_cols];
  
  updateHost(x_support_host, svc.x_support, svc.n_coefs * n_cols); 
  
   float *w = new float[n_cols];
   memset(w, 0, sizeof(float)*n_cols);
   for (int i=0; i<svc.n_coefs; i++) {
       for (int k=0; k<n_cols; k++) {
         w[k] += x_support_host[i + k*svc.n_coefs] * dual_coefs_host[i];   
       }
   }
   
   // for linear separable problems (large C) it should be unique
   // we should norm it and check the direction
  for (int k=0; k<n_cols; k++) {
  //  EXPECT_LT(abs(w[k] - 5.00001139), epsilon) << k;
  }
  
  //EXPECT_FLOAT_EQ(svc.b, -1.50995291e+09f);
 
  
  CUDA_CHECK(cudaFree(x_dev));
  CUDA_CHECK(cudaFree(y_dev));
  delete[] dual_coefs_host;
  delete[] x_support_host;
  delete[] w;
}

}; // end namespace SVM
}; // end namespace ML
