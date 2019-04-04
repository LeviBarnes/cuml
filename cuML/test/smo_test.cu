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

#include "svm/workingset.h"
#include "svm/smosolver.h"
#include "svm/nonlinear.h"
#include <gtest/gtest.h>
#include <cuda_utils.h>
#include <test_utils.h>
#include <iostream>
#include <cub/cub.cuh>   

namespace ML {
namespace SVM {
using namespace MLCommon;

TEST(SmoSolverTestF, SelectWorkingSetTest) {
  WorkingSet<float> *ws;
  
  ws = new WorkingSet<float>(10);
  EXPECT_EQ(ws->GetSize(), 10);
  delete ws;
  
  ws = new WorkingSet<float>(100000);
  EXPECT_EQ(ws->GetSize(), 1024);
  delete ws;

  ws = new WorkingSet<float>(10, 4);
  EXPECT_EQ(ws->GetSize(), 4);
  
  float f_host[10] = {1, 3, 10, 4, 2, 8, 6, 5, 9, 7};
  float *f_dev;

  float y_host[10] = {-1, -1, -1, -1, -1, 1, 1, 1, 1, 1};
  float *y_dev;
  
  float C=1.5;
  
  float alpha_host[10] = {0, 0, 0.1, 0.2, 1.5, 0, 0.2, 0.4, 1.5, 1.5 };
  float *alpha_dev;  //   l  l  l/u  l/u    u  u  l/u  l/u  l    l
    
  int expected_idx[4] = {4, 3, 8, 2};
  allocate(f_dev, 10);
  allocate(y_dev, 10);
  allocate(alpha_dev, 10);
  updateDevice(f_dev, f_host, 10);
  updateDevice(y_dev, y_host, 10); 
  updateDevice(alpha_dev, alpha_host, 10);
  
  ws->Select(f_dev, alpha_dev, y_dev, C);
  int idx[4];
  updateHost(idx, ws->idx, 4);  
  for (int i=0; i<4; i++) {
    EXPECT_EQ(idx[i], expected_idx[i]);
  }
  CUDA_CHECK(cudaFree(f_dev));
  CUDA_CHECK(cudaFree(y_dev));
  CUDA_CHECK(cudaFree(alpha_dev));
  delete ws;
}

TEST(SmoSolverTest, KernelCacheTest) {
    int n_rows = 4;
    int n_cols = 2;
    int n_ws = n_rows;
    
    float *x_dev;
    allocate(x_dev, n_rows*n_cols);
    int *ws_idx_dev;
    allocate(ws_idx_dev, n_ws);
    
    float x_host[] = { 1, 2, 3, 4, 5, 6, 7, 8};
    updateDevice(x_dev, x_host, n_rows*n_cols);
    
    int ws_idx_host[] = {0, 1, 2, 3};
    updateDevice(ws_idx_dev, ws_idx_host, n_ws);
    
    float tile_host[16];
    float tile_host_expected[] = {
      26, 32, 38, 44,
      32, 40, 48, 56,
      38, 48, 58, 68,
      44, 56, 68, 80
    };
    
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    
    KernelCache<float> *cache = new KernelCache<float>(x_dev, n_rows, n_cols, n_ws, cublas_handle);
    float *tile_dev = cache->GetTile(ws_idx_dev);
    updateHost(tile_host, tile_dev, n_ws*n_rows);
    
    for (int i=0; i<n_ws*n_ws; i++) {
      EXPECT_EQ(tile_host[i], tile_host_expected[i])<< "First tile " << i;
    }
    
    // now check with selecting a subset of the rows
    delete cache;
    n_ws = 2;
    cache = new KernelCache<float>(x_dev, n_rows, n_cols, n_ws, cublas_handle);
    ws_idx_host[1] = 3; // i.e. ws_idx_host[] = {0,3}
    updateDevice(ws_idx_dev, ws_idx_host, n_ws);
    tile_dev = cache->GetTile(ws_idx_dev);
    updateHost(tile_host, tile_dev, n_ws*n_rows);
    
    float tile_expected2[] = {
      26, 32, 38, 44,
      44, 56, 68, 80
    };
    for (int i=0; i<n_ws*n_rows; i++) {
      EXPECT_EQ(tile_host[i], tile_expected2[i]) << "third tile " << i;
    }
    delete cache; 
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUDA_CHECK(cudaFree(x_dev));
    CUDA_CHECK(cudaFree(ws_idx_dev));
}

template <typename math_t>
class LambdaContainer {
   //extended lambdas cannot be global scope nor can they be in the scope of a gtest
   //This seems to work.

   public:

   math_t (*polynomial_kernel)(math_t) = [] __device__ __host__ (math_t a) {return (1+a)*(1+a); };

};
TEST(SmoSolverTest, KernelCacheNonLinear) {
    int n_rows = 4;
    int n_cols = 2;
    int n_ws = n_rows;
    
    float *x_dev;
    allocate(x_dev, n_rows*n_cols);
    int *ws_idx_dev;
    allocate(ws_idx_dev, n_ws);
    
    float x_host[] = { 1, 2, 3, 4, 5, 6, 7, 8};
    updateDevice(x_dev, x_host, n_rows*n_cols);
    
    int ws_idx_host[] = {0, 1, 2, 3};
    updateDevice(ws_idx_dev, ws_idx_host, n_ws);
    
    float tile_host[16];
    float tile_host_expected[] = {
      26, 32, 38, 44,
      32, 40, 48, 56,
      38, 48, 58, 68,
      44, 56, 68, 80
    };
    
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    
    //Polynomial kernel with exponent=2
    //auto nonlin = new polynomialKernel<float,int>(2);
    LambdaContainer<float> lambdas;
    auto nonlin = new generalKernel<float>(lambdas.polynomial_kernel);
    for (int z=0;z<16;z++) tile_host_expected[z] = (1+tile_host_expected[z])*(1+tile_host_expected[z]);
    KernelCache<float> *cache = 
           new KernelCache<float>(x_dev, n_rows, n_cols, n_ws, cublas_handle, nonlin);
    float *tile_dev = cache->GetTile(ws_idx_dev);
    updateHost(tile_host, tile_dev, n_ws*n_rows);
    
    for (int i=0; i<n_ws*n_ws; i++) {
      EXPECT_EQ(tile_host[i], tile_host_expected[i])<< "First tile " << i;
    }
    
    // now check with selecting a subset of the rows
    delete cache;
    n_ws = 2;
    cache = new KernelCache<float>(x_dev, n_rows, n_cols, n_ws, cublas_handle);
    ws_idx_host[1] = 3; // i.e. ws_idx_host[] = {0,3}
    updateDevice(ws_idx_dev, ws_idx_host, n_ws);
    tile_dev = cache->GetTile(ws_idx_dev);
    updateHost(tile_host, tile_dev, n_ws*n_rows);
    
    float tile_expected2[] = {
      26, 32, 38, 44,
      44, 56, 68, 80
    };
    for (int i=0; i<n_ws*n_rows; i++) {
      EXPECT_EQ(tile_host[i], tile_expected2[i]) << "third tile " << i;
    }
    delete cache; 
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUDA_CHECK(cudaFree(x_dev));
    CUDA_CHECK(cudaFree(ws_idx_dev));
}

__global__ void init_training_vectors(float * x, int n_rows, int n_cols, int *ws_idx, int n_ws) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n_rows * n_cols) {
      int i = tid % n_rows;
      int k = tid / n_rows;
      x [tid] = tid;
      if (k==0) {
          ws_idx[i] = i;
      }
    }
    
}
TEST(SmoSolverTest, KernelCacheLargeTest) {
    int n_rows = 10;
    int n_cols = 700;
    int n_ws = n_rows;
    
    float *x_dev;
    allocate(x_dev, n_rows*n_cols);
    int *ws_idx_dev;
    allocate(ws_idx_dev, n_ws);
    
    int TPB=256;
    init_training_vectors<<<ceildiv(n_rows*n_cols, TPB), TPB>>>(x_dev, n_rows, n_cols, ws_idx_dev, n_ws);
    CUDA_CHECK(cudaPeekAtLastError());
    
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    
    KernelCache<float> *cache = new KernelCache<float>(x_dev, n_rows, n_cols, n_ws, cublas_handle);
    float *tile_dev = cache->GetTile(ws_idx_dev);
    float *tile_host = new float[n_rows*n_cols];
    updateHost(tile_host, tile_dev, n_ws*n_rows);
    
    /*for (int i=0; i<n_ws*n_ws; i++) {
      EXPECT_EQ(tile_host[i], tile_host_expected[i])<< "First tile " << i;
    }*/
    
    delete cache; 
    delete[] tile_host;
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUDA_CHECK(cudaFree(x_dev));
    CUDA_CHECK(cudaFree(ws_idx_dev));
}

// test a single iteration of the block solver
TEST(SmoSolverTest, SmoBlockSolveSingleTest) {
  int n_rows = 4;
  int n_cols = 2;
  int n_ws = n_rows;
    

  int *ws_idx_dev;
  allocate(ws_idx_dev, n_ws);
  float *y_dev;
  allocate(y_dev, n_rows);
  float *f_dev;
  allocate(f_dev, n_rows);
  float *alpha_dev;
  allocate(alpha_dev, n_rows, true);
  float *delta_alpha_dev;
  allocate(delta_alpha_dev, n_ws, true);
  float *kernel_dev;
  allocate(kernel_dev, n_ws*n_rows);
  float *return_buff_dev;
  allocate(return_buff_dev, 2);
    
  int ws_idx_host[] = {0, 1, 2, 3};
  updateDevice(ws_idx_dev, ws_idx_host, n_ws);
  
  float y_host[] = {1, 1, -1, -1};
  updateDevice(y_dev, y_host, n_rows);

  float f_host[] = {0.4, 0.3, 0.5, 0.1};
  updateDevice(f_dev, f_host, n_rows);

  float kernel_host[] = {
      26, 32, 38, 44,
      32, 40, 48, 56,
      38, 48, 58, 68,
      44, 56, 68, 80
  };
  
  updateDevice(kernel_dev, kernel_host, n_ws*n_rows);

  SmoBlockSolve<float, 1024><<<1, n_ws>>>(y_dev, n_rows, alpha_dev, n_ws, 
      delta_alpha_dev, f_dev, kernel_dev, ws_idx_dev,
      1.5f, 1e-3f, return_buff_dev, 1);
  
  CUDA_CHECK(cudaPeekAtLastError());
  
  float return_buff[2];
  updateHost(return_buff, return_buff_dev, 2);
  EXPECT_FLOAT_EQ(return_buff[0], 0.2f) << return_buff[0];
  EXPECT_EQ(return_buff[1], 1) << "Number of iterations ";
  
  float host_alpha[4], host_dalpha[4];
  updateHost(host_alpha, alpha_dev, n_rows);
  updateHost(host_dalpha, delta_alpha_dev, n_ws);
  
  for (int i=0; i<n_ws; i++) {
      EXPECT_FLOAT_EQ(y_host[i]*host_alpha[i], host_dalpha[i]) << "alpha and delta alpha " << i;
  }
  float alpha_expected[] = {0, 0.1f, 0.1f, 0};
  for (int i=0; i<n_rows; i++) {
      EXPECT_FLOAT_EQ(host_alpha[i], alpha_expected[i]) << "alpha " << i;
  }
  
  // now check if updateF works
  SmoSolver<float> smo;
  cublasHandle_t cublas_handle;
  CUBLAS_CHECK(cublasCreate(&cublas_handle));

  smo.UpdateF(f_dev, n_rows, delta_alpha_dev, n_ws, kernel_dev, cublas_handle);
  updateHost(f_host, f_dev, n_rows);
  float f_host_expected[] = {-0.2, -0.5, -0.5, -1.1};
  for (int i=0; i<n_rows; i++) {
      EXPECT_FLOAT_EQ(f_host[i], f_host_expected[i]) << "UpdateF " << i;
  }   
  CUBLAS_CHECK(cublasDestroy(cublas_handle));
  CUDA_CHECK(cudaFree(y_dev));
  CUDA_CHECK(cudaFree(f_dev));
  CUDA_CHECK(cudaFree(ws_idx_dev));
  CUDA_CHECK(cudaFree(alpha_dev));
  CUDA_CHECK(cudaFree(delta_alpha_dev));
  CUDA_CHECK(cudaFree(kernel_dev));
  CUDA_CHECK(cudaFree(return_buff_dev));
}


TEST(SmoSolverTest, SmoBlockSolveTest) {
  int n_rows = 6;
  int n_cols = 2;
  int n_ws = n_rows;
    
  float *x_dev;
  allocate(x_dev, n_rows*n_cols);
  int *ws_idx_dev;
  allocate(ws_idx_dev, n_ws);
  float *y_dev;
  allocate(y_dev, n_rows);
  float *f_dev;
  allocate(f_dev, n_rows);
  float *alpha_dev;
  allocate(alpha_dev, n_rows, true);
  float *delta_alpha_dev;
  allocate(delta_alpha_dev, n_ws, true);
  float *kernel_dev;
  allocate(kernel_dev, n_ws*n_rows);
  float *return_buff_dev;
  allocate(return_buff_dev, 2);
  
  float x_host[] = {1, 2, 1, 2, 1, 2,   1, 1, 2, 2, 3, 3};
  updateDevice(x_dev, x_host, n_rows*n_cols);
    
  int ws_idx_host[] = {0, 1, 2, 3, 4, 5};
  updateDevice(ws_idx_dev, ws_idx_host, n_ws);
  
  float y_host[] = {-1, -1, 1, -1, 1, 1};
  updateDevice(y_dev, y_host, n_rows);

  float f_host[] = {1, 1, -1, 1, -1, -1};
  updateDevice(f_dev, f_host, n_rows);

  
  float kernel_host[] = {
    2, 3, 3,  4,  4,  5,
    3, 5, 4,  6,  5,  7,
    3, 4, 5,  6,  7,  8,
    4, 6, 6,  8,  8, 10,
    4, 5, 7,  8, 10, 11,
    5, 7, 8, 10, 11, 13
  };
  
  updateDevice(kernel_dev, kernel_host, n_ws*n_rows);

  SmoBlockSolve<float, 1024><<<1, n_ws>>>(y_dev, n_rows, alpha_dev, n_ws, 
      delta_alpha_dev, f_dev, kernel_dev, ws_idx_dev,
      1.0f, 1e-3f, return_buff_dev);
  
  CUDA_CHECK(cudaPeekAtLastError());
  float return_buff[2];
  updateHost(return_buff, return_buff_dev, 2);
  EXPECT_FLOAT_EQ(return_buff[0], 2.0f) << return_buff[0];
  EXPECT_LT(return_buff[1], 100) << return_buff[1];
  
  float host_alpha[6], host_dalpha[6];
  updateHost(host_alpha, alpha_dev, n_rows);
  updateHost(host_dalpha, delta_alpha_dev, n_ws);
  
  for (int i=0; i<n_ws; i++) {
      EXPECT_FLOAT_EQ(y_host[i]*host_alpha[i], host_dalpha[i]) << "alpha and delta alpha " << i;
  }
  float w[] = {0,0};
  
  float alpha_expected[] = {0.6f, 0, 1, 1, 0, 0.6f};
  //for C=10: {0.25f, 0, 2.25f, 3.75f, 0, 1.75f};
  float ay=0;
  for (int i=0; i<n_rows; i++) {
   //   EXPECT_FLOAT_EQ(host_alpha[i], alpha_expected[i]) << "alpha " << i;
      w[0] += x_host[i] * host_alpha[i] * y_host[i]; 
      w[1] += x_host[i + n_rows] * host_alpha[i] * y_host[i];
      ay += host_alpha[i] * y_host[i];
  }
  EXPECT_FLOAT_EQ(ay, 0.0);
  EXPECT_FLOAT_EQ(w[0], -0.4);
  EXPECT_FLOAT_EQ(w[1],  1.2);
  // for C=10
  //EXPECT_FLOAT_EQ(w[0], -2.0);
  //EXPECT_FLOAT_EQ(w[1],  2.0);
  CUDA_CHECK(cudaFree(x_dev));
  CUDA_CHECK(cudaFree(y_dev));
  CUDA_CHECK(cudaFree(f_dev));
  CUDA_CHECK(cudaFree(ws_idx_dev));
  CUDA_CHECK(cudaFree(alpha_dev));
  CUDA_CHECK(cudaFree(delta_alpha_dev));
  CUDA_CHECK(cudaFree(kernel_dev));
  CUDA_CHECK(cudaFree(return_buff_dev));
}


TEST(SmoSolverTest, GetResultsTest) {
  int n_rows = 6;
  int n_cols = 2;
    
  float *x_dev;
  allocate(x_dev, n_rows*n_cols);

    
  float x_host[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  updateDevice(x_dev, x_host, n_rows*n_cols);

  float *y_dev;
  allocate(y_dev, n_rows);
  
  float *alpha_dev;
  allocate(alpha_dev, n_rows);
  float y_host[] = {1, 1, 1, -1, -1, -1};
  updateDevice(y_dev, y_host, n_rows);

  float alpha_host[] = {0.0, 0.5, 0.5, 0, 1.0, 0,0};
  updateDevice(alpha_dev, alpha_host, n_rows);

  SmoSolver<float> smo;
  
  float *dual_coefs;
  int n_coefs;
  int *idx;
  float *x_support;
  float b;
  cublasHandle_t cublas_handle;
  CUBLAS_CHECK(cublasCreate(&cublas_handle));


  smo.GetResults(x_dev, n_rows, n_cols, y_dev, alpha_dev, &dual_coefs, &n_coefs, &idx, 
                 &x_support, &b, cublas_handle);
  
  ASSERT_EQ(n_coefs, 3);

  
  float dual_coefs_host[3];
  updateHost(dual_coefs_host, dual_coefs, n_coefs);
  float dual_coefs_exp[] = { 0.5, 0.5, -1.0 };
  for (int i=0; i<n_coefs; i++) {
    EXPECT_FLOAT_EQ(dual_coefs_host[i], dual_coefs_exp[i]) << "dual coeff idx " << i;
  }

  int idx_host[3];
  updateHost(idx_host, idx, n_coefs);
  float idx_exp[] = { 1, 2, 4 };
  for (int i=0; i<n_coefs; i++) {
    EXPECT_EQ(idx_host[i], idx_exp[i]) << "idx " << i;
  }
 
  float x_support_host[6];
  updateHost(x_support_host, x_support, n_coefs * n_cols);
  float x_support_exp[] = { 2, 3, 5,  8, 9, 11 };
  for (int i=0; i<n_coefs*n_cols; i++) {
    EXPECT_FLOAT_EQ(x_support_host[i], x_support_exp[i]) << "dual coeff idx " << i;
  }

  if (n_coefs > 0) {
    CUDA_CHECK(cudaFree(dual_coefs));
    CUDA_CHECK(cudaFree(idx));
    CUDA_CHECK(cudaFree(x_support));
  }
  
  CUBLAS_CHECK(cublasDestroy(cublas_handle));  
  CUDA_CHECK(cudaFree(x_dev));
  CUDA_CHECK(cudaFree(y_dev));
  CUDA_CHECK(cudaFree(alpha_dev));
}


TEST(SmoSolverTest, SmoUpdateFTest) {
  int n_rows = 6;
  int n_cols = 2;
  int n_ws = 2;
    
  float *kernel_dev;
  allocate(kernel_dev, n_rows*n_ws);
  
  float *f_dev;
  allocate(f_dev, n_rows, true);
  
  float *delta_alpha_dev;
  allocate(delta_alpha_dev, n_ws);
  
  float kernel_host[] = {
    3, 5, 4,  6,  5,  7,
    4, 5, 7,  8, 10, 11
  };
  updateDevice(kernel_dev, kernel_host, n_ws*n_rows);
  
  float delta_alpha_host[] = {-0.1f, 0.1f};
  updateDevice(delta_alpha_dev, delta_alpha_host, n_ws);

  SmoSolver<float> smo(1, 0.001);

  cublasHandle_t cublas_handle;
  CUBLAS_CHECK(cublasCreate(&cublas_handle));

  smo.UpdateF(f_dev, n_rows, delta_alpha_dev, n_ws, kernel_dev, cublas_handle);
  
  float f_host[6];
  updateHost(f_host, f_dev, n_rows);
  
  float f_host_expected[] = {0.1f, 7.4505806e-9f, 0.3f, 0.2f, 0.5f, 0.4f};
  for (int i=0; i<n_rows; i++) {
      EXPECT_FLOAT_EQ(f_host[i], f_host_expected[i]) << "UpdateF " << i;
  }   

  CUDA_CHECK(cudaFree(delta_alpha_dev));
  CUDA_CHECK(cudaFree(kernel_dev));
  CUDA_CHECK(cudaFree(f_dev));
}

TEST(SmoSolverTest, SmoSolveTest) {
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

  SmoSolver<float> smo(1, 0.001);
  
  float *dual_coefs;
  int n_coefs;
  int *idx;
  float *x_support;
  float b;
  cublasHandle_t cublas_handle;
  CUBLAS_CHECK(cublasCreate(&cublas_handle));
  
  smo.Solve(x_dev, n_rows, n_cols, y_dev, &dual_coefs, &n_coefs, &x_support, &idx, &b, 
            cublas_handle, 100,1);
  
  ASSERT_EQ(n_coefs, 4);
  
  float dual_coefs_host[4];
  updateHost(dual_coefs_host, dual_coefs, n_coefs);  
  
  float dual_coefs_exp[] = { -0.6, 1, -1, 0.6 };
  float ay = 0;
  for (int i=0; i<n_coefs; i++) {
    EXPECT_FLOAT_EQ(dual_coefs_host[i], dual_coefs_exp[i]) << "dual coeff idx " << i;
    ay += dual_coefs_host[i];
  }
  
  // \sum \alpha_i y_i = 0
  EXPECT_LT(abs(ay), 1.0e-6f);
  
  int idx_host[4];
  updateHost(idx_host, idx, n_coefs);
  float idx_exp[] = { 0, 2, 3, 5 };
  for (int i=0; i<n_coefs; i++) {
    EXPECT_EQ(idx_host[i], idx_exp[i]) << "idx " << i;
  }
 
  float x_support_host[8];
  updateHost(x_support_host, x_support, n_coefs * n_cols);
  float x_support_exp[] = { 1, 1, 2, 2,  1, 2, 2, 3};
  for (int i=0; i<n_coefs*n_cols; i++) {
    EXPECT_FLOAT_EQ(x_support_host[i], x_support_exp[i]) << "dual coeff idx " << i;
  }
  
  float w[] = {0,0};
  
  for (int i=0; i<n_coefs; i++) {
      w[0] += x_support_host[i] * dual_coefs_host[i]; 
      w[1] += x_support_host[i + n_coefs] * dual_coefs_host[i];      
  }
  EXPECT_FLOAT_EQ(w[0], -0.4);
  EXPECT_FLOAT_EQ(w[1],  1.2);
  
  EXPECT_FLOAT_EQ(b, -1.8);
  
  CUBLAS_CHECK(cublasDestroy(cublas_handle));
  if (n_coefs > 0) {
    CUDA_CHECK(cudaFree(dual_coefs));
    CUDA_CHECK(cudaFree(idx));
    CUDA_CHECK(cudaFree(x_support));
  }
  CUDA_CHECK(cudaFree(x_dev));
  CUDA_CHECK(cudaFree(y_dev));
}

TEST(SmoSolverTest, SmoSolveTestLargeC) {
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
  SmoSolver<float> smo(100, epsilon);
  
  float *dual_coefs;
  int n_coefs;
  int *idx;
  float *x_support;
  float b;
  cublasHandle_t cublas_handle;
  CUBLAS_CHECK(cublasCreate(&cublas_handle));
  smo.Solve(x_dev, n_rows, n_cols, y_dev, &dual_coefs, &n_coefs, &x_support, &idx, &b, 
            cublas_handle, 100, 1);
  
  ASSERT_LE(n_coefs, 4);
  
  float dual_coefs_host[4];
  updateHost(dual_coefs_host, dual_coefs, n_coefs);  
  
  float dual_coefs_exp[] = { -2, 4, -2, 0, 0 };
  float ay = 0;
  for (int i=0; i<n_coefs; i++) {
   // EXPECT_FLOAT_EQ(dual_coefs_host[i], dual_coefs_exp[i]) << "dual coeff idx " << i;
    ay += dual_coefs_host[i];
  }
  // \sum \alpha_i y_i = 0
  EXPECT_LT(abs(ay), 1.0e-6f);
  
  int idx_host[4];
  updateHost(idx_host, idx, n_coefs);
  float idx_exp[] = { 0, 2, 3 };
  for (int i=0; i<n_coefs; i++) {
   // EXPECT_EQ(idx_host[i], idx_exp[i]) << "idx " << i;
  }
 
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
  // for linear problems it should be unique 
  EXPECT_LT(abs(w[0] - (-2)), epsilon);
  EXPECT_LT(abs(w[1] - 2), epsilon);
 
  EXPECT_FLOAT_EQ(b, -1.0f);
  
  CUBLAS_CHECK(cublasDestroy(cublas_handle));
  if (n_coefs > 0) {
    CUDA_CHECK(cudaFree(dual_coefs));
    CUDA_CHECK(cudaFree(idx));
    CUDA_CHECK(cudaFree(x_support));
  }
  CUDA_CHECK(cudaFree(x_dev));
  CUDA_CHECK(cudaFree(y_dev));
}

/*TEST_F(SmoSolverTestF, SelectWorkingSetTest) {
  ASSERT_LT(1, 2);
}*/

}; // end namespace SVM
}; // end namespace ML
