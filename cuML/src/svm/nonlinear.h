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

#include <cuda_utils.h>
#include <linalg/gemm.h>

namespace ML {
namespace SVM {

template <typename math_t, typename exp_t>
__global__ void polynomial_kernel_nopad(math_t *inout, int len, exp_t exponent)
{
   for (int tid=threadIdx.x + blockIdx.x * blockDim.x;
        tid < len;
        tid += blockDim.x * gridDim.x)
   {
      //TODO Is an explicit integer exponentiation faster?
      inout[tid] = pow(inout[tid] + 1, exponent);
   }

} 
template <typename math_t, typename exp_t>
__global__ void polynomial_kernel(math_t *inout, int ld, int rows, int cols, exp_t exponent)
{
   for (int tidy=threadIdx.y + blockIdx.y * blockDim.y;
        tidy < rows;
        tidy += blockDim.y * gridDim.y)
      for (int tidx=threadIdx.x + blockIdx.x * blockDim.x;
           tidx < cols;
           tidx += blockDim.x * gridDim.x)
      {
         //TODO Is an explicit integer exponentiation faster?
         inout[tidx + tidy*ld] = pow(inout[tidx + tidy*ld] + 1, exponent);
      }

}
template <typename math_t>
__global__ void general_kernel_nopad(math_t *inout, int len, math_t (*kernelOp)(math_t))
{
   for (int tid=threadIdx.x + blockIdx.x * blockDim.x;
        tid < len;
        tid += blockDim.x * gridDim.x)
   {
      inout[tid] = kernelOp(inout[tid]);
   }

}
template <typename math_t>
__global__ void general_kernel(math_t *inout, int ld, int rows, int cols, math_t (*kernelOp)(math_t))
{
   for (int tidy=threadIdx.y + blockIdx.y * blockDim.y;
        tidy < rows;
        tidy += blockDim.y * gridDim.y)
      for (int tidx=threadIdx.x + blockIdx.x * blockDim.x;
           tidx < cols;
           tidx += blockDim.x * gridDim.x)
      {
         //TODO Is an explicit integer exponentiation faster?
         inout[tidx + tidy*ld] = kernelOp(inout[tidx + tidy*ld]);
      }

}
template <typename math_t>
class SVMKernelBase {

   public:
   SVMKernelBase() {};
   virtual void operator()(const math_t *x1, int ld1, int n_ws, int n_cols, 
              const math_t *x2, int ld2, int n_rows, math_t *tile, int ld_tile,
              cublasHandle_t cublas_handle) 
   {
      math_t alpha = 1.0;
      math_t beta = 0.0;
      CUBLAS_CHECK(LinAlg::cublasgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, n_ws, n_rows, n_cols,
                   &alpha, x1, ld1, x2, ld2, &beta, tile, ld_tile));
   }
   virtual void evaluate(const math_t *x1, int ld1, int n_ws, int n_cols, 
              const math_t *x2, int ld2, int n_rows, math_t *tile, int ld_tile,
              cublasHandle_t cublas_handle) 
   {
      evaluate(x1, ld1, n_ws, n_cols, x2, ld2, n_rows, tile, ld_tile, cublas_handle);
   }
};
template <typename math_t, typename exp_t>
class polynomialKernel : public SVMKernelBase<math_t> {

  exp_t exponent;

  void applyKernel(math_t* inout, int ld, int rows, int cols)
  {
     if (ld == cols)
        polynomial_kernel_nopad<<<ceildiv(rows*cols, 128), 128>>>(inout, rows*cols, exponent);
     else 
        polynomial_kernel<<< dim3(ceildiv(cols,32),ceildiv(rows,4),1),
                             dim3(32,4,1)                              >>>
                                         (inout, ld, rows, cols, exponent);
  }
   public:
   polynomialKernel (exp_t exponent) : exponent(exponent) {
   }
  /**
   * @brief Compute the kernel value 
   * @param [in] x1 [n_ws x n_cols] workspace vectors
   * @param [in] ld1 
   * @param [in] x2 [n_rows x n_cols] feature vectors
   * @param [out] tile buffer for return values [n_ws x n_rows] (should be already allocated)
   */
   void evaluate(const math_t *x1, int ld1, int n_ws, int n_cols, 
              const math_t *x2, int ld2, int n_rows, math_t *tile, int ld_tile,
              cublasHandle_t cublas_handle)
   {
      math_t alpha = 1.0;
      math_t beta = 0.0;
      CUBLAS_CHECK(LinAlg::cublasgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, n_ws, n_rows, n_cols,
                   &alpha, x1, ld1, x2, ld2, &beta, tile, ld_tile));
      applyKernel(tile, ld_tile, n_ws, n_rows);
   }
   void operator()(const math_t *x1, int ld1, int n_ws, int n_cols, 
              const math_t *x2, int ld2, int n_rows, math_t *tile, int ld_tile,
              cublasHandle_t cublas_handle)
   {
      evaluate(x1,ld1,n_ws,n_cols,x2,ld2,n_rows,tile,ld_tile,cublas_handle);
   }


};
template <typename math_t>
class generalKernel : public SVMKernelBase<math_t> {

  math_t (*kernelOp)(math_t) = 0;
  public:
  void applyKernel(math_t* inout, int ld, int rows, int cols)
  {
     if (true || ld == cols)
        general_kernel_nopad<<<ceildiv(rows*cols, 128), 128>>>(inout, rows*cols, kernelOp);
     else 
        general_kernel<<< dim3(ceildiv(cols,32),ceildiv(rows,4),1),
                               dim3(32,4,1)                              >>>
                                         (inout, ld, rows, cols, kernelOp);
  }
   generalKernel (math_t (*kernelOp)(math_t)) : kernelOp(kernelOp) { }
  /**
   * @brief Compute the kernel value 
   * @param [in] x1 [n_ws x n_cols] workspace vectors
   * @param [in] ld1 
   * @param [in] x2 [n_rows x n_cols] feature vectors
   * @param [out] tile buffer for return values [n_ws x n_rows] (should be already allocated)
   */
   void evaluate(const math_t *x1, int ld1, int n_ws, int n_cols, 
              const math_t *x2, int ld2, int n_rows, math_t *tile, int ld_tile,
              cublasHandle_t cublas_handle)
   {
      math_t alpha = 1.0;
      math_t beta = 0.0;
      CUBLAS_CHECK(LinAlg::cublasgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, n_ws, n_rows, n_cols,
                   &alpha, x1, ld1, x2, ld2, &beta, tile, ld_tile));
      applyKernel(tile, ld_tile, n_ws, n_rows);
   }
   void operator()(const math_t *x1, int ld1, int n_ws, int n_cols, 
              const math_t *x2, int ld2, int n_rows, math_t *tile, int ld_tile,
              cublasHandle_t cublas_handle)
   {
      evaluate(x1,ld1,n_ws,n_cols,x2,ld2,n_rows,tile,ld_tile,cublas_handle);
   }
};


}; // end namespace SVM
}; // end namespace ML
