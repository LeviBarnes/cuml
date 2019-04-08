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

#include <cublas_v2.h>

namespace ML {
namespace SVM {

/**
 * C-Support Vector Classification
 * 
 * The classifier will be fitted using the SMO algorithm in dual space.
 * 
 * The decision function takes the following form 
 * \f[
 *    f(x) = sign\left( \sum_i=1^{N_{support} y_i \alpha_i K(x_i,x) + b \right),
 * \f]
 * where \f$x_i\f$ are the support vectors, and \f$ y_i \alpha_i \f$ are the dual
 * coordinates.
 * 
 * The penalty parameter C limits the values of the dual coefficients
 * \f[ 0 <= \alpha <= C \f]
 * 
 */
template<typename math_t, typename label_t>
class SVC {
public:
  // Public members for easier access during testing (and for Python).
  // TODO Think over how to hide most of this.
    
  math_t C;             //!< Penalty term C
  math_t tol;           //!< Tolerance used to stop fitting.
  
  int n_support = 0;    //!< Number of non-zero dual coefficients
  math_t b;             //!< Constant used in the decision function
  
  // Three device pointers to store the parameters for the classifier
  math_t *dual_coefs = nullptr; //!< Non-zero dual coefficients ( dual_coef[i] = \f$ y_i \alpha_i \f$). Size [n_support].
  math_t *x_support = nullptr;  //!< Support vectors in column major format. Size [n_support x n_cols].
  int *support_idx = nullptr;   //!< Indices (from the traning set) of the non-zero support vectors. Size [n_support].
  
  /**
   * Constructs a support vector classifier
   * @param C penalty term
   * @param tol tolerance to step fitting.
   */
  SVC(math_t C = 1, math_t tol = 1.0e-3);
  ~SVC();
  
  /**
   * Fit a support vector classifier to the training data.
   * 
   * Each row of the input data stores a feature vector.
   * We use the SMO method to fit the SVM.
   * 
   * @param input device pointer for the input data in column major format. Size n_rows x n_cols.
   * @param n_rows number of rows
   * @param n_cols number of colums
   * @param labels device pointer for the labels. Size n_rows.
   */
  void fit(math_t *input, int n_rows, int n_cols, label_t *labels);
  
  /**
   * Predict classes for samples in input.
   * @param input device pointer for the input data in column major format. Size [n_rows x n_cols].
   * @param preds device pointer to store the predicted class labels. Size [n_rows]. Should be allocated on entry.
   */
  void predict(math_t *input, int n_rows, int n_cols, label_t *preds);
  
private:
    
  int n_cols = 0;               //!< Number of columns that was used for fitting the classifier
  int n_classes;                //!< Number of classes found in the input labels
  label_t *unique_labels = nullptr; //!< Device pointer for the unique classes. Size [n_classes]
  cublasHandle_t cublas_handle;
};

}; // end namespace SVM
}; // end namespace ML