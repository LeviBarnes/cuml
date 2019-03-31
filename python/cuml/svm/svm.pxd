# Copyright (c) 2019, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

#cdef extern from "svm/svc.h" namespace "ML::SVM":
cdef extern from "svm/svc_c.h" namespace "ML::SVM":

  # cdef cppclass CppSVC "ML::SVM::SVC" [math_t,label_t]:
  #      int n_coefs
  #      math_t *dual_coefs
  #      int *support_idx
  #      math_t b
  #      math_t C
  #      math_t tol
  #      CppSVC(math_t C, math_t tol) except+
  #      void fit(math_t *input, int n_rows, int n_cols, label_t *labels) except+

   cdef cppclass CppSVC "ML::SVM::SVC_py":
       int n_coefs
       float b
       CppSVC_py(float C, float tol) except+
       void fit(float *input, int n_rows, int n_cols, float *labels) except+

   # cdef cppclass CppSVC "ML::SVM::SVC":
   #     int n_coefs
   #     float *dual_coefs
   #     int *support_idx
   #     float b
   #     float C
   #     float tol
   #     CppSVC(float C, float tol) except+
   #    void fit(float *input, int n_rows, int n_cols, float *labels) except+
