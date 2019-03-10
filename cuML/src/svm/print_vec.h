#pragma once

#include <iostream>
#include <string>
#include "cuda_utils.h"

template<typename math_t>
  void print_vec(math_t* x, int n, std::string msg) {
    std::cout << msg << " ";
    math_t * host_x = (math_t*) malloc(n * sizeof(math_t));
    MLCommon::updateHost(host_x, x, n);
    for(int i=0;i<n; i++) {
        std::cout<<host_x[i]<<" ";
    }
    std::cout<<"\n";  
    free(host_x);
  }
template<typename math_t>  
  void print_host_vec(math_t* x, int n, std::string msg) {
    std::cout << msg << " ";
    for(int i=0;i<n; i++) {
        std::cout<<x[i]<<" ";
    }
    std::cout<<"\n";  
  }
