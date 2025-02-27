#pragma once

#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/types.h"
#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/c10/core/TensorOptions.h"

#include "glog/logging.h"

namespace torch {

class IntArrayRef {
public:
  IntArrayRef() {LOG(FATAL) << "IntArrayRef() is not allowed!";}
  IntArrayRef(const std::initializer_list<int64_t>& Vec) {LOG(FATAL) << "IntArrayRef() is not allowed!";}
};

class MemoryFormat {
  
};

Tensor empty(IntArrayRef, c10::TensorOptions) {
  LOG(FATAL) << "Tensor::empty() is not allowed!";
  return *(Tensor*)nullptr;
}
}