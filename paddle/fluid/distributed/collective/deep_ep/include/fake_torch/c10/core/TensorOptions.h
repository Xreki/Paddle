#pragma once
#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/c10/core/Device.h"
#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/c10/core/ScalarType.h"
#include "glog/logging.h"

namespace c10 {

class TensorOptions {
 public:
  TensorOptions device(
      std::optional<Device> device) const noexcept {
    LOG(FATAL) << "Not implemented";
    return TensorOptions();
  }

};

TensorOptions dtype(ScalarType dtype) noexcept {
  LOG(FATAL) << "Not implemented";
  return TensorOptions();
}
  
}