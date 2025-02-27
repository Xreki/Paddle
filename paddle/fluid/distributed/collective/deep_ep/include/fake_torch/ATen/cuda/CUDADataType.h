#pragma once
#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/c10/core/ScalarType.h"

namespace at::cuda {

inline cudaDataType ScalarTypeToCudaDataType(const c10::ScalarType& scalar_type) {
  LOG(FATAL) << "not implemented";
  return cudaDataType();
}

}