#pragma once

#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/c10/core/DeviceType.h"

namespace c10 {

using DeviceIndex = int8_t;

class Device {
 public:
  /* implicit */ Device(torch::DeviceType type, DeviceIndex index = -1) {
    LOG(FATAL) << "Not implemented";
  }

};

}