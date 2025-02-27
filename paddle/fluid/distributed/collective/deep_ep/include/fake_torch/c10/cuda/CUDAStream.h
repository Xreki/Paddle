#pragma once

#include "glog/logging.h"
#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/c10/core/stream.h"

namespace c10::cuda {

using DeviceIndex = int8_t;
using StreamId = int64_t;

class CUDAStream {
 public:
  Stream unwrap() const {
    LOG(FATAL) << "CUDAStream::unwrap() is not implemented";
    return *(Stream*)nullptr;
  }
  StreamId id() const {
    LOG(FATAL) << "CUDAStream::unwrap() is not implemented";
    return *(StreamId*)nullptr;
  }

  operator cudaStream_t() const {
    LOG(FATAL) << "CUDAStream::operator cudaStream_t() is not implemented";
    return *(cudaStream_t*)nullptr;
  }

};

/**
 * Get the current CUDA stream, for the passed CUDA device, or for the
 * current device if no device index is passed.  The current CUDA stream
 * will usually be the default CUDA stream for the device, but it may
 * be different if someone called 'setCurrentCUDAStream' or used 'StreamGuard'
 * or 'CUDAStreamGuard'.
 */
CUDAStream getCurrentCUDAStream(DeviceIndex device_index = -1) {
  LOG(FATAL) << "getCurrentCUDAStream is not implemented";
  return *(CUDAStream*)nullptr;
}

CUDAStream getStreamFromPool(const bool isHighPriority = false, DeviceIndex device = -1) {
  LOG(FATAL) << "getStreamFromPool is not implemented";
  return *(CUDAStream*)nullptr;
}

void setCurrentCUDAStream(CUDAStream stream) {
  LOG(FATAL) << "setCurrentCUDAStream is not implemented";
}

}