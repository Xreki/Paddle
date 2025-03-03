#pragma once

#include <glog/logging.h>
#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/c10/core/stream.h"
#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/c10/cuda/CUDAStream.h"
#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/c10/core/DeviceType.h"

namespace torch {

class Event {
 public:
  Event() {
    cudaEventCreate(&cuda_event_);
  }
  ~Event() {
    cudaEventDestroy(cuda_event_);
  }
  Event(const DeviceType _device_type) {
    LOG(FATAL) << "Not implemented"; 
  }
  void record(const c10::Stream& stream) {
    LOG(FATAL) << "Not implemented";
  }
  // void record(const c10::cuda::CUDAStream& stream) {
  //   LOG(FATAL) << "Not implemented";
  // }
  void record(const cudaStream_t &stream) {
    cudaEventRecord(cuda_event_, stream);
  }

  cudaEvent_t cuda_event() const {
    return cuda_event_;
  }

 private:
  cudaEvent_t cuda_event_;
};

}