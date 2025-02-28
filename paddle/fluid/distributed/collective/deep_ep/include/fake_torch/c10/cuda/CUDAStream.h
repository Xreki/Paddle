#pragma once

#include "glog/logging.h"
#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/c10/core/stream.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/api/include/context_pool.h"

namespace c10::cuda {

using DeviceIndex = int8_t;
using StreamId = int64_t;

class CUDAStream {
 public:
  CUDAStream() {
    LOG(FATAL) << "CUDAStream::CUDAStream() is not implemented";
  }
  CUDAStream(const cudaStream_t &stream) : raw_stream_(stream) {}
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
  const cudaStream_t& raw_stream() const {return raw_stream_; }
 private:
  cudaStream_t raw_stream_;
};

/**
 * Get the current CUDA stream, for the passed CUDA device, or for the
 * current device if no device index is passed.  The current CUDA stream
 * will usually be the default CUDA stream for the device, but it may
 * be different if someone called 'setCurrentCUDAStream' or used 'StreamGuard'
 * or 'CUDAStreamGuard'.
 */
inline CUDAStream getCurrentCUDAStream(DeviceIndex device_index = -1) {
  // if (device_index == -1) {
  //   device_index = phi::backends::gpu::GetCurrentDeviceId();
  // }

  // return CUDAStream(paddle::GetCurrentCUDAStream(phi::GPUPlace(device_index))->raw_stream());
  LOG(FATAL) << "getCurrentCUDAStream is not implemented";
  return *(CUDAStream*)nullptr;
}

inline CUDAStream getStreamFromPool(const bool isHighPriority = false, DeviceIndex device = -1) {
  // if (device == -1) {
  //   device = phi::backends::gpu::GetCurrentDeviceId();
  // }
  // const auto& place = phi::GPUPlace(device);
  // auto comm_ctx = std::make_unique<phi::GPUContext>(place);
  // // TODO: record ctx?
  // return CUDAStream(comm_ctx->stream());
  LOG(FATAL) << "getStreamFromPool is not implemented";
  return *(CUDAStream*)nullptr;
}

inline void setCurrentCUDAStream(CUDAStream stream) {
  LOG(FATAL) << "setCurrentCUDAStream is not implemented";
}

}