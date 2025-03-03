#pragma once

#include "glog/logging.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"

namespace torch {
class Event;
}

namespace c10 {

class Stream {
 public:
  explicit Stream(const cudaStream_t& stream) : stream_(stream) {}
  Stream(const Stream&) = default;
  Stream(Stream&&) = default;

  template <typename T>
  void wait(const T& event) const {
    wait_event(event.cuda_event());
  }

  cudaStream_t cuda_stream() const { return stream_; }

 private:

  void wait_event(const torch::Event& event);

  cudaStream_t stream_;
};

}