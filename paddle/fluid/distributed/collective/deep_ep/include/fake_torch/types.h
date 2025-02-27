#pragma once

#include "glog/logging.h"
#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/c10/cuda/CUDAStream.h"
#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/c10/core/ScalarType.h"
#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/c10/core/TensorOptions.h"

namespace torch {
struct Tensor;
struct Tensor {
  Tensor() {
    LOG(FATAL) << "Tensor constructor is not allowed!";
  }
  Tensor(const Tensor &) {
    LOG(FATAL) << "Tensor copy constructor is not allowed!";
  }
  Tensor(Tensor &&) {
    LOG(FATAL) << "Tensor move constructor is not allowed!";
  }

  Tensor operator=(const Tensor &x) & noexcept {
    LOG(FATAL) << "Tensor copy assignment operator is not allowed!";
    // return *(*Tensor)nullptr;
    return Tensor();
  }

  int64_t dim() const {
    LOG(FATAL) << "Tensor::dim() is not allowed!";
    return 0;
  }

  bool is_contiguous() const {
    LOG(FATAL) << "Tensor::is_contiguous() is not allowed!";
    return false;
  }

  int64_t size(int64_t d) const {
    LOG(FATAL) << "Tensor::size() is not allowed!";
    return *(int64_t*)nullptr;
  }

  template <typename T>
  T* data_ptr() const {
    LOG(FATAL) << "Tensor::data_ptr() is not allowed!";
    return nullptr;
  }

  void* data_ptr() const {
    LOG(FATAL) << "Tensor::data() is not allowed!";
    return nullptr;
  }

  // code may be generated in torch
  void record_stream(const c10::cuda::CUDAStream &stream) const {
    LOG(FATAL) << "Tensor::record_stream() is not allowed!";
  }

  c10::ScalarType scalar_type() const {
    LOG(FATAL) << "Tensor::scalar_type() is not allowed!";
    return c10::ScalarType::Undefined;
  }

  int64_t element_size() const {
    LOG(FATAL) << "Tensor::element_size() is not allowed!";
    return 0;
  }

  c10::TensorOptions options() const {
    LOG(FATAL) << "Tensor::options() is not allowed!";
    return c10::TensorOptions();
  }

};
}