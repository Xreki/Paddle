#pragma once

#include "glog/logging.h"
#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/c10/cuda/CUDAStream.h"
#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/c10/core/ScalarType.h"
#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/c10/core/TensorOptions.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/memory/malloc.h"

namespace torch {

struct Tensor {
  phi::DenseTensor raw_tensor;

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
    raw_tensor = x.raw_tensor;
    return *this;
  }

  int64_t dim() const {
    return raw_tensor.dims().size();
  }

  bool is_contiguous() const {
    return true;
  }

  int64_t size(int64_t d) const {
    return raw_tensor.dims().at(d);
  }

  template <typename T>
  T* data_ptr() const {
    return const_cast<T*>(raw_tensor.data<T>());
  }

  void* data_ptr() const {
    return const_cast<void*>(raw_tensor.data());
  }

  template <typename T>
  T* data_ptr() {
    return raw_tensor.data<T>();
  }

  void* data_ptr() {
    return raw_tensor.data();
  }

  // code may be generated in torch
  void record_stream(const c10::cuda::CUDAStream &stream) const {
    // paddle::memory::RecordStream(raw_tensor.Holder(), stream.raw_stream());
    LOG(FATAL) << "Tensor::record_stream() is not allowed!";
  }

  c10::ScalarType scalar_type() const {
    LOG(FATAL) << "Tensor::scalar_type() is not allowed!";
    return c10::ScalarType::Undefined;
  }

  int64_t element_size() const {
    return raw_tensor.numel();
  }

  c10::TensorOptions options() const {
    LOG(FATAL) << "Tensor::options() is not allowed!";
    return c10::TensorOptions();
  }

};
}