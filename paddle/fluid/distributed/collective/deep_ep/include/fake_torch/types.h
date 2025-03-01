#pragma once

#include "glog/logging.h"
#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/c10/cuda/CUDAStream.h"
#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/c10/core/ScalarType.h"
#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/c10/core/TensorOptions.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/memory/malloc.h"

namespace torch {

struct Tensor {
  paddle::Tensor raw_tensor;

  explicit Tensor(const paddle::Tensor &t) : raw_tensor(t) {}
  Tensor() {
    LOG(FATAL) << "Tensor constructor is not allowed!";
  }
  Tensor(const Tensor &) = default;
  Tensor(Tensor &&) = default;
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
  void record_stream(const cudaStream_t &stream) const {
    paddle::memory::RecordStream(std::dynamic_pointer_cast<phi::DenseTensor>(raw_tensor.impl())->Holder(), stream);
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