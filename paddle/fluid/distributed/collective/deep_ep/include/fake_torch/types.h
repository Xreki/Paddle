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
  paddle::Tensor raw_tensor_;

  explicit Tensor(const paddle::Tensor &t) : raw_tensor_(t) {}
  Tensor() : raw_tensor_() { }
  Tensor(const Tensor &) = default;
  Tensor(Tensor &&) = default;
  Tensor operator=(const Tensor &x) & noexcept {
    raw_tensor_ = x.raw_tensor_;
    return *this;
  }

  decltype(auto) raw_tensor() const {
    return raw_tensor_;
  }

  decltype(auto) dtype() const {
    return raw_tensor_.dtype();
  }

  decltype(auto) place() const {
    return raw_tensor_.place();
  }

  int64_t dim() const {
    return raw_tensor_.dims().size();
  }

  bool is_contiguous() const {
    return true;
  }

  int64_t size(int64_t d) const {
    return raw_tensor_.dims().at(d);
  }

  template <typename T>
  T* data_ptr() const {
    return const_cast<T*>(raw_tensor_.data<T>());
  }

  void* data_ptr() const {
    return const_cast<void*>(raw_tensor_.data());
  }

  template <typename T>
  T* data_ptr() {
    return raw_tensor_.data<T>();
  }

  void* data_ptr() {
    return raw_tensor_.data();
  }

  // code may be generated in torch
  void record_stream(const cudaStream_t &stream) const {
    paddle::memory::RecordStream(std::dynamic_pointer_cast<phi::DenseTensor>(raw_tensor_.impl())->Holder(), stream);
  }

  c10::ScalarType scalar_type() const {
    return raw_tensor_.dtype();
  }

  int64_t element_size() const {
    return phi::SizeOf(raw_tensor_.dtype());
  }

  c10::TensorOptions options() const {
    LOG(FATAL) << "Tensor::options() is not allowed!";
    return c10::TensorOptions();
  }

};
}