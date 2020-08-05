/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/cast_op.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/gpu_launch_config.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {

template <typename InT, typename OutT>
__global__ void DoCastKernel(int n, void* in, void* out) {
  InT* in_ptr = static_cast<InT*>(in);
  OutT* out_ptr = static_cast<OutT*>(out);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
    out_ptr[i] = static_cast<OutT>(in_ptr[i]);
  }
}

template <typename InT>
struct CastOpFunctor<platform::CUDADeviceContext, InT> {
  const framework::Tensor* in_;
  framework::Tensor* out_;
  const platform::CUDADeviceContext& ctx_;
  CastOpFunctor(const framework::Tensor* in, framework::Tensor* out,
                const platform::CUDADeviceContext& ctx)
      : in_(in), out_(out), ctx_(ctx) {}

  template <typename OutT>
  void apply() const {
    auto* in = in_->data<InT>();
    auto num = in_->numel();
    auto* out = out_->mutable_data<OutT>(ctx_.GetPlace());
    int block = 1024;
    int grid = (block - 1 + num) / block;
    DoCastKernel<InT, OutT><<<grid, block, 0, ctx_.stream()>>>(in, num, out);
  }
};

template <typename DeviceContext, typename InT, typename OutT>
static void CastFunction(const framework::ExecutionContext& context) {
  platform::RecordEvent record_event("cast_function");
  auto* x = context.Input<framework::Tensor>("X");
  auto* out = context.Output<framework::Tensor>("Out");

#if 0
  auto in_t = framework::EigenVector<InT>::Flatten(*in);
  out->mutable_data<OutT>(context.GetPlace());
  auto out_t = framework::EigenVector<OutT>::Flatten(*out);
  auto& place =
    *context.template device_context<DeviceContext>().eigen_device();
  out_t.device(place) = in_t.template cast<OutT>();
#else
  auto* x_data = const_cast<InT*>(x->data<InT>());
  auto n = x->numel();
  auto* out_data = out->mutable_data<OutT>(context.GetPlace());

  {
    platform::RecordEvent record_event("cast_launch_kernel");
    auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();
    //    int block = 1024;
    //    int grid = (block - 1 + num) / block;
    //    DoCastKernel<InT, OutT><<<grid, block, 0, dev_ctx.stream()>>>(n,
    //    x_data, out_data);
    platform::GpuLaunchKernel1D(dev_ctx, DoCastKernel<InT, OutT>, n, x_data,
                                out_data);
  }
#endif
}

template <typename DeviceContext, typename InT>
class CastOpGPUKernel : public framework::OpKernel<InT> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto out_type = static_cast<framework::proto::VarType::Type>(
        context.Attr<int>("out_dtype"));

    if (out_type == paddle::framework::proto::VarType::FP64) {
      CastFunction<DeviceContext, InT, double>(context);
    } else if (out_type == paddle::framework::proto::VarType::FP32) {
      CastFunction<DeviceContext, InT, float>(context);
    } else if (out_type == paddle::framework::proto::VarType::FP16) {
      CastFunction<DeviceContext, InT, paddle::platform::float16>(context);
    } else if (out_type == paddle::framework::proto::VarType::INT64) {
      CastFunction<DeviceContext, InT, int64_t>(context);
    } else if (out_type == paddle::framework::proto::VarType::INT32) {
      CastFunction<DeviceContext, InT, int>(context);
    } else if (out_type == paddle::framework::proto::VarType::UINT8) {
      CastFunction<DeviceContext, InT, uint8_t>(context);
    } else if (out_type == paddle::framework::proto::VarType::BOOL) {
      CastFunction<DeviceContext, InT, bool>(context);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    cast, ops::CastOpGPUKernel<paddle::platform::CUDADeviceContext, float>,
    ops::CastOpGPUKernel<paddle::platform::CUDADeviceContext, double>,
    ops::CastOpGPUKernel<paddle::platform::CUDADeviceContext, int>,
    ops::CastOpGPUKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::CastOpGPUKernel<paddle::platform::CUDADeviceContext, bool>,
    ops::CastOpGPUKernel<paddle::platform::CUDADeviceContext, uint8_t>,
    ops::CastOpGPUKernel<paddle::platform::CUDADeviceContext,
                         paddle::platform::float16>);
