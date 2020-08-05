/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace platform {

struct GpuLaunchConfig {
  // Number of threads per block.
  int threads;
  // Number of blocks for GPU kernel launch.
  int blocks;

  GpuLaunchConfig(int threads, int blocks) : threads(threads), blocks(blocks) {}
};

inline GpuLaunchConfig getGpuLaunchConfig(
    const int N, const framework::ExecutionContext& ctx,
    int max_threads = 1024) {
  int threads =
      std::min(max_threads, ctx.cuda_device_context().GetMaxThreadsPerBlock());
  int physical_thread_count =
      std::min(ctx.cuda_device_context().GetMaxPhysicalThreadCount(), N);
  int blocks = std::min((physical_thread_count + threads - 1) / threads,
                        ctx.cuda_device_context().GetSMCount());

  GpuLaunchConfig config(threads, blocks);

  return config;
}

using GpuKernelHandler =
    std::function<void(const int n, const void* in, void* out)>;
void GpuLaunchKernel1D(const platform::CUDADeviceContext& context,
                       const void* func, int n, void* in, void* out) {
  std::vector<void*> args;
  args.push_back(&n);
  args.push_back(&in);
  args.push_back(&out);

  int num_threads = 1024;
  int num_blocks = (n + num_threads - 1) / num_threads;

  dim3 block_dim = dim3(num_threads, 1, 1);
  dim3 grid_dim = dim3(num_blocks, 1, 1);
  cudaLaunchKernel(func, grid_dim, block_dim, args.data(), 0, context.stream());
}

}  // namespace platform
}  // namespace paddle
