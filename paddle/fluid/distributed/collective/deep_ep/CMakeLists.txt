set(DEEPEP_SRCS
  deep_ep.cpp
  src/fake_torch/c10/cuda/CUDAStream.cc
  kernels/internode_ll.cu
  kernels/internode.cu
  kernels/intranode.cu
  kernels/runtime.cu
)

nv_library(
  deep_ep
  SRCS ${DEEPEP_SRCS}
  DEPS phi common nvshmem)


target_compile_options(deep_ep PRIVATE -Wno-reorder -Wno-error=unused-variable)
