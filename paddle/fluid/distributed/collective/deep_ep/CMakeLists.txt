add_subdirectory(kernels)

set(SRCS
  deep_ep.cpp
  src/fake_torch/c10/cuda/CUDAStream.cc
)

cc_library(
  deep_ep
  SRCS ${SRCS}
  DEPS phi common)

set_target_properties(deep_ep PROPERTIES
  CXX_STANDARD_REQUIRED ON
  CUDA_STANDARD_REQUIRED ON
  CUDA_SEPARABLE_COMPILATION ON
)

# Link CPP and CUDA together
target_compile_options(deep_ep PRIVATE -Wno-reorder)
target_link_libraries(deep_ep ${EP_CUDA_LIBRARIES})