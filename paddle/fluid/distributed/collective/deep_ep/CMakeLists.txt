add_subdirectory(kernels)

cc_library(
  deep_ep
  SRCS deep_ep.cpp
  DEPS phi common)

# Link CPP and CUDA together
target_compile_options(deep_ep PRIVATE -Wno-reorder)
target_link_libraries(deep_ep ${EP_CUDA_LIBRARIES})