#add_subdirectory(kernels)

#set(CMAKE_CUDA_FLAGS
#    "${CMAKE_CUDA_FLAGS} -rdc=true --ptxas-options=--register-usage-level=10,--warn-on-local-memory-usage"
#)

#set(SRCS deep_ep.cpp src/fake_torch/c10/cuda/CUDAStream.cc)
set(DEEPEP_SRCS
    deep_ep.cpp
    src/fake_torch/c10/cuda/CUDAStream.cc
    kernels/intranode.cu
    kernels/runtime.cu
    kernels/internode.cu
    kernels/internode_ll.cu
)

#set(EP_CUDA_LIBRARIES intranode_cuda runtime_cuda internode_cuda
#                      internode_ll_cuda)
#set(EP_CUDA_LIBRARIES deepep_kernels)

cc_library(
  deep_ep
  SRCS ${DEEPEP_SRCS}
  DEPS phi common ${EP_CUDA_LIBRARIES} nvshmem cudadevrt)

set_source_files_properties(deep_ep.cpp PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
set_source_files_properties(src/fake_torch/c10/cuda/CUDAStream.cc PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

#set_source_files_properties(kernels/internode.cu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_source_files_properties(kernels/internode_ll.cu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#set_target_properties(deep_ep PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(deep_ep PROPERTIES CUDA_LINK_LTO ON)

# Link CPP and CUDA together
target_compile_options(deep_ep PRIVATE -Wno-reorder -Wno-unused-variable)
#target_compile_options(deep_ep PRIVATE -Wno-reorder -Wno-unused-variable
#    $<$<COMPILE_LANGUAGE:CUDA>:$<TARGET_OBJECTS:kernels/internode.cu>:-rdc=true>
#    $<$<COMPILE_LANGUAGE:CUDA>:$<TARGET_OBJECTS:kernels/internode_ll.cu>:-rdc=true>
#)
