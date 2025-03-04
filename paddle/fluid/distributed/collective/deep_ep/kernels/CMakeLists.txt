set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} -rdc=true --ptxas-options=--register-usage-level=10,--warn-on-local-memory-usage"
)

function(add_deep_ep_library target_name source_file)
  set(options STATIC static SHARED shared)
  set(oneValueArgs "")
  set(multiValueArgs DEPS)
  cmake_parse_arguments(add_deep_ep_library "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})
  nv_library(
    ${target_name}
    SRCS ${source_file}
    DEPS ${add_deep_ep_library_DEPS})
  #set_target_properties(${target_name} PROPERTIES
  #        POSITION_INDEPENDENT_CODE ON
  #        CXX_STANDARD_REQUIRED ON
  #        CUDA_STANDARD_REQUIRED ON
  #        CXX_STANDARD 17
  #        CUDA_STANDARD 17
  #        CUDA_SEPARABLE_COMPILATION ON
  #)
endfunction()

nv_library(deepep_kernels
    SRCS intranode.cu runtime.cu internode.cu internode_ll.cu
    DEPS nvshmem
)

#add_deep_ep_library(intranode_cuda intranode.cu DEPS nvshmem)
#add_deep_ep_library(runtime_cuda runtime.cu DEPS nvshmem)
#add_deep_ep_library(internode_cuda internode.cu DEPS nvshmem)
#add_deep_ep_library(internode_ll_cuda internode_ll.cu DEPS nvshmem)
