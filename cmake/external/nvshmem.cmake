# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(ExternalProject)

set(GDRCOPY_HOME
    $ENV{GDRCOPY_HOME}
    CACHE PATH "Path to GDRCOPY installation")
if(GDRCOPY_HOME)
  message(STATUS "GDRCOPY_HOME: ${GDRCOPY_HOME}")
else()
  message(
    STATUS
      "You can set GDRCOPY_HOME environment or cmake option to specify your GDRCOPY."
  )
endif()

set(NVSHMEM_INSTALL_DIR ${THIRD_PARTY_PATH}/install/nvshmem)
set(NVSHMEM_PREFIX_DIR ${THIRD_PARTY_PATH}/nvshmem)
set(NVSHMEM_SOURCE_DIR ${NVSHMEM_PREFIX_DIR}/src/extern_nvshmem)
message(STATUS "NVSHMEM_INSTALL_DIR: ${NVSHMEM_INSTALL_DIR}")

set(NVSHMEM_INCLUDE_DIR
    "${NVSHMEM_INSTALL_DIR}/include"
    CACHE PATH "nvshmem include directory." FORCE)
set(NVSHMEM_LIBRARIES
    "${NVSHMEM_INSTALL_DIR}/lib/libnvshmem.a"
    CACHE FILEPATH "NVSHMEM_LIBRARIES" FORCE)

include_directories(${NVSHMEM_INCLUDE_DIR})

set(NVSHMEM_URL
    "https://developer.download.nvidia.com/compute/redist/nvshmem/3.1.7/source/nvshmem_src_3.1.7-1.txz"
    CACHE STRING "" FORCE)

set(NVSHMEM_PATCH_PATH ${PADDLE_SOURCE_DIR}/third_party/nvshmem.patch)
set(NVSHMEM_PATCH_COMMAND
    git init && git config user.name "PaddlePaddle" && git config user.email
    "paddle@baidu.com" && git add . && git commit -m "init" && git apply
    ${NVSHMEM_PATCH_PATH})

ExternalProject_Add(
  extern_nvshmem
  ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
  PREFIX ${NVSHMEM_PREFIX_DIR}
  SOURCE_DIR ${NVSHMEM_SOURCE_DIR}
  DOWNLOAD_DIR ${NVSHMEM_PREFIX_DIR}/src
  DOWNLOAD_COMMAND
    rm -rf extern_nvshmem nvshmem_src_3.1.7-1.txz && wget --no-check-certificate
    -q ${NVSHMEM_URL} && tar xf nvshmem_src_3.1.7-1.txz && mv nvshmem_src
    extern_nvshmem
  PATCH_COMMAND ${NVSHMEM_PATCH_COMMAND}
  UPDATE_COMMAND ""
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${NVSHMEM_INSTALL_DIR}
             -DGDRCOPY_HOME:PATH=${GDRCOPY_HOME}
             -DNVSHMEM_SHMEM_SUPPORT=0
             -DNVSHMEM_UCX_SUPPORT=0
             -DNVSHMEM_USE_NCCL=0
             -DNVSHMEM_IBGDA_SUPPORT=1
             -DNVSHMEM_PMIX_SUPPORT=0
             -DNVSHMEM_TIMEOUT_DEVICE_POLLING=0
             -DNVSHMEM_USE_GDRCOPY=1
             -DNVSHMEM_IBRC_SUPPORT=1
  CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${NVSHMEM_INSTALL_DIR}
  BUILD_BYPRODUCTS ${NVSHMEM_LIBRARIES})

add_library(nvshmem STATIC IMPORTED GLOBAL)
set_property(TARGET nvshmem PROPERTY IMPORTED_LOCATION ${NVSHMEM_LIBRARIES})
add_dependencies(nvshmem extern_nvshmem)
