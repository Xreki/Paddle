// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <Python.h>
#include "pybind11/stl.h"

#include "paddle/fluid/pybind/deep_ep_api.h"
#include "paddle/fluid/distributed/collective/deep_ep/deep_ep.hpp"

namespace py = pybind11;

namespace paddle::pybind {

void BindDeepEPApi(pybind11::module *m) {

  pybind11::class_<deep_ep::Config>(*m, "Config")
      .def(pybind11::init<int, int, int, int, int>(),
           py::arg("num_sms") = 20,
           py::arg("num_max_nvl_chunked_send_tokens") = 6,
           py::arg("num_max_nvl_chunked_recv_tokens") = 256,
           py::arg("num_max_rdma_chunked_send_tokens") = 6,
           py::arg("num_max_rdma_chunked_recv_tokens") = 256);
//   m->def("get_low_latency_rdma_size_hint",
//         &deep_ep::get_low_latency_rdma_size_hint);

  pybind11::class_<deep_ep::EventHandle>(*m, "EventHandle")
      .def(pybind11::init<>())
      .def("current_stream_wait", &deep_ep::EventHandle::current_stream_wait);

  pybind11::class_<deep_ep::Buffer>(*m, "Buffer")
      .def(pybind11::init<int, int, int64_t, int64_t, bool, int>())
    //   .def("is_available", &deep_ep::Buffer::is_available)
      .def("get_num_rdma_ranks", &deep_ep::Buffer::get_num_rdma_ranks)
      .def("get_rdma_rank", &deep_ep::Buffer::get_rdma_rank)
      .def("get_root_rdma_rank", &deep_ep::Buffer::get_root_rdma_rank)
    //   .def("get_local_device_id", &deep_ep::Buffer::get_local_device_id)
    //   .def("get_local_ipc_handle", &deep_ep::Buffer::get_local_ipc_handle)
    //   .def("get_local_buffer_tensor", &deep_ep::Buffer::get_local_buffer_tensor)
      .def("sync", &deep_ep::Buffer::sync)
      .def("get_dispatch_layout", &deep_ep::Buffer::get_dispatch_layout_api)
      .def("intranode_dispatch", &deep_ep::Buffer::intranode_dispatch_api)
      .def("intranode_combine", &deep_ep::Buffer::intranode_combine_api);
}

}  // namespace paddle::pybind