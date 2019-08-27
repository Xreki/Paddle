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

#pragma once

#include <stdint.h>
#include <atomic>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/operators/distributed/request_handler.h"
#include "paddle/fluid/operators/distributed/rpc_server.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {

constexpr char kOptimizeBlocks[] = "optimize_blocks";

void FlRunServer(std::shared_ptr<distributed::RPCServer> service);

template <class TKey, class TValue>
class DoubleFindMap : public std::unordered_map<TKey, TValue> {
 public:
  typename std::unordered_map<TKey, TValue>::iterator find_value(TValue v) {
    return std::find_if(this->begin(), this->end(),
                        [&v](const std::pair<const std::string, int> p) {
                          return p.second == v;
                        });
  }
};

class FlListenAndServOp : public framework::OperatorBase {
 public:
  FlListenAndServOp(const std::string& type,
                    const framework::VariableNameMap& inputs,
                    const framework::VariableNameMap& outputs,
                    const framework::AttributeMap& attrs);
  virtual ~FlListenAndServOp();

  void RunSyncLoop(framework::Executor* executor,
                   framework::ProgramDesc* program,
                   framework::Scope* recv_scope,
                   platform::DeviceContext* dev_ctx) const;

  void SavePort() const;

  int GetSelectedPort() { return rpc_service_->GetSelectedPort(); }

  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override;

 protected:
  mutable std::shared_ptr<distributed::RPCServer> rpc_service_;
  mutable std::shared_ptr<distributed::RequestHandler> request_send_handler_;
  mutable std::shared_ptr<distributed::RequestHandler> request_get_handler_;

  mutable std::shared_ptr<std::thread> server_thread_;
  mutable std::vector<std::string> sparse_vars_;
  mutable std::vector<std::string> dense_vars_;
};

class FlSignalHandler {
 public:
  static void StopAndExit(int signal_num);

 private:
  DISABLE_COPY_AND_ASSIGN(FlSignalHandler);
};

}  // namespace operators
}  // namespace paddle
