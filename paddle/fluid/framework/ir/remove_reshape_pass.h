/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

class RemoveReshapePass : public Pass {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  template <typename T>
  bool Has(const std::vector<T>& vec, T value) const {
    for (auto& elem : vec) {
      if (elem == value) {
        return true;
      }
    }
    return false;
  }

  template <typename T>
  bool IsEqual(const std::vector<T>& a, const std::vector<T>& b, size_t beg_a,
               size_t beg_b) const {
    if (a.size() - beg_a != b.size() - beg_b) {
      return false;
    }
    for (size_t i = 0; i < a.size() - beg_a; ++i) {
      if (a[i + beg_a] != b[i + beg_b]) {
        return false;
      }
    }
    return true;
  }

  bool IsInputOfOtherOps(ir::Graph* graph, framework::OpDesc* op,
                         framework::OpDesc* grad_op,
                         std::string var_name) const;

  Node* FindGradNode(ir::Graph* graph, Node* node) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
