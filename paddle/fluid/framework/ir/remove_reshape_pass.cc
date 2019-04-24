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

#include "paddle/fluid/framework/ir/remove_reshape_pass.h"
#include <unordered_set>
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace framework {
namespace ir {

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& value) {
  os << "{";
  if (value.size() > 0) {
    os << value[0];
  }
  for (size_t i = 1; i < value.size(); ++i) {
    os << ", " << value[i];
  }
  os << "}";
  return os;
}

void RemoveReshapePass::ApplyImpl(ir::Graph* graph) const {
  VLOG(3) << "Remove reshape_op of which input and output has the same shape.";
  std::unordered_set<const Node*> reshape_op_set;
  for (Node* n : graph->Nodes()) {
    if (n->IsOp() && n->Op()) {
      framework::OpDesc* op = n->Op();
      if (op->Type() == "reshape" || op->Type() == "reshape2") {
        PADDLE_ENFORCE_EQ(op->Input("X").size(), 1UL);
        std::string x_name = op->Input("X")[0];
        std::string shape_name = "";
        if (op->Input("Shape").size() > 0UL) {
          PADDLE_ENFORCE_EQ(op->Input("Shape").size(), 1UL);
          shape_name = op->Input("Shape")[0];
        }

        PADDLE_ENFORCE_EQ(op->Output("Out").size(), 1UL);
        std::string out_name = op->Output("Out")[0];
        std::string xshape_name = "";
        if (op->Output("XShape").size() > 0UL) {
          PADDLE_ENFORCE_EQ(op->Output("XShape").size(), 1UL);
          xshape_name = op->Output("XShape")[0];
        }

        Node* grad_n = FindGradNode(graph, n);
        framework::OpDesc* grad_op = grad_n ? grad_n->Op() : nullptr;

        bool can_remove = false;
        if (!shape_name.empty()) {
          can_remove = false;
        } else if (x_name == out_name) {
          // inplace=true
          if (xshape_name.empty()) {
            can_remove = true;
          } else if (!IsInputOfOtherOps(graph, op, grad_op, xshape_name)) {
            std::vector<int64_t> out_shape;
            std::vector<int64_t> xshape_shape;
            for (auto* out : n->outputs) {
              if (out->IsVar() && out->Var()) {
                framework::VarDesc* var_desc = out->Var();
                if (var_desc->Name() == out_name) {
                  out_shape = var_desc->GetShape();
                } else if (var_desc->Name() == xshape_name) {
                  xshape_shape = var_desc->GetShape();
                }
              }
            }
            PADDLE_ENFORCE_GE(out_shape.size(), 1UL);
            PADDLE_ENFORCE_GE(xshape_shape.size(), 2UL);
            PADDLE_ENFORCE_EQ(xshape_shape[0], 0UL);
            if (IsEqual<int64_t>(out_shape, xshape_shape, 0UL, 1UL)) {
              can_remove = true;
            }
          }
        }

        if (can_remove) {
          reshape_op_set.insert(n);
          if (grad_n && grad_op) {
            reshape_op_set.insert(grad_n);
          }
        }
      }
    }
  }

  // for (const Node* n : reshape_op_set) {
  //   std::cout << "Input of " << n->Op()->Type() << std::endl;
  //   for (auto* in : n->inputs) {
  //     if (in->IsVar() && in->Var()) {
  //       framework::VarDesc* var_desc = in->Var();
  //       std::cout << "Var name: " << var_desc->Name()
  //                 << ", shape: " << var_desc->GetShape() << std::endl;
  //     }
  //   }
  //   std::cout << "Output of " << n->Op()->Type() << std::endl;
  //   for (auto* out : n->outputs) {
  //     if (out->IsVar() && out->Var()) {
  //       framework::VarDesc* var_desc = out->Var();
  //       std::cout << "Var name: " << var_desc->Name()
  //                 << ", shape: " << var_desc->GetShape() << std::endl;
  //     }
  //   }
  // }
  GraphSafeRemoveNodes(graph, reshape_op_set);
}

bool RemoveReshapePass::IsInputOfOtherOps(ir::Graph* graph,
                                          framework::OpDesc* op,
                                          framework::OpDesc* grad_op,
                                          std::string var_name) const {
  bool res = false;
  for (const Node* n : graph->Nodes()) {
    if (n->IsOp() && n->Op() && n->Op() != op && n->Op() != grad_op) {
      if (Has(n->Op()->InputArgumentNames(), var_name)) {
        std::cout << "Op " << n->Op()->Type() << " use " << var_name
                  << " as input." << std::endl;
        res = true;
        break;
      }
    }
  }
  return res;
}

Node* RemoveReshapePass::FindGradNode(ir::Graph* graph, Node* node) const {
  if (!node->IsOp() || !node->Op()) {
    return nullptr;
  }

  framework::OpDesc* op = node->Op();
  if (!(op->Type() == "reshape" || op->Type() == "reshape2")) {
    return nullptr;
  }

  std::string x = op->Input("X")[0];
  std::string out = op->Output("Out")[0];

  for (Node* n : graph->Nodes()) {
    if (n->IsOp() && n->Op()) {
      framework::OpDesc* grad_op = n->Op();
      if (grad_op->Type() == op->Type() + "_grad") {
        if (Has(grad_op->InputArgumentNames(), framework::GradVarName(out)) &&
            Has(grad_op->OutputArgumentNames(), framework::GradVarName(x))) {
          return n;
        }
      }
    }
  }

  return nullptr;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(remove_reshape_pass, paddle::framework::ir::RemoveReshapePass);
