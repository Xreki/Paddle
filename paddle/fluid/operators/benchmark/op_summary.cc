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

#include <sstream>
#include <unordered_set>
#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/pybind/pybind.h"

namespace paddle {
namespace operators {
namespace benchmark {

std::string GenSpaces(int count) {
  std::stringstream ss;
  for (int i = 0; i < count; ++i) {
    ss << "  ";
  }
  return ss.str();
}

bool IsGradOp(const std::string& op_type) {
  for (std::string suffix : {"_grad", "_grad2"}) {
    size_t pos = op_type.rfind(suffix);
    if (pos != std::string::npos &&
        pos == (op_type.length() - suffix.length())) {
      return true;
    }
  }
  return false;
}

std::string AttrTypeName(const framework::proto::AttrType& type) {
  switch (type) {
    case framework::proto::AttrType::BOOLEAN:
      return "bool";
    case framework::proto::AttrType::INT:
      return "int32";
    case framework::proto::AttrType::LONG:
      return "int64";
    case framework::proto::AttrType::FLOAT:
      return "float32";
    case framework::proto::AttrType::STRING:
      return "string";
    case framework::proto::AttrType::BOOLEANS:
      return "vector<bool>";
    case framework::proto::AttrType::INTS:
      return "vector<int32>";
    case framework::proto::AttrType::LONGS:
      return "vector<int64>";
    case framework::proto::AttrType::FLOATS:
      return "vector<float32>";
    case framework::proto::AttrType::STRINGS:
      return "vector<string>";
    case framework::proto::AttrType::BLOCK:
      return "block";
    default:
      LOG(WARNING) << "Unsupport attr type " << type;
  }
  std::ostringstream os;
  os << type;
  return os.str();
}

std::string DebugString(const framework::OpInfo& info) {
  const std::unordered_set<std::string> skipped_attrs = {
      framework::OpProtoAndCheckerMaker::OpRoleAttrName(),
      framework::OpProtoAndCheckerMaker::OpRoleVarAttrName(),
      framework::OpProtoAndCheckerMaker::OpNamescopeAttrName(),
      framework::OpProtoAndCheckerMaker::OpCreationCallstackAttrName()};

  std::ostringstream os;
  const framework::proto::OpProto& proto = info.Proto();

  int num_inputs = proto.inputs_size();
  for (int i = 0; i < num_inputs; ++i) {
    if (i > 0) {
      os << ",";
    }
    os << proto.inputs(i).name();
  }
  os << "\t";

  int num_outputs = proto.outputs_size();
  for (int i = 0; i < num_outputs; ++i) {
    if (i > 0) {
      os << ",";
    }
    os << proto.outputs(i).name();
  }
  os << "\t";

  int num_attrs = proto.attrs_size();
  bool is_first = true;
  for (int i = 0; i < num_attrs; ++i) {
    const auto& attr = proto.attrs(i);
    if (skipped_attrs.find(attr.name()) == skipped_attrs.end()) {
      if (!is_first) {
        os << ",";
      }
      os << attr.name() << "(" << AttrTypeName(attr.type()) << ")";
      is_first = false;
    }
  }
  return os.str();
}

void PrintAllOps() {
  std::ostringstream os;
  int num_basic_ops = 0;
  int num_grad_ops = 0;
  auto& op_info_map = framework::OpInfoMap::Instance().map();
  for (auto& item : op_info_map) {
    std::string op_type = item.first;
    if (!IsGradOp(op_type) && op_type != "push_box_sparse") {
      std::string grad_op_type = "none";
      for (std::string suffix : {"_grad", "_grad2"}) {
        if (op_info_map.find(op_type + suffix) != op_info_map.end()) {
          grad_op_type = op_type + suffix;
          break;
        }
      }
      os << op_type << "\t" << grad_op_type << "\t" << DebugString(item.second)
         << "\n";
      num_basic_ops++;
    } else {
      num_grad_ops++;
    }
  }
  os << "";
  os << "Totally " << op_info_map.size() << " operators:\n";
  os << GenSpaces(2) << num_basic_ops << " basic operators.\n";
  os << GenSpaces(2) << num_grad_ops << " grad operators.\n";
  std::cout << os.str();
}

TEST(op_summary, base) { PrintAllOps(); }

}  // namespace benchmark
}  // namespace operators
}  // namespace paddle
