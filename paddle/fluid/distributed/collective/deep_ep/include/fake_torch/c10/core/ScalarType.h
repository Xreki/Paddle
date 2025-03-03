#pragma once

#include "paddle/phi/common/data_type.h"

namespace c10 {

using ScalarType = phi::DataType;

}


namespace torch {
  constexpr auto kInt32 = c10::ScalarType::INT32;
  constexpr auto kBool = c10::ScalarType::BOOL;
  constexpr auto kFloat32 = c10::ScalarType::FLOAT32;
  constexpr auto kByte = c10::ScalarType::INT8;
}
