#pragma once

#include "paddle/phi/common/data_type.h"

namespace c10 {

using ScalarType = phi::DataType;

}

namespace torch {
  constexpr auto kInt32 = c10::ScalarType::INT32;
  constexpr auto kInt64 = c10::ScalarType::INT64;
  constexpr auto kBool = c10::ScalarType::BOOL;
  constexpr auto kFloat8_e4m3fn = c10::ScalarType::FLOAT8_E4M3FN;
  constexpr auto kBFloat16 = c10::ScalarType::BFLOAT16;
  constexpr auto kFloat32 = c10::ScalarType::FLOAT32;
  constexpr auto kByte = c10::ScalarType::INT8;
}
