#pragma once

namespace c10 {

enum class ScalarType : int8_t {
  kInt,
  kBool,
  kByte,
  kFloat32,
  Undefined,
  NumOptions
};

}

namespace torch {
  constexpr auto kInt32 = c10::ScalarType::kInt;
  constexpr auto kBool = c10::ScalarType::kBool;
  constexpr auto kFloat32 = c10::ScalarType::kFloat32;
  constexpr auto kByte = c10::ScalarType::kByte;
};