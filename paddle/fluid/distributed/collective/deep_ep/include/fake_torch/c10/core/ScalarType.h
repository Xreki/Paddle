#pragma once

namespace c10 {

enum class ScalarType : int8_t {
  kInt,
  kBool,
  kByte,
  kInt64,
  kFloat8_e4m3fn,
  kBFloat16,
  kFloat32,
  Undefined,
  NumOptions
};

}

namespace torch {
  constexpr auto kInt32 = c10::ScalarType::kInt;
  constexpr auto kInt64 = c10::ScalarType::kInt64;
  constexpr auto kBool = c10::ScalarType::kBool;
  constexpr auto kFloat8_e4m3fn = c10::ScalarType::kFloat8_e4m3fn;
  constexpr auto kBFloat16 = c10::ScalarType::kBFloat16;
  constexpr auto kFloat32 = c10::ScalarType::kFloat32;
  constexpr auto kByte = c10::ScalarType::kByte;
};
