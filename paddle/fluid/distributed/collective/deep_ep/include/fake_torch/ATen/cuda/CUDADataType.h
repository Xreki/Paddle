#pragma once
#include <variant>
#include <unordered_map>
#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/c10/core/ScalarType.h"
#include "paddle/phi/backends/gpu/cuda/cuda_helper.h"
#include "paddle/common/overloaded.h"

namespace at::cuda {

namespace detail {

template <phi::DataType phi_data_type>
struct PhiDataTypeImpl {
  constexpr static phi::DataType value = phi_data_type;
};

using PhiDataType = std::variant<
#define MAKE_PHI_DATA_TYPE_CASE(_, phi_data_type) PhiDataTypeImpl<phi::phi_data_type>,
    PD_FOR_EACH_DATA_TYPE(MAKE_PHI_DATA_TYPE_CASE)
    PhiDataTypeImpl<phi::DataType::UNDEFINED>
#undef MAKE_PHI_DATA_TYPE_CASE
>;

inline PhiDataType ScalarTypeToPhiDataType(const c10::ScalarType& scalar_type) {
  static std::unordered_map<c10::ScalarType, PhiDataType> map = {
#define MAKE_PHI_DATA_TYPE_CONVERT_CASE(_, phi_data_type) \
    {phi::phi_data_type, PhiDataTypeImpl<phi::phi_data_type>{}},
      PD_FOR_EACH_DATA_TYPE(MAKE_PHI_DATA_TYPE_CONVERT_CASE)
#undef MAKE_PHI_DATA_TYPE_CONVERT_CASE
    {phi::DataType::UNDEFINED, PhiDataTypeImpl<phi::DataType::UNDEFINED>{}},
  };
  const auto iter = map.find(scalar_type);
  if (iter == map.end()) {
    LOG(FATAL) << "unsupported scalar type: " << static_cast<int>(scalar_type);
  }
  return iter->second;
}

}

inline cudaDataType_t ScalarTypeToCudaDataType(const c10::ScalarType& scalar_type) {
  auto phi_data_type = detail::ScalarTypeToPhiDataType(scalar_type);
  auto Converter = ::common::Overloaded{
    [](detail::PhiDataTypeImpl<phi::DataType::PSTRING>) -> cudaDataType_t {
      LOG(FATAL) << "unsupported scalar type: pstring";
      return *(cudaDataType_t*)nullptr;
    },
    [](detail::PhiDataTypeImpl<phi::DataType::UNDEFINED>) -> cudaDataType_t {
      LOG(FATAL) << "unsupported scalar type: undefined";
      return *(cudaDataType_t*)nullptr;
    },
    [](auto phi_data_type_impl) -> cudaDataType_t {
      using T = std::decay_t<decltype(phi_data_type_impl)>;
      using CppT = typename phi::DataTypeToCppType<T::value>::type;
      return phi::backends::gpu::ToCudaDataType<CppT>();
    }
  };
  return std::visit(Converter, phi_data_type);
}

}