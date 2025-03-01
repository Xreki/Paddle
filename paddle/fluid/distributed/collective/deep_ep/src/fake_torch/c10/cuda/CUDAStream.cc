#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/c10/cuda/CUDAStream.h"

namespace c10::cuda {
// CUDAStream::operator const cudaStream_t&() const {
//   return raw_stream_;
// //   LOG(FATAL) << "CUDAStream::operator cudaStream_t() const is not implemented yet.";
// //   return *((cudaStream_t*)nullptr);
// }

// const cudaStream_t& CUDAStream::raw_stream() const {return raw_stream_; }

}