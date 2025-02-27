#pragma once

#include "glog/logging.h"

namespace c10 {

class Stream {
 public:
  template <typename T>
  void wait(const T& event) const {
    LOG(FATAL) << "wait() is not implemented in fake_torch";
  }

};

}