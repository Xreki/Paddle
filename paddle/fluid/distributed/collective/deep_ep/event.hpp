// #include <ATen/cuda/CUDAContext.h>
#include <memory>

#include "kernels/exception.cuh"
#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/event.h"
#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/ATen/cuda/CUDAContext.h"
#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/c10/core/DeviceType.h"
#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/c10/cuda/CUDAStream.h"

namespace deep_ep {

struct EventHandle {
    std::shared_ptr<torch::Event> event;

    EventHandle() {
        event = std::make_shared<torch::Event>(torch::kCUDA);
        event->record(c10::cuda::getCurrentCUDAStream());
    }

    explicit EventHandle(const c10::cuda::CUDAStream& stream) {
        event = std::make_shared<torch::Event>(torch::kCUDA);
        event->record(stream);
    }

    EventHandle(const EventHandle& other) = default;

    void current_stream_wait() const {
        c10::cuda::getCurrentCUDAStream().unwrap().wait(*event);
    }
};

torch::Event create_event(const c10::cuda::CUDAStream &s) {
    auto event = torch::Event(torch::kCUDA);
    event.record(s);
    return event;
}

void stream_wait(const c10::cuda::CUDAStream& s_0, const c10::cuda::CUDAStream& s_1) {
    EP_HOST_ASSERT(s_0.id() != s_1.id());
    s_0.unwrap().wait(create_event(s_1));
}

void stream_wait(const c10::cuda::CUDAStream& s, const EventHandle& event) {
    s.unwrap().wait(*event.event);
}

} // namespace deep_ep
