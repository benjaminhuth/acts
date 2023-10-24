#include "Acts/Plugins/ExaTrkX/detail/CudaInfo.hpp"

#include "torch/torch.h"

std::size_t Acts::detail::cudaNumDevices() {
  if (not torch::cuda::is_available()) {
    return 0;
  }

  return torch::cuda::device_count();
}
