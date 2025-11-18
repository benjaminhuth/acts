// blub

#include <cstdint>
#include <cuda_runtime_api.h>

#include "ActsPlugins/Gnn/detail/CudaUtils.hpp"

//
//   1 - 2 
//        \
//        3 - 4
//       /
//      1

// step 1
// 


__global__ void forward(const int *srcNodes, const int *tgtNodes,
                       std::size_t nEdges, const int *accumulatedPrev,
                       int *accumulatedNext, bool *globalChanged) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= nEdges) {
    return;
  }

  const auto srcNode = srcNodes[i];
  const auto tgtNode = tgtNodes[i];

  const auto nextTgt = accumulatedPrev[srcNode] + 1.f; //nodeWeights[tgtNode];

  // accumulatedNext can only get larger. If local next is already equal or smaller then accumulatedPrev
  // TODO: can we avoid the atomicMax in this case?
  auto prevTgt = atomicMax(&accumulatedNext[tgtNode], nextTgt);

  if (__syncthreads_or(prevTgt != nextTgt) && threadIdx.x == 0) {
    *globalChanged = true;
  }
}

__global__ void backward(const int *srcNodes, const int *tgtNodes,
    std::size_t nEdges, const int *forwardAccumulated,
    int *backwardAccumulated, bool *globalChanged) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= nEdges) {
    return;
  }

  const auto srcNode = srcNodes[i];
  const auto tgtNode = tgtNodes[i];
  
  auto diff = forwardAccumulated[tgtNode] - forwardAccumulated[srcNode];

  float weight = 1.f;
  auto nextSrc = forwardAccumulated[tgtNode] - (diff - weight);
  auto prevSrc = atomicMax(&backwardAccumulated[srcNode], nextSrc);

  if (__syncthreads_or(prevSrc != nextSrc) && threadIdx.x == 0) {
    *globalChanged = true;
  }
}




int filterEdges(const int *srcNodes, const int *dstNodes, std::size_t nEdges, bool *mask, cudaStream_t stream) {
  bool *cudaChanged{};
  ACTS_CUDA_CHECK(cudaMallocAsync(&cudaChanged, sizeof(bool), stream));

  int *forwardAccumulatedNext{}, *forwardAccumulatedPrev{};
  ACTS_CUDA_CHECK(cudaMallocAsync(&forwardAccumulatedNext, sizeof(int)*nEdges, stream));
  ACTS_CUDA_CHECK(cudaMallocAsync(&forwardAccumulatedPrev, sizeof(int)*nEdges, stream));
  ACTS_CUDA_CHECK(cudaMemsetAsync(forwardAccumulatedPrev, 0, sizeof(int)*nEdges, stream));

  const dim3 blockDim = 1024;
  const dim3 gridDimEdges = (nEdges + blockDim.x - 1) / blockDim.x;
  bool changed{};

  // Accumulate forward through graph
  do {
    ACTS_CUDA_CHECK(cudaMemsetAsync(cudaChanged, 0, sizeof(bool), stream));
    
    forward<<<gridDimEdges, blockDim, 0, stream>>>(srcNodes, dstNodes, nEdges,
        forwardAccumulatedPrev, forwardAccumulatedNext, cudaChanged);
    ACTS_CUDA_CHECK(cudaGetLastError());

    
    ACTS_CUDA_CHECK(cudaMemcpyAsync(&changed, cudaChanged, sizeof(int),
                                    cudaMemcpyDeviceToHost, stream));
    ACTS_CUDA_CHECK(cudaStreamSynchronize(stream));

    std::swap(forwardAccumulatedNext, forwardAccumulatedPrev);
  } while(changed);

  // Repurpose pointers, accumulatedNext
  int *forwardAccumulated = forwardAccumulatedNext;
  int *backwardAccumulated = forwardAccumulatedPrev;

  // Propagate backwards
  do {
    ACTS_CUDA_CHECK(cudaMemsetAsync(cudaChanged, 0, sizeof(bool), stream));
    
    backward<<<gridDimEdges, blockDim, 0, stream>>>(srcNodes, dstNodes, nEdges,
      forwardAccumulated, backwardAccumulated, cudaChanged);
      
    ACTS_CUDA_CHECK(cudaMemcpyAsync(&changed, cudaChanged, sizeof(int),
                                    cudaMemcpyDeviceToHost, stream));
    ACTS_CUDA_CHECK(cudaStreamSynchronize(stream));
  } while(changed);

  ACTS_CUDA_CHECK(cudaFreeAsync(cudaChanged, stream));
  ACTS_CUDA_CHECK(cudaFreeAsync(forwardAccumulatedPrev, stream));
  ACTS_CUDA_CHECK(cudaFreeAsync(forwardAccumulatedNext, stream));

  return {};
}


