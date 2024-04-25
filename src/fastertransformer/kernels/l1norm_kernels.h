#pragma once

#include "src/fastertransformer/cuda/cuda_utils.h"
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace fastertransformer
{

template <typename T>
void invokeGeneralL1Norm(T* out, const T* input, const float eps, const int tokens, const int hidden_dim, cudaStream_t stream = 0);

} // namespace fastertransformer
