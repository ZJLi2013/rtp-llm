#pragma once
#include <cuda_runtime_api.h>

namespace fastertransformer {

/*
    This workload computes a batch of GEMM operations with distinct problem sizes. Pointers to matrices
    in Global Memory are passed to the kernel in array (also held in Global Memory). Similarly,
    leading dimensions and problem sizes are stored in arrays in GMEM.

    This differs from "Batched Array" GEMM because the size of each GEMM problem in the Grouped GEMM
    concept may be distinct.
*/


template<typename T>
class CutlassGroupGemmRunner {
public:
    CutlassGroupGemmRunner() = default;
    ~CutlassGroupGemmRunner() = default;

    void gemm(T**                           A,
              T**                           B,
              T**                           C,
              const int*                    m,
              const int*                    n,
              const int*                    k,
              const float                   alpha,
              const float                   beta,
              const int                     count,
              cudaStream_t                  stream);
};

}  // namespace fastertransformer
