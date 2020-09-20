// 20-09-10

#ifndef GATHER_SCATTER
#define GATHER_SCATTER

#include <cuda_runtime.h>
#include <cusparse.h>

namespace cusz {
namespace impl {

template <typename DType>
void GatherAsCSR(DType* d_A, size_t lenA, size_t ldA, size_t m, size_t n, int* nnz, std::string* fo);

template <typename DType>
void ScatterFromCSR(DType* d_A, size_t lenA, size_t ldA, size_t m, size_t n, int* nnz, std::string* fi);

void PruneGatherAsCSR(
    float*       d_A,  //
    size_t       len,
    const int    lda,
    const int    m,
    const int    n,
    int&         nnzC,
    std::string* fo);

}  // namespace impl
}  // namespace cusz

#endif