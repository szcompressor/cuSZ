#ifndef CUSZ_WORKFLOW_CUH
#define CUSZ_WORKFLOW_CUH

#include "argparse.hh"

template <typename T>
__global__ void CountOutlier(T const* const, int*, size_t);

namespace cuSZ {
namespace workflow {

template <typename T, typename Q>
void PdQ(T*, Q*, size_t*, double*);

template <typename T, typename Q>
void ReversedPdQ(T*, Q*, T*, size_t*, double);

template <typename T>
__global__ void Condenser(T*, int*, size_t, size_t);

void DeflateOutlierUsingCuSparse(float*, size_t, int&, int**, int**, float**);

template <typename T>
size_t* DeflateOutlier(T*, T*, int*, size_t, size_t, size_t, int);

template <typename T, typename Q>
void VerifyHuffman(string const&, size_t, Q*, int, size_t*, double*);

// template <typename T, typename Q = uint16_t, typename H = uint32_t>
template <typename T, typename Q, typename H>
void Compress(std::string&, size_t*, double*, size_t&, size_t&, size_t&, size_t&, argpack*);

// template <typename T, typename Q = uint16_t, typename H = uint32_t>
template <typename T, typename Q, typename H>
void Decompress(std::string& fi, size_t*, double*, size_t&, size_t&, size_t&, size_t&, argpack*);

}  // namespace workflow

}  // namespace cuSZ

#endif
