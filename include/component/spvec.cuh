/**
 * @file spvec.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-03-01
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_COMPONENT_SPVEC_CUH
#define CUSZ_COMPONENT_SPVEC_CUH

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include "../common.hh"
#include "../utils.hh"
#include "spcodecs.hh"

#define SPVEC_ALLOCDEV(VAR, SYM)                           \
    CHECK_CUDA(cudaMalloc(&d_##VAR, rte.nbyte[RTE::SYM])); \
    CHECK_CUDA(cudaMemset(d_##VAR, 0x0, rte.nbyte[RTE::SYM]));

#define SPVEC_FREEDEV(VAR)             \
    if (d_##VAR) {                     \
        CHECK_CUDA(cudaFree(d_##VAR)); \
        d_##VAR = nullptr;             \
    }

#define SPVEC_D2DCPY(VAR, FIELD)                                                                       \
    {                                                                                                  \
        auto dst = d_spfmt + header.entry[Header::FIELD];                                              \
        auto src = reinterpret_cast<BYTE*>(d_##VAR);                                                   \
        CHECK_CUDA(cudaMemcpyAsync(dst, src, nbyte[Header::FIELD], cudaMemcpyDeviceToDevice, stream)); \
    }

namespace cusz {

template <typename T, typename M>
struct api::SpCodecVec<T, M>::impl::Header {
    static const int HEADER = 0;
    static const int IDX    = 1;
    static const int VAL    = 2;
    static const int END    = 3;

    int       header_nbyte : 16;
    size_t    uncompressed_len;
    int       nnz;
    MetadataT entry[END + 1];

    MetadataT subfile_size() const { return entry[END]; }
};

template <typename T, typename M>
struct api::SpCodecVec<T, M>::impl::runtime_encode_helper {
    static const int SPFMT = 0;
    static const int IDX   = 1;
    static const int VAL   = 2;
    static const int END   = 3;

    uint32_t nbyte[END];
    int      nnz{0};
};

template <typename T, typename M>
api::SpCodecVec<T, M>::impl::~impl()
{
    SPVEC_FREEDEV(spfmt);
    SPVEC_FREEDEV(idx);
    SPVEC_FREEDEV(val);
}

template <typename T, typename M>
float api::SpCodecVec<T, M>::impl::get_time_elapsed() const
{
    return milliseconds;
}

// template <typename T, typename M>
// MetadataT* api::SpCodecVec<T,M>::impl::expose_idx() const
// {
//     return d_idx;
// }

// template <typename T, typename M>
// T* api::SpCodecVec<T,M>::impl::expose_val() const
// {
//     return d_val;
// }

template <typename T, typename M>
void api::SpCodecVec<T, M>::impl::init(size_t const len, int density_factor, bool dbg_print)
{
    auto max_bytes = [&]() { return len / density_factor * sizeof(T); };
    auto init_nnz  = [&]() { return len / density_factor; };

    memset(rte.nbyte, 0, sizeof(uint32_t) * RTE::END);
    rte.nnz = init_nnz();

    rte.nbyte[RTE::SPFMT] = max_bytes();
    rte.nbyte[RTE::IDX]   = rte.nnz * sizeof(int);
    rte.nbyte[RTE::VAL]   = rte.nnz * sizeof(T);

    SPVEC_ALLOCDEV(spfmt, SPFMT);
    SPVEC_ALLOCDEV(idx, IDX);
    SPVEC_ALLOCDEV(val, VAL);

    // if (dbg_print) debug();
}

template <typename T, typename M>
void api::SpCodecVec<T, M>::impl::subfile_collect(Header& header, size_t len, cudaStream_t stream, bool dbg_print)
{
    header.header_nbyte     = sizeof(Header);
    header.uncompressed_len = len;
    header.nnz              = rte.nnz;

    // update (redundant here)
    rte.nbyte[RTE::IDX] = sizeof(int) * rte.nnz;
    rte.nbyte[RTE::VAL] = sizeof(T) * rte.nnz;

    MetadataT nbyte[Header::END];
    nbyte[Header::HEADER] = 128;
    nbyte[Header::IDX]    = rte.nbyte[RTE::IDX];
    nbyte[Header::VAL]    = rte.nbyte[RTE::VAL];

    header.entry[0] = 0;
    // *.END + 1; need to knwo the ending position
    for (auto i = 1; i < Header::END + 1; i++) { header.entry[i] = nbyte[i - 1]; }
    for (auto i = 1; i < Header::END + 1; i++) { header.entry[i] += header.entry[i - 1]; }

    auto debug_header_entry = [&]() {
        printf("\nCSR11::subfile_collect() debugging:\n");
        printf("%-*s:  %'10ld\n", 16, "final.nnz", rte.nnz);
        printf("  ENTRIES\n");

#define PRINT_ENTRY(VAR) printf("%d %-*s:  %'10u\n", (int)Header::VAR, 14, #VAR, header.entry[Header::VAR]);
        PRINT_ENTRY(HEADER);
        PRINT_ENTRY(IDX);
        PRINT_ENTRY(VAL);
        PRINT_ENTRY(END);
        printf("\n");
#undef PRINT_ENTRY
    };
    if (dbg_print) debug_header_entry();

    CHECK_CUDA(cudaMemcpyAsync(d_spfmt, &header, sizeof(header), cudaMemcpyHostToDevice, stream));

    /* debug */ CHECK_CUDA(cudaStreamSynchronize(stream));

    SPVEC_D2DCPY(idx, IDX)
    SPVEC_D2DCPY(val, VAL)

    /* debug */ CHECK_CUDA(cudaStreamSynchronize(stream));
}

template <typename T, typename M>
void api::SpCodecVec<T, M>::impl::encode(
    T*           in,
    size_t const in_len,
    BYTE*&       out,
    size_t&      out_len,
    cudaStream_t stream,
    bool         dbg_print)
{
    Header header;

    using thrust::placeholders::_1;

    thrust::cuda::par.on(stream);
    thrust::counting_iterator<int> zero(0);

    cuda_timer_t t;
    t.timer_start(stream);

    // find out the indices
    rte.nnz = thrust::copy_if(thrust::device, zero, zero + in_len, in, d_idx, _1 != 0) - d_idx;

    // fetch corresponding values
    thrust::copy(
        thrust::device, thrust::make_permutation_iterator(in, d_idx),
        thrust::make_permutation_iterator(in + rte.nnz, d_idx + rte.nnz), d_val);

    t.timer_end(stream);
    milliseconds = t.get_time_elapsed();

    subfile_collect(header, in_len, stream, dbg_print);
    out     = d_spfmt;
    out_len = header.subfile_size();
}

template <typename T, typename M>
void api::SpCodecVec<T, M>::impl::decode(BYTE* coded, T* decoded, cudaStream_t stream)
{
    header_t header;
    CHECK_CUDA(cudaMemcpyAsync(&header, coded, sizeof(header), cudaMemcpyDeviceToHost, stream));

#define ACCESSOR(SYM, TYPE) reinterpret_cast<TYPE*>(coded + header.entry[Header::SYM])
    auto d_idx = ACCESSOR(IDX, int);
    auto d_val = ACCESSOR(VAL, T);
#undef ACCESSOR
    auto nnz = header.nnz;

    thrust::cuda::par.on(stream);
    cuda_timer_t t;
    t.timer_start(stream);
    thrust::scatter(thrust::device, d_val, d_val + nnz, d_idx, decoded);
    t.timer_end(stream);
    milliseconds = t.get_time_elapsed();
}

template <typename T, typename M>
void api::SpCodecVec<T, M>::impl::clear_buffer()
{
    cudaMemset(d_spfmt, 0x0, rte.nbyte[RTE::SPFMT]);
    cudaMemset(d_idx, 0x0, rte.nbyte[RTE::IDX]);
    cudaMemset(d_val, 0x0, rte.nbyte[RTE::VAL]);
}

}  // namespace cusz

#endif