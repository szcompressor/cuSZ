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
#include "../component/spcodec_vec.hh"
#include "../kernel/launch_spv.cuh"

#include "utils/cuda_err.cuh"
// #include "utils/cuda_mem.cuh"
// #include "utils/format.hh"
// #include "utils/io.hh"
// #include "utils/strhelper.hh"
#include "utils/timer.hh"

#define SPVEC_ALLOCDEV(VAR, SYM)                           \
    CHECK_CUDA(hipMalloc(&d_##VAR, rte.nbyte[RTE::SYM])); \
    CHECK_CUDA(hipMemset(d_##VAR, 0x0, rte.nbyte[RTE::SYM]));

#define SPVEC_FREEDEV(VAR)             \
    if (d_##VAR) {                     \
        CHECK_CUDA(hipFree(d_##VAR)); \
        d_##VAR = nullptr;             \
    }

#define SPVEC_D2DCPY(VAR, FIELD)                                                                       \
    {                                                                                                  \
        auto dst = d_spfmt + header.entry[Header::FIELD];                                              \
        auto src = reinterpret_cast<BYTE*>(d_##VAR);                                                   \
        CHECK_CUDA(hipMemcpyAsync(dst, src, nbyte[Header::FIELD], hipMemcpyDeviceToDevice, stream)); \
    }

namespace cusz {

template <typename T, typename M>
SpcodecVec<T, M>::impl::~impl()
{
    SPVEC_FREEDEV(spfmt);
    SPVEC_FREEDEV(idx);
    SPVEC_FREEDEV(val);
}

// public methods

template <typename T, typename M>
void SpcodecVec<T, M>::impl::init(size_t const len, int density_factor, bool dbg_print)
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
void SpcodecVec<T, M>::impl::encode(
    T*           in,
    size_t const in_len,
    BYTE*&       out,
    size_t&      out_len,
    hipStream_t stream,
    bool         dbg_print)
{
    Header header;

    launch_spv_gather<T, M>(in, in_len, this->d_val, this->d_idx, rte.nnz, milliseconds, stream);

    subfile_collect(header, in_len, stream, dbg_print);
    out     = d_spfmt;
    out_len = header.subfile_size();
}

template <typename T, typename M>
void SpcodecVec<T, M>::impl::decode(BYTE* coded, T* decoded, hipStream_t stream)
{
    header_t header;
    CHECK_CUDA(hipMemcpyAsync(&header, coded, sizeof(header), hipMemcpyDeviceToHost, stream));

#define ACCESSOR(SYM, TYPE) reinterpret_cast<TYPE*>(coded + header.entry[Header::SYM])
    auto d_idx = ACCESSOR(IDX, uint32_t);
    auto d_val = ACCESSOR(VAL, T);
#undef ACCESSOR

    launch_spv_scatter<T, M>(d_val, d_idx, header.nnz, decoded, milliseconds, stream);
}

template <typename T, typename M>
void SpcodecVec<T, M>::impl::clear_buffer()
{
    hipMemset(d_spfmt, 0x0, rte.nbyte[RTE::SPFMT]);
    hipMemset(d_idx, 0x0, rte.nbyte[RTE::IDX]);
    hipMemset(d_val, 0x0, rte.nbyte[RTE::VAL]);
}

// getter
template <typename T, typename M>
float SpcodecVec<T, M>::impl::get_time_elapsed() const
{
    return milliseconds;
}

// helper

template <typename T, typename M>
void SpcodecVec<T, M>::impl::subfile_collect(Header& header, size_t len, hipStream_t stream, bool dbg_print)
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

    CHECK_CUDA(hipMemcpyAsync(d_spfmt, &header, sizeof(header), hipMemcpyHostToDevice, stream));

    /* debug */ CHECK_CUDA(hipStreamSynchronize(stream));

    SPVEC_D2DCPY(idx, IDX)
    SPVEC_D2DCPY(val, VAL)

    /* debug */ CHECK_CUDA(hipStreamSynchronize(stream));
}

}  // namespace cusz

#endif
