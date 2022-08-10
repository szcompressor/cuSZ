/**
 * @file spmat.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-09-28
 * (rev) 2022-01-10
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_SPMAT_CUH
#define CUSZ_SPMAT_CUH

#include <hipsparse.h>

#include "../common.hh"
#include "../component/spcodec.hh"
#include "../kernel/launch_sparse_method.cuh"
#include "../utils.hh"

// clang-format off

// clang-format on

/******************************************************************************
                            macros for shorthand writing
 ******************************************************************************/

#define SPMAT_FREEDEV(VAR)             \
    if (d_##VAR) {                     \
        CHECK_CUDA(hipFree(d_##VAR)); \
        d_##VAR = nullptr;             \
    }

#define SPMAT_D2DCPY(VAR, FIELD)                                                                       \
    {                                                                                                  \
        auto dst = d_csr + header.entry[Header::FIELD];                                                \
        auto src = reinterpret_cast<BYTE*>(d_##VAR);                                                   \
        CHECK_CUDA(hipMemcpyAsync(dst, src, nbyte[Header::FIELD], hipMemcpyDeviceToDevice, stream)); \
    }

#define SPMAT_ALLOCDEV(VAR, SYM)                           \
    CHECK_CUDA(hipMalloc(&d_##VAR, rte.nbyte[RTE::SYM])); \
    CHECK_CUDA(hipMemset(d_##VAR, 0x0, rte.nbyte[RTE::SYM]));

/******************************************************************************
                               class definition
******************************************************************************/

namespace cusz {

template <typename T, typename M>
SpcodecCSR<T, M>::impl::~impl()
{
    SPMAT_FREEDEV(csr);
    SPMAT_FREEDEV(rowptr);
    SPMAT_FREEDEV(colidx);
    SPMAT_FREEDEV(val);
}

// public methods
template <typename T, typename M>
void SpcodecCSR<T, M>::impl::init(size_t const in_uncompressed_len, int density_factor, bool dbg_print)
{
    auto max_compressed_bytes = [&]() { return in_uncompressed_len / density_factor * sizeof(T); };
    auto init_nnz             = [&]() { return in_uncompressed_len / density_factor; };
    auto debug                = [&]() {
        setlocale(LC_NUMERIC, "");

#define PRINT_DBG(VAR) printf("nbyte-%-*s:  %'10u\n", 10, #VAR, rte.nbyte[RTE::VAR]);
        printf("\nCSR11::init() debugging:\n");
        printf("%-*s:  %'10ld\n", 16, "init.nnz", init_nnz());
        PRINT_DBG(CSR);
        PRINT_DBG(ROWPTR);
        PRINT_DBG(COLIDX);
        PRINT_DBG(VAL);
        printf("\n");
#undef PRINT_DBG
    };

    memset(rte.nbyte, 0, sizeof(uint32_t) * RTE::END);

    rte.m   = Reinterpret1DTo2D::get_square_size(in_uncompressed_len);
    rte.nnz = init_nnz();

    rte.nbyte[RTE::CSR]    = max_compressed_bytes();
    rte.nbyte[RTE::ROWPTR] = sizeof(int) * (rte.m + 1);
    rte.nbyte[RTE::COLIDX] = sizeof(int) * init_nnz();
    rte.nbyte[RTE::VAL]    = sizeof(T) * init_nnz();

    SPMAT_ALLOCDEV(csr, CSR);
    SPMAT_ALLOCDEV(rowptr, ROWPTR);
    SPMAT_ALLOCDEV(colidx, COLIDX);
    SPMAT_ALLOCDEV(val, VAL);

    if (dbg_print) debug();
}

template <typename T, typename M>
void SpcodecCSR<T, M>::impl::clear_buffer()
{
    hipMemset(d_csr, 0x0, rte.nbyte[RTE::CSR]);
    hipMemset(d_rowptr, 0x0, rte.nbyte[RTE::ROWPTR]);
    hipMemset(d_colidx, 0x0, rte.nbyte[RTE::COLIDX]);
    hipMemset(d_val, 0x0, rte.nbyte[RTE::VAL]);
}

template <typename T, typename M>
void SpcodecCSR<T, M>::impl::encode(
    T*           in_uncompressed,
    size_t const in_uncompressed_len,
    BYTE*&       out_compressed,
    size_t&      out_compressed_len,
    hipStream_t stream,
    bool         dbg_print)
{
    // cautious!
    Header header;
    rte.ptr_header = &header;

#if CUDART_VERSION >= 11020

    launch_cusparse_gather_cuda11200_onward(
        rte.handle, in_uncompressed, rte.m, rte.m, rte.dnmat, rte.spmat, rte.d_buffer, rte.d_buffer_size, d_rowptr,
        d_colidx, d_val, rte.nnz, milliseconds, stream);

#elif CUDART_VERSION >= 10000

    launch_cusparse_gather_before_cuda11200(
        rte.handle, in_uncompressed, rte.m, rte.m, rte.mat_desc, rte.d_work, rte.lwork_in_bytes, d_rowptr, d_colidx,
        d_val, rte.nnz, milliseconds, stream);

#endif

    subfile_collect(header, in_uncompressed_len, stream, dbg_print);

    out_compressed     = d_csr;
    out_compressed_len = header.subfile_size();
}

template <typename T, typename M>
void SpcodecCSR<T, M>::impl::decode(BYTE* in_compressed, T* out_decompressed, hipStream_t stream)
{
    Header header;
    CHECK_CUDA(hipMemcpyAsync(&header, in_compressed, sizeof(header), hipMemcpyDeviceToHost, stream));

    auto d_rowptr = reinterpret_cast<int*>(in_compressed + header.entry[Header::ROWPTR]);
    auto d_colidx = reinterpret_cast<int*>(in_compressed + header.entry[Header::COLIDX]);
    auto d_val    = reinterpret_cast<T*>(in_compressed + header.entry[Header::VAL]);

#if CUDART_VERSION >= 11020

    launch_cusparse_scatter_cuda11200_onward<T, M>(
        rte.handle, d_rowptr, d_colidx, d_val, rte.m, rte.m, rte.nnz, rte.dnmat, rte.spmat, rte.d_buffer,
        rte.d_buffer_size, out_decompressed, milliseconds, stream);

#elif CUDART_VERSION >= 10000

    launch_cusparse_scatter_before_cuda11200<T, M>(
        rte.handle, d_rowptr, d_colidx, d_val, rte.m, rte.m, rte.nnz, rte.dnmat, rte.spmat, rte.d_work,
        rte.lwork_in_bytes, out_decompressed, milliseconds, stream);

#endif
}

// getter
template <typename T, typename M>
float SpcodecCSR<T, M>::impl::get_time_elapsed() const
{
    return milliseconds;
}

// helper

template <typename T, typename M>
void SpcodecCSR<T, M>::impl::subfile_collect(
    Header&      header,
    size_t       in_uncompressed_len,
    hipStream_t stream,
    bool         dbg_print)
{
    header.header_nbyte     = sizeof(Header);
    header.uncompressed_len = in_uncompressed_len;
    header.nnz              = rte.nnz;
    header.m                = rte.m;

    // update (redundant here)
    rte.nbyte[RTE::COLIDX] = sizeof(int) * rte.nnz;
    rte.nbyte[RTE::VAL]    = sizeof(T) * rte.nnz;

    MetadataT nbyte[Header::END];
    nbyte[Header::HEADER] = 128;
    nbyte[Header::ROWPTR] = rte.nbyte[RTE::ROWPTR];
    nbyte[Header::COLIDX] = rte.nbyte[RTE::COLIDX];
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
        PRINT_ENTRY(ROWPTR);
        PRINT_ENTRY(COLIDX);
        PRINT_ENTRY(VAL);
        PRINT_ENTRY(END);
        printf("\n");
#undef PRINT_ENTRY
    };
    if (dbg_print) debug_header_entry();

    CHECK_CUDA(hipMemcpyAsync(d_csr, &header, sizeof(header), hipMemcpyHostToDevice, stream));

    /* debug */ CHECK_CUDA(hipStreamSynchronize(stream));

    SPMAT_D2DCPY(rowptr, ROWPTR)
    SPMAT_D2DCPY(colidx, COLIDX)
    SPMAT_D2DCPY(val, VAL)

    /* debug */ CHECK_CUDA(hipStreamSynchronize(stream));
}

}  // namespace cusz

#undef SPMAT_D2DCPY
#undef SPMAT_ALLCDEV
#undef SPMAT_FREEDEV

#endif
