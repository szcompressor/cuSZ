/**
 * @file huffman_coarse.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-12-17
 * (created) 2020-04-24 (rev1) 2021-09-05 (rev2) 2021-12-29
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * @copyright (C) 2021 by Washington State University, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef CUSZ_COMPONENT_HUFFMAN_COARSE_CUH
#define CUSZ_COMPONENT_HUFFMAN_COARSE_CUH

#include <cuda.h>
#include <clocale>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <type_traits>
using std::cout;

#include "../common/definition.hh"
#include "../common/type_traits.hh"
#include "../kernel/codec_huffman.cuh"
#include "../kernel/hist.cuh"
#include "../utils.hh"
#include "codecs.hh"
#include "huffman_parbook.cuh"

/******************************************************************************
                            macros for shorthand writing
 ******************************************************************************/

#define EXPORT_NBYTE(FIELD) nbyte[Header::FIELD] = rte.nbyte[RTE::FIELD];

#define DEVICE2DEVICE_COPY(VAR, FIELD)                                            \
    {                                                                             \
        constexpr auto D2D = cudaMemcpyDeviceToDevice;                            \
        auto           dst = d_compressed + header.entry[Header::FIELD];          \
        auto           src = reinterpret_cast<BYTE*>(d_##VAR);                    \
        CHECK_CUDA(cudaMemcpyAsync(dst, src, nbyte[Header::FIELD], D2D, stream)); \
    }

#define ACCESSOR(SYM, TYPE) reinterpret_cast<TYPE*>(in_compressed + header.entry[Header::SYM])

#define HC_ALLOCHOST(VAR, SYM)                     \
    cudaMallocHost(&h_##VAR, rte.nbyte[RTE::SYM]); \
    memset(h_##VAR, 0x0, rte.nbyte[RTE::SYM]);

#define HC_ALLOCDEV(VAR, SYM)                  \
    cudaMalloc(&d_##VAR, rte.nbyte[RTE::SYM]); \
    cudaMemset(d_##VAR, 0x0, rte.nbyte[RTE::SYM]);

#define HC_FREEHOST(VAR)       \
    if (h_##VAR) {             \
        cudaFreeHost(h_##VAR); \
        h_##VAR = nullptr;     \
    }

#define HC_FREEDEV(VAR)    \
    if (d_##VAR) {         \
        cudaFree(d_##VAR); \
        d_##VAR = nullptr; \
    }

/******************************************************************************
                                kernel definition
 ******************************************************************************/

namespace cusz {

template <typename Huff, typename Meta>
__global__ void
huffman_coarse_concatenate(Huff* gapped, Meta* par_entry, Meta* par_ncell, int const cfg_sublen, Huff* non_gapped)
{
    auto n   = par_ncell[blockIdx.x];
    auto src = gapped + cfg_sublen * blockIdx.x;
    auto dst = non_gapped + par_entry[blockIdx.x];

    for (auto i = threadIdx.x; i < n; i += blockDim.x) {  // block-stride
        dst[i] = src[i];
    }
}

}  // namespace cusz

/******************************************************************************
                                class definition
 ******************************************************************************/

#define TEMPLATE_TYPE template <typename T, typename H, typename M>
#define IMPL HuffmanCoarse<T, H, M>::impl

namespace cusz {

TEMPLATE_TYPE
IMPL::~impl()
{
    HC_FREEDEV(tmp);
    HC_FREEDEV(book);
    HC_FREEDEV(revbook);
    HC_FREEDEV(par_nbit);
    HC_FREEDEV(par_ncell);
    HC_FREEDEV(par_entry);
    HC_FREEDEV(bitstream);

    HC_FREEHOST(book);
    HC_FREEHOST(revbook);
    HC_FREEHOST(par_nbit);
    HC_FREEHOST(par_ncell);
    HC_FREEHOST(par_entry);
}

TEMPLATE_TYPE
IMPL::impl() = default;

//------------------------------------------------------------------------------

TEMPLATE_TYPE
void IMPL::init(size_t const in_uncompressed_len, int const booklen, int const pardeg, bool dbg_print)
{
    auto max_compressed_bytes = [&]() { return in_uncompressed_len / 2 * sizeof(H); };

    auto debug = [&]() {
        setlocale(LC_NUMERIC, "");
        printf("\nHuffmanCoarse<T, H, M>::init() debugging:\n");
        printf("CUdeviceptr nbyte: %d\n", (int)sizeof(CUdeviceptr));
        dbg_println("TMP", d_tmp, RTE::TMP);
        dbg_println("BOOK", d_book, RTE::BOOK);
        dbg_println("REVBOOK", d_revbook, RTE::REVBOOK);
        dbg_println("PAR_NBIT", d_par_nbit, RTE::PAR_NBIT);
        dbg_println("PAR_NCELL", d_par_ncell, RTE::PAR_NCELL);
        dbg_println("BITSTREAM", d_bitstream, RTE::BITSTREAM);
        printf("\n");
    };

    memset(rte.nbyte, 0, sizeof(uint32_t) * RTE::END);
    // memset(rte.entry, 0, sizeof(uint32_t) * (RTE::END + 1));

    rte.nbyte[RTE::TMP]       = sizeof(H) * in_uncompressed_len;
    rte.nbyte[RTE::BOOK]      = sizeof(H) * booklen;
    rte.nbyte[RTE::REVBOOK]   = get_revbook_nbyte(booklen);
    rte.nbyte[RTE::PAR_NBIT]  = sizeof(M) * pardeg;
    rte.nbyte[RTE::PAR_NCELL] = sizeof(M) * pardeg;
    rte.nbyte[RTE::PAR_ENTRY] = sizeof(M) * pardeg;
    rte.nbyte[RTE::BITSTREAM] = max_compressed_bytes();

    HC_ALLOCDEV(tmp, TMP);
    HC_ALLOCDEV(book, BOOK);
    HC_ALLOCDEV(revbook, REVBOOK);
    HC_ALLOCDEV(par_nbit, PAR_NBIT);
    HC_ALLOCDEV(par_ncell, PAR_NCELL);
    HC_ALLOCDEV(par_entry, PAR_ENTRY);
    HC_ALLOCDEV(bitstream, BITSTREAM);

    // standalone definition for output
    d_compressed = reinterpret_cast<BYTE*>(d_tmp);

    HC_ALLOCHOST(book, BOOK);
    HC_ALLOCHOST(revbook, REVBOOK);
    HC_ALLOCHOST(par_nbit, PAR_NBIT);
    HC_ALLOCHOST(par_ncell, PAR_NCELL);
    HC_ALLOCHOST(par_entry, PAR_ENTRY);

    if (dbg_print) debug();
}

TEMPLATE_TYPE
void IMPL::build_codebook(cusz::FREQ* freq, int const booklen, cudaStream_t stream)
{
    // This is end-to-end time for parbook.
    cuda_timer_t t;
    t.timer_start(stream);
    kernel_wrapper::par_get_codebook<T, H>(freq, booklen, d_book, d_revbook, stream);
    t.timer_end(stream);
    cudaStreamSynchronize(stream);

    time_book = t.get_time_elapsed();
}

TEMPLATE_TYPE
void IMPL::encode(
    T*           in_uncompressed,
    size_t const in_uncompressed_len,
    cusz::FREQ*  d_freq,
    int const    cfg_booklen,
    int const    cfg_sublen,
    int const    cfg_pardeg,
    BYTE*&       out_compressed,
    size_t&      out_compressed_len,
    cudaStream_t stream)
{
    cuda_timer_t t;
    time_lossless = 0;

    auto BARRIER = [&]() {
        if (stream)
            CHECK_CUDA(cudaStreamSynchronize(stream));
        else
            CHECK_CUDA(cudaDeviceSynchronize());
    };

    struct Header header;

    /* phase 1 */
    {
        auto block_dim = HuffmanHelper::BLOCK_DIM_ENCODE;
        auto grid_dim  = ConfigHelper::get_npart(in_uncompressed_len, block_dim);

        int numSMs;
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

        t.timer_start(stream);

        cusz::coarse_par::detail::kernel::huffman_encode_fixedlen_gridstride<T, H>
            <<<8 * numSMs, 256, sizeof(H) * cfg_booklen, stream>>>  //
            (in_uncompressed, in_uncompressed_len, d_book, cfg_booklen, d_tmp);

        t.timer_end(stream);
        BARRIER();

        time_lossless += t.get_time_elapsed();
    }

    /* phase 2 */
    {
        auto block_dim = HuffmanHelper::BLOCK_DIM_DEFLATE;
        auto grid_dim  = ConfigHelper::get_npart(cfg_pardeg, block_dim);

        t.timer_start(stream);

        cusz::coarse_par::detail::kernel::huffman_encode_deflate<H><<<grid_dim, block_dim, 0, stream>>>  //
            (d_tmp, in_uncompressed_len, d_par_nbit, d_par_ncell, cfg_sublen, cfg_pardeg);

        t.timer_end(stream);
        BARRIER();

        time_lossless += t.get_time_elapsed();
    }

    /* phase 3 */
    {
        CHECK_CUDA(cudaMemcpyAsync(h_par_nbit, d_par_nbit, cfg_pardeg * sizeof(M), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaMemcpyAsync(h_par_ncell, d_par_ncell, cfg_pardeg * sizeof(M), cudaMemcpyDeviceToHost, stream));
        BARRIER();

        memcpy(h_par_entry + 1, h_par_ncell, (cfg_pardeg - 1) * sizeof(M));
        for (auto i = 1; i < cfg_pardeg; i++) h_par_entry[i] += h_par_entry[i - 1];  // inclusive scan

        header.total_nbit  = std::accumulate(h_par_nbit, h_par_nbit + cfg_pardeg, (size_t)0);
        header.total_ncell = std::accumulate(h_par_ncell, h_par_ncell + cfg_pardeg, (size_t)0);

        CHECK_CUDA(cudaMemcpyAsync(d_par_entry, h_par_entry, cfg_pardeg * sizeof(M), cudaMemcpyHostToDevice, stream));
        BARRIER();

        // update with the precise BITSTREAM nbyte
        rte.nbyte[RTE::BITSTREAM] = sizeof(H) * header.total_ncell;
    }

    /* phase 4 */
    {
        t.timer_start(stream);
        cusz::huffman_coarse_concatenate<H, M><<<cfg_pardeg, 128, 0, stream>>>  //
            (d_tmp, d_par_entry, d_par_ncell, cfg_sublen, d_bitstream);
        t.timer_end(stream);
        BARRIER();

        time_lossless += t.get_time_elapsed();
    }

    subfile_collect(header, in_uncompressed_len, cfg_booklen, cfg_sublen, cfg_pardeg, stream);

    out_compressed     = d_compressed;
    out_compressed_len = header.subfile_size();
}

TEMPLATE_TYPE
void IMPL::decode(BYTE* in_compressed, T* out_decompressed, cudaStream_t stream, bool header_on_device)
{
    Header header;
    if (header_on_device)
        CHECK_CUDA(cudaMemcpyAsync(&header, in_compressed, sizeof(header), cudaMemcpyDeviceToHost, stream));

    auto d_revbook   = ACCESSOR(REVBOOK, BYTE);
    auto d_par_nbit  = ACCESSOR(PAR_NBIT, M);
    auto d_par_entry = ACCESSOR(PAR_ENTRY, M);
    auto d_bitstream = ACCESSOR(BITSTREAM, H);

    auto const revbook_nbyte = get_revbook_nbyte(header.booklen);
    auto const pardeg        = header.pardeg;
    auto const block_dim     = HuffmanHelper::BLOCK_DIM_DEFLATE;  // = deflating
    auto const grid_dim      = ConfigHelper::get_npart(pardeg, block_dim);

    cuda_timer_t t;
    t.timer_start(stream);
    cusz::coarse_par::detail::kernel::huffman_decode_new<T, H, M><<<grid_dim, block_dim, revbook_nbyte, stream>>>(
        d_bitstream, d_revbook, d_par_nbit, d_par_entry, revbook_nbyte, header.sublen, pardeg, out_decompressed);
    t.timer_end(stream);
    cudaStreamSynchronize(stream);

    time_lossless = t.get_time_elapsed();
}

TEMPLATE_TYPE
void IMPL::clear_buffer()
{
    cudaMemset(d_tmp, 0x0, rte.nbyte[RTE::TMP]);
    cudaMemset(d_book, 0x0, rte.nbyte[RTE::BOOK]);
    cudaMemset(d_revbook, 0x0, rte.nbyte[RTE::REVBOOK]);
    cudaMemset(d_par_nbit, 0x0, rte.nbyte[RTE::PAR_NBIT]);
    cudaMemset(d_par_ncell, 0x0, rte.nbyte[RTE::PAR_NCELL]);
    cudaMemset(d_par_entry, 0x0, rte.nbyte[RTE::PAR_ENTRY]);
    cudaMemset(d_bitstream, 0x0, rte.nbyte[RTE::BITSTREAM]);
}

// private helper
TEMPLATE_TYPE
void IMPL::subfile_collect(
    Header&      header,
    size_t const in_uncompressed_len,
    int const    cfg_booklen,
    int const    cfg_sublen,
    int const    cfg_pardeg,
    cudaStream_t stream)
{
    auto BARRIER = [&]() {
        if (stream)
            CHECK_CUDA(cudaStreamSynchronize(stream));
        else
            CHECK_CUDA(cudaDeviceSynchronize());
    };

    header.header_nbyte     = sizeof(Header);
    header.booklen          = cfg_booklen;
    header.sublen           = cfg_sublen;
    header.pardeg           = cfg_pardeg;
    header.uncompressed_len = in_uncompressed_len;

    MetadataT nbyte[Header::END];
    nbyte[Header::HEADER] = 128;

    EXPORT_NBYTE(REVBOOK)
    EXPORT_NBYTE(PAR_NBIT)
    EXPORT_NBYTE(PAR_ENTRY)
    EXPORT_NBYTE(BITSTREAM)

    header.entry[0] = 0;
    // *.END + 1: need to know the ending position
    for (auto i = 1; i < Header::END + 1; i++) { header.entry[i] = nbyte[i - 1]; }
    for (auto i = 1; i < Header::END + 1; i++) { header.entry[i] += header.entry[i - 1]; }

    // auto debug_header_entry = [&]() {
    //     for (auto i = 0; i < Header::END + 1; i++) printf("%d, header entry: %d\n", i, header.entry[i]);
    // };
    // debug_header_entry();

    CHECK_CUDA(cudaMemcpyAsync(d_compressed, &header, sizeof(header), cudaMemcpyHostToDevice, stream));

    /* debug */ BARRIER();

    DEVICE2DEVICE_COPY(revbook, REVBOOK)
    DEVICE2DEVICE_COPY(par_nbit, PAR_NBIT)
    DEVICE2DEVICE_COPY(par_entry, PAR_ENTRY)
    DEVICE2DEVICE_COPY(bitstream, BITSTREAM)
}

// getter
TEMPLATE_TYPE
float IMPL::get_time_elapsed() const { return milliseconds; }

TEMPLATE_TYPE
float IMPL::get_time_book() const { return time_book; }
TEMPLATE_TYPE
float IMPL::get_time_lossless() const { return time_lossless; }

TEMPLATE_TYPE
H* IMPL::expose_book() const { return d_book; }

TEMPLATE_TYPE
BYTE* IMPL::expose_revbook() const { return d_revbook; }

// TODO this kind of space will be overlapping with quant-codes
TEMPLATE_TYPE
size_t IMPL::get_workspace_nbyte(size_t len) const { return sizeof(H) * len; }

TEMPLATE_TYPE
size_t IMPL::get_max_output_nbyte(size_t len) const { return sizeof(H) * len / 2; }

TEMPLATE_TYPE
size_t IMPL::get_revbook_nbyte(int dict_size) { return sizeof(BOOK) * (2 * CELL_BITWIDTH) + sizeof(SYM) * dict_size; }

TEMPLATE_TYPE
constexpr bool IMPL::can_overlap_input_and_firstphase_encode() { return sizeof(T) == sizeof(H); }

// auxiliary
TEMPLATE_TYPE
void IMPL::dbg_println(const std::string SYM_name, void* VAR, int SYM)
{
    CUdeviceptr pbase0{0};
    size_t      psize0{0};

    cuMemGetAddressRange(&pbase0, &psize0, (CUdeviceptr)VAR);
    printf(
        "%s:\n"
        "\t(supposed) pointer : %p\n"
        "\t(supposed) bytes   : %'9lu\n"
        "\t(queried)  pbase0  : %p\n"
        "\t(queried)  psize0  : %'9lu\n",
        SYM_name.c_str(), (void*)VAR, (size_t)rte.nbyte[SYM], (void*)&pbase0, psize0);
    pbase0 = 0, psize0 = 0;
}

}  // namespace cusz

#undef HC_ALLOCDEV
#undef HC_ALLOCHOST
#undef HC_FREEDEV
#undef HC_FREEHOST
#undef EXPORT_NBYTE
#undef ACCESSOR
#undef DEVICE2DEVICE_COPY

#undef TEMPLATE_TYPE
#undef IMPL

#endif
