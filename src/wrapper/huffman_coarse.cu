/**
 * @file huffman_coarse.cu
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

#include "../common/type_traits.hh"
#include "../kernel/codec_huffman.cuh"
#include "../kernel/hist.cuh"
#include "../utils.hh"
#include "huffman_coarse.cuh"
#include "huffman_parbook.cuh"

#define nworker blockDim.x

namespace cusz {

template <typename Huff, typename Meta>
__global__ void huffman_coarse_concatenate(
    Huff*     tmp_encspace,
    Meta*     par_entry,
    Meta*     par_ncell,
    int const cfg_sublen,
    Huff*     out_bitstream)
{
    auto       ncell      = par_ncell[blockIdx.x];
    auto       entry      = par_entry[blockIdx.x];
    auto       output_dst = cfg_sublen * blockIdx.x;
    auto const R          = (ncell + nworker - 1) / nworker;

    for (auto i = 0; i < R; i++) {
        auto x = threadIdx.x + i * nworker;
        if (x < ncell) *(out_bitstream + entry + x) = *(tmp_encspace + output_dst + x);
        __syncthreads();
    }
}

}  // namespace cusz

template <typename T, typename H, typename M>
void cusz::HuffmanCoarse<T, H, M>::inspect(
    cusz::FREQ*  tmp_freq,
    H*           tmp_book,
    T*           in_uncompressed,
    size_t const in_uncompressed_len,
    int const    cfg_booklen,
    BYTE*        out_revbook,
    cudaStream_t stream)
{
    kernel_wrapper::get_frequency<T>(in_uncompressed, in_uncompressed_len, tmp_freq, cfg_booklen, time_hist, stream);

    // This is end-to-end time for parbook.
    cuda_timer_t t;
    t.timer_start(stream);
    kernel_wrapper::par_get_codebook<T, H>(tmp_freq, cfg_booklen, tmp_book, out_revbook, stream);
    t.timer_end(stream);
    time_book = t.get_time_elapsed();
    cudaStreamSynchronize(stream);
}

template <typename T, typename H, typename M>
void cusz::HuffmanCoarse<T, H, M>::encode_phase1(
    H*           tmp_book,
    H*           tmp_encspace,
    T*           in_uncompressed,
    size_t const in_uncompressed_len,
    int const    cfg_booklen,
    cudaStream_t stream)
{
    auto block_dim = HuffmanHelper::BLOCK_DIM_ENCODE;
    auto grid_dim  = ConfigHelper::get_npart(in_uncompressed_len, block_dim);

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    cuda_timer_t t;
    t.timer_start(stream);

    cusz::coarse_par::detail::kernel::huffman_encode_fixedlen_gridstride<T, H>
        <<<8 * numSMs, 256, sizeof(H) * cfg_booklen, stream>>>  //
        (in_uncompressed, in_uncompressed_len, tmp_book, cfg_booklen, tmp_encspace);

    t.timer_end(stream);
    time_lossless += t.get_time_elapsed();
    cudaStreamSynchronize(stream);
}

template <typename T, typename H, typename M>
void cusz::HuffmanCoarse<T, H, M>::encode_phase2(
    H*           tmp_encspace,
    size_t const in_uncompressed_len,
    int const    cfg_sublen,
    int const    cfg_pardeg,
    M*           par_nbit,
    M*           par_ncell,
    cudaStream_t stream)
{
    auto block_dim = HuffmanHelper::BLOCK_DIM_DEFLATE;
    auto grid_dim  = ConfigHelper::get_npart(cfg_pardeg, block_dim);

    cuda_timer_t t;
    t.timer_start(stream);

    cusz::coarse_par::detail::kernel::huffman_encode_deflate<H><<<grid_dim, block_dim, 0, stream>>>  //
        (tmp_encspace, in_uncompressed_len, par_nbit, par_ncell, cfg_sublen, cfg_pardeg);

    t.timer_end(stream);
    time_lossless += t.get_time_elapsed();
    cudaStreamSynchronize(stream);
}

template <typename T, typename H, typename M>
void cusz::HuffmanCoarse<T, H, M>::encode_phase3(
    M*           in_meta_deviceview,
    int const    cfg_pardeg,
    M*           out_meta_hostview,
    size_t&      out_total_nbit,
    size_t&      out_total_ncell,
    cudaStream_t stream)
{
    auto d_par_nbit  = in_meta_deviceview;
    auto d_par_ncell = in_meta_deviceview + cfg_pardeg;
    auto d_par_entry = in_meta_deviceview + cfg_pardeg * 2;

    // TODO change order
    auto h_par_ncell = out_meta_hostview;
    auto h_par_nbit  = out_meta_hostview + cfg_pardeg;
    auto h_par_entry = out_meta_hostview + cfg_pardeg * 2;

    cudaMemcpyAsync(h_par_nbit, d_par_nbit, cfg_pardeg * sizeof(M), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_par_ncell, d_par_ncell, cfg_pardeg * sizeof(M), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    memcpy(h_par_entry + 1, h_par_ncell, (cfg_pardeg - 1) * sizeof(M));
    for (auto i = 1; i < cfg_pardeg; i++) h_par_entry[i] += h_par_entry[i - 1];  // inclusive scan

    out_total_nbit  = std::accumulate(h_par_nbit, h_par_nbit + cfg_pardeg, (size_t)0);
    out_total_ncell = std::accumulate(h_par_ncell, h_par_ncell + cfg_pardeg, (size_t)0);

    cudaMemcpyAsync(d_par_entry, h_par_entry, cfg_pardeg * sizeof(M), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
}

template <typename T, typename H, typename M>
void cusz::HuffmanCoarse<T, H, M>::encode_phase4(
    H*           tmp_encspace,
    size_t const in_uncompressed_len,
    int const    cfg_sublen,
    int const    cfg_pardeg,
    M*           out_compressed_meta,
    H*           out_compressed,
    cudaStream_t stream)
{
    auto par_ncell = out_compressed_meta + cfg_pardeg;
    auto par_entry = out_compressed_meta + cfg_pardeg * 2;

    cuda_timer_t t;
    t.timer_start(stream);
    cusz::huffman_coarse_concatenate<H, M><<<cfg_pardeg, 128, 0, stream>>>  //
        (tmp_encspace, par_entry, par_ncell, cfg_sublen, out_compressed);
    t.timer_end(stream);
    time_lossless += t.get_time_elapsed();
    cudaStreamSynchronize(stream);
}

template <typename T, typename H, typename M>
void cusz::HuffmanCoarse<T, H, M>::encode(
    cusz::FREQ*  tmp_freq,
    H*           tmp_book,
    H*           tmp_encspace,
    T*           in_uncompressed,
    size_t const in_uncompressed_len,
    int const    cfg_booklen,
    int const    cfg_sublen,
    BYTE*        out_revbook,
    Capsule<H>&  out_compressed,
    Capsule<M>&  out_compressed_meta,
    size_t&      out_total_nbit,
    size_t&      out_total_ncell,
    cudaStream_t stream)
{
    inspect(tmp_freq, tmp_book, in_uncompressed, in_uncompressed_len, cfg_booklen, out_revbook, stream);

    auto const cfg_pardeg = ConfigHelper::get_npart(in_uncompressed_len, cfg_sublen);

    auto par_nbit  = out_compressed_meta.dptr;
    auto par_ncell = out_compressed_meta.dptr + cfg_pardeg;
    auto par_entry = out_compressed_meta.dptr + cfg_pardeg * 2;

    encode_phase1(tmp_book, tmp_encspace, in_uncompressed, in_uncompressed_len, cfg_booklen, stream);

    encode_phase2(tmp_encspace, in_uncompressed_len, cfg_sublen, cfg_pardeg, par_nbit, par_ncell, stream);

    encode_phase3(
        out_compressed_meta.dptr, cfg_pardeg, out_compressed_meta.hptr, out_total_nbit, out_total_ncell, stream);

    // update with the exact length
    out_compressed.set_len(out_total_ncell);

    encode_phase4(
        tmp_encspace, in_uncompressed_len, cfg_sublen, cfg_pardeg, out_compressed_meta.dptr, out_compressed.dptr,
        stream);
}

template <typename T, typename H, typename M>
void cusz::HuffmanCoarse<T, H, M>::decode(
    H*           in_compressed,
    M*           in_compressed_meta,
    BYTE*        in_revbook,
    size_t const in_uncompressed_len,
    int const    cfg_booklen,
    int const    cfg_sublen,
    T*           out_uncompressed,
    cudaStream_t stream)
{
    auto const pardeg        = ConfigHelper::get_npart(in_uncompressed_len, cfg_sublen);
    auto       revbook_nbyte = get_revbook_nbyte(cfg_booklen);

    auto block_dim = HuffmanHelper::BLOCK_DIM_DEFLATE;  // = deflating
    auto grid_dim  = ConfigHelper::get_npart(pardeg, block_dim);

    cuda_timer_t t;
    t.timer_start(stream);
    cusz::coarse_par::detail::kernel::huffman_decode<T, H, M><<<grid_dim, block_dim, revbook_nbyte, stream>>>(  //
        in_compressed, in_compressed_meta, in_revbook, revbook_nbyte, cfg_sublen, pardeg, out_uncompressed);
    t.timer_end(stream);
    milliseconds = t.get_time_elapsed();
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

#define HUFFCOARSE(E, H, M) \
    template class cusz::HuffmanCoarse<ErrCtrlTrait<E>::type, HuffTrait<H>::type, MetadataTrait<M>::type>;

HUFFCOARSE(2, 4, 4)
HUFFCOARSE(2, 4, 8)
HUFFCOARSE(2, 8, 4)
HUFFCOARSE(2, 8, 8)

HUFFCOARSE(4, 4, 4)
HUFFCOARSE(4, 4, 8)
HUFFCOARSE(4, 8, 4)
HUFFCOARSE(4, 8, 8)
