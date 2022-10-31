/**
 * @file launch_lossless.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-06-13
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_KERNEL_LAUNCH_LOSSLESS_CUH
#define CUSZ_KERNEL_LAUNCH_LOSSLESS_CUH

#include <cuda_runtime.h>
#include <cstdlib>

#include "../hf/hf_struct.h"
#include "../utils/cuda_err.cuh"
#include "../utils/timer.hh"
#include "codec_huffman.cuh"
#include "huffman_parbook.cuh"

template <typename T, typename H, typename M>
void launch_gpu_parallel_build_codebook(
    uint32_t*    freq,
    H*           book,
    int const    booklen,
    uint8_t*     revbook,
    int const    revbook_nbyte,
    float&       time_book,
    cudaStream_t stream)
{
    cuda_timer_t t;
    t.timer_start(stream);

    float placeholder;

    // TODO internal malloc & free takes much time
    kernel_wrapper::parallel_get_codebook<T, H>(freq, book, booklen, revbook, placeholder, stream);
    t.timer_end(stream);
    cudaStreamSynchronize(stream);

    time_book = t.get_time_elapsed();
}

template <typename T, typename H, typename M>
void launch_coarse_grained_Huffman_encoding(
    T*           uncompressed,
    H*           d_internal_coded,
    size_t const len,
    uint32_t*    d_freq,
    H*           d_book,
    int const    booklen,
    H*           d_bitstream,
    M*           d_par_metadata,
    M*           h_par_metadata,
    int const    sublen,
    int const    pardeg,
    int          numSMs,
    uint8_t*&    out_compressed,
    size_t&      out_compressed_len,
    float&       time_lossless,
    cudaStream_t stream)
{
    cuda_timer_t t;

    auto d_par_nbit  = d_par_metadata;
    auto d_par_ncell = d_par_metadata + pardeg;
    auto d_par_entry = d_par_metadata + pardeg * 2;

    auto h_par_nbit  = h_par_metadata;
    auto h_par_ncell = h_par_metadata + pardeg;
    auto h_par_entry = h_par_metadata + pardeg * 2;

    /* phase 1 */
    {
        auto block_dim = HuffmanHelper::BLOCK_DIM_ENCODE;
        auto grid_dim  = ConfigHelper::get_npart(len, block_dim);

        t.timer_start(stream);

        cusz::coarse_grained_Huffman_encode_phase1_fill<T, H>   //
            <<<8 * numSMs, 256, sizeof(H) * booklen, stream>>>  //
            (uncompressed, len, d_book, booklen, d_internal_coded);

        t.timer_end(stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        time_lossless += t.get_time_elapsed();
    }

    /* phase 2 */
    {
        auto block_dim = HuffmanHelper::BLOCK_DIM_DEFLATE;
        auto grid_dim  = ConfigHelper::get_npart(pardeg, block_dim);

        t.timer_start(stream);

        cusz::coarse_grained_Huffman_encode_phase2_deflate<H>  //
            <<<grid_dim, block_dim, 0, stream>>>               //
            (d_internal_coded, len, d_par_nbit, d_par_ncell, sublen, pardeg);

        t.timer_end(stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        time_lossless += t.get_time_elapsed();
    }

    /* phase 3 */
    {
        CHECK_CUDA(cudaMemcpyAsync(h_par_nbit, d_par_nbit, pardeg * sizeof(M), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaMemcpyAsync(h_par_ncell, d_par_ncell, pardeg * sizeof(M), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        memcpy(h_par_entry + 1, h_par_ncell, (pardeg - 1) * sizeof(M));
        for (auto i = 1; i < pardeg; i++) h_par_entry[i] += h_par_entry[i - 1];  // inclusive scan

        CHECK_CUDA(cudaMemcpyAsync(d_par_entry, h_par_entry, pardeg * sizeof(M), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    /* phase 4 */
    {
        t.timer_start(stream);
        cusz::coarse_grained_Huffman_encode_phase4_concatenate<H, M><<<pardeg, 128, 0, stream>>>  //
            (d_internal_coded, d_par_entry, d_par_ncell, sublen, d_bitstream);
        t.timer_end(stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        time_lossless += t.get_time_elapsed();
    }
}

template <typename T, typename H, typename M>
void launch_coarse_grained_Huffman_encoding_rev1(
    T*            uncompressed,
    size_t const  len,
    hf_book*      book_desc,
    hf_bitstream* bitstream_desc,
    uint8_t*&     out_compressed,      // 22-10-12 buggy
    size_t&       out_compressed_len,  // 22-10-12 buggy
    float&        time_lossless,
    cudaStream_t  stream)
{
    cuda_timer_t t;

    H*        d_buffer    = (H*)bitstream_desc->buffer;
    H*        d_bitstream = (H*)bitstream_desc->bitstream;
    H*        d_book      = (H*)book_desc->book;
    int const booklen     = book_desc->booklen;
    int const sublen      = bitstream_desc->sublen;
    int const pardeg      = bitstream_desc->pardeg;
    int const numSMs      = bitstream_desc->numSMs;
    // uint32_t* d_freq      = book_desc->freq;

    auto d_par_nbit  = (M*)bitstream_desc->d_metadata->bits;
    auto d_par_ncell = (M*)bitstream_desc->d_metadata->cells;
    auto d_par_entry = (M*)bitstream_desc->d_metadata->entries;

    auto h_par_nbit  = (M*)bitstream_desc->h_metadata->bits;
    auto h_par_ncell = (M*)bitstream_desc->h_metadata->cells;
    auto h_par_entry = (M*)bitstream_desc->h_metadata->entries;

    /* phase 1 */
    {
        auto block_dim = HuffmanHelper::BLOCK_DIM_ENCODE;
        auto grid_dim  = ConfigHelper::get_npart(len, block_dim);

        t.timer_start(stream);

        cusz::coarse_grained_Huffman_encode_phase1_fill<T, H>   //
            <<<8 * numSMs, 256, sizeof(H) * booklen, stream>>>  //
            (uncompressed, len, d_book, booklen, d_buffer);

        t.timer_end(stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        time_lossless += t.get_time_elapsed();
    }

    /* phase 2 */
    {
        auto block_dim = HuffmanHelper::BLOCK_DIM_DEFLATE;
        auto grid_dim  = ConfigHelper::get_npart(pardeg, block_dim);

        t.timer_start(stream);

        cusz::coarse_grained_Huffman_encode_phase2_deflate<H>  //
            <<<grid_dim, block_dim, 0, stream>>>               //
            (d_buffer, len, d_par_nbit, d_par_ncell, sublen, pardeg);

        t.timer_end(stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        time_lossless += t.get_time_elapsed();
    }

    /* phase 3 */
    {
        CHECK_CUDA(cudaMemcpyAsync(h_par_nbit, d_par_nbit, pardeg * sizeof(M), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaMemcpyAsync(h_par_ncell, d_par_ncell, pardeg * sizeof(M), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        memcpy(h_par_entry + 1, h_par_ncell, (pardeg - 1) * sizeof(M));
        for (auto i = 1; i < pardeg; i++) h_par_entry[i] += h_par_entry[i - 1];  // inclusive scan

        CHECK_CUDA(cudaMemcpyAsync(d_par_entry, h_par_entry, pardeg * sizeof(M), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    /* phase 4 */
    {
        t.timer_start(stream);
        cusz::coarse_grained_Huffman_encode_phase4_concatenate<H, M><<<pardeg, 128, 0, stream>>>  //
            (d_buffer, d_par_entry, d_par_ncell, sublen, d_bitstream);
        t.timer_end(stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        time_lossless += t.get_time_elapsed();
    }
}

template <typename T, typename H, typename M>
void launch_coarse_grained_Huffman_decoding(
    H*           d_bitstream,
    uint8_t*     d_revbook,
    int const    revbook_nbyte,
    M*           d_par_nbit,
    M*           d_par_entry,
    int const    sublen,
    int const    pardeg,
    T*           out_decompressed,
    float&       time_lossless,
    cudaStream_t stream)
{
    auto const block_dim = HuffmanHelper::BLOCK_DIM_DEFLATE;  // = deflating
    auto const grid_dim  = ConfigHelper::get_npart(pardeg, block_dim);

    cuda_timer_t t;
    t.timer_start(stream);
    cusz::huffman_decode<T, H, M>                         //
        <<<grid_dim, block_dim, revbook_nbyte, stream>>>  //
        (d_bitstream, d_revbook, d_par_nbit, d_par_entry, revbook_nbyte, sublen, pardeg, out_decompressed);
    t.timer_end(stream);
    cudaStreamSynchronize(stream);

    time_lossless = t.get_time_elapsed();
}

#endif
