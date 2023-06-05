/**
 * @file hf_codecg.cu
 * @author Jiannan Tian
 * @brief kernel wrappers; launching Huffman kernels
 * @version 0.3
 * @date 2022-11-02
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include "detail/hf_codecg.inl"
#include "hf/hf_bookg.hh"
#include "hf/hf_codecg.hh"

template <typename T, typename H, typename M>
void psz::hf_encode_coarse_rev2(
    T*            uncompressed,
    size_t const  len,
    hf_book*      book_desc,
    hf_bitstream* bitstream_desc,
    size_t*       outlen_nbit,
    size_t*       outlen_ncell,
    float*        time_lossless,
    cudaStream_t  stream)
{
    auto div = [](auto whole, auto part) -> uint32_t {
        if (whole == 0) throw std::runtime_error("Dividend is zero.");
        if (part == 0) throw std::runtime_error("Divisor is zero.");
        return (whole - 1) / part + 1;
    };
    static const int BLOCK_DIM_ENCODE  = 256;
    static const int BLOCK_DIM_DEFLATE = 256;

    CREATE_CUDAEVENT_PAIR;

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
        auto block_dim = BLOCK_DIM_ENCODE;
        auto grid_dim  = div(len, block_dim);

        START_CUDAEVENT_RECORDING(stream);

        psz::detail::hf_encode_phase1_fill<T, H>                //
            <<<8 * numSMs, 256, sizeof(H) * booklen, stream>>>  //
            (uncompressed, len, d_book, booklen, d_buffer);

        STOP_CUDAEVENT_RECORDING(stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        float stage_time;
        TIME_ELAPSED_CUDAEVENT(&stage_time);
        if (time_lossless) *time_lossless += stage_time;
    }

    /* phase 2 */
    {
        auto block_dim = BLOCK_DIM_DEFLATE;
        auto grid_dim  = div(pardeg, block_dim);

        START_CUDAEVENT_RECORDING(stream);

        psz::detail::hf_encode_phase2_deflate<H>  //
            <<<grid_dim, block_dim, 0, stream>>>  //
            (d_buffer, len, d_par_nbit, d_par_ncell, sublen, pardeg);

        STOP_CUDAEVENT_RECORDING(stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        float stage_time;
        TIME_ELAPSED_CUDAEVENT(&stage_time);
        if (time_lossless) *time_lossless += stage_time;
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
        START_CUDAEVENT_RECORDING(stream);

        psz::detail::hf_encode_phase4_concatenate<H, M><<<pardeg, 128, 0, stream>>>  //
            (d_buffer, d_par_entry, d_par_ncell, sublen, d_bitstream);

        STOP_CUDAEVENT_RECORDING(stream);

        CHECK_CUDA(cudaStreamSynchronize(stream));

        float stage_time;
        TIME_ELAPSED_CUDAEVENT(&stage_time);
        if (time_lossless) *time_lossless += stage_time;
    }

    /* phase 5: gather out sizes without memcpy */
    {
        if (outlen_nbit) *outlen_nbit = std::accumulate(h_par_nbit, h_par_nbit + pardeg, (size_t)0);
        if (outlen_ncell) *outlen_ncell = std::accumulate(h_par_ncell, h_par_ncell + pardeg, (size_t)0);
    }
}

template <typename T, typename H, typename M>
void psz::hf_decode_coarse(
    H*           d_bitstream,
    uint8_t*     d_revbook,
    int const    revbook_nbyte,
    M*           d_par_nbit,
    M*           d_par_entry,
    int const    sublen,
    int const    pardeg,
    T*           out_decompressed,
    float*       time_lossless,
    cudaStream_t stream)
{
    auto const block_dim = HuffmanHelper::BLOCK_DIM_DEFLATE;  // = deflating
    auto const grid_dim  = ConfigHelper::get_npart(pardeg, block_dim);

    CREATE_CUDAEVENT_PAIR;
    START_CUDAEVENT_RECORDING(stream)

    hf_decode_kernel<T, H, M>                             //
        <<<grid_dim, block_dim, revbook_nbyte, stream>>>  //
        (d_bitstream, d_revbook, d_par_nbit, d_par_entry, revbook_nbyte, sublen, pardeg, out_decompressed);

    STOP_CUDAEVENT_RECORDING(stream)
    cudaStreamSynchronize(stream);

    TIME_ELAPSED_CUDAEVENT(time_lossless);
    DESTROY_CUDAEVENT_PAIR;
}

#define HF_CODEC_INIT(T, H, M)                                                              \
    template void psz::hf_encode_coarse_rev2<T, H, M>(                                      \
        T*, size_t const, hf_book*, hf_bitstream*, size_t*, size_t*, float*, cudaStream_t); \
                                                                                            \
    template void psz::hf_decode_coarse<T, H, M>(                                           \
        H*, uint8_t*, int const, M*, M*, int const, int const, T*, float*, cudaStream_t);

// 23-06-04 restricted to u4 for quantization code

// HF_CODEC_INIT(uint8_t, uint32_t, uint32_t);
// HF_CODEC_INIT(uint16_t, uint32_t, uint32_t);
HF_CODEC_INIT(uint32_t, uint32_t, uint32_t);
// HF_CODEC_INIT(float, uint32_t, uint32_t);
// HF_CODEC_INIT(uint8_t, unsigned long long, uint32_t);
// HF_CODEC_INIT(uint16_t, unsigned long long, uint32_t);
HF_CODEC_INIT(uint32_t, unsigned long long, uint32_t);
// HF_CODEC_INIT(float, unsigned long long, uint32_t);

#undef HFBOOK_INIT
#undef HF_CODEC_INIT
