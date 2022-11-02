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
void asz::hf_buildbook_g(
    uint32_t*    freq,
    H*           book,
    int const    booklen,
    uint8_t*     revbook,
    int const    revbook_nbyte,
    float*       time_book,
    cudaStream_t stream)
{
    CREATE_CUDAEVENT_PAIR;
    START_CUDAEVENT_RECORDING(stream);

    float end_to_end;
    // TODO internal malloc & free takes much time
    asz::parallel_get_codebook<T, H>(freq, booklen, book, revbook, revbook_nbyte, &end_to_end, stream);

    STOP_CUDAEVENT_RECORDING(stream);
    TIME_ELAPSED_CUDAEVENT(time_book);
    DESTROY_CUDAEVENT_PAIR;
}

template <typename T, typename H, typename M>
void asz::hf_encode_coarse(
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
    auto d_par_nbit  = d_par_metadata;
    auto d_par_ncell = d_par_metadata + pardeg;
    auto d_par_entry = d_par_metadata + pardeg * 2;

    auto h_par_nbit  = h_par_metadata;
    auto h_par_ncell = h_par_metadata + pardeg;
    auto h_par_entry = h_par_metadata + pardeg * 2;

    CREATE_CUDAEVENT_PAIR;

    /* phase 1 */
    {
        auto block_dim = HuffmanHelper::BLOCK_DIM_ENCODE;
        auto grid_dim  = ConfigHelper::get_npart(len, block_dim);

        START_CUDAEVENT_RECORDING(stream);

        asz::detail::hf_encode_phase1_fill<T, H>                //
            <<<8 * numSMs, 256, sizeof(H) * booklen, stream>>>  //
            (uncompressed, len, d_book, booklen, d_internal_coded);

        STOP_CUDAEVENT_RECORDING(stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        float stage_time;
        TIME_ELAPSED_CUDAEVENT(&stage_time);
        time_lossless += stage_time;
    }

    /* phase 2 */
    {
        auto block_dim = HuffmanHelper::BLOCK_DIM_DEFLATE;
        auto grid_dim  = ConfigHelper::get_npart(pardeg, block_dim);

        START_CUDAEVENT_RECORDING(stream);

        asz::detail::hf_encode_phase2_deflate<H>  //
            <<<grid_dim, block_dim, 0, stream>>>  //
            (d_internal_coded, len, d_par_nbit, d_par_ncell, sublen, pardeg);

        STOP_CUDAEVENT_RECORDING(stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        float stage_time;
        TIME_ELAPSED_CUDAEVENT(&stage_time);
        time_lossless += stage_time;
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

        asz::detail::hf_encode_phase4_concatenate<H, M><<<pardeg, 128, 0, stream>>>  //
            (d_internal_coded, d_par_entry, d_par_ncell, sublen, d_bitstream);

        STOP_CUDAEVENT_RECORDING(stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        float stage_time;
        TIME_ELAPSED_CUDAEVENT(&stage_time);
        time_lossless += stage_time;
    }

    DESTROY_CUDAEVENT_PAIR;
}

template <typename T, typename H, typename M>
void asz::hf_encode_coarse_rev1(
    T*            uncompressed,
    size_t const  len,
    hf_book*      book_desc,
    hf_bitstream* bitstream_desc,
    uint8_t*&     out_compressed,      // 22-10-12 buggy
    size_t&       out_compressed_len,  // 22-10-12 buggy
    float&        time_lossless,
    cudaStream_t  stream)
{
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
        auto block_dim = HuffmanHelper::BLOCK_DIM_ENCODE;
        auto grid_dim  = ConfigHelper::get_npart(len, block_dim);

        START_CUDAEVENT_RECORDING(stream);

        asz::detail::hf_encode_phase1_fill<T, H>                //
            <<<8 * numSMs, 256, sizeof(H) * booklen, stream>>>  //
            (uncompressed, len, d_book, booklen, d_buffer);

        STOP_CUDAEVENT_RECORDING(stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        float stage_time;
        TIME_ELAPSED_CUDAEVENT(&stage_time);
        time_lossless += stage_time;
    }

    /* phase 2 */
    {
        auto block_dim = HuffmanHelper::BLOCK_DIM_DEFLATE;
        auto grid_dim  = ConfigHelper::get_npart(pardeg, block_dim);

        START_CUDAEVENT_RECORDING(stream);

        asz::detail::hf_encode_phase2_deflate<H>  //
            <<<grid_dim, block_dim, 0, stream>>>  //
            (d_buffer, len, d_par_nbit, d_par_ncell, sublen, pardeg);

        STOP_CUDAEVENT_RECORDING(stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        float stage_time;
        TIME_ELAPSED_CUDAEVENT(&stage_time);
        time_lossless += stage_time;
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

        asz::detail::hf_encode_phase4_concatenate<H, M><<<pardeg, 128, 0, stream>>>  //
            (d_buffer, d_par_entry, d_par_ncell, sublen, d_bitstream);

        STOP_CUDAEVENT_RECORDING(stream);

        CHECK_CUDA(cudaStreamSynchronize(stream));

        float stage_time;
        TIME_ELAPSED_CUDAEVENT(&stage_time);
        time_lossless += stage_time;
    }
}

template <typename T, typename H, typename M>
void asz::hf_decode_coarse(
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

    CREATE_CUDAEVENT_PAIR;
    START_CUDAEVENT_RECORDING(stream)

    hf_decode_kernel<T, H, M>                             //
        <<<grid_dim, block_dim, revbook_nbyte, stream>>>  //
        (d_bitstream, d_revbook, d_par_nbit, d_par_entry, revbook_nbyte, sublen, pardeg, out_decompressed);

    STOP_CUDAEVENT_RECORDING(stream)
    cudaStreamSynchronize(stream);

    TIME_ELAPSED_CUDAEVENT(&time_lossless);
    DESTROY_CUDAEVENT_PAIR;
}

// TODO 22-11-02 remove reference; use ptr instead
// TODO return status

#define HFBOOK_INIT(T, H, M) \
    template void asz::hf_buildbook_g<T, H, M>(uint32_t*, H*, int const, uint8_t*, int const, float*, cudaStream_t);

HFBOOK_INIT(uint8_t, uint32_t, uint32_t);
HFBOOK_INIT(uint16_t, uint32_t, uint32_t);
HFBOOK_INIT(uint32_t, uint32_t, uint32_t);
HFBOOK_INIT(float, uint32_t, uint32_t);

HFBOOK_INIT(uint8_t, uint64_t, uint32_t);
HFBOOK_INIT(uint16_t, uint64_t, uint32_t);
HFBOOK_INIT(uint32_t, uint64_t, uint32_t);
HFBOOK_INIT(float, uint64_t, uint32_t);

HFBOOK_INIT(uint8_t, unsigned long long, uint32_t);
HFBOOK_INIT(uint16_t, unsigned long long, uint32_t);
HFBOOK_INIT(uint32_t, unsigned long long, uint32_t);
HFBOOK_INIT(float, unsigned long long, uint32_t);

#define HF_CODEC_INIT(T, H, M)                                                                                     \
    template void asz::hf_encode_coarse<T, H, M>(                                                                  \
        T*, H*, size_t const, uint32_t*, H*, int const, H*, M*, M*, int const, int const, int, uint8_t*&, size_t&, \
        float&, cudaStream_t);                                                                                     \
                                                                                                                   \
    template void asz::hf_encode_coarse_rev1<T, H, M>(                                                             \
        T*, size_t const, hf_book*, hf_bitstream*, uint8_t*&, size_t&, float&, cudaStream_t);                      \
                                                                                                                   \
    template void asz::hf_decode_coarse<T, H, M>(                                                                  \
        H*, uint8_t*, int const, M*, M*, int const, int const, T*, float&, cudaStream_t);

HF_CODEC_INIT(uint8_t, uint32_t, uint32_t);
HF_CODEC_INIT(uint16_t, uint32_t, uint32_t);
HF_CODEC_INIT(uint32_t, uint32_t, uint32_t);
HF_CODEC_INIT(float, uint32_t, uint32_t);
HF_CODEC_INIT(uint8_t, uint64_t, uint32_t);
HF_CODEC_INIT(uint16_t, uint64_t, uint32_t);
HF_CODEC_INIT(uint32_t, uint64_t, uint32_t);
HF_CODEC_INIT(float, uint64_t, uint32_t);
HF_CODEC_INIT(uint8_t, unsigned long long, uint32_t);
HF_CODEC_INIT(uint16_t, unsigned long long, uint32_t);
HF_CODEC_INIT(uint32_t, unsigned long long, uint32_t);
HF_CODEC_INIT(float, unsigned long long, uint32_t);

#undef HFBOOK_INIT
#undef HF_CODEC_INIT
