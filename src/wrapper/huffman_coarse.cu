/**
 * @file huffman_coarse.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-12-17
 * (created) 2020-04-24 (rev1) 2021-09-05 (rev2) 2021-12-17
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

template <typename Huff>
__global__ void cusz::huffman_enc_concatenate(
    Huff*   in_enc_space,
    Huff*   out_bitstream,
    size_t* sp_entries,
    size_t* sp_uints,
    size_t  chunk_size)
{
    auto len      = sp_uints[blockIdx.x];
    auto sp_entry = sp_entries[blockIdx.x];
    auto dn_entry = chunk_size * blockIdx.x;

    for (auto i = 0; i < (len + nworker - 1) / nworker; i++) {
        auto _tid = threadIdx.x + i * nworker;
        if (_tid < len) *(out_bitstream + sp_entry + _tid) = *(in_enc_space + dn_entry + _tid);
        __syncthreads();
    }
}

template <typename Huff>
void cusz::huffman_process_metadata(
    size_t* _counts,
    size_t* dev_bits,
    size_t  nchunk,
    size_t& num_bits,
    size_t& num_uints)
{
    constexpr auto TYPE_BITCOUNT = sizeof(Huff) * 8;

    auto sp_uints = _counts, sp_bits = _counts + nchunk, sp_entries = _counts + nchunk * 2;

    cudaMemcpy(sp_bits, dev_bits, nchunk * sizeof(size_t), cudaMemcpyDeviceToHost);
    memcpy(sp_uints, sp_bits, nchunk * sizeof(size_t));
    for_each(sp_uints, sp_uints + nchunk, [&](size_t& i) { i = (i + TYPE_BITCOUNT - 1) / TYPE_BITCOUNT; });
    memcpy(sp_entries + 1, sp_uints, (nchunk - 1) * sizeof(size_t));
    for (auto i = 1; i < nchunk; i++) sp_entries[i] += sp_entries[i - 1];  // inclusive scan

    num_bits  = std::accumulate(sp_bits, sp_bits + nchunk, (size_t)0);
    num_uints = std::accumulate(sp_uints, sp_uints + nchunk, (size_t)0);
}

template <typename T, typename H, typename M>
void cusz::HuffmanCoarse<T, H, M>::huffman_encode_proxy1(
    H*      dev_enc_space,
    size_t* dev_bits,
    size_t* dev_uints,
    size_t* dev_entries,
    size_t* host_counts,
    T*      dev_input,
    H*      dev_book,
    size_t  len,
    int     chunk_size,
    int     dict_size,
    size_t* ptr_num_bits,
    size_t* ptr_num_uints,
    float&  milliseconds)
{
    auto nchunk = ConfigHelper::get_npart(len, chunk_size);

    {
        auto block_dim = HuffmanHelper::BLOCK_DIM_ENCODE;
        auto grid_dim  = ConfigHelper::get_npart(len, block_dim);

        int numSMs;
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

        cuda_timer_t t;
        t.timer_start();

        cusz::encode_fixedlen_gridstride  //
            <T, H><<<8 * numSMs, 256>>>   //
            (dev_input, dev_enc_space, len, dev_book, dict_size);

        t.timer_end();
        milliseconds += t.get_time_elapsed();
        cudaDeviceSynchronize();
    }

    {
        auto block_dim = HuffmanHelper::BLOCK_DIM_DEFLATE;
        auto grid_dim  = ConfigHelper::get_npart(nchunk, block_dim);

        cuda_timer_t t;
        t.timer_start();
        cusz::encode_deflate<H><<<grid_dim, block_dim>>>(dev_enc_space, len, dev_bits, chunk_size);
        t.timer_end();
        milliseconds += t.get_time_elapsed();
        cudaDeviceSynchronize();
    }

    cusz::huffman_process_metadata<H>(host_counts, dev_bits, nchunk, *ptr_num_bits, *ptr_num_uints);
    cudaMemcpy(dev_uints, host_counts, nchunk * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_entries, (host_counts + nchunk * 2), nchunk * sizeof(size_t), cudaMemcpyHostToDevice);
}

template <typename T, typename H, typename M>
void cusz::HuffmanCoarse<T, H, M>::huffman_encode_proxy2(
    H*      dev_enc_space,
    size_t* dev_uints,
    size_t* dev_entries,
    H*      dev_out_bitstream,
    size_t  len,
    int     chunk_size,
    float&  milliseconds)
{
    auto nchunk = ConfigHelper::get_npart(len, chunk_size);

    cuda_timer_t t;
    t.timer_start();
    cusz::huffman_enc_concatenate<<<nchunk, 128>>>(
        dev_enc_space, dev_out_bitstream, dev_entries, dev_uints, chunk_size);
    t.timer_end();
    milliseconds += t.get_time_elapsed();
    cudaDeviceSynchronize();
}

template <typename T, typename H, typename M>
void cusz::HuffmanCoarse<T, H, M>::encode(
    H*               workspace,    //  intermediate
    T*               in,           // input 1
    size_t           in_len,       // input 1 size
    uint32_t*        freq,         // input 2
    H*               book,         // input 3
    int              dict_size,    // input 2&2 size
    BYTE*            revbook,      // output 1
    Capsule<H>&      huff_data,    // output 2
    Capsule<size_t>& huff_counts,  // output 3
    int              chunk_size,   // related
    size_t&          num_bits,     // output
    size_t&          num_uints     // output
)
{
    {
        wrapper::get_frequency<T>(in, in_len, freq, dict_size, time_hist);

        {  // This is end-to-end time for parbook.
            cuda_timer_t t;
            t.timer_start();
            lossless::par_get_codebook<T, H>(dict_size, freq, book, revbook);
            t.timer_end();
            time_book = t.get_time_elapsed();
            cudaDeviceSynchronize();
        }
    }

    {
        auto const nchunk = ConfigHelper::get_npart(in_len, chunk_size);

        // fix-length space, padding improvised
        // H* workspace;
        // cudaMalloc(&workspace, sizeof(H) * in_len);

        auto d_bits    = huff_counts.dptr;
        auto d_uints   = huff_counts.dptr + nchunk;
        auto d_entries = huff_counts.dptr + nchunk * 2;

        huffman_encode_proxy1(
            workspace, d_bits, d_uints, d_entries, huff_counts.hptr, in, book, in_len, chunk_size, dict_size, &num_bits,
            &num_uints, time_lossless);

        // --------------------------------------------------------------------------------
        // update with the exact length
        huff_data.set_len(num_uints);

        huffman_encode_proxy2(workspace, d_uints, d_entries, huff_data.dptr, in_len, chunk_size, time_lossless);

        // cudaFree(workspace);

        // return *this;
    }
}

template <typename T, typename H, typename M>
void cusz::HuffmanCoarse<T, H, M>::decode(
    uint32_t _orilen,
    BYTE*    _dump,
    uint32_t _chunk_size,
    uint32_t _num_uints,
    uint32_t _dict_size,
    H*       d_in_bitstream,
    M*       d_chunkwise_metadata,
    BYTE*    d_revbook,
    T*       out_decoded)
{
    dump          = _dump;
    orilen        = _orilen;
    chunk_size    = _chunk_size;
    nchunk        = ConfigHelper::get_npart(orilen, chunk_size);
    num_uints     = _num_uints;
    revbook_nbyte = get_revbook_nbyte(_dict_size);

    auto block_dim = HuffmanHelper::BLOCK_DIM_DEFLATE;  // = deflating
    auto grid_dim  = ConfigHelper::get_npart(nchunk, block_dim);

    cuda_timer_t t;
    t.timer_start();
    cusz::decode_newtype<T, H, M><<<grid_dim, block_dim, revbook_nbyte>>>(  //
        d_in_bitstream, d_chunkwise_metadata, out_decoded, orilen, chunk_size, nchunk, d_revbook,
        (size_t)revbook_nbyte);
    t.timer_end();
    milliseconds += t.get_time_elapsed();
    CHECK_CUDA(cudaDeviceSynchronize());
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
