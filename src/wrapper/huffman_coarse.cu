/**
 * @file huffman_coarse.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-09-18
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include "../kernel/codec_huffman.cuh"
#include "../type_trait.hh"
#include "../utils.hh"
#include "huffman_coarse.cuh"

namespace cusz {

template <typename T, typename H, typename M>
void HuffmanWork<T, H, M>::decode(WHERE w, H* in_bitstream, M* chunkwise_metadata, BYTE* revbook, T* out_decoded)
{
    H*    d_in_bitstream;
    M*    d_chunkwise_metadata;
    BYTE* d_revbook;

    if (w == WHERE::HOST) {
        d_in_bitstream       = mem::create_devspace_memcpy_h2d(in_bitstream, num_uints);
        d_chunkwise_metadata = mem::create_devspace_memcpy_h2d(chunkwise_metadata, 2 * nchunk);
        d_revbook            = mem::create_devspace_memcpy_h2d(revbook, revbook_nbyte);
    }
    else if (w == WHERE::DEVICE) {
        d_in_bitstream       = in_bitstream;
        d_chunkwise_metadata = chunkwise_metadata;
        d_revbook            = revbook;
    }

    auto block_dim = HuffmanHelper::BLOCK_DIM_DEFLATE;  // = deflating
    auto grid_dim  = ConfigHelper::get_npart(nchunk, block_dim);

    auto t = new cuda_timer_t;
    t->timer_start();
    cusz::decode_newtype<T, H, M><<<grid_dim, block_dim, revbook_nbyte>>>(  //
        d_in_bitstream, d_chunkwise_metadata, out_decoded, orilen, chunk_size, nchunk, d_revbook,
        (size_t)revbook_nbyte);
    milliseconds += t->timer_end_get_elapsed_time();
    CHECK_CUDA(cudaDeviceSynchronize());
    delete t;

    cudaFree(d_in_bitstream);
    cudaFree(d_chunkwise_metadata);
    cudaFree(d_revbook);
}

}  // namespace cusz

template class cusz::HuffmanWork<ErrCtrlTrait<2>::type, HuffTrait<4>::type, MetadataTrait<4>::type>;
template class cusz::HuffmanWork<ErrCtrlTrait<2>::type, HuffTrait<4>::type, MetadataTrait<8>::type>;
template class cusz::HuffmanWork<ErrCtrlTrait<2>::type, HuffTrait<8>::type, MetadataTrait<4>::type>;
template class cusz::HuffmanWork<ErrCtrlTrait<2>::type, HuffTrait<8>::type, MetadataTrait<8>::type>;
