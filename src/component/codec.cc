/**
 * @file codec.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-04-23
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include "component/codec.hh"
#include "common/type_traits.hh"

namespace cusz {

#define TEMPLATE_TYPE template <typename T, typename H, typename M>
#define HUFFMAN_COARSE HuffmanCoarse<T, H, M>

TEMPLATE_TYPE
HUFFMAN_COARSE::~HuffmanCoarse() { pimpl.reset(); }

TEMPLATE_TYPE
HUFFMAN_COARSE::HuffmanCoarse() : pimpl{std::make_unique<impl>()} {}

TEMPLATE_TYPE
HUFFMAN_COARSE::HuffmanCoarse(const HUFFMAN_COARSE& old) : pimpl{std::make_unique<impl>(*old.pimpl)}
{
    // TODO allocation/deep copy
}

TEMPLATE_TYPE
HUFFMAN_COARSE& HUFFMAN_COARSE::operator=(const HUFFMAN_COARSE& old)
{
    *pimpl = *old.pimpl;
    // TODO allocation/deep copy
    return *this;
}

TEMPLATE_TYPE
HUFFMAN_COARSE::HuffmanCoarse(HUFFMAN_COARSE&&) = default;

TEMPLATE_TYPE
HUFFMAN_COARSE& HUFFMAN_COARSE::operator=(HUFFMAN_COARSE&&) = default;

//------------------------------------------------------------------------------

TEMPLATE_TYPE
void HUFFMAN_COARSE::init(size_t const in_uncompressed_len, int const booklen, int const pardeg, bool dbg_print)
{
    pimpl->init(in_uncompressed_len, booklen, pardeg, dbg_print);
}

TEMPLATE_TYPE
void HUFFMAN_COARSE::build_codebook(uint32_t* freq, int const booklen, cudaStream_t stream)
{
    pimpl->build_codebook(freq, booklen, stream);
}

TEMPLATE_TYPE
void HUFFMAN_COARSE::encode(
    T*           in_uncompressed,
    size_t const in_uncompressed_len,
    uint32_t*    d_freq,
    int const    cfg_booklen,
    int const    cfg_sublen,
    int const    cfg_pardeg,
    BYTE*&       out_compressed,
    size_t&      out_compressed_len,
    cudaStream_t stream)
{
    pimpl->encode(
        in_uncompressed, in_uncompressed_len, d_freq, cfg_booklen, cfg_sublen, cfg_pardeg, out_compressed,
        out_compressed_len, stream);
}

TEMPLATE_TYPE
void HUFFMAN_COARSE::decode(BYTE* in_compressed, T* out_decompressed, cudaStream_t stream, bool header_on_device)
{
    pimpl->decode(in_compressed, out_decompressed, stream, header_on_device);
}

TEMPLATE_TYPE
void HUFFMAN_COARSE::clear_buffer() { pimpl->clear_buffer(); }

TEMPLATE_TYPE
float HUFFMAN_COARSE::get_time_elapsed() const { return pimpl->get_time_elapsed(); }

TEMPLATE_TYPE
float HUFFMAN_COARSE::get_time_book() const { return pimpl->get_time_book(); }
TEMPLATE_TYPE
float HUFFMAN_COARSE::get_time_lossless() const { return pimpl->get_time_lossless(); }

#undef TEMPLATE_TYPE
#undef HUFFMAN_COARSE

}  // namespace cusz

#define HUFFCOARSE_CC(E, ETF, H, M) \
    template class cusz::HuffmanCoarse<ErrCtrlTrait<E, ETF>::type, HuffTrait<H>::type, MetadataTrait<M>::type>;

HUFFCOARSE_CC(2, false, 4, 4)  // deprecated
HUFFCOARSE_CC(2, false, 8, 4)  // deprecated
HUFFCOARSE_CC(4, false, 4, 4)  // deprecated
HUFFCOARSE_CC(4, false, 8, 4)  // deprecated

HUFFCOARSE_CC(4, true, 4, 4)  // float
HUFFCOARSE_CC(4, true, 8, 4)  // float

#undef HUFFCOARSE_CC
