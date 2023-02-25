/**
 * @file v2_compressor.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-01-29
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include "pipeline/v2_compressor.hh"
#include "common/configs.hh"
#include "framework.hh"

namespace parsz {

template <class B>
v2_Compressor<B>::~v2_Compressor()
{
    pimpl.reset();
}

template <class B>
v2_Compressor<B>::v2_Compressor() : pimpl{std::make_unique<impl>()}
{
}

template <class B>
v2_Compressor<B>::v2_Compressor(const v2_Compressor<B>& old) : pimpl{std::make_unique<impl>(*old.pimpl)}
{
}

template <class B>
v2_Compressor<B>& v2_Compressor<B>::operator=(const v2_Compressor<B>& old)
{
    *pimpl = *old.pimpl;
    return *this;
}

template <class B>
v2_Compressor<B>::v2_Compressor(v2_Compressor<B>&&) = default;

template <class B>
v2_Compressor<B>& v2_Compressor<B>::operator=(v2_Compressor<B>&&) = default;

//------------------------------------------------------------------------------

template <class B>
void v2_Compressor<B>::init(Context* config)
{
    pimpl->init(config);
}

template <class B>
void v2_Compressor<B>::init(v2_header* config)
{
    pimpl->init(config);
}

template <class B>
void v2_Compressor<B>::compress(
    Context*             config,
    v2_Compressor<B>::T* uncompressed,
    BYTE*&               compressed,
    size_t&              compressed_len,
    cudaStream_t         stream,
    bool                 dbg_print)
{
    pimpl->compress(config, uncompressed, compressed, compressed_len, stream, dbg_print);
}

template <class B>
void v2_Compressor<B>::decompress(
    v2_header*           config,
    BYTE*                compressed,
    v2_Compressor<B>::T* decompressed,
    cudaStream_t         stream,
    bool                 dbg_print)
{
    pimpl->decompress(config, compressed, decompressed, stream, dbg_print);
}

// template <class B>
// void v2_Compressor<B>::clear_buffer()
// {
//     pimpl->clear_buffer();
// }

// getter

template <class B>
void v2_Compressor<B>::export_header(v2_header& header)
{
    pimpl->export_header(header);
}

template <class B>
void v2_Compressor<B>::export_header(v2_header* header)
{
    pimpl->export_header(header);
}

// template <class B>
// void v2_Compressor<B>::export_timerecord(TimeRecord* ext_timerecord)
// {
//     pimpl->export_timerecord(ext_timerecord);
// }

}  // namespace parsz

template class parsz::v2_Compressor<cusz::Framework<float>>;