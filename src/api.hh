/**
 * @file api.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-04-05
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef API_HH
#define API_HH

#include <type_traits>

#include "context.hh"
#include "header.hh"

namespace cusz {

using context_t = cuszCTX*;
using header_t  = cuszHEADER*;

template <class compressor_t, typename T>
void core_compress(
    context_t&    ctx,
    compressor_t& compressor,
    T*&           uncompressed,
    BYTE*&        compressed,
    size_t&       compressed_len,
    header_t&     header,
    cudaStream_t  stream,
    bool          report_time)
{
    using inner_T = typename std::remove_pointer<compressor_t>::type::T;
    static_assert(std::is_same<inner_T, T>::value, "Compressor::T and T must match");

    AutoconfigHelper::autotune(ctx);
    (*compressor).init(ctx);
    (*compressor).compress(ctx, uncompressed, compressed, compressed_len, stream, report_time);
    (*compressor).export_header(header);
}

template <class compressor_t, typename T>
void core_decompress(
    header_t&     header,
    compressor_t& compressor,
    BYTE*&        compressed,
    T*&           decompressed,
    cudaStream_t  stream,
    bool          report_time)
{
    using inner_T = typename std::remove_pointer<compressor_t>::type::T;
    static_assert(std::is_same<inner_T, T>::value, "Compressor::T and T must match");

    (*compressor).init(header);
    (*compressor).decompress(header, compressed, decompressed, stream, report_time);
}

}  // namespace cusz

#endif