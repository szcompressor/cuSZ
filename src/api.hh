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

#include "common/types.hh"
#include "context.hh"
#include "header.hh"

namespace cusz {

using TimeRecord = std::vector<std::tuple<const char*, double>>;

template <class compressor_t, typename T>
void core_compress(
    compressor_t& compressor,
    context_t&    config,
    T*&           uncompressed,
    BYTE*&        compressed,
    size_t&       compressed_len,
    header_t&     header,
    cudaStream_t  stream)
{
    using inner_T = typename std::remove_pointer<compressor_t>::type::T;
    static_assert(std::is_same<inner_T, T>::value, "Compressor::T and T must match");

    AutoconfigHelper::autotune(config);
    (*compressor).init(config);
    (*compressor).compress(config, uncompressed, compressed, compressed_len, stream);
    (*compressor).export_header(header);
}

template <class compressor_t, typename T>
void core_compress(
    compressor_t& compressor,
    context_t&    config,
    T*&           uncompressed,
    BYTE*&        compressed,
    size_t&       compressed_len,
    header_t&     header,
    cudaStream_t  stream,
    timerecord_t& timerecord)
{
    using inner_T = typename std::remove_pointer<compressor_t>::type::T;
    static_assert(std::is_same<inner_T, T>::value, "Compressor::T and T must match");

    core_compress(compressor, config, uncompressed, compressed, compressed_len, header, stream);
    (*compressor).export_timerecord(timerecord);
}

template <class compressor_t, typename T>
void core_decompress(
    compressor_t& compressor,
    header_t&     config,
    BYTE*&        compressed,
    T*&           decompressed,
    cudaStream_t  stream)
{
    using inner_T = typename std::remove_pointer<compressor_t>::type::T;
    static_assert(std::is_same<inner_T, T>::value, "Compressor::T and T must match");

    (*compressor).init(config);
    (*compressor).decompress(config, compressed, decompressed, stream);
}

template <class compressor_t, typename T>
void core_decompress(
    compressor_t& compressor,
    header_t&     config,
    BYTE*&        compressed,
    T*&           decompressed,
    cudaStream_t  stream,
    timerecord_t& timerecord)
{
    using inner_T = typename std::remove_pointer<compressor_t>::type::T;
    static_assert(std::is_same<inner_T, T>::value, "Compressor::T and T must match");

    core_decompress(compressor, config, compressed, decompressed, stream);
    (*compressor).export_timerecord(timerecord);
}

}  // namespace cusz

#endif