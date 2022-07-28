/**
 * @file cuszapi.hh
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

#include <stdexcept>
#include <type_traits>

#include "context.hh"
#include "framework.hh"
#include "header.h"

#define STASTIC_ASSERT() \
    static_assert(std::is_same<typename Compressor::T, T>::value, "Compressor::T and T must match");

namespace cusz {

/**
 * @brief Core compression API for cuSZ, requiring that input and output are on device pointers/iterators.
 *
 * @tparam Compressor predefined Compressor type, accessible via cusz::Framework<T>::XFeaturedCompressor
 * @tparam T uncompressed data type
 * @param compressor Compressor instance
 * @param config (host) cusz::Context as configuration type
 * @param uncompressed (device) input uncompressed type
 * @param uncompressed_alloc_len (host) for checking; >1.03x the original data size to ensure the legal memory access
 * @param compressed (device) exposed compressed array in Compressor (shallow copy); need to transfer before Compressor
 * destruction
 * @param compressed_len (host) output compressed array length
 * @param header (host) header for compressed binary description; aquired by a deep copy
 * @param stream CUDA stream
 * @param timerecord collected time information for compressor; aquired by a deep copy
 */
template <class Compressor, typename T>
void core_compress(
    Compressor*  compressor,
    Context*     config,
    T*           uncompressed,
    size_t       uncompressed_alloc_len,
    uint8_t*&    compressed,
    size_t&      compressed_len,
    Header&      header,
    hipStream_t stream     = nullptr,
    TimeRecord*  timerecord = nullptr);

/**
 * @brief Core decompression API for cuSZ, requiring that input and output are on device pointers/iterators.
 *
 * @tparam Compressor predefined Compressor type, accessible via cusz::Framework<T>::XFeaturedCompressor
 * @tparam T uncompressed data type
 * @param compressor Compressor instance
 * @param config (host) cusz::Header as configuration type
 * @param compressed (device) input compressed array
 * @param compressed_len (host) input compressed length for checking
 * @param decompressed (device) output decompressed array
 * @param decompressed_alloc_len (host) for checking; >1.03x the original data size to ensure the legal memory access
 * @param stream CUDA stream
 * @param timerecord collected time information for compressor; aquired by a deep copy
 */
template <class Compressor, typename T>
void core_decompress(
    Compressor*  compressor,
    Header*      config,
    uint8_t*     compressed,
    size_t       compressed_len,
    T*           decompressed,
    size_t       decompressed_alloc_len,
    hipStream_t stream     = nullptr,
    TimeRecord*  timerecord = nullptr);

}  // namespace cusz

#endif