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

#include <stdexcept>
#include <type_traits>

#include "common/definition.hh"
#include "compressor_impl.cuh"
#include "context.hh"
#include "framework.hh"
#include "header.hh"

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
    cudaStream_t stream     = nullptr,
    TimeRecord*  timerecord = nullptr)
{
    STASTIC_ASSERT();

    {  // runtime check
        if (compressor == nullptr) throw std::runtime_error("`compressor` cannot be null.");
        if (config == nullptr) throw std::runtime_error("`config` cannot be null.");
        if (uncompressed == nullptr) throw std::runtime_error("Input `uncompressed` cannot be null.");
        if (not(uncompressed_alloc_len > 1.0299 * config->get_len()))
            throw std::runtime_error(
                "cuSZ requires the allocation for `uncompressed` to at least 1.03x the original size.");
    }

    cusz::CompressorHelper::autotune_coarse_parvle(config);
    (*compressor).init(config);
    (*compressor).compress(config, uncompressed, compressed, compressed_len, stream);
    (*compressor).export_header(header);
    (*compressor).export_timerecord(timerecord);
}

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
    cudaStream_t stream     = nullptr,
    TimeRecord*  timerecord = nullptr)
{
    STASTIC_ASSERT();

    {  // runtime check
        if (compressor == nullptr) throw std::runtime_error("`compressor` cannot be null.");
        if (config == nullptr) throw std::runtime_error("`config` cannot be null.");
        if (compressed == nullptr) throw std::runtime_error("Input `compressed` cannot be null.");
        if (compressed_len != config->get_filesize())
            throw std::runtime_error("`compressed_len` mismatches the description in header.");
        if (decompressed == nullptr)
            throw std::runtime_error("Output `decompressed` cannot be null: must be allocated before API call.");
        if (not(decompressed_alloc_len > 1.0299 * config->get_len_uncompressed()))
            throw std::runtime_error(
                "cuSZ requires the allocation for `decompressed` to at least 1.03x the original size.");
    }

    (*compressor).init(config);
    (*compressor).decompress(config, compressed, decompressed, stream);
    (*compressor).export_timerecord(timerecord);
}

}  // namespace cusz

#endif