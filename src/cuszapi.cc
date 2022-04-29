/**
 * @file cuszapi.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-04-29
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include "cuszapi.hh"

namespace cusz {

template <class Compressor, typename T>
void core_compress(
    Compressor*  compressor,
    Context*     config,
    T*           uncompressed,
    size_t       uncompressed_alloc_len,
    uint8_t*&    compressed,
    size_t&      compressed_len,
    Header&      header,
    cudaStream_t stream,
    TimeRecord*  timerecord)
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

template <class Compressor, typename T>
void core_decompress(
    Compressor*  compressor,
    Header*      config,
    uint8_t*     compressed,
    size_t       compressed_len,
    T*           decompressed,
    size_t       decompressed_alloc_len,
    cudaStream_t stream,
    TimeRecord*  timerecord)
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

namespace cusz {

using fp32lorenzo = Framework<float>::LorenzoFeaturedCompressor;
using fp32spline3 = Framework<float>::Spline3FeaturedCompressor;

// clang-format off

template void
core_compress<fp32lorenzo, float>(fp32lorenzo*, Context*, float*, size_t, uint8_t*&, size_t&, Header&, cudaStream_t, TimeRecord*);

template void
core_compress<fp32spline3, float>(fp32spline3*, Context*, float*, size_t, uint8_t*&, size_t&, Header&, cudaStream_t, TimeRecord*);

template void
core_decompress<fp32lorenzo, float>(fp32lorenzo*, Header*, uint8_t*, size_t, float*, size_t, cudaStream_t, TimeRecord*);

template void
core_decompress<fp32spline3, float>(fp32spline3*, Header*, uint8_t*, size_t, float*, size_t, cudaStream_t, TimeRecord*);

// clang-format on

}  // namespace cusz
