/**
 * @file cc2c.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-05-01
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include "cusz/cc2c.h"
#include "compressor.hh"
#include "context.hh"
#include "framework.hh"

extern "C" {
#include "cusz/custom.h"
#include "cusz/type.h"
}

cusz_error_status cusz_compressor::commit_framework(cusz_framework* _framework)
{
    framework = _framework;
    return CUSZ_SUCCESS;
}

cusz_compressor::cusz_compressor(cusz_framework* _framework, cusz_datatype _type) : type(_type)
{
    framework = _framework;

    if (type == FP32) {
        using DATA       = float;
        using Compressor = cusz::Framework<DATA>::DefaultCompressor;

        this->compressor = new Compressor();
    }
    else {
        throw std::runtime_error("Type is not supported.");
    }
}

cusz_error_status cusz_compressor::commit_space(cusz_len const reserved_mem, cusz_framework* adjusted)
{
    // TODO adjust framework here

    // end of adjusting framework
    return CUSZ_SUCCESS;
}

void cc2c_record(cusz::TimeRecord cpp_record, cusz_record** record)
{
    *record = new cusz_record;
    auto it = (*record)->head;

    for (auto const& i : cpp_record) {
        it = new cusz_record_entry{.name = std::get<0>(i), .time = std::get<1>(i)};
        it = it->next;
        (*record)->n += 1;
    }
}

cusz_error_status cusz_compressor::compress(
    cusz_config*   config,
    void*          uncompressed,
    cusz_len const uncomp_len,
    uint8_t**      compressed,
    size_t*        comp_bytes,
    cusz_header*   header,
    void*          record,
    cudaStream_t   stream)
{
    // cusz::TimeRecord cpp_record;

    context = new cusz_context();
    static_cast<cusz_context*>(context)
        ->set_len(uncomp_len.x, uncomp_len.y, uncomp_len.z, uncomp_len.w)
        .set_eb(config->eb)
        .set_control_string(config->eb == Rel ? "mode=r2r" : "mode=abs");

    // Be cautious of autotuning! The default value of pardeg is not robust.
    cusz::CompressorHelper::autotune_coarse_parvle(static_cast<cusz_context*>(context));

    // TODO how to check effectively?
    size_t len1d = uncomp_len.x * uncomp_len.y * uncomp_len.z * uncomp_len.factor;

    if (type == FP32) {
        using DATA       = float;
        using Compressor = cusz::Framework<DATA>::DefaultCompressor;

        // TODO add memlen & datalen comparison
        static_cast<Compressor*>(this->compressor)->init(static_cast<cusz_context*>(context));
        static_cast<Compressor*>(this->compressor)
            ->compress(
                static_cast<cusz_context*>(context), static_cast<DATA*>(uncompressed), *compressed, *comp_bytes,
                stream);
        static_cast<Compressor*>(this->compressor)->export_header(*header);
        static_cast<Compressor*>(this->compressor)->export_timerecord((cusz::TimeRecord*)record);
    }
    else {
        throw std::runtime_error(std::string(__FUNCTION__) + ": Type is not supported.");
    }

    return CUSZ_SUCCESS;
}

cusz_error_status cusz_compressor::decompress(
    cusz_header*   header,
    uint8_t*       compressed,
    size_t const   comp_len,
    void*          decompressed,
    cusz_len const decomp_len,
    void*          record,
    cudaStream_t   stream)
{
    // cusz::TimeRecord cpp_record;

    if (type == FP32) {
        using DATA       = float;
        using Compressor = cusz::Framework<DATA>::DefaultCompressor;

        static_cast<Compressor*>(this->compressor)->init(header);
        static_cast<Compressor*>(this->compressor)
            ->decompress(header, compressed, static_cast<DATA*>(decompressed), stream);
        static_cast<Compressor*>(this->compressor)->export_timerecord((cusz::TimeRecord*)record);
    }
    else {
        throw std::runtime_error(std::string(__FUNCTION__) + ": Type is not supported.");
    }

    return CUSZ_SUCCESS;
}
