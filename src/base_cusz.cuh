/**
 * @file base_cusz.cuh
 * @author Jiannan Tian
 * @brief Predictor-only Base Compressor; can also be used for dryrun.
 * @version 0.3
 * @date 2021-10-05
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef BASE_CUSZ_CUH
#define BASE_CUSZ_CUH

#include "common.hh"
#include "context.hh"
#include "header.hh"
#include "kernel/dryrun.cuh"
#include "utils.hh"

/**
 * @brief bare metal, can run predictor to check data quality and compressibility
 *
 * @tparam T for data type
 * @tparam E for error control type
 */

template <class Predictor>
class BaseCompressor {
   public:
    using BYTE = uint8_t;
    using T    = typename Predictor::Origin;
    using FP   = typename Predictor::Precision;
    using E    = typename Predictor::ErrCtrl;

   protected:
    DataSeg dataseg;

    // clang-format off
    struct { double eb; FP ebx2, ebx2_r, eb_r; } config;
    struct { float lossy{0.0}, sparsity{0.0}, hist{0.0}, book{0.0}, lossless{0.0}; } time;
    // clang-format on

    // data fields
    Capsule<T>*         in_data;  // compress-time, TODO rename
    Capsule<BYTE>*      in_dump;  // decompress-time, TODO rename
    Capsule<E>          quant;    // for compressor
    Capsule<T>          anchor;   // for compressor
    Capsule<cusz::FREQ> freq;     // for compressibility

    cuszCTX*    ctx;
    cuszHEADER* header;
    cusz::WHEN  timing;

    dim3 xyz;

   protected:
    BaseCompressor& dryrun()
    {
        if (ctx->task_is.dryrun and ctx->str_predictor == "lorenzo") {
            auto len = ctx->data_len;

            LOGGING(LOG_INFO, "invoke dry-run");
            constexpr auto SEQ       = 4;
            constexpr auto SUBSIZE   = 256;
            auto           dim_block = SUBSIZE / SEQ;
            auto           dim_grid  = ConfigHelper::get_npart(len, SUBSIZE);

            cusz::dual_quant_dryrun<T, float, SUBSIZE, SEQ>
                <<<dim_grid, dim_block>>>(in_data->dptr, len, config.ebx2_r, config.ebx2);
            HANDLE_ERROR(cudaDeviceSynchronize());

            T* dryrun_result;
            cudaMallocHost(&dryrun_result, len * sizeof(T));
            cudaMemcpy(dryrun_result, in_data->dptr, len * sizeof(T), cudaMemcpyDeviceToHost);

            analysis::verify_data<T>(&ctx->stat, dryrun_result, in_data->hptr, len);
            analysis::print_data_quality_metrics<T>(&ctx->stat, 0, false);

            cudaFreeHost(dryrun_result);

            exit(0);
        }
        return *this;

        return *this;
    }
    BaseCompressor& prescan();
    BaseCompressor& noncritical__optional__report_compress_time();
    BaseCompressor& noncritical__optional__report_decompress_time();
    BaseCompressor& noncritical__optional__compare_with_original(T* xdata);
    BaseCompressor& noncritical__optional__write2disk(T* host_xdata);

    BaseCompressor& pack_metadata();
    BaseCompressor& unpack_metadata();

    template <cusz::LOC SRC, cusz::LOC DST>
    BaseCompressor& consolidate(BYTE** dump)
    {  // no impl temporarily
        return *this;
    }

   public:
    BaseCompressor() = default;

    BaseCompressor(cuszCTX* _ctx, Capsule<T>* _in_data)
    {  // dummy
    }
    BaseCompressor(cuszCTX* _ctx, Capsule<BYTE>* _in_dump)
    {  // dummy
    }
    ~BaseCompressor()
    {  // dummy
    }
};

#endif