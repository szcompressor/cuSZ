/**
 * @file base_cusz.cu
 * @author Jiannan Tian
 * @brief Predictor-only Base Compressor; can also be used for dryrun.
 * @version 0.3
 * @date 2021-10-05
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include "analysis/analyzer.hh"
#include "base_cusz.cuh"
#include "utils.hh"
#include "wrapper.hh"

#define BASE_COMPRESSOR_TYPE template <class Predictor>
#define BASE_COMPRESSOR BaseCompressor<Predictor>

BASE_COMPRESSOR_TYPE BASE_COMPRESSOR& BASE_COMPRESSOR::prescan()
{
    if (ctx->mode == "r2r") {
        auto result = Analyzer::get_maxmin_rng                         //
            <T, ExecutionPolicy::cuda_device, AnalyzerMethod::thrust>  //
            (in_data->dptr, in_data->len);
        if (ctx->verbose) LOGGING(LOG_DBG, "time scanning:", result.seconds, "sec");
        if (ctx->mode == "r2r") ctx->eb *= result.rng;
    }

    // TODO data "policy": (non-)destructive
    if (ctx->preprocess.binning) {
        LOGGING(LOG_ERR, "Binning is not working temporarily  (ver. 0.2.9)");
        exit(1);
    }

    return *this;
}

BASE_COMPRESSOR_TYPE
BASE_COMPRESSOR& BASE_COMPRESSOR::try_report_compress_time()
{
    if (not ctx->report.time) return *this;

    auto  nbyte   = ctx->data_len * sizeof(T);
    float nonbook = time.lossy + time.outlier + time.hist + time.lossless;

    ReportHelper::print_throughput_tablehead("compression");

    ReportHelper::print_throughput_line("construct", time.lossy, nbyte);
    ReportHelper::print_throughput_line("gather-outlier", time.outlier, nbyte);
    ReportHelper::print_throughput_line("histogram", time.hist, nbyte);
    ReportHelper::print_throughput_line("Huff-encode", time.lossless, nbyte);
    ReportHelper::print_throughput_line("(subtotal)", nonbook, nbyte);
    printf("\e[2m");
    ReportHelper::print_throughput_line("book", time.book, nbyte);
    ReportHelper::print_throughput_line("(total)", nonbook + time.book, nbyte);
    printf("\e[0m");

    return *this;
}

BASE_COMPRESSOR_TYPE
BASE_COMPRESSOR& BASE_COMPRESSOR::try_report_decompress_time()
{
    if (not ctx->report.time) return *this;

    auto  nbyte = ctx->data_len * sizeof(T);
    float all   = time.lossy + time.outlier + time.lossless;

    ReportHelper::print_throughput_tablehead("decompression");
    ReportHelper::print_throughput_line("scatter-outlier", time.outlier, nbyte);
    ReportHelper::print_throughput_line("Huff-decode", time.lossless, nbyte);
    ReportHelper::print_throughput_line("reconstruct", time.lossy, nbyte);
    ReportHelper::print_throughput_line("(total)", all, nbyte);
    printf("\n");

    return *this;
}

BASE_COMPRESSOR_TYPE
BASE_COMPRESSOR& BASE_COMPRESSOR::try_compare_with_origin(T* xdata)
{
    if (not ctx->fnames.origin_cmp.empty() and ctx->report.quality) {
        LOGGING(LOG_INFO, "compare to the original");

        auto odata = io::read_binary_to_new_array<T>(ctx->fnames.origin_cmp, ctx->data_len);

        analysis::verify_data(&ctx->stat, xdata, odata, ctx->data_len);
        analysis::print_data_quality_metrics<T>(&ctx->stat, in_dump->nbyte(), false);

        delete[] odata;
    }

    return *this;
}

BASE_COMPRESSOR_TYPE
BASE_COMPRESSOR& BASE_COMPRESSOR::try_write2disk(T* host_xdata)
{
    if (ctx->to_skip.write2disk)
        LOGGING(LOG_INFO, "output: skipped");
    else {
        LOGGING(LOG_INFO, "output:", ctx->fnames.path_basename + ".cuszx");
        io::write_array_to_binary(ctx->fnames.path_basename + ".cuszx", host_xdata, ctx->data_len);
    }

    return *this;
}

BASE_COMPRESSOR_TYPE
BASE_COMPRESSOR& BASE_COMPRESSOR::pack_metadata()
{
    ConfigHelper::deep_copy_config_items(/* dst */ header, /* src */ ctx);
    DatasegHelper::header_nbyte_from_dataseg(header, dataseg);
    return *this;
}

BASE_COMPRESSOR_TYPE
BASE_COMPRESSOR& BASE_COMPRESSOR::unpack_metadata()
{
    DatasegHelper::dataseg_nbyte_from_header(header, dataseg);

    /* 0 header */ dataseg.offset.push_back(0);

    if (ctx->verbose) { printf("(verification)"), ReportHelper::print_datasegment_tablehead(), setlocale(LC_ALL, ""); }

    for (auto i = 0; i < 8; i++) {
        const auto& name       = dataseg.order2name.at(i);
        auto        this_nbyte = dataseg.nbyte.at(name);
        auto        o          = dataseg.offset.back() + __cusz_get_alignable_len<BYTE, 128>(this_nbyte);
        dataseg.offset.push_back(o);

        if (ctx->verbose) {
            if (this_nbyte != 0)
                printf(
                    "  %-18s\t%'12lu\t%'15lu\t%'15lu\n",  //
                    dataseg.get_namestr(name).c_str(),    //
                    (size_t)this_nbyte,                   //
                    (size_t)dataseg.offset.at(i),         //
                    (size_t)dataseg.offset.back());
        }
    }
    if (ctx->verbose) printf("\n");

    ConfigHelper::deep_copy_config_items(/* dst */ ctx, /* src */ header);
    ConfigHelper::set_eb_series(ctx->eb, config);

    return *this;
}

template class BaseCompressor<cusz::PredictorLorenzo<  //
    DataTrait<4>::type,
    ErrCtrlTrait<2>::type,
    FastLowPrecisionTrait<true>::type>>;

template class BaseCompressor<cusz::PredictorLorenzo<  //
    DataTrait<4>::type,
    ErrCtrlTrait<4, true>::type,
    FastLowPrecisionTrait<true>::type>>;

template class BaseCompressor<cusz::Spline3<  //
    DataTrait<4>::type,
    ErrCtrlTrait<2>::type,
    FastLowPrecisionTrait<true>::type>>;

template class BaseCompressor<cusz::Spline3<  //
    DataTrait<4>::type,
    ErrCtrlTrait<4, true>::type,
    FastLowPrecisionTrait<true>::type>>;