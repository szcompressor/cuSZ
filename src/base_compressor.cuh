/**
 * @file base_compressor.cuh
 * @author Jiannan Tian
 * @brief Predictor-only Base Compressor; can also be used for dryrun.
 * @version 0.3
 * @date 2021-10-05
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef BASE_COMPRESSOR_CUH
#define BASE_COMPRESSOR_CUH

#include "analysis/analyzer.hh"
#include "common.hh"
#include "context.hh"
#include "header.hh"
#include "kernel/dryrun.cuh"
#include "utils.hh"
#include "wrapper.hh"

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

   private:
    struct NonCritical {
        Predictor* p;
        Capsule<T> original;
        Capsule<E> errctrl;  // TODO change to 4-byte
        Capsule<T> outlier;
        Capsule<T> anchor;
        Capsule<T> reconst;

        NonCritical(dim3 size) { p = new Predictor(size, false); }
    };

    struct NonCritical* nc;

   protected:
    DataSeg dataseg;

    // clang-format off
    struct { double eb; FP ebx2, ebx2_r, eb_r; } config;
    struct { float lossy{0.0}, sparsity{0.0}, hist{0.0}, book{0.0}, lossless{0.0}; } time;
    // clang-format on

    // data fields
    Capsule<T>*    original;
    Capsule<BYTE>* compressed;
    Capsule<T>*    reconstructed;

    Capsule<E>          quant;   // for compressor
    Capsule<T>          anchor;  // for compressor
    Capsule<cusz::FREQ> freq;    // for compressibility

    cuszCTX*    ctx;
    cuszHEADER* header;
    cusz::WHEN  timing;

    int    dict_size;
    double eb;

    dim3 xyz;

   public:
    /**
     * @brief Generic dryrun; performing predictor.construct() and .reconstruct()
     *
     * @param fname filename
     * @param eb (host variable) error bound; future: absolute error bound only
     * @param radius (host variable) limiting radius
     * @param r2r if relative-to-value-range
     * @param stream CUDA stream
     * @return BaseCompressor& this object instance
     */
    BaseCompressor& generic_dryrun(const std::string fname, double eb, int radius, bool r2r, cudaStream_t stream)
    {
        if (not nc) throw std::runtime_error("NonCritical struct has no instance.");

        LOGGING(LOG_INFO, "invoke dry-run");

        nc->original.template from_file<cusz::LOC::HOST>(fname).host2device_async(stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        if (r2r) {
            double max, min, rng;
            nc->original.prescan(max, min, rng);
            eb *= rng;
        }

        nc->p->construct(nc->original.dptr, nc->anchor.dptr, nc->errctrl.dptr, eb, radius, stream, nc->outlier.dptr);
        nc->p->reconstruct(nc->anchor.dptr, nc->errctrl.dptr, nc->reconst.dptr, eb, radius, stream, nc->outlier.dptr);

        nc->reconst.device2host_async(stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        stat_t stat;
        verify_data_GPU<T>(&stat, nc->reconst.hptr, nc->original.hptr, nc->p->get_data_len());
        analysis::print_data_quality_metrics<T>(&stat, 0, true);

        return *this;
    }

    /**
     * @brief Dual-quant dryrun; performing integerization & its reverse procedure
     *
     * @param eb (host variable) error bound; future: absolute error bound only
     * @param r2r if relative-to-value-range
     * @param stream CUDA stream
     * @return BaseCompressor& this object instance
     */
    BaseCompressor& dualquant_dryrun(const std::string fname, double eb, bool r2r, cudaStream_t stream)
    {
        auto len = nc->original.len;

        nc->original.template from_file<cusz::LOC::HOST>(fname).host2device_async(stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        if (r2r) {
            double max, min, rng;
            nc->original.prescan(max, min, rng);
            eb *= rng;
        }

        auto ebx2_r = 1 / (eb * 2);
        auto ebx2   = eb * 2;

        cusz::dualquant_dryrun_kernel                                              //
            <<<ConfigHelper::get_npart(len, 256), 256, 256 * sizeof(T), stream>>>  //
            (nc->original.dptr, nc->reconst.dptr, len, ebx2_r, ebx2);

        nc->reconst.device2host_async(stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        stat_t stat;
        verify_data_GPU(&stat, nc->reconst.hptr, nc->original.hptr, len);
        analysis::print_data_quality_metrics<T>(&stat, 0, true);

        return *this;
    }

   protected:
    /**
     * @brief
     * @deprecated use Capsule.prescan() instead
     *
     * @return BaseCompressor&
     */
    BaseCompressor& prescan()
    {
        // if (ctx->mode == "r2r")
        //     auto result = Analyzer::get_maxmin_rng<T, ExecutionPolicy::cuda_device, AnalyzerMethod::thrust>(
        //         original->dptr, original->len);
        // if (ctx->verbose) LOGGING(LOG_DBG, "time scanning:", result.seconds, "sec");
        // if (ctx->mode == "r2r") ctx->eb *= result.rng;

        if (ctx->mode == "r2r") {
            auto rng = original->prescan().get_rng();
            ctx->eb *= rng;
            // LOGGING(LOG_DBG, "data range:", rng);
        }

        // TODO data "policy": (non-)destructive
        if (ctx->preprocess.binning) {
            LOGGING(LOG_ERR, "Binning is not working temporarily  (ver. 0.2.9)");
            exit(1);
        }

        return *this;
    }

    /**
     * @brief Report compress time.
     *
     * @return BaseCompressor&
     */
    BaseCompressor& noncritical__optional__report_compress_time()
    {
        if (not ctx->report.time) return *this;

        auto  nbyte   = ctx->data_len * sizeof(T);
        float nonbook = time.lossy + time.sparsity + time.hist + time.lossless;

        ReportHelper::print_throughput_tablehead("compression");

        ReportHelper::print_throughput_line("construct", time.lossy, nbyte);
        ReportHelper::print_throughput_line("gather-outlier", time.sparsity, nbyte);
        ReportHelper::print_throughput_line("histogram", time.hist, nbyte);
        ReportHelper::print_throughput_line("Huff-encode", time.lossless, nbyte);
        ReportHelper::print_throughput_line("(subtotal)", nonbook, nbyte);
        printf("\e[2m");
        ReportHelper::print_throughput_line("book", time.book, nbyte);
        ReportHelper::print_throughput_line("(total)", nonbook + time.book, nbyte);
        printf("\e[0m");

        return *this;
    }

    /**
     * @brief Report decompress time.
     *
     * @return BaseCompressor&
     */
    BaseCompressor& noncritical__optional__report_decompress_time()
    {
        if (not ctx->report.time) return *this;

        auto  nbyte = ctx->data_len * sizeof(T);
        float all   = time.lossy + time.sparsity + time.lossless;

        ReportHelper::print_throughput_tablehead("decompression");
        ReportHelper::print_throughput_line("scatter-outlier", time.sparsity, nbyte);
        ReportHelper::print_throughput_line("Huff-decode", time.lossless, nbyte);
        ReportHelper::print_throughput_line("reconstruct", time.lossy, nbyte);
        ReportHelper::print_throughput_line("(total)", all, nbyte);
        printf("\n");

        return *this;
    }

    /**
     * @brief Compare with the original for data quality measure.
     *
     * @param xdata (device array)
     * @param use_gpu (host variable)
     * @return BaseCompressor&
     */
    BaseCompressor& noncritical__optional__compare_with_original(T* xdata, bool use_gpu = true)
    {
        if (not ctx->fname.origin_cmp.empty() and ctx->report.quality) {
            LOGGING(LOG_INFO, "compare to the original");

            auto odata = io::read_binary_to_new_array<T>(ctx->fname.origin_cmp, ctx->data_len);

            if (use_gpu) {
                // TODO redundant memory use
                T *_xdata, *_odata;
                cudaMalloc(&_xdata, sizeof(T) * ctx->data_len);
                cudaMalloc(&_odata, sizeof(T) * ctx->data_len);

                cudaMemcpy(_xdata, xdata, sizeof(T) * ctx->data_len, cudaMemcpyHostToDevice);
                cudaMemcpy(_odata, odata, sizeof(T) * ctx->data_len, cudaMemcpyHostToDevice);

                verify_data_GPU<T>(&ctx->stat, _xdata, _odata, ctx->data_len);

                cudaFree(_xdata);
                cudaFree(_odata);
            }
            else
                analysis::verify_data(&ctx->stat, xdata, odata, ctx->data_len);

            analysis::print_data_quality_metrics<T>(&ctx->stat, compressed->nbyte(), use_gpu);

            delete[] odata;
        }

        return *this;
    }

    /**
     * @brief
     * @deprecated should be handled by Capsule outside a compressor
     *
     * @param host_xdata
     * @return BaseCompressor&
     */
    BaseCompressor& noncritical__optional__write2disk(T* host_xdata)
    {
        if (ctx->to_skip.write2disk)
            LOGGING(LOG_INFO, "output: skipped");
        else {
            LOGGING(LOG_INFO, "output:", ctx->fname.path_basename + ".cuszx");
            io::write_array_to_binary(ctx->fname.path_basename + ".cuszx", host_xdata, ctx->data_len);
        }

        return *this;
    }

    /**
     * @brief
     * @deprecated There would be hierachy file format.
     *
     * @return BaseCompressor&
     */
    BaseCompressor& pack_metadata()
    {
        ConfigHelper::deep_copy_config_items(/* dst */ header, /* src */ ctx);
        DatasegHelper::header_nbyte_from_dataseg(header, dataseg);
        return *this;
    }

    /**
     * @brief
     * @deprecated There would be hierachy file format.
     *
     * @return BaseCompressor&
     */
    BaseCompressor& unpack_metadata()
    {
        DatasegHelper::dataseg_nbyte_from_header(header, dataseg);

        /* 0 header */ dataseg.offset.push_back(0);

        if (ctx->verbose) {
            printf("(verification)"), ReportHelper::print_datasegment_tablehead(), setlocale(LC_ALL, "");
        }

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

    template <cusz::LOC SRC, cusz::LOC DST>
    BaseCompressor& consolidate(BYTE** dump)
    {  // no impl temporarily
        return *this;
    }

   public:
    BaseCompressor() = default;

    ~BaseCompressor() {}

   public:
    // dry run
    void init_generic_dryrun(dim3 size)
    {  //
        auto len = size.x * size.y * size.z;
        nc       = new struct NonCritical(size);

        nc->original.set_len(len).template alloc<cusz::LOC::HOST_DEVICE>();
        nc->outlier.set_len(len).template alloc<cusz::LOC::HOST_DEVICE>();
        nc->errctrl.set_len(len).template alloc<cusz::LOC::HOST_DEVICE>();
        nc->anchor.set_len(nc->p->get_anchor_len()).template alloc<cusz::LOC::HOST_DEVICE>();
        nc->reconst.set_len(len).template alloc<cusz::LOC::HOST_DEVICE>();
    }

    void destroy_generic_dryrun()
    {
        delete nc->p;
        nc->original.template free<cusz::LOC::HOST_DEVICE>();
        nc->outlier.template free<cusz::LOC::HOST_DEVICE>();
        nc->errctrl.template free<cusz::LOC::HOST_DEVICE>();
        nc->anchor.template free<cusz::LOC::HOST_DEVICE>();
        nc->reconst.template free<cusz::LOC::HOST_DEVICE>();
        delete nc;
    }

    void init_dualquant_dryrun(dim3 size)
    {
        auto len = size.x * size.y * size.z;
        nc       = new struct NonCritical(size);
        nc->original.set_len(len).template alloc<cusz::LOC::HOST_DEVICE>();
        nc->reconst.set_len(len).template alloc<cusz::LOC::HOST_DEVICE>();
    }

    void destroy_dualquant_dryrun()
    {
        nc->original.template free<cusz::LOC::HOST_DEVICE>();
        nc->reconst.template free<cusz::LOC::HOST_DEVICE>();

        delete nc;
    }
};

#endif
