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
    cuszCTX* ctx;

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
