/**
 * @file predictor_boilerplate.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-09-15
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_INCLUDE_PREDICTOR_HH
#define CUSZ_INCLUDE_PREDICTOR_HH

#include <cstdint>
#include <cstdio>
#include <stdexcept>

#include "../common/configs.hh"
#include "../cusz/type.h"

namespace cusz {

class PredictorBoilerplate {
   protected:
    struct DerivedLengths {
        struct Interpretion3D {
            dim3   len3, leap;
            size_t serialized;

            void set_leap() { leap = ConfigHelper::get_leap(len3); }
            void set_serialized() { serialized = ConfigHelper::get_serialized_len(len3); }
        };

        struct Interpretion3D base, anchor, aligned;

        dim3 nblock;
        int  ndim;

        struct {
            size_t data, quant, outlier, anchor;
        } assigned;

        dim3 get_len3() const { return base.len3; }
        dim3 get_leap() const { return base.leap; }
    };

    template <class DERIVED>
    void __derive_len(dim3 base, DERIVED& derived)
    {
        int sublen[3]      = {1, 1, 1};
        int anchor_step[3] = {1, 1, 1};
        __derive_len(base, derived, sublen, anchor_step, false);
    }

    template <class DERIVED>
    void
    __derive_len(dim3 base, DERIVED& derived, int const sublen3[3], int const anchor_step3[3], bool use_anchor = false)
    {
        derived.base.len3 = base;
        derived.base.set_leap();
        derived.base.set_serialized();
        derived.ndim = ConfigHelper::get_ndim(base);

        if (not use_anchor) {
            derived.assigned.data    = derived.base.serialized;
            derived.assigned.quant   = derived.base.serialized;
            derived.assigned.outlier = derived.base.serialized;
            derived.assigned.anchor  = 0;
        }
        else {
            derived.nblock = ConfigHelper::get_pardeg3(base, sublen3);

            derived.aligned.len3 = ConfigHelper::multiply_dim3(derived.nblock, sublen3);
            derived.aligned.set_leap();
            derived.aligned.set_serialized();

            derived.anchor.len3 = ConfigHelper::get_pardeg3(base, anchor_step3);
            derived.anchor.set_leap();
            derived.anchor.set_serialized();

            derived.assigned.data    = derived.base.serialized;
            derived.assigned.quant   = derived.aligned.serialized;
            derived.assigned.outlier = std::max(derived.base.serialized, derived.aligned.serialized);  // TODO
            derived.assigned.anchor  = derived.anchor.serialized;
        }
    }

    template <class DERIVED, typename T, typename E, typename FP = float>
    void __debug_list_derived(DERIVED const& derived, bool use_anchor = false)
    {
        auto base    = derived.base;
        auto aligned = derived.aligned;
        auto anchor  = derived.anchor;
        auto nblock  = derived.nblock;

        printf("%-*s:  (%u, %u, %u)\n", 16, "sizeof.{T,E,FP}", (int)sizeof(T), (int)sizeof(E), (int)sizeof(FP));
        printf("%-*s:  (%u, %u, %u)\n", 16, "base.len3", base.len3.x, base.len3.y, base.len3.z);
        printf("%-*s:  (%u, %u, %u)\n", 16, "base.leap", base.leap.x, base.leap.y, base.leap.z);
        printf("%-*s:  %'zu\n", 16, "base.serial", base.serialized);

        if (use_anchor) {
            printf("%-*s:  (%u, %u, %u)\n", 16, "nblock", nblock.x, nblock.y, nblock.z);

            printf("%-*s:  (%u, %u, %u)\n", 16, "aligned.len3", aligned.len3.x, aligned.len3.y, aligned.len3.z);
            printf("%-*s:  (%u, %u, %u)\n", 16, "aligned.leap", aligned.leap.x, aligned.leap.y, aligned.leap.z);
            printf("%-*s:  %'zu\n", 16, "aligned.serial", aligned.serialized);

            printf("%-*s:  (%u, %u, %u)\n", 16, "anchor.len3", anchor.len3.x, anchor.len3.y, anchor.len3.z);
            printf("%-*s:  (%u, %u, %u)\n", 16, "anchor.leap", anchor.leap.x, anchor.leap.y, anchor.leap.z);
            printf("%-*s:  %'zu\n", 16, "anchor.serial", anchor.serialized);
        }

        printf("%-*s:  %'zu\n", 16, "len.data", derived.assigned.data);
        printf("%-*s:  %'zu\n", 16, "len.quant", derived.assigned.quant);
        printf("%-*s:  %'zu\n", 16, "len.outlier", derived.assigned.outlier);
        printf("%-*s:  %'zu\n", 16, "len.anchor", derived.assigned.anchor);
    }

    void check_rtlen()
    {
        auto rtlen3    = rtlen.get_len3();
        auto alloclen3 = alloclen.get_len3();

        if (rtlen3.x > alloclen3.x or rtlen3.y > alloclen3.y or rtlen3.z > alloclen3.z or
            rtlen.base.serialized > alloclen.base.serialized)
            throw std::runtime_error("Predictor: the runtime lengths cannot be greater than the allocation lengths.");
    }

    template <typename T, typename E, typename FP = float>
    void debug_list_alloclen(bool use_anchor = false)
    {
        printf("\ndebugging, listing allocation lengths:\n");
        __debug_list_derived<decltype(alloclen), T, E, FP>(alloclen, use_anchor);
    }

    template <typename T, typename E, typename FP = float>
    void debug_list_rtlen(bool use_anchor = false)
    {
        printf("\ndebugging, listing runtime lengths:\n");
        __debug_list_derived<decltype(rtlen), T, E, FP>(rtlen, use_anchor);
    }

   protected:
    struct DerivedLengths alloclen, rtlen;

    float time_elapsed;

    // -----------------------------------------------------------------------------
    //                                  accessor
    // -----------------------------------------------------------------------------
   public:
    // helper
    size_t get_alloclen_data() const { return alloclen.assigned.data; }
    size_t get_alloclen_anchor() const { return alloclen.assigned.anchor; }
    size_t get_alloclen_quant() const { return alloclen.assigned.quant; }
    size_t get_alloclen_outlier() const { return alloclen.assigned.outlier; }

    dim3   get_len3() const { return rtlen.base.len3; }
    dim3   get_leap3() const { return rtlen.base.leap; }
    size_t get_len_data() const { return rtlen.assigned.data; }
    size_t get_len_anchor() const { return rtlen.assigned.anchor; }
    size_t get_len_quant() const { return rtlen.assigned.quant; }
    size_t get_len_outlier() const { return rtlen.assigned.outlier; }

    float get_time_elapsed() const { return time_elapsed; }

    size_t get_x() const { return this->rtlen.get_len3().x; }
    size_t get_y() const { return this->rtlen.get_len3().y; }
    size_t get_z() const { return this->rtlen.get_len3().z; }

    dim3 get_leap() const { return this->rtlen.get_leap(); }
    int  get_ndim() const { return this->rtlen.ndim; }

    void derive_alloclen(cusz_predictortype predictor, dim3 base)
    {
        if (predictor == LorenzoI) {
            // normal
            this->__derive_len(base, this->alloclen);
        }

        else if (predictor == Spline3) {
            // maximum possible
            int sublen[3]      = {32, 8, 8};
            int anchor_step[3] = {8, 8, 8};
            this->__derive_len(base, this->alloclen, sublen, anchor_step, true);
        }
    }

    void derive_rtlen(cusz_predictortype predictor, dim3 base)
    {
        if (predictor == LorenzoI) {
            // normal
            this->__derive_len(base, this->rtlen);
        }
        else if (predictor == Spline3) {
            // maximum possible
            int sublen[3]      = {32, 8, 8};
            int anchor_step[3] = {8, 8, 8};
            this->__derive_len(base, this->rtlen, sublen, anchor_step, true);
        }
    }

    // "real" methods
    virtual ~PredictorBoilerplate() = default;
};

}  // namespace cusz

#endif
