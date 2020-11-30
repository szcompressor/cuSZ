// 200213

#ifndef PSZ_DUALQUANT_CUH
#define PSZ_DUALQUANT_CUH

#include <cstddef>

#include "types.hh"

namespace pSZ {
namespace PredictionQuantizationReconstructionBlocked {

template <typename T, typename Q, int B>
void c_lorenzo_1d1l(T*                  data,
                    T*                  outlier,
                    Q*                  bincode,
                    size_t const* const dims_L16,
                    double const* const ebs_L4,
                    T*                  pred_err,
                    T*                  comp_err,
                    size_t              b0) {
    auto   radius = static_cast<Q>(dims_L16[RADIUS]);
    size_t _idx0 = b0 * B;
    for (size_t i0 = 0; i0 < B; i0++) {
        size_t id = _idx0 + i0;
        if (id >= dims_L16[DIM0]) continue;
        T    pred        = id < _idx0 + 1 ? 0 : data[id - 1];
        T    err         = data[id] - pred;
        T    dup         = data[id];
        Q    bin_count   = fabs(err) * ebs_L4[EBr] + 1;
        bool quantizable = fabs(bin_count) < dims_L16[CAP];
        if (err < 0) bin_count = -bin_count;
        Q _code  = static_cast<Q>(bin_count / 2) + radius;
        data[id] = pred + (_code - radius) * ebs_L4[EBx2];
#ifdef PRED_COMP_ERR
        pred_err[id] = err;
        comp_err[id] = dup - data[id];  // origin - decompressed
#endif
        outlier[id] = (1 - quantizable) * data[id];
        bincode[id] = quantizable * _code;
    }
}

template <typename T, typename Q, int B>
void c_lorenzo_2d1l(T*                  data,
                    T*                  outlier,
                    Q*                  bincode,
                    size_t const* const dims_L16,
                    double const* const ebs_L4,
                    T*                  pred_err,
                    T*                  comp_err,
                    size_t              b0,
                    size_t              b1) {
    T __s[B + 1][B + 1];
    memset(__s, 0, (B + 1) * (B + 1) * sizeof(T));
    auto radius = static_cast<Q>(dims_L16[RADIUS]);

    size_t _idx1 = b1 * B;
    size_t _idx0 = b0 * B;

    for (size_t i1 = 0; i1 < B; i1++) {
        for (size_t i0 = 0; i0 < B; i0++) {
            size_t gi1 = _idx1 + i1;
            size_t gi0 = _idx0 + i0;

            if (gi1 >= dims_L16[DIM1] or gi0 >= dims_L16[DIM0]) continue;
            size_t id           = gi0 + gi1 * dims_L16[DIM0];
            __s[i1 + 1][i0 + 1] = data[id];

            T    pred        = __s[i1 + 1][i0] + __s[i1][i0 + 1] - __s[i1][i0];
            T    err         = __s[i1 + 1][i0 + 1] - pred;
            Q    bin_count   = fabs(err) * ebs_L4[EBr] + 1;
            bool quantizable = fabs(bin_count) < dims_L16[CAP];

            if (err < 0) bin_count = -bin_count;
            Q _code             = static_cast<Q>(bin_count / 2) + radius;
            __s[i1 + 1][i0 + 1] = pred + (_code - radius) * ebs_L4[EBx2];
#ifdef PRED_COMP_ERR
            pred_err[id] = err;
            comp_err[id] = data[id] - __s[i1 + 1][i0 + 1];  // origin - decompressed
#endif
            outlier[id] = (1 - quantizable) * __s[i1 + 1][i0 + 1];
            bincode[id] = quantizable * _code;
        }
    }
}

template <typename T, typename Q, int B>
void c_lorenzo_3d1l(T*                  data,
                    T*                  outlier,
                    Q*                  bincode,
                    size_t const* const dims_L16,
                    double const* const ebs_L4,
                    T*                  pred_err,
                    T*                  comp_err,
                    size_t              b0,
                    size_t              b1,
                    size_t              b2) {
    T __s[B + 1][B + 1][B + 1];
    memset(__s, 0, (B + 1) * (B + 1) * (B + 1) * sizeof(T));
    auto radius = static_cast<Q>(dims_L16[RADIUS]);

    size_t _idx2 = b2 * B;
    size_t _idx1 = b1 * B;
    size_t _idx0 = b0 * B;

    for (size_t i2 = 0; i2 < B; i2++) {
        for (size_t i1 = 0; i1 < B; i1++) {
            for (size_t i0 = 0; i0 < B; i0++) {
                size_t gi2 = _idx2 + i2;
                size_t gi1 = _idx1 + i1;
                size_t gi0 = _idx0 + i0;

                if (gi2 >= dims_L16[DIM2] or gi1 >= dims_L16[DIM1] or gi0 >= dims_L16[DIM0]) continue;
                size_t id                   = gi0 + gi1 * dims_L16[DIM0] + gi2 * dims_L16[DIM1] * dims_L16[DIM0];
                __s[i2 + 1][i1 + 1][i0 + 1] = data[id];

                T pred = __s[i2][i1][i0]                                                                //
                         - __s[i2 + 1][i1][i0 + 1] - __s[i2 + 1][i1 + 1][i0] - __s[i2][i1 + 1][i0 + 1]  //
                         + __s[i2 + 1][i1][i0] + __s[i2][i1 + 1][i0] + __s[i2][i1][i0 + 1];
                T    err         = __s[i2 + 1][i1 + 1][i0 + 1] - pred;
                Q    bin_count   = fabs(err) * ebs_L4[EBr] + 1;
                bool quantizable = fabs(bin_count) < dims_L16[CAP];

                if (err < 0) bin_count = -bin_count;
                Q _code = static_cast<Q>(bin_count / 2) + radius;

                __s[i2 + 1][i1 + 1][i0 + 1] = pred + (_code - radius) * ebs_L4[EBx2];
#ifdef PRED_COMP_ERR
                pred_err[id] = err;
                comp_err[id] = data[id] - __s[i2 + 1][i1 + 1][i0 + 1];  // origin - decompressed
#endif
                outlier[id] = (1 - quantizable) * __s[i2 + 1][i1 + 1][i0 + 1];
                bincode[id] = quantizable * _code;
            }
        }
    }
}

template <typename T, typename Q, int B>
void x_lorenzo_1d1l(T* xdata, T* outlier, Q* bincode, size_t const* const dims_L16, double const* const ebs_L4, size_t b0) {
    auto   radius = static_cast<Q>(dims_L16[RADIUS]);
    size_t _idx0 = b0 * B;
    for (size_t i0 = 0; i0 < B; i0++) {
        size_t id = _idx0 + i0;
        if (id >= dims_L16[DIM0]) continue;
        T pred    = id < _idx0 + 1 ? 0 : xdata[id - 1];
        xdata[id] = bincode[id] == 0 ? outlier[id] : static_cast<T>(pred + (bincode[id] - radius) * ebs_L4[EBx2]);
    }
}

template <typename T, typename Q, int B>
void x_lorenzo_2d1l(T* xdata, T* outlier, Q* bincode, size_t const* const dims_L16, double const* const ebs_L4, size_t b0, size_t b1) {
    T __s[B + 1][B + 1];
    memset(__s, 0, (B + 1) * (B + 1) * sizeof(T));
    auto radius = static_cast<Q>(dims_L16[RADIUS]);

    size_t _idx1 = b1 * B;
    size_t _idx0 = b0 * B;

    for (size_t i1 = 0; i1 < B; i1++) {
        for (size_t i0 = 0; i0 < B; i0++) {
            size_t gi1 = _idx1 + i1;
            size_t gi0 = _idx0 + i0;
            if (gi1 >= dims_L16[DIM1] or gi0 >= dims_L16[DIM0]) continue;
            const size_t id     = gi0 + gi1 * dims_L16[DIM0];
            T            pred   = __s[i1][i0 + 1] + __s[i1 + 1][i0] - __s[i1][i0];
            __s[i1 + 1][i0 + 1] = bincode[id] == 0 ? outlier[id] : static_cast<T>(pred + (bincode[id] - radius) * ebs_L4[EBx2]);
            xdata[id]           = __s[i1 + 1][i0 + 1];
        }
    }
}

template <typename T, typename Q, int B>
void x_lorenzo_3d1l(T*                  xdata,
                    T*                  outlier,
                    Q*                  bincode,
                    size_t const* const dims_L16,
                    double const* const ebs_L4,
                    size_t              b0,
                    size_t              b1,
                    size_t              b2) {
    T __s[B + 1][B + 1][B + 1];
    memset(__s, 0, (B + 1) * (B + 1) * (B + 1) * sizeof(T));
    auto radius = static_cast<Q>(dims_L16[RADIUS]);

    size_t _idx2 = b2 * B;
    size_t _idx1 = b1 * B;
    size_t _idx0 = b0 * B;

    for (size_t i2 = 0; i2 < B; i2++) {
        for (size_t i1 = 0; i1 < B; i1++) {
            for (size_t i0 = 0; i0 < B; i0++) {
                size_t gi2 = _idx2 + i2;
                size_t gi1 = _idx1 + i1;
                size_t gi0 = _idx0 + i0;
                if (gi2 >= dims_L16[DIM2] or gi1 >= dims_L16[DIM1] or gi0 >= dims_L16[DIM0]) continue;
                size_t id   = gi0 + gi1 * dims_L16[DIM0] + gi2 * dims_L16[DIM1] * dims_L16[DIM0];
                T      pred = __s[i2][i1][i0]                                                           //
                         - __s[i2 + 1][i1][i0 + 1] - __s[i2 + 1][i1 + 1][i0] - __s[i2][i1 + 1][i0 + 1]  //
                         + __s[i2 + 1][i1][i0] + __s[i2][i1 + 1][i0] + __s[i2][i1][i0 + 1];

                __s[i2 + 1][i1 + 1][i0 + 1] = bincode[id] == 0 ? outlier[id] : static_cast<T>(pred + (bincode[id] - radius) * ebs_L4[EBx2]);

                xdata[id] = __s[i2 + 1][i1 + 1][i0 + 1];
            }
        }
    }
}

}  // namespace PredictionQuantizationReconstructionBlocked
}  // namespace pSZ

#endif
