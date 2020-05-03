// 200211

#ifndef PSZ_DUALQUANT_HH
#define PSZ_DUALQUANT_HH

#include <cstddef>

#include "types.hh"

namespace pSZ {
namespace PredictionDualQuantization {

template <typename T, typename Q, int B>
void c_lorenzo_1d1l(T* data, T* outlier, Q* bcode, size_t const* const dims_L16, double const* const ebs_L4, T* pred_err, T* comp_err, size_t b0) {
    auto   radius = static_cast<Q>(dims_L16[RADIUS]);
    size_t _idx0  = b0 * B;
    // prequantization
    for (size_t i0 = 0; i0 < B; i0++) {
        size_t id = _idx0 + i0;
        if (id >= dims_L16[DIM0]) continue;
#ifdef PRED_COMP_ERR
        pred_err[id] = data[id];  // for recording pred error
        comp_err[id] = data[id];
#endif
        data[id] = round(data[id] * ebs_L4[EBx2_r]);
    }
    // postquantization
    for (size_t i0 = 0; i0 < B; i0++) {
        size_t id = _idx0 + i0;
        if (id >= dims_L16[DIM0]) continue;
        T    pred        = id < _idx0 + 1 ? 0 : data[id - 1];
        T    posterr     = data[id] - pred;
        bool quantizable = fabs(posterr) < radius;
        Q    _code       = static_cast<Q>(posterr + radius);
        outlier[id]      = (1 - quantizable) * data[id];
        bcode[id]        = quantizable * _code;
#ifdef PRED_COMP_ERR
        pred_err[id] -= pred * ebs_L4[__2EB];  // for recording pred error
        comp_err[id] -= bcode[id] == 0 ? outlier[id] : static_cast<T>(pred + (bcode[id] - radius)) * ebs_L4[__2EB];
#endif
    }
}

template <typename T, typename Q, int B>
void c_lorenzo_2d1l(T*                  data,
                    T*                  outlier,
                    Q*                  bcode,
                    size_t const* const dims_L16,
                    double const* const ebs_L4,
                    T*                  pred_err,
                    T*                  comp_err,
                    size_t              b0,
                    size_t              b1) {
    T _s[B + 1][B + 1];  // 2D interpretation of data
    memset(_s, 0, (B + 1) * (B + 1) * sizeof(T));
    auto radius = static_cast<Q>(dims_L16[RADIUS]);

    size_t _idx1 = b1 * B;
    size_t _idx0 = b0 * B;

    // prequantization
    for (size_t i1 = 0; i1 < B; i1++) {
        for (size_t i0 = 0; i0 < B; i0++) {
            size_t gi1 = _idx1 + i1;
            size_t gi0 = _idx0 + i0;
            if (gi1 >= dims_L16[DIM1] or gi0 >= dims_L16[DIM0]) continue;
            size_t id          = gi0 + gi1 * dims_L16[DIM0];
            _s[i1 + 1][i0 + 1] = round(data[id] * ebs_L4[EBx2_r]);
#ifdef PRED_COMP_ERR
            pred_err[id] = data[id];  // for recording pred error
            comp_err[id] = data[id];
#endif
        }
    }
    // postquantization
    for (size_t i1 = 0; i1 < B; i1++) {
        for (size_t i0 = 0; i0 < B; i0++) {
            size_t gi1 = _idx1 + i1;
            size_t gi0 = _idx0 + i0;
            if (gi1 >= dims_L16[DIM1] or gi0 >= dims_L16[DIM0]) continue;
            size_t id          = gi0 + gi1 * dims_L16[DIM0];
            T      pred        = _s[i1 + 1][i0] + _s[i1][i0 + 1] - _s[i1][i0];
            T      posterr     = _s[i1 + 1][i0 + 1] - pred;
            bool   quantizable = fabs(posterr) < radius;
            Q      _code       = static_cast<Q>(posterr + radius);
            outlier[id]        = (1 - quantizable) * _s[i1 + 1][i0 + 1];
            bcode[id]          = quantizable * _code;
#ifdef PRED_COMP_ERR
            pred_err[id] -= pred * ebs_L4[__2EB];  // for recording pred error
            comp_err[id] -= bincode[id] == 0 ? outlier[id] : static_cast<T>(pred + (bincode[id] - radius)) * ebs_L4[__2EB];
#endif
        }
    }
}

template <typename T, typename Q, int B>
void c_lorenzo_3d1l(T*                  data,
                    T*                  outlier,
                    Q*                  bcode,
                    size_t const* const dims_L16,
                    double const* const ebs_L4,
                    T*                  pred_err,
                    T*                  comp_err,
                    size_t              b0,
                    size_t              b1,
                    size_t              b2) {
    T _s[B + 1][B + 1][B + 1];
    memset(_s, 0, (B + 1) * (B + 1) * (B + 1) * sizeof(T));
    auto radius = static_cast<Q>(dims_L16[RADIUS]);

    size_t _idx2 = b2 * B;
    size_t _idx1 = b1 * B;
    size_t _idx0 = b0 * B;

    // prequantization
    for (size_t i2 = 0; i2 < B; i2++) {
        for (size_t i1 = 0; i1 < B; i1++) {
            for (size_t i0 = 0; i0 < B; i0++) {
                size_t gi2 = _idx2 + i2;
                size_t gi1 = _idx1 + i1;
                size_t gi0 = _idx0 + i0;
                if (gi2 >= dims_L16[DIM2] or gi1 >= dims_L16[DIM1] or gi0 >= dims_L16[DIM0]) continue;
                size_t id                  = gi0 + gi1 * dims_L16[DIM0] + gi2 * dims_L16[DIM1] * dims_L16[DIM0];
                _s[i2 + 1][i1 + 1][i0 + 1] = round(data[id] * ebs_L4[EBx2_r]);
#ifdef PRED_COMP_ERR
                pred_err[id] = data[id];  // for recording pred error
                comp_err[id] = data[id];
#endif
            }
        }
    }
    // postquantization
    for (size_t i2 = 0; i2 < B; i2++) {
        for (size_t i1 = 0; i1 < B; i1++) {
            for (size_t i0 = 0; i0 < B; i0++) {
                size_t gi2 = _idx2 + i2;
                size_t gi1 = _idx1 + i1;
                size_t gi0 = _idx0 + i0;
                if (gi2 >= dims_L16[DIM2] or gi1 >= dims_L16[DIM1] or gi0 >= dims_L16[DIM0]) continue;
                size_t id   = gi0 + gi1 * dims_L16[DIM0] + gi2 * dims_L16[DIM1] * dims_L16[DIM0];
                T      pred = _s[i2][i1][i0]                                                          // +, dist=3
                         - _s[i2 + 1][i1][i0] - _s[i2][i1 + 1][i0] - _s[i2][i1][i0 + 1]               // -, dist=2
                         + _s[i2 + 1][i1 + 1][i0] + _s[i2 + 1][i1][i0 + 1] + _s[i2][i1 + 1][i0 + 1];  // +, dist=1
                T    posterr     = _s[i2 + 1][i1 + 1][i0 + 1] - pred;
                bool quantizable = fabs(posterr) < radius;
                Q    _code       = static_cast<Q>(posterr + radius);
                outlier[id]      = (1 - quantizable) * _s[i2 + 1][i1 + 1][i0 + 1];
                bcode[id]        = quantizable * _code;
#ifdef PRED_COMP_ERR
                pred_err[id] -= pred * ebs_L4[__2EB];  // for recording pred error
                comp_err[id] -= bincode[id] == 0 ? outlier[id] : static_cast<T>(pred + (bincode[id] - radius)) * ebs_L4[__2EB];
#endif
            }
        }
    }
}

template <typename T, typename Q, int B>
void x_lorenzo_1d1l(T* xdata, T* outlier, Q* bcode, size_t const* const dims_L16, double _2EB, size_t b0) {
    auto   radius = static_cast<Q>(dims_L16[RADIUS]);
    size_t _idx0  = b0 * B;
    for (size_t i0 = 0; i0 < B; i0++) {
        size_t id = _idx0 + i0;
        if (id >= dims_L16[DIM0]) continue;
        T pred    = id < _idx0 + 1 ? 0 : xdata[id - 1];
        xdata[id] = bcode[id] == 0 ? outlier[id] : static_cast<T>(pred + (bcode[id] - radius));
    }
    for (size_t i0 = 0; i0 < B; i0++) {
        size_t id = _idx0 + i0;
        if (id >= dims_L16[DIM0]) continue;
        xdata[id] = xdata[id] * _2EB;
    }
}

template <typename T, typename Q, int B>
void x_lorenzo_2d1l(T* xdata, T* outlier, Q* bcode, size_t const* const dims_L16, double _2EB, size_t b0, size_t b1) {
    T _s[B + 1][B + 1];
    memset(_s, 0, (B + 1) * (B + 1) * sizeof(T));
    auto radius = static_cast<Q>(dims_L16[RADIUS]);

    size_t _idx1 = b1 * B;
    size_t _idx0 = b0 * B;

    for (size_t i1 = 0; i1 < B; i1++) {
        for (size_t i0 = 0; i0 < B; i0++) {
            size_t gi1 = _idx1 + i1;
            size_t gi0 = _idx0 + i0;
            if (gi1 >= dims_L16[DIM1] or gi0 >= dims_L16[DIM0]) continue;
            const size_t id    = gi0 + gi1 * dims_L16[DIM0];
            T            pred  = _s[i1][i0 + 1] + _s[i1 + 1][i0] - _s[i1][i0];
            _s[i1 + 1][i0 + 1] = bcode[id] == 0 ? outlier[id] : static_cast<T>(pred + (bcode[id] - radius));
            xdata[id]          = _s[i1 + 1][i0 + 1] * _2EB;
        }
    }
}

template <typename T, typename Q, int B>
void x_lorenzo_3d1l(T*                  xdata,
                    T*                  outlier,
                    Q*                  bcode,
                    size_t const* const dims_L16,  //
                    double              _2EB,
                    size_t              b0,
                    size_t              b1,
                    size_t              b2) {
    T _s[B + 1][B + 1][B + 1];
    memset(_s, 0, (B + 1) * (B + 1) * (B + 1) * sizeof(T));
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
                T      pred = _s[i2][i1][i0]                                                          // +, dist=3
                         - _s[i2 + 1][i1][i0] - _s[i2][i1 + 1][i0] - _s[i2][i1][i0 + 1]               // -, dist=2
                         + _s[i2 + 1][i1 + 1][i0] + _s[i2 + 1][i1][i0 + 1] + _s[i2][i1 + 1][i0 + 1];  // +, dist=1
                _s[i2 + 1][i1 + 1][i0 + 1] = bcode[id] == 0 ? outlier[id] : static_cast<T>(pred + (bcode[id] - radius));
                xdata[id]                  = _s[i2 + 1][i1 + 1][i0 + 1] * _2EB;
            }
        }
    }
}

}  // namespace PredictionDualQuantization
}  // namespace pSZ

#endif
