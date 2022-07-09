#ifndef CUSZ_PSZ_DUALQUANT_HH
#define CUSZ_PSZ_DUALQUANT_HH

/**
 * @file psz_dualquant.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.1.4
 * @date 2020-02-11
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cstddef>

#include "../common.hh"

namespace psz {
namespace dualquant {

template <typename Data, typename Quant, int B>
void c_lorenzo_1d1l(
    Data*               d,
    Data*               outlier,
    Quant*              q,
    size_t const* const dims,
    double const* const eb_variants,
    Data*               pred_err,
    Data*               comp_err,
    size_t              b0)
{
    auto   radius = static_cast<Quant>(dims[RADIUS]);
    size_t _idx0  = b0 * B;
    // prequant
    for (size_t i0 = 0; i0 < B; i0++) {
        size_t id = _idx0 + i0;
        if (id >= dims[DIM0]) continue;
#ifdef PRED_COMP_ERR
        pred_err[id] = data[id];  // for recording pred error
        comp_err[id] = data[id];
#endif
        d[id] = round(d[id] * eb_variants[EBx2_r]);
    }
    // postquant
    for (size_t i0 = 0; i0 < B; i0++) {
        size_t id = _idx0 + i0;
        if (id >= dims[DIM0]) continue;
        Data pred        = id < _idx0 + 1 ? 0 : d[id - 1];
        Data delta       = d[id] - pred;
        bool quantizable = fabs(delta) < radius;
        auto _code       = static_cast<Quant>(delta + radius);
        outlier[id]      = (1 - quantizable) * d[id];
        q[id]            = quantizable * _code;
#ifdef PRED_COMP_ERR
        pred_err[id] -= pred * eb_variants[__2EB];  // for recording pred error
        comp_err[id] -= q[id] == 0 ? outlier[id] : static_cast<T>(pred + (q[id] - radius)) * eb_variants[__2EB];
#endif
    }
}

template <typename Data, typename Quant, int B>
void c_lorenzo_2d1l(
    Data*               d,
    Data*               outlier,
    Quant*              q,
    size_t const* const dims,
    double const* const eb_variants,
    Data*               pred_err,
    Data*               comp_err,
    size_t              b0,
    size_t              b1)
{
    Data _s[B + 1][B + 1];  // 2D interpretation of data
    memset(_s, 0, (B + 1) * (B + 1) * sizeof(Data));
    auto radius = static_cast<Quant>(dims[RADIUS]);

    size_t _idx1 = b1 * B;
    size_t _idx0 = b0 * B;

    // prequant
    for (size_t i1 = 0; i1 < B; i1++) {
        for (size_t i0 = 0; i0 < B; i0++) {
            size_t gi1 = _idx1 + i1;
            size_t gi0 = _idx0 + i0;
            if (gi1 >= dims[DIM1] or gi0 >= dims[DIM0]) continue;
            size_t id          = gi0 + gi1 * dims[DIM0];
            _s[i1 + 1][i0 + 1] = round(d[id] * eb_variants[EBx2_r]);
#ifdef PRED_COMP_ERR
            pred_err[id] = data[id];  // for recording pred error
            comp_err[id] = data[id];
#endif
        }
    }
    // postquant
    for (size_t i1 = 0; i1 < B; i1++) {
        for (size_t i0 = 0; i0 < B; i0++) {
            size_t gi1 = _idx1 + i1;
            size_t gi0 = _idx0 + i0;
            if (gi1 >= dims[DIM1] or gi0 >= dims[DIM0]) continue;
            size_t id          = gi0 + gi1 * dims[DIM0];
            Data   pred        = _s[i1 + 1][i0] + _s[i1][i0 + 1] - _s[i1][i0];
            Data   delta       = _s[i1 + 1][i0 + 1] - pred;
            bool   quantizable = fabs(delta) < radius;
            auto   _code       = static_cast<Quant>(delta + radius);
            outlier[id]        = (1 - quantizable) * _s[i1 + 1][i0 + 1];
            q[id]              = quantizable * _code;
#ifdef PRED_COMP_ERR
            pred_err[id] -= pred * eb_variants[__2EB];  // for recording pred error
            comp_err[id] -=
                bincode[id] == 0 ? outlier[id] : static_cast<T>(pred + (bincode[id] - radius)) * eb_variants[__2EB];
#endif
        }
    }
}

template <typename Data, typename Quant, int B>
void c_lorenzo_3d1l(
    Data*               d,
    Data*               outlier,
    Quant*              q,
    size_t const* const dims,
    double const* const eb_variants,
    Data*               pred_err,
    Data*               comp_err,
    size_t              b0,
    size_t              b1,
    size_t              b2)
{
    Data _s[B + 1][B + 1][B + 1];
    memset(_s, 0, (B + 1) * (B + 1) * (B + 1) * sizeof(Data));
    auto radius = static_cast<Quant>(dims[RADIUS]);

    size_t _idx2 = b2 * B;
    size_t _idx1 = b1 * B;
    size_t _idx0 = b0 * B;

    // prequant
    for (size_t i2 = 0; i2 < B; i2++) {
        for (size_t i1 = 0; i1 < B; i1++) {
            for (size_t i0 = 0; i0 < B; i0++) {
                size_t gi2 = _idx2 + i2;
                size_t gi1 = _idx1 + i1;
                size_t gi0 = _idx0 + i0;
                if (gi2 >= dims[DIM2] or gi1 >= dims[DIM1] or gi0 >= dims[DIM0]) continue;
                size_t id                  = gi0 + gi1 * dims[DIM0] + gi2 * dims[DIM1] * dims[DIM0];
                _s[i2 + 1][i1 + 1][i0 + 1] = round(d[id] * eb_variants[EBx2_r]);
#ifdef PRED_COMP_ERR
                pred_err[id] = data[id];  // for recording pred error
                comp_err[id] = data[id];
#endif
            }
        }
    }
    // postquant
    for (size_t i2 = 0; i2 < B; i2++) {
        for (size_t i1 = 0; i1 < B; i1++) {
            for (size_t i0 = 0; i0 < B; i0++) {
                size_t gi2 = _idx2 + i2;
                size_t gi1 = _idx1 + i1;
                size_t gi0 = _idx0 + i0;
                if (gi2 >= dims[DIM2] or gi1 >= dims[DIM1] or gi0 >= dims[DIM0]) continue;
                size_t id   = gi0 + gi1 * dims[DIM0] + gi2 * dims[DIM1] * dims[DIM0];
                Data   pred = _s[i2][i1][i0]                                                             // +, dist=3
                            - _s[i2 + 1][i1][i0] - _s[i2][i1 + 1][i0] - _s[i2][i1][i0 + 1]               // -, dist=2
                            + _s[i2 + 1][i1 + 1][i0] + _s[i2 + 1][i1][i0 + 1] + _s[i2][i1 + 1][i0 + 1];  // +, dist=1
                Data  delta       = _s[i2 + 1][i1 + 1][i0 + 1] - pred;
                bool  quantizable = fabs(delta) < radius;
                Quant _code       = static_cast<Quant>(delta + radius);
                outlier[id]       = (1 - quantizable) * _s[i2 + 1][i1 + 1][i0 + 1];
                q[id]             = quantizable * _code;
#ifdef PRED_COMP_ERR
                pred_err[id] -= pred * eb_variants[__2EB];  // for recording pred error
                comp_err[id] -=
                    bincode[id] == 0 ? outlier[id] : static_cast<T>(pred + (bincode[id] - radius)) * eb_variants[__2EB];
#endif
            }
        }
    }
}

template <typename Data, typename Quant, int B>
void x_lorenzo_1d1l(Data* xd, Data* outlier, Quant* q, size_t const* const dims, double _2EB, size_t b0)
{
    auto   radius = static_cast<Quant>(dims[RADIUS]);
    size_t _idx0  = b0 * B;
    for (size_t i0 = 0; i0 < B; i0++) {
        size_t id = _idx0 + i0;
        if (id >= dims[DIM0]) continue;
        Data pred = id < _idx0 + 1 ? 0 : xd[id - 1];
        xd[id]    = q[id] == 0 ? outlier[id] : static_cast<Data>(pred + (q[id] - radius));
    }
    for (size_t i0 = 0; i0 < B; i0++) {
        size_t id = _idx0 + i0;
        if (id >= dims[DIM0]) continue;
        xd[id] = xd[id] * _2EB;
    }
}

template <typename Data, typename Quant, int B>
void x_lorenzo_2d1l(Data* xd, Data* outlier, Quant* q, size_t const* const dims, double _2EB, size_t b0, size_t b1)
{
    Data _s[B + 1][B + 1];
    memset(_s, 0, (B + 1) * (B + 1) * sizeof(Data));
    auto radius = static_cast<Quant>(dims[RADIUS]);

    size_t _idx1 = b1 * B;
    size_t _idx0 = b0 * B;

    for (size_t i1 = 0; i1 < B; i1++) {
        for (size_t i0 = 0; i0 < B; i0++) {
            size_t gi1 = _idx1 + i1;
            size_t gi0 = _idx0 + i0;
            if (gi1 >= dims[DIM1] or gi0 >= dims[DIM0]) continue;
            const size_t id    = gi0 + gi1 * dims[DIM0];
            Data         pred  = _s[i1][i0 + 1] + _s[i1 + 1][i0] - _s[i1][i0];
            _s[i1 + 1][i0 + 1] = q[id] == 0 ? outlier[id] : static_cast<Data>(pred + (q[id] - radius));
            xd[id]             = _s[i1 + 1][i0 + 1] * _2EB;
        }
    }
}

template <typename Data, typename Quant, int B>
void x_lorenzo_3d1l(
    Data*               xd,
    Data*               outlier,
    Quant*              q,
    size_t const* const dims,  //
    double              _2EB,
    size_t              b0,
    size_t              b1,
    size_t              b2)
{
    Data _s[B + 1][B + 1][B + 1];
    memset(_s, 0, (B + 1) * (B + 1) * (B + 1) * sizeof(Data));
    auto radius = static_cast<Quant>(dims[RADIUS]);

    size_t _idx2 = b2 * B;
    size_t _idx1 = b1 * B;
    size_t _idx0 = b0 * B;

    for (size_t i2 = 0; i2 < B; i2++) {
        for (size_t i1 = 0; i1 < B; i1++) {
            for (size_t i0 = 0; i0 < B; i0++) {
                size_t gi2 = _idx2 + i2;
                size_t gi1 = _idx1 + i1;
                size_t gi0 = _idx0 + i0;
                if (gi2 >= dims[DIM2] or gi1 >= dims[DIM1] or gi0 >= dims[DIM0]) continue;
                size_t id   = gi0 + gi1 * dims[DIM0] + gi2 * dims[DIM1] * dims[DIM0];
                Data   pred = _s[i2][i1][i0]                                                             // +, dist=3
                            - _s[i2 + 1][i1][i0] - _s[i2][i1 + 1][i0] - _s[i2][i1][i0 + 1]               // -, dist=2
                            + _s[i2 + 1][i1 + 1][i0] + _s[i2 + 1][i1][i0 + 1] + _s[i2][i1 + 1][i0 + 1];  // +, dist=1
                _s[i2 + 1][i1 + 1][i0 + 1] = q[id] == 0 ? outlier[id] : static_cast<Data>(pred + (q[id] - radius));
                xd[id]                     = _s[i2 + 1][i1 + 1][i0 + 1] * _2EB;
            }
        }
    }
}

}  // namespace dualquant
}  // namespace psz

#endif
