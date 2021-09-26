#ifndef CUSZ_PSZ_SZ1_4_BLOCKED_HH
#define CUSZ_PSZ_SZ1_4_BLOCKED_HH

/**
 * @file psz_14blocked.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.1.4
 * @date 2020-02-13
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cstddef>

#include "../common.hh"

namespace psz {
namespace sz1_4_blocked {

template <typename Data, typename Quant, int B>
void c_lorenzo_1d1l(
    Data*               data,
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
    for (size_t i0 = 0; i0 < B; i0++) {
        size_t id = _idx0 + i0;
        if (id >= dims[DIM0]) continue;
        Data  pred        = id < _idx0 + 1 ? 0 : data[id - 1];
        Data  err         = data[id] - pred;
        Data  dup         = data[id];
        Quant bin_count   = fabs(err) * eb_variants[EBr] + 1;
        bool  quantizable = fabs(bin_count) < dims[CAP];
        if (err < 0) bin_count = -bin_count;
        auto _code = static_cast<Quant>(bin_count / 2) + radius;
        data[id]   = pred + (_code - radius) * eb_variants[EBx2];
#ifdef PRED_COMP_ERR
        pred_err[id] = err;
        comp_err[id] = dup - data[id];  // origin - decompressed
#endif
        outlier[id] = (1 - quantizable) * data[id];
        q[id]       = quantizable * _code;
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
    Data __s[B + 1][B + 1];
    memset(__s, 0, (B + 1) * (B + 1) * sizeof(Data));
    auto radius = static_cast<Quant>(dims[RADIUS]);

    size_t _idx1 = b1 * B;
    size_t _idx0 = b0 * B;

    for (size_t i1 = 0; i1 < B; i1++) {
        for (size_t i0 = 0; i0 < B; i0++) {
            size_t gi1 = _idx1 + i1;
            size_t gi0 = _idx0 + i0;

            if (gi1 >= dims[DIM1] or gi0 >= dims[DIM0]) continue;
            size_t id           = gi0 + gi1 * dims[DIM0];
            __s[i1 + 1][i0 + 1] = d[id];

            Data  pred        = __s[i1 + 1][i0] + __s[i1][i0 + 1] - __s[i1][i0];
            Data  err         = __s[i1 + 1][i0 + 1] - pred;
            Quant bin_count   = fabs(err) * eb_variants[EBr] + 1;
            bool  quantizable = fabs(bin_count) < dims[CAP];

            if (err < 0) bin_count = -bin_count;
            Quant _code         = static_cast<Quant>(bin_count / 2) + radius;
            __s[i1 + 1][i0 + 1] = pred + (_code - radius) * eb_variants[EBx2];
#ifdef PRED_COMP_ERR
            pred_err[id] = err;
            comp_err[id] = data[id] - __s[i1 + 1][i0 + 1];  // origin - decompressed
#endif
            outlier[id] = (1 - quantizable) * __s[i1 + 1][i0 + 1];
            q[id]       = quantizable * _code;
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
    Data __s[B + 1][B + 1][B + 1];
    memset(__s, 0, (B + 1) * (B + 1) * (B + 1) * sizeof(Data));
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
                size_t id                   = gi0 + gi1 * dims[DIM0] + gi2 * dims[DIM1] * dims[DIM0];
                __s[i2 + 1][i1 + 1][i0 + 1] = d[id];

                Data pred = __s[i2][i1][i0]                                                                //
                            - __s[i2 + 1][i1][i0 + 1] - __s[i2 + 1][i1 + 1][i0] - __s[i2][i1 + 1][i0 + 1]  //
                            + __s[i2 + 1][i1][i0] + __s[i2][i1 + 1][i0] + __s[i2][i1][i0 + 1];
                Data  err         = __s[i2 + 1][i1 + 1][i0 + 1] - pred;
                Quant bin_count   = fabs(err) * eb_variants[EBr] + 1;
                bool  quantizable = fabs(bin_count) < dims[CAP];

                if (err < 0) bin_count = -bin_count;
                Quant _code = static_cast<Quant>(bin_count / 2) + radius;

                __s[i2 + 1][i1 + 1][i0 + 1] = pred + (_code - radius) * eb_variants[EBx2];
#ifdef PRED_COMP_ERR
                pred_err[id] = err;
                comp_err[id] = data[id] - __s[i2 + 1][i1 + 1][i0 + 1];  // origin - decompressed
#endif
                outlier[id] = (1 - quantizable) * __s[i2 + 1][i1 + 1][i0 + 1];
                q[id]       = quantizable * _code;
            }
        }
    }
}

template <typename Data, typename Quant, int B>
void x_lorenzo_1d1l(
    Data*               xd,
    Data*               outlier,
    Quant*              q,
    size_t const* const dims,
    double const* const eb_variants,
    size_t              b0)
{
    auto   radius = static_cast<Quant>(dims[RADIUS]);
    size_t _idx0  = b0 * B;
    for (size_t i0 = 0; i0 < B; i0++) {
        size_t id = _idx0 + i0;
        if (id >= dims[DIM0]) continue;
        Data pred = id < _idx0 + 1 ? 0 : xd[id - 1];
        xd[id]    = q[id] == 0 ? outlier[id] : static_cast<Data>(pred + (q[id] - radius) * eb_variants[EBx2]);
    }
}

template <typename Data, typename Quant, int B>
void x_lorenzo_2d1l(
    Data*               xd,
    Data*               outlier,
    Quant*              q,
    size_t const* const dims,
    double const* const eb_variants,
    size_t              b0,
    size_t              b1)
{
    Data __s[B + 1][B + 1];
    memset(__s, 0, (B + 1) * (B + 1) * sizeof(Data));
    auto radius = static_cast<Quant>(dims[RADIUS]);

    size_t _idx1 = b1 * B;
    size_t _idx0 = b0 * B;

    for (size_t i1 = 0; i1 < B; i1++) {
        for (size_t i0 = 0; i0 < B; i0++) {
            size_t gi1 = _idx1 + i1;
            size_t gi0 = _idx0 + i0;
            if (gi1 >= dims[DIM1] or gi0 >= dims[DIM0]) continue;
            const size_t id   = gi0 + gi1 * dims[DIM0];
            Data         pred = __s[i1][i0 + 1] + __s[i1 + 1][i0] - __s[i1][i0];
            __s[i1 + 1][i0 + 1] =
                q[id] == 0 ? outlier[id] : static_cast<Data>(pred + (q[id] - radius) * eb_variants[EBx2]);
            xd[id] = __s[i1 + 1][i0 + 1];
        }
    }
}

template <typename Data, typename Quant, int B>
void x_lorenzo_3d1l(
    Data*               xd,
    Data*               outlier,
    Quant*              q,
    size_t const* const dims,
    double const* const eb_variants,
    size_t              b0,
    size_t              b1,
    size_t              b2)
{
    Data __s[B + 1][B + 1][B + 1];
    memset(__s, 0, (B + 1) * (B + 1) * (B + 1) * sizeof(Data));
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
                Data   pred = __s[i2][i1][i0]                                                              //
                            - __s[i2 + 1][i1][i0 + 1] - __s[i2 + 1][i1 + 1][i0] - __s[i2][i1 + 1][i0 + 1]  //
                            + __s[i2 + 1][i1][i0] + __s[i2][i1 + 1][i0] + __s[i2][i1][i0 + 1];

                __s[i2 + 1][i1 + 1][i0 + 1] =
                    q[id] == 0 ? outlier[id] : static_cast<Data>(pred + (q[id] - radius) * eb_variants[EBx2]);

                xd[id] = __s[i2 + 1][i1 + 1][i0 + 1];
            }
        }
    }
}

}  // namespace sz1_4_blocked
}  // namespace psz

#endif
