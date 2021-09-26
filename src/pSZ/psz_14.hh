#ifndef CUSZ_PSZ_14_HH
#define CUSZ_PSZ_14_HH

/**
 * @file psz_14.hh
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
namespace sz1_4 {

template <typename Data, typename Quant, int B>
void c_lorenzo_1d1l(
    Data*               d,
    Data*               outlier,
    Quant*              q,
    size_t const* const dims,
    double const* const eb_variants,
    Data*               pred_err,
    Data*               comp_err)
{
    auto radius = static_cast<Quant>(dims[RADIUS]);
    for (ptrdiff_t id = 0; id < dims[DIM0]; id++) {
        auto current = d + id;
        Data pred    = id == 0 ? 0 : d[id - 1];
        Data err     = *current - pred;
#ifdef PRED_COMP_ERR
        comp_err[id] = *current;
#endif
        Quant bin_count   = fabs(err) * eb_variants[EBr] + 1;
        bool  quantizable = fabs(bin_count) < dims[CAP];
        if (err < 0) bin_count = -bin_count;
        auto _code = static_cast<Quant>(bin_count / 2) + radius;
        *current   = pred + (_code - radius) * eb_variants[EBx2];
#ifdef PRED_COMP_ERR
        pred_err[id] = err;
        comp_err[id] = comp_err[id] - (*current);  // origin - decompressed
#endif
        outlier[id] = (1 - quantizable) * (*current);
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
    Data*               comp_err)
{
    auto  radius = static_cast<Quant>(dims[RADIUS]);
    Data *NW = new Data, *NE = new Data, *SW = new Data, *SE;

    for (ptrdiff_t i1 = 0; i1 < dims[DIM1]; i1++) {      // NW  NE
        for (ptrdiff_t i0 = 0; i0 < dims[DIM0]; i0++) {  // SW (SE)<- to predict
            *NW       = i1 == 0 or i0 == 0 ? 0.0 : *(d + (i0 - 1) + (i1 - 1) * dims[DIM0]);
            *NE       = i1 == 0 ? 0.0 : *(d + i0 + (i1 - 1) * dims[DIM0]);
            *SW       = i0 == 0 ? 0.0 : *(d + (i0 - 1) + i1 * dims[DIM0]);
            size_t id = i0 + i1 * dims[DIM0];
            SE        = d + id;

            Data pred = (*NE) + (*SW) - (*NW);
            Data err  = (*SE) - pred;
#ifdef PRED_COMP_ERR
            comp_err[id] = *SE;
#endif
            Quant bin_count   = fabs(err) * eb_variants[EBr] + 1;
            bool  quantizable = fabs(bin_count) < dims[CAP];
            if (err < 0) bin_count = -bin_count;
            Quant _code = static_cast<Quant>(bin_count / 2) + radius;
            *SE         = pred + (_code - radius) * eb_variants[EBx2];
#ifdef PRED_COMP_ERR
            pred_err[id] = err;
            comp_err[id] = comp_err[id] - (*SE);  // origin - decompressed
#endif
            outlier[id] = (1 - quantizable) * (*SE);
            q[id]       = quantizable * _code;
        }
    }
}

template <typename Data, typename Quant, int B>
void c_lorenzo_3d1l(
    Data*               data,
    Data*               outlier,
    Quant*              q,
    size_t const* const dims,
    double const* const eb_variants,
    Data*               pred_err,
    Data*               comp_err)
{
    auto      radius = static_cast<Quant>(dims[RADIUS]);
    Data *    NWo = new Data, *NEo = new Data, *SWo = new Data, *SEo = new Data;
    Data *    NWi = new Data, *NEi = new Data, *SWi = new Data, *SEi;
    ptrdiff_t w0 = 1, w1 = dims[DIM0], w2 = dims[DIM0] * dims[DIM1];

    for (ptrdiff_t i2 = 0; i2 < dims[DIM2]; i2++) {          //  | \---> x  NWo NEo
        for (ptrdiff_t i1 = 0; i1 < dims[DIM1]; i1++) {      //  v  v       SWo SEo  NWi  NEi
            for (ptrdiff_t i0 = 0; i0 < dims[DIM0]; i0++) {  //  y   z               SWi (SEi)<- to predict
                *NWo = i2 == 0 or i1 == 0 or i0 == 0 ? 0 : *(data + (i0 - 1) * w0 + (i1 - 1) * w1 + (i2 - 1) * w2);
                *NEo = i2 == 0 or i1 == 0 ? 0.0 : *(data + i0 * w0 + (i1 - 1) * w1 + (i2 - 1) * w2);
                *SWo = i2 == 0 or i0 == 0 ? 0.0 : *(data + (i0 - 1) * w0 + i1 * w1 + (i2 - 1) * w2);
                *SEo = i2 == 0 ? 0.0 : *(data + i0 * w0 + i1 * w1 + (i2 - 1) * w2);
                *NWi = i1 == 0 or i0 == 0 ? 0.0 : *(data + (i0 - 1) * w0 + (i1 - 1) * w1 + i2 * w2);
                *NEi = i1 == 0 ? 0.0 : *(data + i0 * w0 + (i1 - 1) * w1 + i2 * w2);
                *SWi = i0 == 0 ? 0.0 : *(data + (i0 - 1) * w0 + i1 * w1 + i2 * w2);

                size_t id = i0 * w0 + i1 * w1 + i2 * w2;
                SEi       = data + id;

                Data pred = +(*NWo) - (*NEo) - (*SWo) + (*SEo) + (*SWi) + (*NEi) - (*NWi);
                Data err  = (*SEi) - pred;
#ifdef PRED_COMP_ERR
                comp_err[id] = (*SEi);
#endif
                Quant bin_count   = fabs(err) * eb_variants[EBr] + 1;
                bool  quantizable = fabs(bin_count) < dims[CAP];
                if (err < 0) bin_count = -bin_count;
                Quant _code = static_cast<Quant>(bin_count / 2) + radius;
                *SEi        = pred + (_code - radius) * eb_variants[EBx2];
#ifdef PRED_COMP_ERR
                pred_err[id] = err;
                comp_err[id] = comp_err[id] - (*SEi);  // origin - decompressed
#endif
                outlier[id] = (1 - quantizable) * (*SEi);
                q[id]       = quantizable * _code;
            }
        }
    }
}

template <typename Data, typename Quant, int B>
void x_lorenzo_1d1l(Data* xdata, Data* outlier, Quant* q, size_t const* const dims, double const* const eb_variants)
{
    auto radius = static_cast<Quant>(dims[RADIUS]);
    for (ptrdiff_t id = 0; id < dims[DIM0]; id++) {
        Data pred = id == 0 ? 0 : xdata[id - 1];
        xdata[id] = q[id] == 0 ? outlier[id] : static_cast<Data>(pred + (q[id] - radius) * eb_variants[EBx2]);
    }
}
template <typename T, typename Q, int B>
void x_lorenzo_2d1l(T* xdata, T* outlier, Q* q, size_t const* const dims, double const* const eb_variants)
{
    auto radius = static_cast<Q>(dims[RADIUS]);
    T *  NW = new T, *NE = new T, *SW = new T;

    for (ptrdiff_t i1 = 0; i1 < dims[DIM1]; i1++) {      // NW  NE
        for (ptrdiff_t i0 = 0; i0 < dims[DIM0]; i0++) {  // SW (SE)<- to predict
            *NW = i1 == 0 or i0 == 0 ? 0.0 : *(xdata + (i0 - 1) + (i1 - 1) * dims[DIM0]);
            *NE = i1 == 0 ? 0.0 : *(xdata + i0 + (i1 - 1) * dims[DIM0]);
            *SW = i0 == 0 ? 0.0 : *(xdata + (i0 - 1) + i1 * dims[DIM0]);

            T      pred = (*NE) + (*SW) - (*NW);
            size_t id   = i0 + i1 * dims[DIM0];
            xdata[id]   = q[id] == 0 ? outlier[id] : static_cast<T>(pred + (q[id] - radius) * eb_variants[EBx2]);
        }
    }
}

template <typename Data, typename Quant, int B>
void x_lorenzo_3d1l(Data* xdata, Data* outlier, Quant* q, size_t const* const dims, double const* const eb_variants)
{
    auto      radius = static_cast<Quant>(dims[RADIUS]);
    Data *    NWo = new Data, *NEo = new Data, *SWo = new Data, *SEo = new Data;
    Data *    NWi = new Data, *NEi = new Data, *SWi = new Data, *SEi;
    ptrdiff_t w0 = 1, w1 = dims[DIM0], w2 = dims[DIM0] * dims[DIM1];

    for (ptrdiff_t i2 = 0; i2 < dims[DIM2]; i2++) {          // NW  NE
        for (ptrdiff_t i1 = 0; i1 < dims[DIM1]; i1++) {      // NW  NE
            for (ptrdiff_t i0 = 0; i0 < dims[DIM0]; i0++) {  // SW (SE)<- to predict
                *NWo = i2 == 0 or i1 == 0 or i0 == 0 ? 0.0 : *(xdata + (i0 - 1) * w0 + (i1 - 1) * w1 + (i2 - 1) * w2);
                *NEo = i2 == 0 or i1 == 0 ? 0.0 : *(xdata + i0 * w0 + (i1 - 1) * w1 + (i2 - 1) * w2);
                *SWo = i2 == 0 or i0 == 0 ? 0.0 : *(xdata + (i0 - 1) * w0 + i1 * w1 + (i2 - 1) * w2);
                *SEo = i2 == 0 ? 0.0 : *(xdata + i0 * w0 + i1 * w1 + (i2 - 1) * w2);
                *NWi = i1 == 0 or i0 == 0 ? 0.0 : *(xdata + (i0 - 1) * w0 + (i1 - 1) * w1 + i2 * w2);
                *NEi = i1 == 0 ? 0.0 : *(xdata + i0 * w0 + (i1 - 1) * w1 + i2 * w2);
                *SWi = i0 == 0 ? 0.0 : *(xdata + (i0 - 1) * w0 + i1 * w1 + i2 * w2);

                size_t id = i0 * w0 + i1 * w1 + i2 * w2;
                SEi       = xdata + id;

                Data pred = +(*NWo) - *(NEo) - *(SWo) + (*SEo) + (*SWi) + (*NEi) - (*NWi);
                xdata[id] = q[id] == 0 ? outlier[id] : static_cast<Data>(pred + (q[id] - radius) * eb_variants[EBx2]);
            }
        }
    }
}

}  // namespace sz1_4
}  // namespace psz

#endif
