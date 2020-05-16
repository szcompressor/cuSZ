// 200213

#ifndef PSZ_14_HH
#define PSZ_14_HH

#include <cstddef>

#include "types.hh"

namespace pSZ {
namespace PredictionQuantizationReconstructionSingleton {

template <typename T, typename Q, int B>
void c_lorenzo_1d1l(T* data, T* outlier, Q* bincode, size_t const* const dims_L16, double const* const ebs_L4, T* pred_err, T* comp_err) {
    auto radius = static_cast<Q>(dims_L16[RADIUS]);
    for (ptrdiff_t id = 0; id < dims_L16[DIM0]; id++) {
        auto current = data + id;
        T    pred    = id == 0 ? 0 : data[id - 1];
        T    err     = *current - pred;
#ifdef PRED_COMP_ERR
        comp_err[id] = *current;
#endif
        Q    bin_count   = fabs(err) * ebs_L4[EBr] + 1;
        bool quantizable = fabs(bin_count) < dims_L16[CAP];
        if (err < 0) bin_count = -bin_count;
        Q _code  = static_cast<Q>(bin_count / 2) + radius;
        *current = pred + (_code - radius) * ebs_L4[EBx2];
#ifdef PRED_COMP_ERR
        pred_err[id] = err;
        comp_err[id] = comp_err[id] - (*current);  // origin - decompressed
#endif
        outlier[id] = (1 - quantizable) * (*current);
        bincode[id] = quantizable * _code;
    }
}

template <typename T, typename Q, int B>
void c_lorenzo_2d1l(T* data, T* outlier, Q* bincode, size_t const* const dims_L16, double const* const ebs_L4, T* pred_err, T* comp_err) {
    auto radius = static_cast<Q>(dims_L16[RADIUS]);
    T *  NW = new T, *NE = new T, *SW = new T, *SE;

    for (ptrdiff_t i1 = 0; i1 < dims_L16[DIM1]; i1++) {      // NW  NE
        for (ptrdiff_t i0 = 0; i0 < dims_L16[DIM0]; i0++) {  // SW (SE)<- to predict
            *NW       = i1 == 0 or i0 == 0 ? 0.0 : *(data + (i0 - 1) + (i1 - 1) * dims_L16[DIM0]);
            *NE       = i1 == 0 ? 0.0 : *(data + i0 + (i1 - 1) * dims_L16[DIM0]);
            *SW       = i0 == 0 ? 0.0 : *(data + (i0 - 1) + i1 * dims_L16[DIM0]);
            size_t id = i0 + i1 * dims_L16[DIM0];
            SE        = data + id;

            T pred = (*NE) + (*SW) - (*NW);
            T err  = (*SE) - pred;
#ifdef PRED_COMP_ERR
            comp_err[id] = *SE;
#endif
            Q    bin_count   = fabs(err) * ebs_L4[EBr] + 1;
            bool quantizable = fabs(bin_count) < dims_L16[CAP];
            if (err < 0) bin_count = -bin_count;
            Q _code = static_cast<Q>(bin_count / 2) + radius;
            *SE     = pred + (_code - radius) * ebs_L4[EBx2];
#ifdef PRED_COMP_ERR
            pred_err[id] = err;
            comp_err[id] = comp_err[id] - (*SE);  // origin - decompressed
#endif
            outlier[id] = (1 - quantizable) * (*SE);
            bincode[id] = quantizable * _code;
        }
    }
}

template <typename T, typename Q, int B>
void c_lorenzo_3d1l(T* data, T* outlier, Q* bincode, size_t const* const dims_L16, double const* const ebs_L4, T* pred_err, T* comp_err) {
    auto      radius = static_cast<Q>(dims_L16[RADIUS]);
    T *       NWo = new T, *NEo = new T, *SWo = new T, *SEo = new T;
    T *       NWi = new T, *NEi = new T, *SWi = new T, *SEi;
    ptrdiff_t w0 = 1, w1 = dims_L16[DIM0], w2 = dims_L16[DIM0] * dims_L16[DIM1];

    for (ptrdiff_t i2 = 0; i2 < dims_L16[DIM2]; i2++) {          //  | \---> x  NWo NEo
        for (ptrdiff_t i1 = 0; i1 < dims_L16[DIM1]; i1++) {      //  v  v       SWo SEo  NWi  NEi
            for (ptrdiff_t i0 = 0; i0 < dims_L16[DIM0]; i0++) {  //  y   z               SWi (SEi)<- to predict
                *NWo = i2 == 0 or i1 == 0 or i0 == 0 ? 0 : *(data + (i0 - 1) * w0 + (i1 - 1) * w1 + (i2 - 1) * w2);
                *NEo = i2 == 0 or i1 == 0 ? 0.0 : *(data + i0 * w0 + (i1 - 1) * w1 + (i2 - 1) * w2);
                *SWo = i2 == 0 or i0 == 0 ? 0.0 : *(data + (i0 - 1) * w0 + i1 * w1 + (i2 - 1) * w2);
                *SEo = i2 == 0 ? 0.0 : *(data + i0 * w0 + i1 * w1 + (i2 - 1) * w2);
                *NWi = i1 == 0 or i0 == 0 ? 0.0 : *(data + (i0 - 1) * w0 + (i1 - 1) * w1 + i2 * w2);
                *NEi = i1 == 0 ? 0.0 : *(data + i0 * w0 + (i1 - 1) * w1 + i2 * w2);
                *SWi = i0 == 0 ? 0.0 : *(data + (i0 - 1) * w0 + i1 * w1 + i2 * w2);

                size_t id = i0 * w0 + i1 * w1 + i2 * w2;
                SEi       = data + id;

                T pred = +(*NWo) - (*NEo) - (*SWo) + (*SEo) + (*SWi) + (*NEi) - (*NWi);
                T err  = (*SEi) - pred;
#ifdef PRED_COMP_ERR
                comp_err[id] = (*SEi);
#endif
                Q    bin_count   = fabs(err) * ebs_L4[EBr] + 1;
                bool quantizable = fabs(bin_count) < dims_L16[CAP];
                if (err < 0) bin_count = -bin_count;
                Q _code = static_cast<Q>(bin_count / 2) + radius;
                *SEi    = pred + (_code - radius) * ebs_L4[EBx2];
#ifdef PRED_COMP_ERR
                pred_err[id] = err;
                comp_err[id] = comp_err[id] - (*SEi);  // origin - decompressed
#endif
                outlier[id] = (1 - quantizable) * (*SEi);
                bincode[id] = quantizable * _code;
            }
        }
    }
}

template <typename T, typename Q, int B>
void x_lorenzo_1d1l(T* xdata, T* outlier, Q* bincode, size_t const* const dims_L16, double const* const ebs_L4) {
    auto radius = static_cast<Q>(dims_L16[RADIUS]);
    for (ptrdiff_t id = 0; id < dims_L16[DIM0]; id++) {
        T pred    = id == 0 ? 0 : xdata[id - 1];
        xdata[id] = bincode[id] == 0 ? outlier[id] : static_cast<T>(pred + (bincode[id] - radius) * ebs_L4[EBx2]);
    }
}
template <typename T, typename Q, int B>
void x_lorenzo_2d1l(T* xdata, T* outlier, Q* bincode, size_t const* const dims_L16, double const* const ebs_L4) {
    auto radius = static_cast<Q>(dims_L16[RADIUS]);
    T *  NW = new T, *NE = new T, *SW = new T;

    for (ptrdiff_t i1 = 0; i1 < dims_L16[DIM1]; i1++) {      // NW  NE
        for (ptrdiff_t i0 = 0; i0 < dims_L16[DIM0]; i0++) {  // SW (SE)<- to predict
            *NW = i1 == 0 or i0 == 0 ? 0.0 : *(xdata + (i0 - 1) + (i1 - 1) * dims_L16[DIM0]);
            *NE = i1 == 0 ? 0.0 : *(xdata + i0 + (i1 - 1) * dims_L16[DIM0]);
            *SW = i0 == 0 ? 0.0 : *(xdata + (i0 - 1) + i1 * dims_L16[DIM0]);

            T      pred = (*NE) + (*SW) - (*NW);
            size_t id   = i0 + i1 * dims_L16[DIM0];
            xdata[id]   = bincode[id] == 0 ? outlier[id] : static_cast<T>(pred + (bincode[id] - radius) * ebs_L4[EBx2]);
        }
    }
}

template <typename T, typename Q, int B>
void x_lorenzo_3d1l(T* xdata, T* outlier, Q* bincode, size_t const* const dims_L16, double const* const ebs_L4) {
    auto      radius = static_cast<Q>(dims_L16[RADIUS]);
    T *       NWo = new T, *NEo = new T, *SWo = new T, *SEo = new T;
    T *       NWi = new T, *NEi = new T, *SWi = new T, *SEi;
    ptrdiff_t w0 = 1, w1 = dims_L16[DIM0], w2 = dims_L16[DIM0] * dims_L16[DIM1];

    for (ptrdiff_t i2 = 0; i2 < dims_L16[DIM2]; i2++) {          // NW  NE
        for (ptrdiff_t i1 = 0; i1 < dims_L16[DIM1]; i1++) {      // NW  NE
            for (ptrdiff_t i0 = 0; i0 < dims_L16[DIM0]; i0++) {  // SW (SE)<- to predict
                *NWo = i2 == 0 or i1 == 0 or i0 == 0 ? 0.0 : *(xdata + (i0 - 1) * w0 + (i1 - 1) * w1 + (i2 - 1) * w2);
                *NEo = i2 == 0 or i1 == 0 ? 0.0 : *(xdata + i0 * w0 + (i1 - 1) * w1 + (i2 - 1) * w2);
                *SWo = i2 == 0 or i0 == 0 ? 0.0 : *(xdata + (i0 - 1) * w0 + i1 * w1 + (i2 - 1) * w2);
                *SEo = i2 == 0 ? 0.0 : *(xdata + i0 * w0 + i1 * w1 + (i2 - 1) * w2);
                *NWi = i1 == 0 or i0 == 0 ? 0.0 : *(xdata + (i0 - 1) * w0 + (i1 - 1) * w1 + i2 * w2);
                *NEi = i1 == 0 ? 0.0 : *(xdata + i0 * w0 + (i1 - 1) * w1 + i2 * w2);
                *SWi = i0 == 0 ? 0.0 : *(xdata + (i0 - 1) * w0 + i1 * w1 + i2 * w2);

                size_t id = i0 * w0 + i1 * w1 + i2 * w2;
                SEi       = xdata + id;

                T pred    = +(*NWo) - *(NEo) - *(SWo) + (*SEo) + (*SWi) + (*NEi) - (*NWi);
                xdata[id] = bincode[id] == 0 ? outlier[id] : static_cast<T>(pred + (bincode[id] - radius) * ebs_L4[EBx2]);
            }
        }
    }
}

/*
template <typename T, typename Q, int B>
void c_lorenzo_2d1l(T* data, T* outlier, Q* bincode, size_t const* const dims_L16, double const* const ebs_L4, size_t b0, size_t b1) {
    T __s[B + 1][B + 1];  // 2D interpretation of data
    memset(__s, 0, (B + 1) * (B + 1) * sizeof(T));
    auto radius = static_cast<Q>(dims_L16[__RADIUS]);

    size_t _idx1 = b1 * B;
    size_t _idx0 = b0 * B;

    //    initiate border points
    for (ptrdiff_t i1 = -1; i1 < B; i1++) {
        for (ptrdiff_t i0 = -1; i0 < B; i0++) {
            ptrdiff_t gi1 = _idx1 + i1;
            ptrdiff_t gi0 = _idx0 + i0;
            if (gi1 >= static_cast<ptrdiff_t>(dims_L16[__DIM1]) or gi0 >= static_cast<ptrdiff_t>(dims_L16[__DIM0])) continue;
            // border condition control
            if (b1 == 0 and i1 == -1) continue;
            if (b0 == 0 and i0 == -1) continue;

            size_t id           = gi0 + gi1 * dims_L16[__DIM0];
            __s[i1 + 1][i0 + 1] = data[id];
        }
    }

    for (size_t i1 = 0; i1 < B; i1++) {
        for (size_t i0 = 0; i0 < B; i0++) {
            size_t gi1 = _idx1 + i1;
            size_t gi0 = _idx0 + i0;
            if (gi1 >= dims_L16[__DIM1] or gi0 >= dims_L16[__DIM0]) continue;
            size_t id   = gi0 + gi1 * dims_L16[__DIM0];
            T      pred = __s[i1 + 1][i0] + __s[i1][i0 + 1] - __s[i1][i0];
            T      err  = __s[i1 + 1][i0 + 1] - pred;
            // sz14 specific begin
            Q    bin_count   = fabs(err) * ebs_L4[__EBr] + 1;
            bool quantizable = fabs(bin_count) < dims_L16[__CAP];

            //            // ********************************************************************************
            //            if (b0 == 1 and b1 == 1) {
            //                printf("\e[7m%zu, %zu\e[27m\t", i1, i0);
            //                cout << "origin: " << __s[i1 + 1][i0 + 1] << "\t";
            //            }
            //            T __a = __s[i1 + 1][i0 + 1];
            //            // ********************************************************************************

            if (err < 0) bin_count = -bin_count;
            Q _code             = static_cast<Q>(bin_count / 2) + radius;
            __s[i1 + 1][i0 + 1] = pred + (_code - radius) * ebs_L4[__2EB];

            //            // ********************************************************************************
            //            if (b0 == 1 and b1 == 1) {
            //                cout << "writen by: " << __s[i1 + 1][i0 + 1] << "\t"
            //                     << "<= eb? " << (fabs(__a - __s[i1 + 1][i0 + 1]) <= ebs_L4[__EB] ? "yes" : "no") << "\n";
            //                cout << setw(5) << " - " << __s[i1][i0] << " + " << __s[i1][i0 + 1] << endl;
            //                cout << setw(5) << " + " << __s[i1 + 1][i0] << " -> " << pred;
            //                cout << "\terror: " << err << "\t code dist: " << _code - radius << endl;
            //            }
            //            // ********************************************************************************

            outlier[id] = (1 - quantizable) * __s[i1 + 1][i0 + 1];
            bincode[id] = quantizable * _code;
        }
    }
}

template <typename T, typename Q, int B>
void x_lorenzo_2d1l(T* xdata, T* outlier, Q* bincode, size_t const* const dims_L16, double const* const ebs_L4, size_t b0, size_t b1) {
    T __s[B + 1][B + 1];  // 2D interpretation of data
    memset(__s, 0, (B + 1) * (B + 1) * sizeof(T));
    auto radius = static_cast<Q>(dims_L16[__RADIUS]);

    size_t _idx0 = b0 * B;
    size_t _idx1 = b1 * B;

    for (ptrdiff_t i1 = -1; i1 < B; i1++) {
        for (ptrdiff_t i0 = -1; i0 < B; i0++) {
            ptrdiff_t gi1 = _idx1 + i1;
            ptrdiff_t gi0 = _idx0 + i0;
            if (gi1 >= static_cast<ptrdiff_t>(dims_L16[__DIM1]) or gi0 >= static_cast<ptrdiff_t>(dims_L16[__DIM0])) continue;
            // border condition control
            if (b1 == 0  and i1 == -1) continue;
            if (b0 == 0  and i0 == -1) continue;

            size_t id           = gi0 + gi1 * dims_L16[__DIM0];
            __s[i1 + 1][i0 + 1] = xdata[id];
        }
    }

    for (size_t i1 = 0; i1 < B; i1++) {
        for (size_t i0 = 0; i0 < B; i0++) {
            size_t gi1 = _idx1 + i1;
            size_t gi0 = _idx0 + i0;
            if (gi1 >= dims_L16[__DIM1] or gi0 >= dims_L16[__DIM0]) continue;
            const size_t id     = gi0 + gi1 * dims_L16[__DIM0];
            T            pred   = __s[i1][i0 + 1] + __s[i1 + 1][i0] - __s[i1][i0];
            __s[i1 + 1][i0 + 1] = bincode[id] == 0 ? outlier[id] : static_cast<T>(pred + (bincode[id] - radius) * ebs_L4[__2EB]);
            xdata[id]           = __s[i1 + 1][i0 + 1];
        }
    }

}
*/

}  // namespace PredictionQuantizationReconstructionSingleton
}  // namespace pSZ

#endif
