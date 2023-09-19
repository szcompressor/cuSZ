/**
 * @file l23.seq.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-03-16
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "detail/l23.seq.inl"
#include "cusz/type.h"

template <typename T, typename EQ, typename FP, typename OUTLIER = psz_outlier_serial<T>>
cusz_error_status psz_comp_l23ser(
    T* const       data,
    psz_dim3 const len3,
    double const   eb,
    int const      radius,
    EQ* const      eq,
    OUTLIER*       outlier,
    float*         time_elapsed)
{
    auto divide3 = [](psz_dim3 len, psz_dim3 sublen) {
        return psz_dim3{(len.x - 1) / sublen.x + 1, (len.y - 1) / sublen.y + 1, (len.z - 1) / sublen.z + 1};
    };

    auto ndim = [&]() {
        if (len3.z == 1 and len3.y == 1)
            return 1;
        else if (len3.z == 1 and len3.y != 1)
            return 2;
        else
            return 3;
    };

    auto d = ndim();

    // error bound
    auto ebx2   = eb * 2;
    auto ebx2_r = 1 / ebx2;
    auto leap3  = psz_dim3{1, len3.x, len3.x * len3.y};

    if (d == 1) {
        psz::serial::__kernel::c_lorenzo_1d1l<T, EQ, FP, 256>(data, len3, leap3, radius, ebx2_r, eq, outlier);
    }
    else if (d == 2) {
        psz::serial::__kernel::c_lorenzo_2d1l<T, EQ, FP, 16>(data, len3, leap3, radius, ebx2_r, eq, outlier);
    }
    else if (d == 3) {
        psz::serial::__kernel::c_lorenzo_3d1l<T, EQ, FP, 8>(data, len3, leap3, radius, ebx2_r, eq, outlier);
    }

    return CUSZ_SUCCESS;
}

template <typename T, typename EQ, typename FP>
cusz_error_status psz_decomp_l23ser(
    EQ*            eq,
    psz_dim3 const len3,
    T*             outlier,
    double const   eb,
    int const      radius,
    T*             xdata,
    float*         time_elapsed)
{
    auto divide3 = [](psz_dim3 len, psz_dim3 sublen) {
        return psz_dim3{(len.x - 1) / sublen.x + 1, (len.y - 1) / sublen.y + 1, (len.z - 1) / sublen.z + 1};
    };

    auto ndim = [&]() {
        if (len3.z == 1 and len3.y == 1)
            return 1;
        else if (len3.z == 1 and len3.y != 1)
            return 2;
        else
            return 3;
    };

    // error bound
    auto ebx2   = eb * 2;
    auto ebx2_r = 1 / ebx2;
    auto leap3  = psz_dim3{1, len3.x, len3.x * len3.y};

    auto d = ndim();

    if (d == 1) {
        psz::serial::__kernel::x_lorenzo_1d1l<T, EQ, FP, 256>(eq, outlier, len3, leap3, radius, ebx2, xdata);
    }
    else if (d == 2) {
        psz::serial::__kernel::x_lorenzo_2d1l<T, EQ, FP, 16>(eq, outlier, len3, leap3, radius, ebx2, xdata);
    }
    else if (d == 3) {
        psz::serial::__kernel::x_lorenzo_3d1l<T, EQ, FP, 8>(eq, outlier, len3, leap3, radius, ebx2, xdata);
    }

    return CUSZ_SUCCESS;
}

#define CPP_INS(Tliteral, Eliteral, FPliteral, T, EQ, FP)                      \
    template cusz_error_status psz_comp_l23ser<T, EQ, FP>(                           \
        T* const, psz_dim3 const, double const, int const, EQ* const, psz_outlier_serial<T>*, float*); \
                                                                                                       \
    template cusz_error_status psz_decomp_l23ser<T, EQ, FP>(                         \
        EQ*, psz_dim3 const, T*, double const, int const, T*, float*);

CPP_INS(fp32, ui8, fp32, float, uint8_t, float);
CPP_INS(fp32, ui16, fp32, float, uint16_t, float);
CPP_INS(fp32, ui32, fp32, float, uint32_t, float);
CPP_INS(fp32, fp32, fp32, float, float, float);

CPP_INS(fp64, ui8, fp64, double, uint8_t, double);
CPP_INS(fp64, ui16, fp64, double, uint16_t, double);
CPP_INS(fp64, ui32, fp64, double, uint32_t, double);
CPP_INS(fp64, fp32, fp64, double, float, double);

#undef CPP_INS
