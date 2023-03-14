/**
 * @file lorenzo_serial.inl
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-03-13
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef E0B87BA8_BEDC_4CBE_B5EE_C0C5875E07D6
#define E0B87BA8_BEDC_4CBE_B5EE_C0C5875E07D6

#include <iostream>
#include "cusz/it.hh"
#include "cusz/nd.h"

using std::cout;
using std::endl;

#define SETUP_1D_BASIC                                                                        \
    psz_dim3 grid_dim, block_idx, thread_idx;                                                 \
    auto     gx             = [&]() -> uint32_t { return block_idx.x * BLK + thread_idx.x; }; \
    auto     gidx           = [&]() -> uint32_t { return block_idx.x * BLK + thread_idx.x; }; \
    auto     check_boundary = [&]() { return gx() < len3.x; };                                \
    grid_dim.x              = (len3.x - 1) / BLK + 1;
#define SETUP_1D_DATABUF                                            \
    constexpr auto PADDING    = 1;                                  \
    auto           _buf1      = new psz_buf<T, 1, BLK + PADDING>(); \
    auto&          buf1       = *_buf1;                             \
    auto           databuf_it = [&](auto x) -> T& { return buf1(thread_idx.x + x + PADDING); };
#define SETUP_1D_EQBUF                          \
    auto  _buf2    = new psz_buf<EQ, 1, BLK>(); \
    auto& buf2     = *_buf2;                    \
    auto  eqbuf_it = [&](auto dx) -> EQ& { return buf2(thread_idx.x + dx); };
#define PFOR_GRID_1D() for (block_idx.x = 0; block_idx.x < grid_dim.x; block_idx.x++)
#define PFOR_BLOCK_1D() for (thread_idx.x = 0; thread_idx.x < BLK; thread_idx.x++)

#define SETUP_2D_BASIC                                                                        \
    psz_dim3 grid_dim, block_idx, thread_idx;                                                 \
    auto     gx             = [&]() -> uint32_t { return block_idx.x * BLK + thread_idx.x; }; \
    auto     gy             = [&]() -> uint32_t { return block_idx.y * BLK + thread_idx.y; }; \
    auto     gidx           = [&]() -> uint32_t { return gy() * stride3.y + gx(); };          \
    auto     check_boundary = [&]() { return gx() < len3.x and gy() < len3.y; };              \
    grid_dim.x              = (len3.x - 1) / BLK + 1;                                         \
    grid_dim.y              = (len3.y - 1) / BLK + 1;
#define SETUP_2D_DATABUF                                                                 \
    constexpr auto PADDING    = 1;                                                       \
    auto           _buf1      = new psz_buf<T, 2, BLK + PADDING>();                      \
    auto&          buf1       = *_buf1;                                                  \
    auto           databuf_it = [&](auto dx, auto dy) -> T& {                            \
        return buf1(thread_idx.x + dx + PADDING, thread_idx.y + dy + PADDING); \
    };
#define SETUP_2D_EQBUF                          \
    auto  _buf2    = new psz_buf<EQ, 2, BLK>(); \
    auto& buf2     = *_buf2;                    \
    auto  eqbuf_it = [&](auto dx, auto dy) -> EQ& { return buf2(thread_idx.x + dx, thread_idx.y + dy); };
#define PFOR_GRID_2D()                                             \
    for (block_idx.y = 0; block_idx.y < grid_dim.y; block_idx.y++) \
        for (block_idx.x = 0; block_idx.x < grid_dim.x; block_idx.x++)
#define PFOR_BLOCK_2D()                                        \
    for (thread_idx.y = 0; thread_idx.y < BLK; thread_idx.y++) \
        for (thread_idx.x = 0; thread_idx.x < BLK; thread_idx.x++)

#define SETUP_3D_BASIC                                                                                  \
    psz_dim3 grid_dim, block_idx, thread_idx;                                                           \
    auto     gx             = [&]() -> uint32_t { return block_idx.x * BLK + thread_idx.x; };           \
    auto     gy             = [&]() -> uint32_t { return block_idx.y * BLK + thread_idx.y; };           \
    auto     gz             = [&]() -> uint32_t { return block_idx.z * BLK + thread_idx.z; };           \
    auto     gidx           = [&]() -> uint32_t { return gz() * stride3.z + gy() * stride3.y + gx(); }; \
    auto     check_boundary = [&]() { return gx() < len3.x and gy() < len3.y and gz() < len3.z; };      \
    grid_dim.x              = (len3.x - 1) / BLK + 1;                                                   \
    grid_dim.y              = (len3.y - 1) / BLK + 1;                                                   \
    grid_dim.z              = (len3.z - 1) / BLK + 1;
#define SETUP_3D_DATABUF                                                                                              \
    constexpr auto PADDING    = 1;                                                                                    \
    auto           _buf1      = new psz_buf<T, 3, BLK + PADDING>();                                                   \
    auto&          buf1       = *_buf1;                                                                               \
    auto           databuf_it = [&](auto dx, auto dy, auto dz) -> T& {                                                \
        return buf1(thread_idx.x + dx + PADDING, thread_idx.y + dy + PADDING, thread_idx.z + dz + PADDING); \
    };
#define SETUP_3D_EQBUF                                                         \
    auto  _buf2    = new psz_buf<EQ, 3, BLK>();                                \
    auto& buf2     = *_buf2;                                                   \
    auto  eqbuf_it = [&](auto dx, auto dy, auto dz) -> EQ& {                   \
        return buf2(thread_idx.x + dx, thread_idx.y + dy, thread_idx.z + dz); \
    };
#define PFOR_GRID_3D()                                                 \
    for (block_idx.z = 0; block_idx.z < grid_dim.z; block_idx.z++)     \
        for (block_idx.y = 0; block_idx.y < grid_dim.y; block_idx.y++) \
            for (block_idx.x = 0; block_idx.x < grid_dim.x; block_idx.x++)
#define PFOR_BLOCK_3D()                                            \
    for (thread_idx.z = 0; thread_idx.z < BLK; thread_idx.z++)     \
        for (thread_idx.y = 0; thread_idx.y < BLK; thread_idx.y++) \
            for (thread_idx.x = 0; thread_idx.x < BLK; thread_idx.x++)

namespace psz {
namespace serial {
namespace __kernel {

template <
    typename T,
    typename EQ      = int32_t,
    typename FP      = T,
    int BLK          = 256,
    typename OUTLIER = struct psz_outlier_serial<T>>
void c_lorenzo_1d1l(T* data, psz_dim3 len3, psz_dim3 stride3, int radius, FP ebx2_r, EQ* eq, OUTLIER* outlier) {
    SETUP_1D_BASIC;
    SETUP_1D_DATABUF;
    SETUP_1D_EQBUF;

    // per-thread ("real" kernel)
    auto threadview_load = [&]() {
        if (check_boundary()) databuf_it(0) = data[gidx()] * ebx2_r;
    };
    auto threadview_process = [&]() {
        auto delta = databuf_it(0) - databuf_it(-1);
        if (delta > radius) {
            outlier->record(delta, gidx());
            eqbuf_it(0) = 0;
        }
        else {
            eqbuf_it(0) = delta;
        }
    };
    auto threadview_store = [&]() {
        if (check_boundary()) eq[gidx()] = eqbuf_it(0);
    };

    ////////////////////////////////////////
    PFOR_GRID_1D() { PFOR_BLOCK_1D() threadview_load(); }
    PFOR_GRID_1D() { PFOR_BLOCK_1D() threadview_process(); }
    PFOR_GRID_1D() { PFOR_BLOCK_1D() threadview_store(); }

    delete _buf1;
    delete _buf2;

}

template <typename T, typename EQ = int32_t, typename FP = T, int BLK = 256>
void x_lorenzo_1d1l(EQ* eq, T* scattered_outlier, psz_dim3 len3, psz_dim3 stride3, int radius, FP ebx2, T* xdata)
{
    SETUP_1D_BASIC;
    SETUP_1D_DATABUF;

    // per-thread ("real" kernel)
    auto threadview_load = [&]() {
        if (check_boundary()) databuf_it(0) = eq[gidx()] + scattered_outlier[gidx()];
    };
    auto threadview_partial_sum = [&]() {
        if (thread_idx.x > 0) databuf_it(0) += databuf_it(-1);
    };
    auto threadview_store = [&]() {
        if (check_boundary()) xdata[gidx()] = databuf_it(0) * ebx2;
    };

    ////////////////////////////////////////
    PFOR_GRID_1D() { PFOR_BLOCK_1D() threadview_load(); }
    PFOR_GRID_1D() { PFOR_BLOCK_1D() threadview_partial_sum(); }
    PFOR_GRID_1D() { PFOR_BLOCK_1D() threadview_store(); }

    delete _buf1;
}

template <
    typename T,
    typename EQ      = int32_t,
    typename FP      = T,
    int BLK          = 16,
    typename OUTLIER = struct psz_outlier_serial<T>>
void c_lorenzo_2d1l(T* data, psz_dim3 len3, psz_dim3 stride3, int radius, FP ebx2_r, EQ* eq, OUTLIER* outlier) {
    SETUP_2D_BASIC;
    SETUP_2D_DATABUF;
    SETUP_2D_EQBUF;

    // per-thread ("real" kernel)
    auto threadview_load = [&]() {
        if (check_boundary()) databuf_it(0, 0) = data[gidx()] * ebx2_r;
    };
    auto threadview_process = [&]() {
        auto delta = databuf_it(0, 0) - (databuf_it(-1, 0) + databuf_it(0, -1) - databuf_it(-1, -1));
        if (delta > radius) {
            outlier->record(delta, gidx());
            eqbuf_it(0, 0) = 0;
        }
        else {
            eqbuf_it(0, 0) = delta;
        }
    };
    auto threadview_store = [&]() {
        if (check_boundary()) eq[gidx()] = eqbuf_it(0, 0);
    };

    ////////////////////////////////////////
    PFOR_GRID_2D() { PFOR_BLOCK_2D() threadview_load(); }
    PFOR_GRID_2D() { PFOR_BLOCK_2D() threadview_process(); }
    PFOR_GRID_2D() { PFOR_BLOCK_2D() threadview_store(); }

    delete _buf1;
    delete _buf2;
}

template <typename T, typename EQ = int32_t, typename FP = T, int BLK = 16>
void x_lorenzo_2d1l(EQ* eq, T* scattered_outlier, psz_dim3 len3, psz_dim3 stride3, int radius, FP ebx2, T* xdata)
{
    SETUP_2D_BASIC;
    SETUP_2D_DATABUF;

    // per-thread ("real" kernel)
    auto threadview_load = [&]() {
        if (check_boundary()) databuf_it(0, 0) = eq[gidx()] + scattered_outlier[gidx()];
    };
    auto threadview_partial_sum_x = [&]() {
        if (thread_idx.x > 0) databuf_it(0, 0) += databuf_it(-1, 0);
    };
    auto threadview_partial_sum_y = [&]() {
        if (thread_idx.y > 0) databuf_it(0, 0) += databuf_it(0, -1);
    };
    auto threadview_store = [&]() {
        if (check_boundary()) xdata[gidx()] = databuf_it(0, 0) * ebx2;
    };

    ////////////////////////////////////////
    PFOR_GRID_2D() { PFOR_BLOCK_2D() threadview_load(); }
    PFOR_GRID_2D()
    {
        PFOR_BLOCK_2D() threadview_partial_sum_x();
        PFOR_BLOCK_2D() threadview_partial_sum_y();
    }
    PFOR_GRID_2D() { PFOR_BLOCK_2D() threadview_store(); }

    delete _buf1;
}

template <
    typename T,
    typename EQ      = int32_t,
    typename FP      = T,
    int BLK          = 8,
    typename OUTLIER = struct psz_outlier_serial<T>>
void c_lorenzo_3d1l(T* data, psz_dim3 len3, psz_dim3 stride3, int radius, FP ebx2_r, EQ* eq, OUTLIER* outlier) {
    SETUP_3D_BASIC;
    SETUP_3D_DATABUF;
    SETUP_3D_EQBUF;

    // per-thread ("real" kernel)
    auto threadview_load = [&]() {
        if (check_boundary()) databuf_it(0, 0, 0) = data[gidx()] * ebx2_r;
    };
    auto threadview_process = [&]() {
        auto delta = databuf_it(0, 0, 0) -
                     (databuf_it(-1, -1, -1) - databuf_it(0, -1, -1) - databuf_it(-1, 0, -1) - databuf_it(-1, -1, 0) +
                      databuf_it(0, 0, -1) + databuf_it(0, -1, 0) + databuf_it(-1, 0, 0));
        if (delta > radius) {
            outlier->record(delta, gidx());
            eqbuf_it(0, 0, 0) = 0;
        }
        else {
            eqbuf_it(0, 0, 0) = delta;
        }
    };
    auto threadview_store = [&]() {
        if (check_boundary()) eq[gidx()] = eqbuf_it(0, 0, 0);
    };

    ////////////////////////////////////////
    PFOR_GRID_3D() { PFOR_BLOCK_3D() threadview_load(); }
    PFOR_GRID_3D() { PFOR_BLOCK_3D() threadview_process(); }
    PFOR_GRID_3D() { PFOR_BLOCK_3D() threadview_store(); }

    delete _buf1;
    delete _buf2;
}

template <typename T, typename EQ = int32_t, typename FP = T, int BLK = 8>
void x_lorenzo_3d1l(EQ* eq, T* scattered_outlier, psz_dim3 len3, psz_dim3 stride3, int radius, FP ebx2, T* xdata)
{
    SETUP_3D_BASIC;
    SETUP_3D_DATABUF;

    // per-thread ("real" kernel)
    auto threadview_load = [&]() {
        if (check_boundary()) databuf_it(0, 0, 0) = eq[gidx()] + scattered_outlier[gidx()];
    };
    auto threadview_partial_sum_x = [&]() {
        if (thread_idx.x > 0) databuf_it(0, 0, 0) += databuf_it(-1, 0, 0);
    };
    auto threadview_partial_sum_y = [&]() {
        if (thread_idx.y > 0) databuf_it(0, 0, 0) += databuf_it(0, -1, 0);
    };
    auto threadview_partial_sum_z = [&]() {
        if (thread_idx.z > 0) databuf_it(0, 0, 0) += databuf_it(0, 0, -1);
    };
    auto threadview_store = [&]() {
        if (check_boundary()) xdata[gidx()] = databuf_it(0, 0, 0) * ebx2;
    };

    ////////////////////////////////////////
    PFOR_GRID_3D() { PFOR_BLOCK_3D() threadview_load(); }
    PFOR_GRID_3D()
    {
        PFOR_BLOCK_3D() threadview_partial_sum_x();
        PFOR_BLOCK_3D() threadview_partial_sum_y();
        PFOR_BLOCK_3D() threadview_partial_sum_z();
    }
    PFOR_GRID_3D() { PFOR_BLOCK_3D() threadview_store(); }

    delete _buf1;
}

}  // namespace __kernel
}  // namespace serial
}  // namespace psz

#undef SETUP_1D
#undef PFOR_GRID_1D
#undef PFOR_BLOCK_1D
#undef SETUP_2D_BASIC
#undef PFOR_GRID_2D
#undef PFOR_BLOCK_2D
#undef SETUP_3D
#undef PFOR_GRID_3D
#undef PFOR_BLOCK_3D

#endif /* E0B87BA8_BEDC_4CBE_B5EE_C0C5875E07D6 */
