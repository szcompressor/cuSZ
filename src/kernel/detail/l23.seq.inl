/**
 * @file l23.seq.inl
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

#include "../src/utils/it_serial.hh"
#include "cusz/it.hh"
#include "cusz/nd.h"
#include "mem/compact/compact.seq.hh"

using std::cout;
using std::endl;

#define SETUP_1D_DATABUF                           \
  constexpr auto PADDING = 1;                      \
  auto _buf1 = new psz_buf<T, 1, BLK + PADDING>(); \
  auto& buf1 = *_buf1;                             \
  auto databuf_it = [&](auto x) -> T& { return buf1(t.x + x + PADDING); };
#define SETUP_1D_EQBUF                    \
  auto _buf2 = new psz_buf<EQ, 1, BLK>(); \
  auto& buf2 = *_buf2;                    \
  auto eqbuf_it = [&](auto dx) -> EQ& { return buf2(t.x + dx); };

#define SETUP_2D_DATABUF                                 \
  constexpr auto PADDING = 1;                            \
  auto _buf1 = new psz_buf<T, 2, BLK + PADDING>();       \
  auto& buf1 = *_buf1;                                   \
  auto databuf_it = [&](auto dx, auto dy) -> T& {        \
    return buf1(t.x + dx + PADDING, t.y + dy + PADDING); \
  };
#define SETUP_2D_EQBUF                           \
  auto _buf2 = new psz_buf<EQ, 2, BLK>();        \
  auto& buf2 = *_buf2;                           \
  auto eqbuf_it = [&](auto dx, auto dy) -> EQ& { \
    return buf2(t.x + dx, t.y + dy);             \
  };

#define SETUP_3D_DATABUF                                                     \
  constexpr auto PADDING = 1;                                                \
  auto _buf1 = new psz_buf<T, 3, BLK + PADDING>();                           \
  auto& buf1 = *_buf1;                                                       \
  auto databuf_it = [&](auto dx, auto dy, auto dz) -> T& {                   \
    return buf1(t.x + dx + PADDING, t.y + dy + PADDING, t.z + dz + PADDING); \
  };
#define SETUP_3D_EQBUF                                    \
  auto _buf2 = new psz_buf<EQ, 3, BLK>();                 \
  auto& buf2 = *_buf2;                                    \
  auto eqbuf_it = [&](auto dx, auto dy, auto dz) -> EQ& { \
    return buf2(t.x + dx, t.y + dy, t.z + dz);            \
  };

namespace psz {
namespace seq {
namespace __kernel {

template <
    typename T, typename EQ = int32_t, typename FP = T, int BLK = 256,
    typename OUTLIER = struct CompactSerial<T>>
void c_lorenzo_1d1l(
    T* data, psz_dim3 len3, psz_dim3 stride3, int radius, FP ebx2_r, EQ* eq,
    OUTLIER* outlier) {
  SETUP_ND_CPU_SERIAL;
  SETUP_1D_DATABUF;
  SETUP_1D_EQBUF;

  // per-thread ("real" kernel)
  auto threadview_load = [&]() {
    if (check_boundary1()) databuf_it(0) = data[gid1()] * ebx2_r;
  };
  auto threadview_process = [&]() {
    auto delta = databuf_it(0) - databuf_it(-1);
    bool quantizable = fabs(delta) < radius;
    T candidate = delta + radius;

    if (not quantizable) {
      auto cur_idx = outlier->num()++;
      outlier->idx(cur_idx) = gid1();
      outlier->val(cur_idx) = delta;
      eqbuf_it(0) = 0;
    }
    else {
      eqbuf_it(0) = delta;
    }
  };
  auto threadview_store = [&]() {
    if (check_boundary1()) eq[gid1()] = eqbuf_it(0);
  };

  ////////////////////////////////////////
  data_partition();

  PFOR1_GRID()
  PFOR1_BLOCK()
  {
    threadview_load();
    threadview_process();
    threadview_store();
  }

  delete _buf1;
  delete _buf2;

}

template <typename T, typename EQ = int32_t, typename FP = T, int BLK = 256>
void x_lorenzo_1d1l(
    EQ* eq, T* scattered_outlier, psz_dim3 len3, psz_dim3 stride3, int radius,
    FP ebx2, T* xdata)
{
  SETUP_ND_CPU_SERIAL;
  SETUP_1D_DATABUF;

  // per-thread ("real" kernel)
  auto threadview_load = [&]() {
    if (check_boundary1())
      databuf_it(0) = eq[gid1()] + scattered_outlier[gid1()];
  };
  auto threadview_partial_sum = [&]() {
    if (t.x > 0) databuf_it(0) += databuf_it(-1);
  };
  auto threadview_store = [&]() {
    if (check_boundary1()) xdata[gid1()] = databuf_it(0) * ebx2;
  };

  ////////////////////////////////////////
  data_partition();

  PFOR1_GRID()
  PFOR1_BLOCK()
  {
    threadview_load();
    threadview_partial_sum();
    threadview_store();
  }

  delete _buf1;
}

template <
    typename T, typename EQ = int32_t, typename FP = T, int BLK = 16,
    typename OUTLIER = struct CompactSerial<T>>
void c_lorenzo_2d1l(
    T* data, psz_dim3 len3, psz_dim3 stride3, int radius, FP ebx2_r, EQ* eq,
    OUTLIER* outlier) {
  SETUP_ND_CPU_SERIAL;
  SETUP_2D_DATABUF;
  SETUP_2D_EQBUF;

  // per-thread ("real" kernel)
  auto threadview_load = [&]() {
    if (check_boundary2()) databuf_it(0, 0) = data[gid2()] * ebx2_r;
  };
  auto threadview_process = [&]() {
    auto delta = databuf_it(0, 0) -
                 (databuf_it(-1, 0) + databuf_it(0, -1) - databuf_it(-1, -1));

    bool quantizable = fabs(delta) < radius;
    T candidate = delta + radius;

    if (not quantizable) {
      auto cur_idx = outlier->num()++;
      outlier->idx(cur_idx) = gid2();
      outlier->val(cur_idx) = delta;
      eqbuf_it(0, 0) = 0;
    }
    else {
      eqbuf_it(0, 0) = delta;
    }
  };
  auto threadview_store = [&]() {
    if (check_boundary2()) eq[gid2()] = eqbuf_it(0, 0);
  };

  ////////////////////////////////////////
  data_partition();

  PFOR2_GRID()
  PFOR2_BLOCK()
  {
    threadview_load();
    threadview_process();
    threadview_store();
  }

  delete _buf1;
  delete _buf2;
}

template <typename T, typename EQ = int32_t, typename FP = T, int BLK = 16>
void x_lorenzo_2d1l(
    EQ* eq, T* scattered_outlier, psz_dim3 len3, psz_dim3 stride3, int radius,
    FP ebx2, T* xdata)
{
  SETUP_ND_CPU_SERIAL;
  SETUP_2D_DATABUF;

  // per-thread ("real" kernel)
  auto threadview_load = [&]() {
    if (check_boundary2())
      databuf_it(0, 0) = eq[gid2()] + scattered_outlier[gid2()];
  };
  auto threadview_partial_sum_x = [&]() {
    if (t.x > 0) databuf_it(0, 0) += databuf_it(-1, 0);
  };
  auto threadview_partial_sum_y = [&]() {
    if (t.y > 0) databuf_it(0, 0) += databuf_it(0, -1);
  };
  auto threadview_store = [&]() {
    if (check_boundary2()) xdata[gid2()] = databuf_it(0, 0) * ebx2;
  };

  ////////////////////////////////////////
  data_partition();

  PFOR2_GRID()
  PFOR2_BLOCK()
  {
    threadview_load();
    threadview_partial_sum_x();
    threadview_partial_sum_y();
    threadview_store();
  }

  delete _buf1;
}

template <
    typename T, typename EQ = int32_t, typename FP = T, int BLK = 8,
    typename OUTLIER = struct CompactSerial<T>>
void c_lorenzo_3d1l(
    T* data, psz_dim3 len3, psz_dim3 stride3, int radius, FP ebx2_r, EQ* eq,
    OUTLIER* outlier) {
  SETUP_ND_CPU_SERIAL;
  SETUP_3D_DATABUF;
  SETUP_3D_EQBUF;

  // per-thread ("real" kernel)
  auto threadview_load = [&]() {
    if (check_boundary3()) databuf_it(0, 0, 0) = data[gid3()] * ebx2_r;
  };
  auto threadview_process = [&]() {
    auto delta =
        databuf_it(0, 0, 0) -
        (databuf_it(-1, -1, -1) - databuf_it(0, -1, -1) -
         databuf_it(-1, 0, -1) - databuf_it(-1, -1, 0) + databuf_it(0, 0, -1) +
         databuf_it(0, -1, 0) + databuf_it(-1, 0, 0));
    bool quantizable = fabs(delta) < radius;
    T candidate = delta + radius;

    if (not quantizable) {
      auto cur_idx = outlier->num()++;
      outlier->idx(cur_idx) = gid3();
      outlier->val(cur_idx) = delta;
      eqbuf_it(0, 0, 0) = 0;
    }
    else {
      eqbuf_it(0, 0, 0) = delta;
    }
  };
  auto threadview_store = [&]() {
    if (check_boundary3()) eq[gid3()] = eqbuf_it(0, 0, 0);
  };

  ////////////////////////////////////////
  data_partition();

  PFOR3_GRID()
  PFOR3_BLOCK()
  {
    threadview_load();
    threadview_process();
    threadview_store();
  }

  delete _buf1;
  delete _buf2;
}

template <typename T, typename EQ = int32_t, typename FP = T, int BLK = 8>
void x_lorenzo_3d1l(
    EQ* eq, T* scattered_outlier, psz_dim3 len3, psz_dim3 stride3, int radius,
    FP ebx2, T* xdata)
{
  SETUP_ND_CPU_SERIAL;
  SETUP_3D_DATABUF;

  // per-thread ("real" kernel)
  auto threadview_load = [&]() {
    if (check_boundary3())
      databuf_it(0, 0, 0) = eq[gid3()] + scattered_outlier[gid3()];
  };
  auto threadview_partial_sum_x = [&]() {
    if (t.x > 0) databuf_it(0, 0, 0) += databuf_it(-1, 0, 0);
  };
  auto threadview_partial_sum_y = [&]() {
    if (t.y > 0) databuf_it(0, 0, 0) += databuf_it(0, -1, 0);
  };
  auto threadview_partial_sum_z = [&]() {
    if (t.z > 0) databuf_it(0, 0, 0) += databuf_it(0, 0, -1);
  };
  auto threadview_store = [&]() {
    if (check_boundary3()) xdata[gid3()] = databuf_it(0, 0, 0) * ebx2;
  };

  ////////////////////////////////////////
  data_partition();

  PFOR3_GRID()
  PFOR3_BLOCK()
  {
    threadview_load();
    threadview_partial_sum_x();
    threadview_partial_sum_y();
    threadview_partial_sum_z();
    threadview_store();
  }

  delete _buf1;
}

}  // namespace __kernel
}  // namespace seq
}  // namespace psz

#undef SETUP_1D
#undef PFOR1_GRID
#undef PFOR1_BLOCK
#undef SETUP_2D_BASIC
#undef PFOR2_GRID
#undef PFOR2_BLOCK
#undef SETUP_3D
#undef PFOR3_GRID
#undef PFOR3_BLOCK

#endif /* E0B87BA8_BEDC_4CBE_B5EE_C0C5875E07D6 */
