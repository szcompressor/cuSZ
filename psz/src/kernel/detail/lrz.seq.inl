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
#include "cusz/suint.hh"
#include "mem/cxx_sp_cpu.h"

auto div3 = [](psz_dim3 len, psz_dim3 sublen) {
  return psz_dim3{
      (len.x - 1) / sublen.x + 1, (len.y - 1) / sublen.y + 1, (len.z - 1) / sublen.z + 1};
};

using std::cout;
using std::endl;

#define SETUP_1D_DATABUF                           \
  constexpr auto PADDING = 1;                      \
  auto _buf1 = new psz_buf<T, 1, BLK + PADDING>(); \
  auto& buf1 = *_buf1;                             \
  auto databuf_it = [&](auto x) -> T& { return buf1(t.x + x + PADDING); };
#define SETUP_1D_EQBUF                    \
  auto _buf2 = new psz_buf<Eq, 1, BLK>(); \
  auto& buf2 = *_buf2;                    \
  auto eqbuf_it = [&](auto dx) -> Eq& { return buf2(t.x + dx); };

#define SETUP_2D_DATABUF                                 \
  constexpr auto PADDING = 1;                            \
  auto _buf1 = new psz_buf<T, 2, BLK + PADDING>();       \
  auto& buf1 = *_buf1;                                   \
  auto databuf_it = [&](auto dx, auto dy) -> T& {        \
    return buf1(t.x + dx + PADDING, t.y + dy + PADDING); \
  };
#define SETUP_2D_EQBUF                    \
  auto _buf2 = new psz_buf<Eq, 2, BLK>(); \
  auto& buf2 = *_buf2;                    \
  auto eqbuf_it = [&](auto dx, auto dy) -> Eq& { return buf2(t.x + dx, t.y + dy); };

#define SETUP_3D_DATABUF                                                     \
  constexpr auto PADDING = 1;                                                \
  auto _buf1 = new psz_buf<T, 3, BLK + PADDING>();                           \
  auto& buf1 = *_buf1;                                                       \
  auto databuf_it = [&](auto dx, auto dy, auto dz) -> T& {                   \
    return buf1(t.x + dx + PADDING, t.y + dy + PADDING, t.z + dz + PADDING); \
  };
#define SETUP_3D_EQBUF                                    \
  auto _buf2 = new psz_buf<Eq, 3, BLK>();                 \
  auto& buf2 = *_buf2;                                    \
  auto eqbuf_it = [&](auto dx, auto dy, auto dz) -> Eq& { \
    return buf2(t.x + dx, t.y + dy, t.z + dz);            \
  };

#define SETUP_ZIGZAG                    \
  using ZigZag = psz::ZigZag<Eq>;       \
  using EqUInt = typename ZigZag::UInt; \
  using EqSInt = typename ZigZag::SInt;

namespace psz {

template <typename T, bool UseZigZag, typename Eq = uint16_t, int BLK = 256>
void KERNEL_SEQ_c_lorenzo_1d1l(
    T* data, psz_dim3 len3, psz_dim3 stride3, uint16_t radius, f8 ebx2_r, Eq* in_eq,
    void* in_outlier)
{
  SETUP_ND_CPU_SERIAL;
  SETUP_1D_DATABUF;
  SETUP_1D_EQBUF;
  SETUP_ZIGZAG;

  auto outlier = (struct _portable::compact_seq<T>*)in_outlier;

  // per-thread ("real" kernel)
  auto threadview_load = [&]() {
    if (check_boundary1()) databuf_it(0) = data[gid1()] * ebx2_r;
  };
  auto threadview_process = [&]() {
    auto delta = databuf_it(0) - databuf_it(-1);
    bool quantizable = fabs(delta) < radius;
    T candidate;

    if constexpr (UseZigZag) {
      candidate = delta;
      eqbuf_it(0) = ZigZag::encode(static_cast<EqSInt>(quantizable * candidate));
    }
    else {
      candidate = delta + radius;
      eqbuf_it(0) = quantizable * (EqUInt)candidate;
    }

    if (not quantizable) {
      auto cur_idx = outlier->num()++;
      outlier->idx(cur_idx) = gid1();
      outlier->val(cur_idx) = candidate;
    }
  };
  auto threadview_store = [&]() {
    if (check_boundary1()) in_eq[gid1()] = eqbuf_it(0);
  };

  ////////////////////////////////////////
  data_partition();

  PARFOR1_GRID()
  PARFOR1_BLOCK()
  {
    threadview_load();
    threadview_process();
    threadview_store();
  }

  delete _buf1;
  delete _buf2;
}

template <typename T, bool UseZigZag, typename Eq = uint16_t, int BLK = 256>
void KERNEL_SEQ_x_lorenzo_1d1l(
    Eq* in_eq, T* in_outlier, psz_dim3 len3, psz_dim3 stride3, uint16_t radius, f8 ebx2, T* xdata)
{
  SETUP_ND_CPU_SERIAL;
  SETUP_1D_DATABUF;
  SETUP_ZIGZAG;

  // per-thread ("real" kernel)
  auto threadview_load = [&]() {
    if (check_boundary1()) {
      if constexpr (not UseZigZag) {
        databuf_it(0) = in_outlier[gid1()] + static_cast<T>(in_eq[gid1()]) - radius;
      }
      else {
        auto e = in_eq[gid1()];
        databuf_it(0) =
            in_outlier[gid1()] + static_cast<T>(ZigZag::decode(static_cast<EqUInt>(e)));
      }
    }
  };
  auto threadview_partial_sum = [&]() {
    if (t.x > 0) databuf_it(0) += databuf_it(-1);
  };
  auto threadview_store = [&]() {
    if (check_boundary1()) xdata[gid1()] = databuf_it(0) * ebx2;
  };

  ////////////////////////////////////////
  data_partition();

  PARFOR1_GRID()
  PARFOR1_BLOCK()
  {
    threadview_load();
    threadview_partial_sum();
    threadview_store();
  }

  delete _buf1;
}

template <typename T, bool UseZigZag, typename Eq = uint16_t, int BLK = 16>
void KERNEL_SEQ_c_lorenzo_2d1l(
    T* data, psz_dim3 len3, psz_dim3 stride3, uint16_t radius, f8 ebx2_r, Eq* in_eq,
    void* in_outlier)
{
  SETUP_ND_CPU_SERIAL;
  SETUP_2D_DATABUF;
  SETUP_2D_EQBUF;
  SETUP_ZIGZAG;

  auto outlier = (struct _portable::compact_seq<T>*)in_outlier;

  // per-thread ("real" kernel)
  auto threadview_load = [&]() {
    if (check_boundary2()) databuf_it(0, 0) = data[gid2()] * ebx2_r;
  };
  auto threadview_process = [&]() {
    auto delta = databuf_it(0, 0) - (databuf_it(-1, 0) + databuf_it(0, -1) - databuf_it(-1, -1));

    bool quantizable = fabs(delta) < radius;
    T candidate;

    if constexpr (UseZigZag) {
      candidate = delta;
      eqbuf_it(0, 0) = ZigZag::encode(static_cast<EqSInt>(quantizable * candidate));
    }
    else {
      candidate = delta + radius;
      eqbuf_it(0, 0) = quantizable * (EqUInt)candidate;
    }

    if (not quantizable) {
      auto cur_idx = outlier->num()++;
      outlier->idx(cur_idx) = gid2();
      outlier->val(cur_idx) = candidate;
    }
  };
  auto threadview_store = [&]() {
    if (check_boundary2()) in_eq[gid2()] = eqbuf_it(0, 0);
  };

  ////////////////////////////////////////
  data_partition();

  PARFOR2_GRID()
  PARFOR2_BLOCK()
  {
    threadview_load();
    threadview_process();
    threadview_store();
  }

  delete _buf1;
  delete _buf2;
}

template <typename T, bool UseZigZag, typename Eq = uint16_t, int BLK = 16>
void KERNEL_SEQ_x_lorenzo_2d1l(
    Eq* in_eq, T* in_outlier, psz_dim3 len3, psz_dim3 stride3, uint16_t radius, f8 ebx2, T* xdata)
{
  SETUP_ND_CPU_SERIAL;
  SETUP_2D_DATABUF;
  SETUP_ZIGZAG;

  // per-thread ("real" kernel)
  auto threadview_load = [&]() {
    if (check_boundary2()) {
      if constexpr (not UseZigZag) {
        databuf_it(0, 0) = in_outlier[gid2()] + static_cast<T>(in_eq[gid2()]) - radius;
      }
      else {
        auto e = in_eq[gid2()];
        databuf_it(0, 0) =
            in_outlier[gid2()] + static_cast<T>(ZigZag::decode(static_cast<EqUInt>(e)));
      }
    }
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

  PARFOR2_GRID()
  {
    PARFOR2_BLOCK() { threadview_load(); }
    PARFOR2_BLOCK() { threadview_partial_sum_x(); }
    PARFOR2_BLOCK() { threadview_partial_sum_y(); }
    PARFOR2_BLOCK() { threadview_store(); }
  }

  delete _buf1;
}

template <typename T, bool UseZigZag, typename Eq = uint16_t, int BLK = 8>
void KERNEL_SEQ_c_lorenzo_3d1l(
    T* data, psz_dim3 len3, psz_dim3 stride3, uint16_t radius, f8 ebx2_r, Eq* in_eq,
    void* in_outlier)
{
  SETUP_ND_CPU_SERIAL;
  SETUP_3D_DATABUF;
  SETUP_3D_EQBUF;
  SETUP_ZIGZAG;

  auto outlier = (struct _portable::compact_seq<T>*)in_outlier;

  // per-thread ("real" kernel)
  auto threadview_load = [&]() {
    if (check_boundary3()) databuf_it(0, 0, 0) = data[gid3()] * ebx2_r;
  };
  auto threadview_process = [&]() {
    auto delta =
        databuf_it(0, 0, 0) - (databuf_it(-1, -1, -1) - databuf_it(0, -1, -1) -
                               databuf_it(-1, 0, -1) - databuf_it(-1, -1, 0) +
                               databuf_it(0, 0, -1) + databuf_it(0, -1, 0) + databuf_it(-1, 0, 0));
    bool quantizable = fabs(delta) < radius;
    T candidate;

    if constexpr (UseZigZag) {
      candidate = delta;
      eqbuf_it(0, 0, 0) = ZigZag::encode(static_cast<EqSInt>(quantizable * candidate));
    }
    else {
      candidate = delta + radius;
      eqbuf_it(0, 0, 0) = quantizable * (EqUInt)candidate;
    }

    if (not quantizable) {
      auto cur_idx = outlier->num()++;
      outlier->idx(cur_idx) = gid3();
      outlier->val(cur_idx) = candidate;
    }
  };
  auto threadview_store = [&]() {
    if (check_boundary3()) in_eq[gid3()] = eqbuf_it(0, 0, 0);
  };

  ////////////////////////////////////////
  data_partition();

  PARFOR3_GRID()
  PARFOR3_BLOCK()
  {
    threadview_load();
    threadview_process();
    threadview_store();
  }

  delete _buf1;
  delete _buf2;
}

template <typename T, bool UseZigZag, typename Eq = uint16_t, int BLK = 8>
void KERNEL_SEQ_x_lorenzo_3d1l(
    Eq* in_eq, T* in_outlier, psz_dim3 len3, psz_dim3 stride3, uint16_t radius, f8 ebx2, T* xdata)
{
  SETUP_ND_CPU_SERIAL;
  SETUP_3D_DATABUF;
  SETUP_ZIGZAG;

  // per-thread ("real" kernel)
  auto threadview_load = [&]() {
    if (check_boundary3()) {
      if constexpr (not UseZigZag) {
        databuf_it(0, 0, 0) = in_outlier[gid3()] + static_cast<T>(in_eq[gid3()]) - radius;
      }
      else {
        auto e = in_eq[gid3()];
        databuf_it(0, 0, 0) =
            in_outlier[gid3()] + static_cast<T>(ZigZag::decode(static_cast<EqUInt>(e)));
      }
    }
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

  PARFOR3_GRID()
  {
    PARFOR3_BLOCK() { threadview_load(); }
    PARFOR3_BLOCK() { threadview_partial_sum_x(); }
    PARFOR3_BLOCK() { threadview_partial_sum_y(); }
    PARFOR3_BLOCK() { threadview_partial_sum_z(); }
    PARFOR3_BLOCK() { threadview_store(); }
  }

  delete _buf1;
}

}  // namespace psz

#undef SETUP_1D
#undef PARFOR1_GRID
#undef PARFOR1_BLOCK
#undef SETUP_2D_BASIC
#undef PARFOR2_GRID
#undef PARFOR2_BLOCK
#undef SETUP_3D
#undef PARFOR3_GRID
#undef PARFOR3_BLOCK

#endif /* E0B87BA8_BEDC_4CBE_B5EE_C0C5875E07D6 */
