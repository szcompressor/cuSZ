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

#define SETUP_ND_CPU_SERIAL                                                   \
                                                                              \
  /* fake thread-block setup */                                               \
  psz_dim3_seq b, t;                /* (fake) threadblock-related indices */  \
  psz_dim3_seq grid_dim, block_dim; /* threadblock-related dimensions */      \
                                                                              \
  /* threadblock-related strides */                                           \
  auto TWy = [&]() { return block_dim.x; };                                   \
  auto TWz = [&]() { return block_dim.x * block_dim.y; };                     \
  auto BWy = [&]() { return grid_dim.x; };                                    \
  auto BWz = [&]() { return grid_dim.x * grid_dim.y; };                       \
                                                                              \
  /* threadblock idx, linearized */                                           \
  auto bid1 = [&]() { return b.x; };                                          \
  auto bid2 = [&]() { return b.x + b.y * BWy(); };                            \
  auto bid3 = [&]() { return b.x + b.y * BWy() + b.z * BWz(); };              \
                                                                              \
  /* thread idx, linearized */                                                \
  auto tid1 = [&]() { return t.x; };                                          \
  auto tid2 = [&]() { return t.x + t.y * TWy(); };                            \
  auto tid3 = [&]() { return t.x + t.y * TWy() + t.z * TWz(); };              \
                                                                              \
  /* global data id, BLK is defined in function template */                   \
  auto gx = [&]() { return t.x + b.x * BLK; };                                \
  auto gy = [&]() { return t.y + b.y * BLK; };                                \
  auto gz = [&]() { return t.z + b.z * BLK; };                                \
  auto gid1 = [&]() { return gx(); };                                         \
  auto gid2 = [&]() { return gx() + gy() * stride3.y; };                      \
  auto gid3 = [&]() { return gx() + gy() * stride3.y + gz() * stride3.z; };   \
                                                                              \
  /* partition */                                                             \
  auto data_partition = [&]() {                                               \
    grid_dim.x = (len3.x - 1) / BLK + 1, grid_dim.y = (len3.y - 1) / BLK + 1, \
    grid_dim.z = (len3.z - 1) / BLK + 1;                                      \
    block_dim.x = BLK, block_dim.y = BLK, block_dim.z = BLK;                  \
  };                                                                          \
                                                                              \
  /* check data access validity */                                            \
  auto check_boundary1 = [&]() { return gx() < len3.x; };                     \
  auto check_boundary2 = [&]() { return gx() < len3.x and gy() < len3.y; };   \
  auto check_boundary3 = [&]() { return check_boundary2() and gz() < len3.z; };

#define PARFOR1_GRID() for (b.x = 0; b.x < grid_dim.x; b.x++)
#define PARFOR1_BLOCK() for (t.x = 0; t.x < BLK; t.x++)

#define PARFOR2_GRID()                   \
  for (b.y = 0; b.y < grid_dim.y; b.y++) \
    for (b.x = 0; b.x < grid_dim.x; b.x++)
#define PARFOR2_BLOCK()                   \
  for (t.y = 0; t.y < block_dim.y; t.y++) \
    for (t.x = 0; t.x < block_dim.x; t.x++)
#define PARFOR2_BLOCK_along()             \
  for (t.y = 0; t.y < block_dim.y; t.y++) \
    for (t.x = 0; t.x < block_dim.x; t.x++)

#define PARFOR3_GRID()                     \
  for (b.z = 0; b.z < grid_dim.z; b.z++)   \
    for (b.y = 0; b.y < grid_dim.y; b.y++) \
      for (b.x = 0; b.x < grid_dim.x; b.x++)
#define PARFOR3_BLOCK()                     \
  for (t.z = 0; t.z < block_dim.z; t.z++)   \
    for (t.y = 0; t.y < block_dim.y; t.y++) \
      for (t.x = 0; t.x < block_dim.x; t.x++)

#include <cmath>
#include <cstring>
#include <iostream>

#include "detail/composite.hh"
#include "kernel/predictor.hh"
#include "mem/cxx_sp_cpu.h"

template <typename T, int DIM, int BLOCK>
struct psz_buf_seq {
 private:
  T* _buf;
  size_t _len{1};
  static const int stridey{BLOCK};
  static const int stridez{BLOCK * BLOCK};

 public:
  psz_buf_seq(bool do_memset = true)
  {
    if (DIM == 1) _len = BLOCK;
    if (DIM == 2) _len = BLOCK * BLOCK;
    if (DIM == 3) _len = BLOCK * BLOCK * BLOCK;
    _buf = new T[_len];
    if (do_memset) memset(_buf, 0x0, sizeof(T) * _len);
  }

  ~psz_buf_seq() { delete[] _buf; }

  T*& buf() { return _buf; }

  T& operator()(int x) { return _buf[x]; }
  T& operator()(int x, int y) { return _buf[x + y * stridey]; }
  T& operator()(int x, int y, int z) { return _buf[x + y * stridey + z * stridez]; }
};

auto div3 = [](psz_dim3_seq len, psz_dim3_seq sublen) {
  return psz_dim3_seq{
      (len.x - 1) / sublen.x + 1, (len.y - 1) / sublen.y + 1, (len.z - 1) / sublen.z + 1};
};

using std::cout;
using std::endl;

#define SETUP_1D_DATABUF                               \
  constexpr auto PADDING = 1;                          \
  auto _buf1 = new psz_buf_seq<T, 1, BLK + PADDING>(); \
  auto& buf1 = *_buf1;                                 \
  auto databuf_it = [&](auto x) -> T& { return buf1(t.x + x + PADDING); };
#define SETUP_1D_EQBUF                        \
  auto _buf2 = new psz_buf_seq<Eq, 1, BLK>(); \
  auto& buf2 = *_buf2;                        \
  auto eqbuf_it = [&](auto dx) -> Eq& { return buf2(t.x + dx); };

#define SETUP_2D_DATABUF                                 \
  constexpr auto PADDING = 1;                            \
  auto _buf1 = new psz_buf_seq<T, 2, BLK + PADDING>();   \
  auto& buf1 = *_buf1;                                   \
  auto databuf_it = [&](auto dx, auto dy) -> T& {        \
    return buf1(t.x + dx + PADDING, t.y + dy + PADDING); \
  };
#define SETUP_2D_EQBUF                        \
  auto _buf2 = new psz_buf_seq<Eq, 2, BLK>(); \
  auto& buf2 = *_buf2;                        \
  auto eqbuf_it = [&](auto dx, auto dy) -> Eq& { return buf2(t.x + dx, t.y + dy); };

#define SETUP_3D_DATABUF                                                     \
  constexpr auto PADDING = 1;                                                \
  auto _buf1 = new psz_buf_seq<T, 3, BLK + PADDING>();                       \
  auto& buf1 = *_buf1;                                                       \
  auto databuf_it = [&](auto dx, auto dy, auto dz) -> T& {                   \
    return buf1(t.x + dx + PADDING, t.y + dy + PADDING, t.z + dz + PADDING); \
  };
#define SETUP_3D_EQBUF                                    \
  auto _buf2 = new psz_buf_seq<Eq, 3, BLK>();             \
  auto& buf2 = *_buf2;                                    \
  auto eqbuf_it = [&](auto dx, auto dy, auto dz) -> Eq& { \
    return buf2(t.x + dx, t.y + dy, t.z + dz);            \
  };

#define SETUP_ZIGZAG                    \
  using ZigZag = psz::ZigZag<Eq>;       \
  using EqUInt = typename ZigZag::UInt; \
  using EqSInt = typename ZigZag::SInt;

namespace psz {

template <typename T, bool ZigZagEnabled, typename Eq = uint16_t, int BLK = 256>
void KERNEL_SEQ_c_lorenzo_1d1l(
    T* data, psz_dim3_seq len3, psz_dim3_seq stride3, uint16_t radius, f8 ebx2_r, Eq* in_eq,
    void* in_outlier)
{
  SETUP_ND_CPU_SERIAL;
  SETUP_1D_DATABUF;
  SETUP_1D_EQBUF;
  SETUP_ZIGZAG;

  auto outlier = (struct _portable::compact_CPU<T>*)in_outlier;

  // per-thread ("real" kernel)
  auto threadview_load = [&]() {
    if (check_boundary1()) databuf_it(0) = data[gid1()] * ebx2_r;
  };
  auto threadview_process = [&]() {
    auto delta = databuf_it(0) - databuf_it(-1);
    bool quantizable = fabs(delta) < radius;
    T candidate;

    if constexpr (ZigZagEnabled) {
      candidate = delta;
      eqbuf_it(0) = ZigZag::encode(static_cast<EqSInt>(quantizable * candidate));
    }
    else {
      candidate = delta + radius;
      eqbuf_it(0) = quantizable * (EqUInt)candidate;
    }

    if (not quantizable) {
      auto cur_idx = outlier->num()++;
      outlier->val_idx(cur_idx) = {(float)candidate, gid1()};
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

template <typename T, bool ZigZagEnabled, typename Eq = uint16_t, int BLK = 256>
void KERNEL_SEQ_x_lorenzo_1d1l(
    Eq* in_eq, T* in_outlier, psz_dim3_seq len3, psz_dim3_seq stride3, uint16_t radius, f8 ebx2,
    T* xdata)
{
  SETUP_ND_CPU_SERIAL;
  SETUP_1D_DATABUF;
  SETUP_ZIGZAG;

  // per-thread ("real" kernel)
  auto threadview_load = [&]() {
    if (check_boundary1()) {
      if constexpr (not ZigZagEnabled) {
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

template <typename T, bool ZigZagEnabled, typename Eq = uint16_t, int BLK = 16>
void KERNEL_SEQ_c_lorenzo_2d1l(
    T* data, psz_dim3_seq len3, psz_dim3_seq stride3, uint16_t radius, f8 ebx2_r, Eq* in_eq,
    void* in_outlier)
{
  SETUP_ND_CPU_SERIAL;
  SETUP_2D_DATABUF;
  SETUP_2D_EQBUF;
  SETUP_ZIGZAG;

  auto outlier = (struct _portable::compact_CPU<T>*)in_outlier;

  // per-thread ("real" kernel)
  auto threadview_load = [&]() {
    if (check_boundary2()) databuf_it(0, 0) = data[gid2()] * ebx2_r;
  };
  auto threadview_process = [&]() {
    auto delta = databuf_it(0, 0) - (databuf_it(-1, 0) + databuf_it(0, -1) - databuf_it(-1, -1));

    bool quantizable = fabs(delta) < radius;
    T candidate;

    if constexpr (ZigZagEnabled) {
      candidate = delta;
      eqbuf_it(0, 0) = ZigZag::encode(static_cast<EqSInt>(quantizable * candidate));
    }
    else {
      candidate = delta + radius;
      eqbuf_it(0, 0) = quantizable * (EqUInt)candidate;
    }

    if (not quantizable) {
      auto cur_idx = outlier->num()++;
      outlier->val_idx(cur_idx) = {(float)candidate, gid2()};
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

template <typename T, bool ZigZagEnabled, typename Eq = uint16_t, int BLK = 16>
void KERNEL_SEQ_x_lorenzo_2d1l(
    Eq* in_eq, T* in_outlier, psz_dim3_seq len3, psz_dim3_seq stride3, uint16_t radius, f8 ebx2,
    T* xdata)
{
  SETUP_ND_CPU_SERIAL;
  SETUP_2D_DATABUF;
  SETUP_ZIGZAG;

  // per-thread ("real" kernel)
  auto threadview_load = [&]() {
    if (check_boundary2()) {
      if constexpr (not ZigZagEnabled) {
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

template <typename T, bool ZigZagEnabled, typename Eq = uint16_t, int BLK = 8>
void KERNEL_SEQ_c_lorenzo_3d1l(
    T* data, psz_dim3_seq len3, psz_dim3_seq stride3, uint16_t radius, f8 ebx2_r, Eq* in_eq,
    void* in_outlier)
{
  SETUP_ND_CPU_SERIAL;
  SETUP_3D_DATABUF;
  SETUP_3D_EQBUF;
  SETUP_ZIGZAG;

  auto outlier = (struct _portable::compact_CPU<T>*)in_outlier;

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

    if constexpr (ZigZagEnabled) {
      candidate = delta;
      eqbuf_it(0, 0, 0) = ZigZag::encode(static_cast<EqSInt>(quantizable * candidate));
    }
    else {
      candidate = delta + radius;
      eqbuf_it(0, 0, 0) = quantizable * (EqUInt)candidate;
    }

    if (not quantizable) {
      auto cur_idx = outlier->num()++;
      outlier->val_idx(cur_idx) = {(float)candidate, gid3()};
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

template <typename T, bool ZigZagEnabled, typename Eq = uint16_t, int BLK = 8>
void KERNEL_SEQ_x_lorenzo_3d1l(
    Eq* in_eq, T* in_outlier, psz_dim3_seq len3, psz_dim3_seq stride3, uint16_t radius, f8 ebx2,
    T* xdata)
{
  SETUP_ND_CPU_SERIAL;
  SETUP_3D_DATABUF;
  SETUP_ZIGZAG;

  // per-thread ("real" kernel)
  auto threadview_load = [&]() {
    if (check_boundary3()) {
      if constexpr (not ZigZagEnabled) {
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
