// Author: Jinyang Liu, Shixun Wu, Jiannan Tian

#ifndef CUSZ_KERNEL_SPLINE_Y24_CUH
#define CUSZ_KERNEL_SPLINE_Y24_CUH

#include <cstdint>
#include <cstdio>

#include "cusz/type.h"
#include "utils/err.hh"

constexpr auto SPLINE3_COMPR = true;
constexpr auto SPLINE3_DECOMPR = false;
constexpr auto SPLINE3_PRED_ATT = true;
constexpr auto SPLINE3_AB_ATT = false;

#define TIX threadIdx.x
#define TIY threadIdx.y
#define TIZ threadIdx.z
#define BIX blockIdx.x
#define BIY blockIdx.y
#define BIZ blockIdx.z
#define BDX blockDim.x
#define BDY blockDim.y
#define BDZ blockDim.z
#define GDX gridDim.x
#define GDY gridDim.y
#define GDZ gridDim.z

constexpr int BLOCK8 = 8;
constexpr int BLOCK32 = 32;
constexpr int DEFAULT_LINEAR_BLOCK_SIZE = 384;

namespace psz {

template <
    typename T, typename E, typename FP = float, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE,
    typename CompactValIdx = void*, typename CompactNum = uint32_t*>
__global__ void KCU_c_spline3d_infprecis_32x8x8data(
    T* data, dim3 data_size, dim3 data_leap, E* eq, dim3 eq_size, dim3 eq_leap, T* anchor,
    dim3 anchor_leap, CompactValIdx cvi, CompactNum cn, FP eb_r, FP ebx2, int radius);

template <
    typename E, typename T, typename FP = float, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__global__ void KCU_x_spline3d_infprecis_32x8x8data(
    E* eq, dim3 eq_size, dim3 eq_leap, T* anchor, dim3 anchor_size, dim3 anchor_leap, T* data,
    dim3 data_size, dim3 data_leap, FP eb_r, FP ebx2, int radius);

}  // namespace psz

namespace psz {

template <
    typename T1, typename T2, typename FP, int LINEAR_BLOCK_SIZE, bool WORKFLOW = SPLINE3_COMPR,
    bool PROBE_PRED_ERROR = false>
__device__ void spline3d_layout2_interpolate(
    volatile T1 s_data[9][9][33], volatile T2 s_eq[9][9][33], FP eb_r, FP ebx2, int radius);
}  // namespace psz

namespace {

template <bool INCLUSIVE = true>
__forceinline__ __device__ bool xyz33x9x9_predicate(
    unsigned int x, unsigned int y, unsigned int z, const dim3& data_size)
{
  if constexpr (INCLUSIVE) {  //

    return (x <= 32 and y <= 8 and z <= 8) and BIX * BLOCK32 + x < data_size.x and
           BIY * BLOCK8 + y < data_size.y and BIZ * BLOCK8 + z < data_size.z;
  }
  else {
    return x < 32 + (BIX == GDX - 1) and y < 8 + (BIY == GDY - 1) and z < 8 + (BIZ == GDZ - 1) and
           BIX * BLOCK32 + x < data_size.x and BIY * BLOCK8 + y < data_size.y and
           BIZ * BLOCK8 + z < data_size.z;
  }
}

// control block_id3 in function call
template <typename T, bool PRINT_FP = true, int XEND = 33, int YEND = 9, int ZEND = 9>
__device__ void spline3d_print_block_from_GPU(
    T volatile a[9][9][33], int radius = 512, bool compress = true, bool print_ectrl = true)
{
  for (auto z = 0; z < ZEND; z++) {
    printf("\nprint from GPU, z=%d\n", z);
    printf("    ");
    for (auto i = 0; i < 33; i++) printf("%3d", i);
    printf("\n");

    for (auto y = 0; y < YEND; y++) {
      printf("y=%d ", y);
      for (auto x = 0; x < XEND; x++) {  //
        if constexpr (PRINT_FP) { printf("%.2e\t", (float)a[z][y][x]); }
        else {
          T c = print_ectrl ? a[z][y][x] - radius : a[z][y][x];
          if (compress) {
            if (c == 0) { printf("%3c", '.'); }
            else {
              if (abs(c) >= 10) { printf("%3c", '*'); }
              else {
                if (print_ectrl) { printf("%3d", c); }
                else {
                  printf("%4.2f", c);
                }
              }
            }
          }
          else {
            if (print_ectrl) { printf("%3d", c); }
            else {
              printf("%4.2f", c);
            }
          }
        }
      }
      printf("\n");
    }
  }
  printf("\nGPU print end\n\n");
}

template <typename T1, typename T2, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void c_reset_scratch_33x9x9data(
    volatile T1 s_data[9][9][33], volatile T2 s_eq[9][9][33], int radius)
{
  // alternatively, reinterprete cast volatile T?[][][] to 1D
  for (auto _tix = TIX; _tix < 33 * 9 * 9; _tix += LINEAR_BLOCK_SIZE) {
    auto x = (_tix % 33);
    auto y = (_tix / 33) % 9;
    auto z = (_tix / 33) / 9;

    s_data[z][y][x] = 0;
    /*****************************************************************************
     okay to use
     ******************************************************************************/
    if (x % 8 == 0 and y % 8 == 0 and z % 8 == 0) s_eq[z][y][x] = radius;
    /*****************************************************************************
     alternatively
     ******************************************************************************/
    // s_eq[z][y][x] = radius;
  }
  __syncthreads();
}

template <typename T1, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void c_gather_anchor(
    T1* data, dim3 data_size, dim3 data_leap, T1* anchor, dim3 anchor_leap, dim3 begin)
{
  auto x = begin.x + (TIX % 32) + BIX * 32;
  auto y = begin.y + (TIX / 32) % 8 + BIY * 8;
  auto z = begin.z + (TIX / 32) / 8 + BIZ * 8;

  bool pred1 = x % 8 == 0 and y % 8 == 0 and z % 8 == 0;
  bool pred2 = x < data_size.x and y < data_size.y and z < data_size.z;

  if (pred1 and pred2) {
    auto data_id = x + y * data_leap.y + z * data_leap.z;
    auto anchor_id = (x / 8) + (y / 8) * anchor_leap.y + (z / 8) * anchor_leap.z;
    anchor[anchor_id] = data[data_id];
  }
  __syncthreads();
}

template <typename T1, typename T2 = T1, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void x_reset_scratch_33x9x9data(
    volatile T1 s_xdata[9][9][33], volatile T2 s_eq[9][9][33], T1* anchor, dim3 anchor_size,
    dim3 anchor_leap, dim3 begin)
{
  for (auto _tix = TIX; _tix < 33 * 9 * 9; _tix += LINEAR_BLOCK_SIZE) {
    auto x = (_tix % 33);
    auto y = (_tix / 33) % 9;
    auto z = (_tix / 33) / 9;

    s_eq[z][y][x] = 0;  // TODO explicitly handle zero-padding
    /*****************************************************************************
     okay to use
     ******************************************************************************/
    if (x % 8 == 0 and y % 8 == 0 and z % 8 == 0) {
      s_xdata[z][y][x] = 0;

      auto ax = (begin.x / 8 + (x / 8) + BIX * 4);
      auto ay = (begin.y / 8 + (y / 8) + BIY);
      auto az = (begin.z / 8 + (z / 8) + BIZ);

      if (ax < anchor_size.x and ay < anchor_size.y and az < anchor_size.z)
        s_xdata[z][y][x] = anchor[ax + ay * anchor_leap.y + az * anchor_leap.z];
    }
    /*****************************************************************************
     alternatively
     ******************************************************************************/
    // s_eq[z][y][x] = radius;
  }

  __syncthreads();
}

template <typename T1, typename T2, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void global2shmem_33x9x9data(
    T1* data, dim3 data_size, dim3 data_leap, dim3 begin, volatile T2 s_data[9][9][33])
{
  constexpr auto TOTAL = 33 * 9 * 9;

  for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
    auto x = (_tix % 33);
    auto y = (_tix / 33) % 9;
    auto z = (_tix / 33) / 9;
    auto gx = (begin.x + x + BIX * BLOCK32);
    auto gy = (begin.y + y + BIY * BLOCK8);
    auto gz = (begin.z + z + BIZ * BLOCK8);
    auto gid = gx + gy * data_leap.y + gz * data_leap.z;

    if (gx < data_size.x and gy < data_size.y and gz < data_size.z) s_data[z][y][x] = data[gid];
  }
  __syncthreads();
}

template <typename T = float, typename E = u4, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void global2shmem_fuse(
    E* eq, dim3 eq_size, dim3 eq_leap, T* scattered_outlier, dim3 begin, volatile T s_eq[9][9][33])
{
  constexpr auto TOTAL = 33 * 9 * 9;

  for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
    auto x = (_tix % 33);
    auto y = (_tix / 33) % 9;
    auto z = (_tix / 33) / 9;
    auto gx = (begin.x + x + BIX * BLOCK32);
    auto gy = (begin.y + y + BIY * BLOCK8);
    auto gz = (begin.z + z + BIZ * BLOCK8);
    auto gid = gx + gy * eq_leap.y + gz * eq_leap.z;

    if (gx < eq_size.x and gy < eq_size.y and gz < eq_size.z)
      s_eq[z][y][x] = static_cast<T>(eq[gid]) + scattered_outlier[gid];
  }
  __syncthreads();
}

// dram_outlier should be the same in type with shared memory buf
template <typename T1, typename T2, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void shmem2global_32x8x8data(
    volatile T1 s_buf[9][9][33], T2* dram_buf, dim3 buf_size, dim3 buf_leap, dim3 begin)
{
  auto x_size = BLOCK32 + (BIX == GDX - 1);
  auto y_size = BLOCK8 + (BIY == GDY - 1);
  auto z_size = BLOCK8 + (BIZ == GDZ - 1);
  // constexpr auto TOTAL = 32 * 8 * 8;
  auto TOTAL = x_size * y_size * z_size;

  for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
    auto x = (_tix % x_size);
    auto y = (_tix / x_size) % y_size;
    auto z = (_tix / x_size) / y_size;
    auto gx = (begin.x + x + BIX * BLOCK32);
    auto gy = (begin.y + y + BIY * BLOCK8);
    auto gz = (begin.z + z + BIZ * BLOCK8);
    auto gid = gx + gy * buf_leap.y + gz * buf_leap.z;

    if (gx < buf_size.x and gy < buf_size.y and gz < buf_size.z) dram_buf[gid] = s_buf[z][y][x];
  }
  __syncthreads();
}

// dram_outlier should be the same in type with shared memory buf
template <
    typename T1, typename T2, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE,
    typename CompactValIdx>
__device__ void shmem2global_32x8x8data_with_compaction(
    volatile T1 s_buf[9][9][33], T2* dram_buf, dim3 buf_size, dim3 buf_leap, dim3 begin, int radius,
    CompactValIdx* dram_compact = nullptr, uint32_t* dram_compactnum = nullptr)
{
  auto x_size = BLOCK32 + (BIX == GDX - 1);
  auto y_size = BLOCK8 + (BIY == GDY - 1);
  auto z_size = BLOCK8 + (BIZ == GDZ - 1);
  auto TOTAL = x_size * y_size * z_size;

  for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
    auto x = (_tix % x_size);
    auto y = (_tix / x_size) % y_size;
    auto z = (_tix / x_size) / y_size;
    auto gx = (begin.x + x + BIX * BLOCK32);
    auto gy = (begin.y + y + BIY * BLOCK8);
    auto gz = (begin.z + z + BIZ * BLOCK8);
    auto gid = gx + gy * buf_leap.y + gz * buf_leap.z;

    auto candidate = s_buf[z][y][x];
    bool quantizable = (candidate >= 0) and (candidate < 2 * radius);

    if (gx < buf_size.x and gy < buf_size.y and gz < buf_size.z) {
      // TODO this is for algorithmic demo by reading from shmem
      // For performance purpose, it can be inlined in quantization
      dram_buf[gid] = quantizable * static_cast<T2>(candidate);

      if (not quantizable) {
        using Val = typename CompactValIdx::OutlierValT;
        auto cur_idx = atomicAdd(dram_compactnum, 1);
        dram_compact[cur_idx] = {(Val)candidate, gid};
      }
    }
  }
  __syncthreads();
}

template <
    typename T1, typename T2, typename FP, typename LAMBDAX, typename LAMBDAY, typename LAMBDAZ,
    bool BLUE, bool YELLOW, bool HOLLOW, int LINEAR_BLOCK_SIZE, int BLOCK_DIMX, int BLOCK_DIMY,
    bool COARSEN, int BLOCK_DIMZ, bool BORDER_INCLUSIVE, bool WORKFLOW>
__forceinline__ __device__ void interpolate_stage(
    volatile T1 s_data[9][9][33], volatile T2 s_eq[9][9][33], dim3 data_size, LAMBDAX xmap,
    LAMBDAY ymap, LAMBDAZ zmap, int unit, FP eb_r, FP ebx2, int radius, bool cubic)
{
  static_assert(BLOCK_DIMX * BLOCK_DIMY * (COARSEN ? 1 : BLOCK_DIMZ) <= 384, "block oversized");
  static_assert((BLUE or YELLOW or HOLLOW) == true, "must be one hot");
  static_assert((BLUE and YELLOW) == false, "must be only one hot (1)");
  static_assert((BLUE and YELLOW) == false, "must be only one hot (2)");
  static_assert((YELLOW and HOLLOW) == false, "must be only one hot (3)");

  auto run = [&](auto x, auto y, auto z) {
    if (xyz33x9x9_predicate<BORDER_INCLUSIVE>(x, y, z, data_size)) {
      T1 pred = 0;

      auto global_x = BIX * BLOCK32 + x, global_y = BIY * BLOCK8 + y, global_z = BIZ * BLOCK8 + z;
      if (cubic) {
        if constexpr (BLUE) {  //

          if (BIZ != GDZ - 1) {
            if (z >= 3 * unit and z + 3 * unit <= BLOCK8)
              pred = (-s_data[z - 3 * unit][y][x] + 9 * s_data[z - unit][y][x] +
                      9 * s_data[z + unit][y][x] - s_data[z + 3 * unit][y][x]) /
                     16;
            else if (z + 3 * unit <= BLOCK8)
              pred = (3 * s_data[z - unit][y][x] + 6 * s_data[z + unit][y][x] -
                      s_data[z + 3 * unit][y][x]) /
                     8;
            else if (z >= 3 * unit)
              pred = (-s_data[z - 3 * unit][y][x] + 6 * s_data[z - unit][y][x] +
                      3 * s_data[z + unit][y][x]) /
                     8;

            else
              pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x]) / 2;
          }
          else {
            if (z >= 3 * unit) {
              if (z + 3 * unit <= BLOCK8 and global_z + 3 * unit < data_size.z)
                pred = (-s_data[z - 3 * unit][y][x] + 9 * s_data[z - unit][y][x] +
                        9 * s_data[z + unit][y][x] - s_data[z + 3 * unit][y][x]) /
                       16;
              else if (global_z + unit < data_size.z)
                pred = (-s_data[z - 3 * unit][y][x] + 6 * s_data[z - unit][y][x] +
                        3 * s_data[z + unit][y][x]) /
                       8;
              else
                pred = s_data[z - unit][y][x];
            }
            else {
              if (z + 3 * unit <= BLOCK8 and global_z + 3 * unit < data_size.z)
                pred = (3 * s_data[z - unit][y][x] + 6 * s_data[z + unit][y][x] -
                        s_data[z + 3 * unit][y][x]) /
                       8;
              else if (global_z + unit < data_size.z)
                pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x]) / 2;
              else
                pred = s_data[z - unit][y][x];
            }
          }
        }
        if constexpr (YELLOW) {  //

          if (BIY != GDY - 1) {
            if (y >= 3 * unit and y + 3 * unit <= BLOCK8)
              pred = (-s_data[z][y - 3 * unit][x] + 9 * s_data[z][y - unit][x] +
                      9 * s_data[z][y + unit][x] - s_data[z][y + 3 * unit][x]) /
                     16;
            else if (y + 3 * unit <= BLOCK8)
              pred = (3 * s_data[z][y - unit][x] + 6 * s_data[z][y + unit][x] -
                      s_data[z][y + 3 * unit][x]) /
                     8;
            else if (y >= 3 * unit)
              pred = (-s_data[z][y - 3 * unit][x] + 6 * s_data[z][y - unit][x] +
                      3 * s_data[z][y + unit][x]) /
                     8;
            else
              pred = (s_data[z][y - unit][x] + s_data[z][y + unit][x]) / 2;
          }
          else {
            if (y >= 3 * unit) {
              if (y + 3 * unit <= BLOCK8 and global_y + 3 * unit < data_size.y)
                pred = (-s_data[z][y - 3 * unit][x] + 9 * s_data[z][y - unit][x] +
                        9 * s_data[z][y + unit][x] - s_data[z][y + 3 * unit][x]) /
                       16;
              else if (global_y + unit < data_size.y)
                pred = (-s_data[z][y - 3 * unit][x] + 6 * s_data[z][y - unit][x] +
                        3 * s_data[z][y + unit][x]) /
                       8;
              else
                pred = s_data[z][y - unit][x];
            }
            else {
              if (y + 3 * unit <= BLOCK8 and global_y + 3 * unit < data_size.y)
                pred = (3 * s_data[z][y - unit][x] + 6 * s_data[z][y + unit][x] -
                        s_data[z][y + 3 * unit][x]) /
                       8;
              else if (global_y + unit < data_size.y)
                pred = (s_data[z][y - unit][x] + s_data[z][y + unit][x]) / 2;
              else
                pred = s_data[z][y - unit][x];
            }
          }
        }

        if constexpr (HOLLOW) {  //
          // if(BIX == 5 and BIY == 22 and BIZ == 6 and unit==1)
          //     printf("%d %d %d\n",x,y,z);
          if (BIX != GDX - 1) {
            if (x >= 3 * unit and x + 3 * unit <= BLOCK32)
              pred = (-s_data[z][y][x - 3 * unit] + 9 * s_data[z][y][x - unit] +
                      9 * s_data[z][y][x + unit] - s_data[z][y][x + 3 * unit]) /
                     16;
            else if (x + 3 * unit <= BLOCK32)
              pred = (3 * s_data[z][y][x - unit] + 6 * s_data[z][y][x + unit] -
                      s_data[z][y][x + 3 * unit]) /
                     8;
            else if (x >= 3 * unit)
              pred = (-s_data[z][y][x - 3 * unit] + 6 * s_data[z][y][x - unit] +
                      3 * s_data[z][y][x + unit]) /
                     8;
            else
              pred = (s_data[z][y][x - unit] + s_data[z][y][x + unit]) / 2;
          }
          else {
            if (x >= 3 * unit) {
              if (x + 3 * unit <= BLOCK32 and global_x + 3 * unit < data_size.x)
                pred = (-s_data[z][y][x - 3 * unit] + 9 * s_data[z][y][x - unit] +
                        9 * s_data[z][y][x + unit] - s_data[z][y][x + 3 * unit]) /
                       16;
              else if (global_x + unit < data_size.x)
                pred = (-s_data[z][y][x - 3 * unit] + 6 * s_data[z][y][x - unit] +
                        3 * s_data[z][y][x + unit]) /
                       8;
              else
                pred = s_data[z][y][x - unit];
            }
            else {
              if (x + 3 * unit <= BLOCK32 and global_x + 3 * unit < data_size.x)
                pred = (3 * s_data[z][y][x - unit] + 6 * s_data[z][y][x + unit] -
                        s_data[z][y][x + 3 * unit]) /
                       8;
              else if (global_x + unit < data_size.x)
                pred = (s_data[z][y][x - unit] + s_data[z][y][x + unit]) / 2;
              else
                pred = s_data[z][y][x - unit];
            }
          }
        }
      }
      else {
        if constexpr (BLUE) {  //
          if (global_z + unit < data_size.z)

            pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x]) / 2;
          else
            pred = s_data[z - unit][y][x];
        }
        if constexpr (YELLOW) {  //
          if (global_y + unit < data_size.y)

            pred = (s_data[z][y - unit][x] + s_data[z][y + unit][x]) / 2;
          else
            pred = s_data[z][y - unit][x];
        }

        if constexpr (HOLLOW) {  //
          if (global_x + unit < data_size.x)
            pred = (s_data[z][y][x - unit] + s_data[z][y][x + unit]) / 2;
          else
            pred = s_data[z][y][x - unit];
        }
      }

      if constexpr (WORKFLOW == SPLINE3_COMPR) {
        auto err = s_data[z][y][x] - pred;
        decltype(err) code;
        // TODO unsafe, did not deal with the out-of-cap case
        {
          code = fabs(err) * eb_r + 1;
          code = err < 0 ? -code : code;
          code = int(code / 2) + radius;
        }
        s_eq[z][y][x] = code;  // TODO double check if unsigned type works

        s_data[z][y][x] = pred + (code - radius) * ebx2;
      }
      else {  // TODO == DECOMPRESSS and static_assert
        auto code = s_eq[z][y][x];
        s_data[z][y][x] = pred + (code - radius) * ebx2;
      }
    }
  };
  // -------------------------------------------------------------------------------- //

  if constexpr (COARSEN) {
    constexpr auto TOTAL = BLOCK_DIMX * BLOCK_DIMY * BLOCK_DIMZ;
    // if( BLOCK_DIMX *BLOCK_DIMY<= LINEAR_BLOCK_SIZE){
    for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
      auto itix = (_tix % BLOCK_DIMX);
      auto itiy = (_tix / BLOCK_DIMX) % BLOCK_DIMY;
      auto itiz = (_tix / BLOCK_DIMX) / BLOCK_DIMY;
      auto x = xmap(itix, unit);
      auto y = ymap(itiy, unit);
      auto z = zmap(itiz, unit);

      run(x, y, z);
    }
  }
  else {
    auto itix = (TIX % BLOCK_DIMX);
    auto itiy = (TIX / BLOCK_DIMX) % BLOCK_DIMY;
    auto itiz = (TIX / BLOCK_DIMX) / BLOCK_DIMY;
    auto x = xmap(itix, unit);
    auto y = ymap(itiy, unit);
    auto z = zmap(itiz, unit);

    run(x, y, z);
  }
  __syncthreads();
}

}  // namespace

/********************************************************************************/

template <
    typename T1, typename T2, typename FP, int LINEAR_BLOCK_SIZE, bool WORKFLOW,
    bool PROBE_PRED_ERROR>
__device__ void psz::spline3d_layout2_interpolate(
    volatile T1 s_data[9][9][33], volatile T2 s_eq[9][9][33], dim3 data_size, FP eb_r, FP ebx2,
    int radius

)
{
  double alpha = 1.25;
  double beta = 2.0;
  bool interpolators[3] = {true, true, true};
  // bool reverse[3]={true,true,true};//{false,true,true};
  bool reverse[3] = {false, false, false};
  auto xblue = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2); };
  auto yblue = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2); };
  auto zblue = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2 + 1); };

  auto xblue_reverse = [] __device__(int _tix, int unit) -> int { return unit * (_tix); };
  auto yblue_reverse = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy); };
  auto zblue_reverse = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2 + 1); };

  auto xyellow = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2); };
  auto yyellow = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2 + 1); };
  auto zyellow = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz); };

  auto xyellow_reverse = [] __device__(int _tix, int unit) -> int { return unit * (_tix); };
  auto yyellow_reverse = [] __device__(int _tiy, int unit) -> int {
    return unit * (_tiy * 2 + 1);
  };
  auto zyellow_reverse = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2); };

  auto xhollow = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2 + 1); };
  auto yhollow = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy); };
  auto zhollow = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz); };

  auto xhollow_reverse = [] __device__(int _tix, int unit) -> int {
    return unit * (_tix * 2 + 1);
  };
  auto yhollow_reverse = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2); };
  auto zhollow_reverse = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2); };

  constexpr auto COARSEN = true;
  constexpr auto NO_COARSEN = false;
  constexpr auto BORDER_INCLUSIVE = true;
  constexpr auto BORDER_EXCLUSIVE = false;

  FP cur_ebx2 = ebx2, cur_eb_r = eb_r;

  auto calc_eb = [&](auto unit) {
    cur_ebx2 = ebx2, cur_eb_r = eb_r;
    int temp = 1;
    while (temp < unit) {
      temp *= 2;
      cur_eb_r *= alpha;
      cur_ebx2 /= alpha;
    }
    if (cur_ebx2 < ebx2 / beta) {
      cur_ebx2 = ebx2 / beta;
      cur_eb_r = eb_r * beta;
    }
  };

  // iteration 1

  int unit = 4;
  calc_eb(unit);

  if (reverse[2]) {
    interpolate_stage<
        T1, T2, FP, decltype(xhollow_reverse), decltype(yhollow_reverse),
        decltype(zhollow_reverse),  //
        false, false, true, LINEAR_BLOCK_SIZE, 4, 2, NO_COARSEN, 2, BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_eq, data_size, xhollow_reverse, yhollow_reverse, zhollow_reverse, unit, cur_eb_r,
        cur_ebx2, radius, interpolators[2]);

    interpolate_stage<
        T1, T2, FP, decltype(xyellow_reverse), decltype(yyellow_reverse),
        decltype(zyellow_reverse),  //
        false, true, false, LINEAR_BLOCK_SIZE, 9, 1, NO_COARSEN, 2, BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_eq, data_size, xyellow_reverse, yyellow_reverse, zyellow_reverse, unit, cur_eb_r,
        cur_ebx2, radius, interpolators[2]);
    interpolate_stage<
        T1, T2, FP, decltype(xblue_reverse), decltype(yblue_reverse), decltype(zblue_reverse),  //
        true, false, false, LINEAR_BLOCK_SIZE, 9, 3, NO_COARSEN, 1, BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_eq, data_size, xblue_reverse, yblue_reverse, zblue_reverse, unit, cur_eb_r,
        cur_ebx2, radius, interpolators[2]);
  }
  else {
    interpolate_stage<
        T1, T2, FP, decltype(xblue), decltype(yblue), decltype(zblue),  //
        true, false, false, LINEAR_BLOCK_SIZE, 5, 2, NO_COARSEN, 1, BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_eq, data_size, xblue, yblue, zblue, unit, cur_eb_r, cur_ebx2, radius,
        interpolators[2]);

    interpolate_stage<
        T1, T2, FP, decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
        false, true, false, LINEAR_BLOCK_SIZE, 5, 1, NO_COARSEN, 3, BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_eq, data_size, xyellow, yyellow, zyellow, unit, cur_eb_r, cur_ebx2, radius,
        interpolators[2]);

    interpolate_stage<
        T1, T2, FP, decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
        false, false, true, LINEAR_BLOCK_SIZE, 4, 3, NO_COARSEN, 3, BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_eq, data_size, xhollow, yhollow, zhollow, unit, cur_eb_r, cur_ebx2, radius,
        interpolators[2]);
  }

  unit = 2;
  calc_eb(unit);

  // iteration 2, TODO switch y-z order
  if (reverse[1]) {
    interpolate_stage<
        T1, T2, FP, decltype(xhollow_reverse), decltype(yhollow_reverse),
        decltype(zhollow_reverse),  //
        false, false, true, LINEAR_BLOCK_SIZE, 8, 3, NO_COARSEN, 3, BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_eq, data_size, xhollow_reverse, yhollow_reverse, zhollow_reverse, unit, cur_eb_r,
        cur_ebx2, radius, interpolators[1]);
    interpolate_stage<
        T1, T2, FP, decltype(xyellow_reverse), decltype(yyellow_reverse),
        decltype(zyellow_reverse),  //
        false, true, false, LINEAR_BLOCK_SIZE, 17, 2, NO_COARSEN, 3, BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_eq, data_size, xyellow_reverse, yyellow_reverse, zyellow_reverse, unit, cur_eb_r,
        cur_ebx2, radius, interpolators[1]);
    interpolate_stage<
        T1, T2, FP, decltype(xblue_reverse), decltype(yblue_reverse), decltype(zblue_reverse),  //
        true, false, false, LINEAR_BLOCK_SIZE, 17, 5, NO_COARSEN, 2, BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_eq, data_size, xblue_reverse, yblue_reverse, zblue_reverse, unit, cur_eb_r,
        cur_ebx2, radius, interpolators[1]);
  }
  else {
    interpolate_stage<
        T1, T2, FP, decltype(xblue), decltype(yblue), decltype(zblue),  //
        true, false, false, LINEAR_BLOCK_SIZE, 9, 3, NO_COARSEN, 2, BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_eq, data_size, xblue, yblue, zblue, unit, cur_eb_r, cur_ebx2, radius,
        interpolators[1]);
    interpolate_stage<
        T1, T2, FP, decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
        false, true, false, LINEAR_BLOCK_SIZE, 9, 2, NO_COARSEN, 5, BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_eq, data_size, xyellow, yyellow, zyellow, unit, cur_eb_r, cur_ebx2, radius,
        interpolators[1]);
    interpolate_stage<
        T1, T2, FP, decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
        false, false, true, LINEAR_BLOCK_SIZE, 8, 5, NO_COARSEN, 5, BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_eq, data_size, xhollow, yhollow, zhollow, unit, cur_eb_r, cur_ebx2, radius,
        interpolators[1]);
  }
  unit = 1;
  calc_eb(unit);

  // iteration 3
  if (reverse[0]) {
    interpolate_stage<
        T1, T2, FP, decltype(xhollow_reverse), decltype(yhollow_reverse),
        decltype(zhollow_reverse),  //
        false, false, true, LINEAR_BLOCK_SIZE, 16, 5, COARSEN, 5, BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_eq, data_size, xhollow_reverse, yhollow_reverse, zhollow_reverse, unit, cur_eb_r,
        cur_ebx2, radius, interpolators[0]);
    interpolate_stage<
        T1, T2, FP, decltype(xyellow_reverse), decltype(yyellow_reverse),
        decltype(zyellow_reverse),  //
        false, true, false, LINEAR_BLOCK_SIZE, 33, 4, COARSEN, 5, BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_eq, data_size, xyellow_reverse, yyellow_reverse, zyellow_reverse, unit, cur_eb_r,
        cur_ebx2, radius, interpolators[0]);
    interpolate_stage<
        T1, T2, FP, decltype(xblue_reverse), decltype(yblue_reverse), decltype(zblue_reverse),  //
        true, false, false, LINEAR_BLOCK_SIZE, 33, 9, COARSEN, 4, BORDER_EXCLUSIVE, WORKFLOW>(
        s_data, s_eq, data_size, xblue_reverse, yblue_reverse, zblue_reverse, unit, cur_eb_r,
        cur_ebx2, radius, interpolators[0]);
  }
  else {
    interpolate_stage<
        T1, T2, FP, decltype(xblue), decltype(yblue), decltype(zblue),  //
        true, false, false, LINEAR_BLOCK_SIZE, 17, 5, COARSEN, 4, BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_eq, data_size, xblue, yblue, zblue, unit, cur_eb_r, cur_ebx2, radius,
        interpolators[0]);
    interpolate_stage<
        T1, T2, FP, decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
        false, true, false, LINEAR_BLOCK_SIZE, 17, 4, COARSEN, 9, BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_eq, data_size, xyellow, yyellow, zyellow, unit, cur_eb_r, cur_ebx2, radius,
        interpolators[0]);

    interpolate_stage<
        T1, T2, FP, decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
        false, false, true, LINEAR_BLOCK_SIZE, 16, 9, COARSEN, 9, BORDER_EXCLUSIVE, WORKFLOW>(
        s_data, s_eq, data_size, xhollow, yhollow, zhollow, unit, cur_eb_r, cur_ebx2, radius,
        interpolators[0]);
  }
  //  if(TIX==0 and TIY==0 and TIZ==0 and BIX==0 and BIY==0 and BIZ==0)
  // printf("lv1\n");

  /******************************************************************************
  test only: last step inclusive
  ******************************************************************************/
  // interpolate_stage<
  //     T1, T2, FP, decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
  //     false, false, true, LINEAR_BLOCK_SIZE, 33, 4, COARSEN, 9, BORDER_INCLUSIVE, WORKFLOW>(
  //     s_data, s_eq, xhollow, yhollow, zhollow, unit, eb_r, ebx2, radius);
  /******************************************************************************
   production
   ******************************************************************************/

  /******************************************************************************
   test only: print a block
   ******************************************************************************/
  // if (TIX == 0 and BIX == 7 and BIY == 47 and BIZ == 15) {
  // spline3d_print_block_from_GPU(s_eq); }
  //  if (TIX == 0 and BIX == 4 and BIY == 20 and BIZ == 20) {
  //  spline3d_print_block_from_GPU(s_data); }
}

/********************************************************************************
 * host API/kernel
 ********************************************************************************/

template <
    typename T, typename E, typename FP, int LINEAR_BLOCK_SIZE, typename CompactValIdx,
    typename CompactNum>
__global__ void psz::KCU_c_spline3d_infprecis_32x8x8data(
    T* data, dim3 data_size, dim3 data_leap, E* eq, dim3 eq_size, dim3 eq_leap, T* anchor,
    dim3 anchor_leap, CompactValIdx cvi, CompactNum cn, FP eb_r, FP ebx2, int radius)
{
  // compile time variables

  {
    __shared__ struct {
      T data[9][9][33];
      T eq[9][9][33];
    } shmem;

    dim3 begin{0, 0, 0};  // local frame; the offset lives in the (pre-offset) pointers
    auto sub_extent = data_size;

    c_reset_scratch_33x9x9data<T, T, LINEAR_BLOCK_SIZE>(shmem.data, shmem.eq, radius);

    global2shmem_33x9x9data<T, T, LINEAR_BLOCK_SIZE>(data, data_size, data_leap, begin, shmem.data);

    c_gather_anchor<T>(data, data_size, data_leap, anchor, anchor_leap, begin);

    psz::spline3d_layout2_interpolate<T, T, FP, LINEAR_BLOCK_SIZE, SPLINE3_COMPR, false>(
        shmem.data, shmem.eq, sub_extent, eb_r, ebx2, radius);

    shmem2global_32x8x8data_with_compaction<T, E, LINEAR_BLOCK_SIZE>(
        shmem.eq, eq, eq_size, eq_leap, begin, radius, cvi, cn);
  }
}

template <
    typename E, typename T, typename FP,
    int LINEAR_BLOCK_SIZE>
__global__ void psz::KCU_x_spline3d_infprecis_32x8x8data(
    E* eq,             // input 1
    dim3 eq_size,      //
    dim3 eq_leap,      //
    T* anchor,         // input 2
    dim3 anchor_size,  //
    dim3 anchor_leap,  //
    T* data,           // output
    dim3 data_size,    //
    dim3 data_leap,    //
    FP eb_r, FP ebx2, int radius)
{
  // compile time variables

  __shared__ struct {
    T data[9][9][33];
    T eq[9][9][33];
  } shmem;

  dim3 begin{0, 0, 0};  // local frame; the offset lives in the (pre-offset) pointers
  auto sub_extent = data_size;

  x_reset_scratch_33x9x9data<T, T, LINEAR_BLOCK_SIZE>(
      shmem.data, shmem.eq, anchor, anchor_size, anchor_leap, begin);

  global2shmem_fuse<T, E, LINEAR_BLOCK_SIZE>(eq, eq_size, eq_leap, data, begin, shmem.eq);

  psz::spline3d_layout2_interpolate<T, T, FP, LINEAR_BLOCK_SIZE, SPLINE3_DECOMPR, false>(
      shmem.data, shmem.eq, sub_extent, eb_r, ebx2, radius);

  shmem2global_32x8x8data<T, T, LINEAR_BLOCK_SIZE>(shmem.data, data, data_size, data_leap, begin);
}

#undef TIX
#undef TIY
#undef TIZ
#undef BIX
#undef BIY
#undef BIZ
#undef BDX
#undef BDY
#undef BDZ
#undef GDX
#undef GDY
#undef GDZ

#endif
