/**
 * @file utils_spline.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-05-30
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_KERNEL_UTILS_SPLINE_CUH
#define CUSZ_KERNEL_UTILS_SPLINE_CUH

#if __cplusplus >= 201703L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

#ifndef __CUDACC__
#define __global__
#define __device__
#define __host__
#define __shared__
#define __forceinline__
#endif

#define tix threadIdx.x
#define tiy threadIdx.y
#define tiz threadIdx.z
#define bix blockIdx.x
#define biy blockIdx.y
#define biz blockIdx.z
#define bdx blockDim.x
#define bdy blockDim.y
#define bdz blockDim.z

using DIM     = unsigned int;
using STRIDE  = unsigned int;
using DIM3    = dim3;
using STRIDE3 = dim3;

constexpr int Block8  = 8;
constexpr int Block32 = 32;

namespace kernel {
namespace internal {

template <typename Data, typename FP>
__forceinline__ __device__ Data infprecis_quantcode(Data err, FP eb_r, int radius)
{
    Data code = fabs(err) * eb_r + 1;
    code      = err < 0 ? -code : code;
    code      = int(code / 2) + radius;
    return code;
}

}  // namespace internal
}  // namespace kernel

////////////////////////////////////////////////////////////////////////////////
// misc
////////////////////////////////////////////////////////////////////////////////

namespace kernel {
namespace internal {

// control block_id3 in function call
template <typename T, bool PrintFP = false, int Xend = 33, int Yend = 9, int Zend = 9>
__device__ void spline3d_print_block_from_GPU(
    T volatile a[9][9][33],
    int  radius        = 512,
    bool compress      = true,
    bool print_errctrl = true)
{
    for (auto z = 0; z < Zend; z++) {
        printf("\nprint from GPU, z=%d\n", z);
        printf("    ");
        for (auto i = 0; i < 33; i++) printf("%3d", i);
        printf("\n");

        for (auto y = 0; y < Yend; y++) {
            printf("y=%d ", y);

            for (auto x = 0; x < Xend; x++) {  //
                if CONSTEXPR (PrintFP) {       //
                    printf("%.2e\t", (float)a[z][y][x]);
                }
                else {
                    T c;

                    if (print_errctrl)
                        c = a[z][y][x] - radius;
                    else
                        c = a[z][y][x];

                    if CONSTEXPR (compress) {
                        if (c == 0)
                            printf("%3c", '.');
                        else {
                            if (abs(c) >= 10) { printf("%3c", '*'); }
                            else {
                                if (print_errctrl)
                                    printf("%3d", c);
                                else
                                    printf("%4.2f", c);
                            }
                        }
                    }
                    else {
                        if (print_errctrl)
                            printf("%3d", c);
                        else
                            printf("%4.2f", c);
                    }
                }
            }
            printf("\n");
        }
    }
    printf("\nGPU print end\n\n");
}

}  // namespace internal
}  // namespace kernel

namespace kernel {
namespace internal {
namespace memops {

template <typename T1, typename T2, int NumThreads = 256>
__device__ void
c_spline3d_reset_scratch_33x9x9data(volatile T1 shm_data[9][9][33], volatile T2 shm_errctrl[9][9][33], int radius = 512)
{
    constexpr auto NumIters = 33 * 9 * 9 / NumThreads + 1;  // 11 iterations
    // alternatively, reinterprete cast volatile T?[][][] to 1D
    for (auto i = 0; i < NumIters; i++) {
        auto _tix = i * NumThreads + tix;

        if (_tix < 33 * 9 * 9) {
            auto x = (_tix % 33);
            auto y = (_tix / 33) % 9;
            auto z = (_tix / 33) / 9;

            shm_data[z][y][x] = 0;

            /*****************************************************************************
             okay to use
             ******************************************************************************/
            if (x % 8 == 0 and y % 8 == 0 and z % 8 == 0) shm_errctrl[z][y][x] = radius;
            /*****************************************************************************
             alternatively
             ******************************************************************************/
            // shm_errctrl[z][y][x] = radius;
        }
    }
    __syncthreads();
}

template <typename T1, typename T2, int NumThreads = 256>
__device__ void x_spline3d_reset_scratch_33x9x9data(
    volatile T1 shm_xdata[9][9][33],
    volatile T2 shm_errctrl[9][9][33],
    T1*         anchor,
    DIM3        anchor_dim3,  // TODO unused
    STRIDE3     anchor_stride3,
    int         radius = 512)
{
    constexpr auto NumIters = 33 * 9 * 9 / NumThreads + 1;  // 11 iterations
    // alternatively, reinterprete cast volatile T?[][][] to 1D
    for (auto i = 0; i < NumIters; i++) {
        auto _tix = i * NumThreads + tix;

        if (_tix < 33 * 9 * 9) {
            auto x = (_tix % 33);
            auto y = (_tix / 33) % 9;
            auto z = (_tix / 33) / 9;

            shm_errctrl[z][y][x] = 0;  // TODO explicitly handle zero-padding

            /*****************************************************************************
             okay to use
             ******************************************************************************/
            if (x % 8 == 0 and y % 8 == 0 and z % 8 == 0) {
                shm_xdata[z][y][x] = 0;

                auto aid = ((x / 8) + bix * 4) +                 //
                           ((y / 8) + biy) * anchor_stride3.y +  //
                           ((z / 8) + biz) * anchor_stride3.z;   //
                shm_xdata[z][y][x] = anchor[aid];
            }
            /*****************************************************************************
             alternatively
             ******************************************************************************/
            // shm_errctrl[z][y][x] = radius;
        }
    }
    __syncthreads();
}

template <typename Input, int NumThreads = 256>
__device__ void
spline3d_global2shmem_33x9x9data(Input* data, volatile Input shm_data[9][9][33], DIM3 dim3d, STRIDE3 stride3d)
{
    constexpr auto Total    = 33 * 9 * 9;
    constexpr auto NumIters = 33 * 9 * 9 / NumThreads + 1;  // 11 iterations

    for (auto i = 0; i < NumIters; i++) {
        auto _tix = i * NumThreads + tix;

        if (_tix < Total) {
            auto x   = (_tix % 33);
            auto y   = (_tix / 33) % 9;
            auto z   = (_tix / 33) / 9;
            auto gx  = (x + bix * Block32);
            auto gy  = (y + biy * Block8);
            auto gz  = (z + biz * Block8);
            auto gid = gx + gy * stride3d.y + gz * stride3d.z;

            if (gx < dim3d.x and gy < dim3d.y and gz < dim3d.z) { shm_data[z][y][x] = data[gid]; }
        }
    }
    __syncthreads();
}

template <typename Output, int NumThreads = 256>
__device__ void
spline3d_shmem2global_32x8x8data(volatile Output shm_data[9][9][33], Output* data, DIM3 dim3d, STRIDE3 stride3d)
{
    constexpr auto Total    = 32 * 8 * 8;
    constexpr auto NumIters = Total / NumThreads + 1;  // 11 iterations

    for (auto i = 0; i < NumIters; i++) {
        auto _tix = i * NumThreads + tix;

        if (_tix < Total) {
            auto x   = (_tix % 32);
            auto y   = (_tix / 32) % 8;
            auto z   = (_tix / 32) / 8;
            auto gx  = (x + bix * Block32);
            auto gy  = (y + biy * Block8);
            auto gz  = (z + biz * Block8);
            auto gid = (x + bix * Block32) + (y + biy * Block8) * stride3d.y + (z + biz * Block8) * stride3d.z;

            if (gx < dim3d.x and gy < dim3d.y and gz < dim3d.z) { data[gid] = shm_data[z][y][x]; }
        }
    }
    __syncthreads();
}

}  // namespace memops
}  // namespace internal
}  // namespace kernel

#endif