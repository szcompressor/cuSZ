/**
 * @file lorenzo23.inl
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2022-12-22
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "subroutine.inl"

namespace parsz {
namespace cuda {
namespace __kernel {

////////////////////////////////////////////////////////////////////////////////
// 1D

namespace v0 {

template <typename T, typename EQ, typename FP, int BLOCK, int SEQ>
__global__ void c_lorenzo_1d1l(T* data, dim3 len3, dim3 stride3, int radius, FP ebx2_r, EQ* quant, T* outlier);

template <typename T, typename EQ, typename FP, int BLOCK, int SEQ>
__global__ void x_lorenzo_1d1l(EQ* quant, T* outlier, dim3 len3, dim3 stride3, int radius, FP ebx2, T* xdata);

}  // namespace v0

namespace v1 {

template <typename T, typename EQ, typename FP, int BLOCK, int SEQ>
__global__ void c_lorenzo_1d1l(T* data, dim3 len3, dim3 stride3, int radius, FP ebx2_r, EQ* quant, T* outlier);

}

////////////////////////////////////////////////////////////////////////////////
// 2D

namespace v0 {

template <typename T, typename EQ, typename FP>
__global__ void c_lorenzo_2d1l(T* data, dim3 len3, dim3 stride3, int radius, FP ebx2_r, EQ* quant, T* outlier);

template <typename T, typename EQ, typename FP>
__global__ void x_lorenzo_2d1l(EQ* quant, T* outlier, dim3 len3, dim3 stride3, int radius, FP ebx2, T* xdata);

}  // namespace v0

}  // namespace __kernel
}  // namespace cuda
}  // namespace parsz

////////////////////////////////////////////////////////////////////////////////
// 1D definition

template <typename T, typename EQ, typename FP, int BLOCK, int SEQ>
__global__ void parsz::cuda::__kernel::v0::c_lorenzo_1d1l(
    T*   data,
    dim3 len3,
    dim3 stride3,
    int  radius,
    FP   ebx2_r,
    EQ*  quant,
    T*   outlier)
{
    namespace subr_v0 = parsz::cuda::__device::v0;

    constexpr auto NTHREAD = BLOCK / SEQ;

    __shared__ struct {
        union {
            T data[BLOCK];
            T outlier[BLOCK];
        };
        EQ quant[BLOCK];
    } s;

    T prev{0};
    T thp_data[SEQ];

    auto id_base = blockIdx.x * BLOCK;

    subr_v0::load_prequant_1d<T, FP, NTHREAD, SEQ>(data, len3.x, id_base, s.data, thp_data, prev, ebx2_r);
    subr_v0::predict_quantize_1d<T, EQ, SEQ, true>(thp_data, s.quant, s.outlier, radius, prev);
    subr_v0::predict_quantize_1d<T, EQ, SEQ, false>(thp_data, s.quant, s.outlier, radius);
    subr_v0::write_1d<EQ, T, NTHREAD, SEQ, false>(s.quant, s.outlier, len3.x, id_base, quant, outlier);
}

template <typename T, typename EQ, typename FP, int BLOCK, int SEQ>
__global__ void parsz::cuda::__kernel::v1::c_lorenzo_1d1l(  //
    T*   data,
    dim3 len3,
    dim3 stride3,
    int  radius,
    FP   ebx2_r,
    EQ*  quant,
    T*   outlier)
{
    namespace subr_v0 = parsz::cuda::__device::v0;
    namespace subr_v1 = parsz::cuda::__device::v1_pncodec;

    constexpr auto NTHREAD = BLOCK / SEQ;

    __shared__ struct {
        union {
            T data[BLOCK];
            T outlier[BLOCK];
        };
        EQ quant[BLOCK];
    } s;

    T prev{0};
    T thp_data[SEQ];

    auto id_base = blockIdx.x * BLOCK;

    subr_v0::load_prequant_1d<T, FP, NTHREAD, SEQ>(data, len3.x, id_base, s.data, thp_data, prev, ebx2_r);
    subr_v1::predict_quantize_1d<T, EQ, SEQ, true>(thp_data, s.quant, s.outlier, radius, prev);
    subr_v1::predict_quantize_1d<T, EQ, SEQ, false>(thp_data, s.quant, s.outlier, radius);
    subr_v0::write_1d<EQ, T, NTHREAD, SEQ, false>(s.quant, s.outlier, len3.x, id_base, quant, outlier);
}

template <typename T, typename EQ, typename FP, int BLOCK, int SEQ>
__global__ void parsz::cuda::__kernel::v0::x_lorenzo_1d1l(  //
    EQ*  quant,
    T*   outlier,
    dim3 len3,
    dim3 stride3,
    int  radius,
    FP   ebx2,
    T*   xdata)
{
    namespace subr_v0 = parsz::cuda::__device::v0;
    namespace wave32  = parsz::cuda::__device::wave32;

    constexpr auto NTHREAD = BLOCK / SEQ;  // equiv. to blockDim.x

    __shared__ struct {
        union {
            T outlier[BLOCK];
            T xdata[BLOCK];
        };
        // even if it's wave64, "/32" works
        T exchange_in[NTHREAD / 32];
        T exchange_out[NTHREAD / 32];
    } s;

    T thp_data[SEQ];

    auto id_base = blockIdx.x * BLOCK;

    subr_v0::load_fuse_1d<T, EQ, NTHREAD, SEQ>(quant, outlier, len3.x, id_base, radius, s.xdata, thp_data);
    subr_v0::block_scan_1d<T, SEQ, NTHREAD>(thp_data, ebx2, s.exchange_in, s.exchange_out, s.xdata);
    subr_v0::write_1d<T, T, NTHREAD, SEQ, true>(s.xdata, nullptr, len3.x, id_base, xdata, nullptr);
}

////////////////////////////////////////////////////////////////////////////////
// 2D definition

template <typename T, typename EQ, typename FP>
__global__ void parsz::cuda::__kernel::v0::c_lorenzo_2d1l(
    T*   data,
    dim3 len3,
    dim3 stride3,
    int  radius,
    FP   ebx2_r,
    EQ*  quant,
    T*   outlier)
{
    namespace subr_v0 = parsz::cuda::__device::v0;

    constexpr auto BLOCK = 16;
    constexpr auto YSEQ  = 8;

    T center[YSEQ + 1] = {0};  // NW  N       first element <- 0
                               //  W  center

    auto gix      = blockIdx.x * BLOCK + threadIdx.x;         // BDX == BLOCK == 16
    auto giy_base = blockIdx.y * BLOCK + threadIdx.y * YSEQ;  // BDY * YSEQ = BLOCK == 16

    subr_v0::load_prequant_2d<T, FP, YSEQ>(data, len3.x, gix, len3.y, giy_base, stride3.y, ebx2_r, center);
    subr_v0::predict_2d<T, EQ, YSEQ>(center);
    subr_v0::quantize_write_2d<T, EQ, YSEQ>(center, len3.x, gix, len3.y, giy_base, stride3.y, radius, quant, outlier);
}

// 16x16 data block maps to 16x2 (one warp) thread block
template <typename T, typename EQ, typename FP>
__global__ void parsz::cuda::__kernel::v0::x_lorenzo_2d1l(  //
    EQ*  quant,
    T*   outlier,
    dim3 len3,
    dim3 stride3,
    int  radius,
    FP   ebx2,
    T*   xdata)
{
    namespace subr_v0 = parsz::cuda::__device::v0;

    constexpr auto BLOCK = 16;
    constexpr auto YSEQ  = BLOCK / 2;  // sequentiality in y direction
    static_assert(BLOCK == 16, "In one case, we need BLOCK for 2D == 16");

    __shared__ T intermediate[BLOCK];  // TODO use warp shuffle to eliminate this
    T            thread_private[YSEQ];

    auto gix      = blockIdx.x * BLOCK + threadIdx.x;
    auto giy_base = blockIdx.y * BLOCK + threadIdx.y * YSEQ;  // BDY * YSEQ = BLOCK == 16

    auto get_gid = [&](auto i) { return (giy_base + i) * stride3.y + gix; };

    subr_v0::load_fuse_2d<T, EQ, YSEQ>(
        quant, outlier, len3.x, gix, len3.y, giy_base, stride3.y, radius, thread_private);
    subr_v0::block_scan_2d<T, EQ, FP, YSEQ>(thread_private, intermediate, ebx2);
    subr_v0::decomp_write_2d<T, YSEQ>(thread_private, len3.x, gix, len3.y, giy_base, stride3.y, xdata);
}
