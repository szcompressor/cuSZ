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

namespace subr = parsz::cuda::__device;

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

namespace compaction {

template <
    typename T,
    typename EQ,
    typename FP,
    int BLOCK,
    int SEQ,
    typename OutlierDesc = OutlierDescriptionGlobalMemory<T>>
__global__ void c_lorenzo_1d1l(T* data, dim3 len3, dim3 stride3, int radius, FP ebx2_r, EQ* quant, OutlierDesc outlier);

}

namespace delta_only {

template <typename T, typename EQ, typename FP, int BLOCK, int SEQ>
__global__ void c_lorenzo_1d1l(T* data, dim3 len3, dim3 stride3, FP ebx2_r, EQ* delta);

template <typename T, typename EQ, typename FP, int BLOCK, int SEQ>
__global__ void x_lorenzo_1d1l(EQ* delta, dim3 len3, dim3 stride3, FP ebx2, T* xdata);

}  // namespace delta_only

}  // namespace v0

namespace v1_pn {

namespace compaction {

template <
    typename T,
    typename EQ,
    typename FP,
    int BLOCK,
    int SEQ,
    typename OutlierDesc = OutlierDescriptionGlobalMemory<T>>
__global__ void c_lorenzo_1d1l(T* data, dim3 len3, dim3 stride3, int radius, FP ebx2_r, EQ* quant, OutlierDesc outlier);

}  // namespace compaction

template <typename T, typename EQ, typename FP, int BLOCK, int SEQ>
__global__ void x_lorenzo_1d1l(EQ* quant, T* outlier, dim3 len3, dim3 stride3, int radius, FP ebx2, T* xdata);

}  // namespace v1_pn

namespace delta_only {

template <typename T, typename EQ, typename FP, int BLOCK, int SEQ>
__global__ void c_lorenzo_1d1l(T* data, dim3 len3, dim3 stride3, FP ebx2_r, EQ* delta);

template <typename T, typename EQ, typename FP, int BLOCK, int SEQ>
__global__ void x_lorenzo_1d1l(EQ* delta, dim3 len3, dim3 stride3, FP ebx2, T* xdata);

}  // namespace delta_only

////////////////////////////////////////////////////////////////////////////////
// 2D

namespace v0 {

template <typename T, typename EQ, typename FP>
__global__ void c_lorenzo_2d1l(T* data, dim3 len3, dim3 stride3, int radius, FP ebx2_r, EQ* quant, T* outlier);

template <typename T, typename EQ, typename FP>
__global__ void x_lorenzo_2d1l(EQ* quant, T* outlier, dim3 len3, dim3 stride3, int radius, FP ebx2, T* xdata);

namespace delta_only {

template <typename T, typename EQ, typename FP>
__global__ void c_lorenzo_2d1l(T* data, dim3 len3, dim3 stride3, FP ebx2_r, EQ* delta);

template <typename T, typename EQ, typename FP>
__global__ void x_lorenzo_2d1l(EQ* delta, dim3 len3, dim3 stride3, FP ebx2, T* xdata);

}  // namespace delta_only

namespace compaction {

template <typename T, typename EQ, typename FP, typename OutlierDesc = OutlierDescriptionGlobalMemory<T>>
__global__ void c_lorenzo_2d1l(T* data, dim3 len3, dim3 stride3, int radius, FP ebx2_r, EQ* quant, OutlierDesc outlier);

}  // namespace compaction

}  // namespace v0

namespace v1_pn {

namespace compaction {

template <typename T, typename EQ, typename FP, typename OutlierDesc = OutlierDescriptionGlobalMemory<T>>
__global__ void c_lorenzo_2d1l(T* data, dim3 len3, dim3 stride3, int radius, FP ebx2_r, EQ* quant, OutlierDesc outlier);

}  // namespace compaction

template <typename T, typename EQ, typename FP>
__global__ void x_lorenzo_2d1l(EQ* quant, T* outlier, dim3 len3, dim3 stride3, int radius, FP ebx2, T* xdata);

namespace delta_only {

template <typename T, typename EQ, typename FP>
__global__ void c_lorenzo_2d1l(T* data, dim3 len3, dim3 stride3, FP ebx2_r, EQ* delta);

template <typename T, typename EQ, typename FP>
__global__ void x_lorenzo_2d1l(EQ* delta, dim3 len3, dim3 stride3, FP ebx2, T* xdata);

}  // namespace delta_only

}  // namespace v1_pn

////////////////////////////////////////////////////////////////////////////////
// 3D

namespace v0 {

// TODO -> `legacy`
namespace obsolete {
template <typename T, typename EQ, typename FP>
__global__ void c_lorenzo_3d1l(T* data, dim3 len3, dim3 stride3, int radius, FP ebx2_r, EQ* quant, T* outlier);

}

template <typename T, typename EQ, typename FP>
__global__ void c_lorenzo_3d1l(T* data, dim3 len3, dim3 stride3, int radius, FP ebx2_r, EQ* quant, T* outlier);

template <typename T, typename EQ, typename FP>
__global__ void x_lorenzo_3d1l(EQ* quant, T* outlier, dim3 len3, dim3 stride3, int radius, FP ebx2, T* xdata);

namespace delta_only {

template <typename T, typename EQ, typename FP>
__global__ void c_lorenzo_3d1l(T* data, dim3 len3, dim3 stride3, FP ebx2_r, EQ* quant);

template <typename T, typename EQ, typename FP>
__global__ void x_lorenzo_3d1l(EQ* quant, dim3 len3, dim3 stride3, FP ebx2, T* xdata);

}  // namespace delta_only

namespace compaction {

template <typename T, typename EQ, typename FP, typename OutlierDesc = OutlierDescriptionGlobalMemory<T>>
__global__ void c_lorenzo_3d1l(T* data, dim3 len3, dim3 stride3, int radius, FP ebx2_r, EQ* quant, OutlierDesc outlier);

}

}  // namespace v0

namespace v1_pn {

namespace compaction {

template <typename T, typename EQ, typename FP, typename OutlierDesc = OutlierDescriptionGlobalMemory<T>>
__global__ void c_lorenzo_3d1l(T* data, dim3 len3, dim3 stride3, int radius, FP ebx2_r, EQ* quant, OutlierDesc outlier);

}

template <typename T, typename EQ, typename FP>
__global__ void x_lorenzo_3d1l(EQ* quant, T* outlier, dim3 len3, dim3 stride3, int radius, FP ebx2, T* xdata);

namespace delta_only {

template <typename T, typename EQ, typename FP>
__global__ void c_lorenzo_3d1l(T* data, dim3 len3, dim3 stride3, FP ebx2_r, EQ* quant);

template <typename T, typename EQ, typename FP>
__global__ void x_lorenzo_3d1l(EQ* quant, dim3 len3, dim3 stride3, FP ebx2, T* xdata);

}  // namespace delta_only

}  // namespace v1_pn

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
__global__ void
parsz::cuda::__kernel::v0::delta_only::c_lorenzo_1d1l(T* data, dim3 len3, dim3 stride3, FP ebx2_r, EQ* quant)
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
    subr_v0::predict_quantize__no_outlier_1d<T, EQ, SEQ, true>(thp_data, s.quant, prev);
    subr_v0::predict_quantize__no_outlier_1d<T, EQ, SEQ, false>(thp_data, s.quant);
    subr_v0::write_1d<EQ, T, NTHREAD, SEQ, false>(s.quant, nullptr, len3.x, id_base, quant, nullptr);
}

template <typename T, typename EQ, typename FP, int BLOCK, int SEQ, typename OutlierDesc>
__global__ void parsz::cuda::__kernel::v0::compaction::c_lorenzo_1d1l(
    T*          data,
    dim3        len3,
    dim3        stride3,
    int         radius,
    FP          ebx2_r,
    EQ*         quant,
    OutlierDesc outlier_desc)
{
    namespace subr_v0  = parsz::cuda::__device::v0;
    namespace subr_v0c = parsz::cuda::__device::v0::compaction;

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
    subr_v0c::predict_quantize_1d<T, EQ, SEQ, true>(thp_data, s.quant, len3.x, radius, id_base, outlier_desc, prev);
    subr_v0c::predict_quantize_1d<T, EQ, SEQ, false>(thp_data, s.quant, len3.x, radius, id_base, outlier_desc);
    subr_v0::write_1d<EQ, T, NTHREAD, SEQ, true>(s.quant, nullptr, len3.x, id_base, quant, nullptr);
}

template <typename T, typename EQ, typename FP, int BLOCK, int SEQ, typename Compaction>
__global__ void parsz::cuda::__kernel::v1_pn::compaction::c_lorenzo_1d1l(  //
    T*         data,
    dim3       len3,
    dim3       stride3,
    int        radius,
    FP         ebx2_r,
    EQ*        quant,
    Compaction outlier)
{
    namespace subr_v0  = parsz::cuda::__device::v0;
    namespace subr_v1c = parsz::cuda::__device::v1_pn::compaction;

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
    subr_v1c::predict_quantize_1d<T, EQ, SEQ, true>(thp_data, s.quant, s.outlier, radius, prev);
    subr_v1c::predict_quantize_1d<T, EQ, SEQ, false>(thp_data, s.quant, s.outlier, radius);
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

template <typename T, typename EQ, typename FP, int BLOCK, int SEQ>
__global__ void parsz::cuda::__kernel::v0::delta_only::x_lorenzo_1d1l(  //
    EQ*  quant,
    dim3 len3,
    dim3 stride3,
    FP   ebx2,
    T*   xdata)
{
    namespace subr_v0 = parsz::cuda::__device::v0;

    constexpr auto NTHREAD = BLOCK / SEQ;  // equiv. to blockDim.x

    __shared__ struct {
        T xdata[BLOCK];
        // even if it's wave64, "/32" works
        T exchange_in[NTHREAD / 32];
        T exchange_out[NTHREAD / 32];
    } s;

    T thp_data[SEQ];

    auto id_base = blockIdx.x * BLOCK;

    subr_v0::delta_only::load_1d<T, EQ, NTHREAD, SEQ>(quant, len3.x, id_base, s.xdata, thp_data);
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

template <typename T, typename EQ, typename FP>
__global__ void
parsz::cuda::__kernel::v0::delta_only::c_lorenzo_2d1l(T* data, dim3 len3, dim3 stride3, FP ebx2_r, EQ* quant)
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
    subr_v0::delta_only::quantize_write_2d<T, EQ, YSEQ>(center, len3.x, gix, len3.y, giy_base, stride3.y, quant);
}

template <typename T, typename EQ, typename FP>
__global__ void
parsz::cuda::__kernel::v1_pn::delta_only::c_lorenzo_2d1l(T* data, dim3 len3, dim3 stride3, FP ebx2_r, EQ* quant)
{
    namespace subr_v0  = parsz::cuda::__device::v0;
    namespace subr_v1d = parsz::cuda::__device::v1_pn::delta_only;

    constexpr auto BLOCK = 16;
    constexpr auto YSEQ  = 8;

    T center[YSEQ + 1] = {0};  // NW  N       first element <- 0
                               //  W  center

    auto gix      = blockIdx.x * BLOCK + threadIdx.x;         // BDX == BLOCK == 16
    auto giy_base = blockIdx.y * BLOCK + threadIdx.y * YSEQ;  // BDY * YSEQ = BLOCK == 16

    subr_v0::load_prequant_2d<T, FP, YSEQ>(data, len3.x, gix, len3.y, giy_base, stride3.y, ebx2_r, center);
    subr_v0::predict_2d<T, EQ, YSEQ>(center);
    subr_v1d::quantize_write_2d<T, EQ, YSEQ>(center, len3.x, gix, len3.y, giy_base, stride3.y, quant);
}

template <typename T, typename EQ, typename FP, typename OutlierDesc>
__global__ void parsz::cuda::__kernel::v0::compaction::c_lorenzo_2d1l(
    T*          data,
    dim3        len3,
    dim3        stride3,
    int         radius,
    FP          ebx2_r,
    EQ*         quant,
    OutlierDesc outlier)
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
    subr_v0::compaction::quantize_write_2d<T, EQ, YSEQ>(
        center, len3.x, gix, len3.y, giy_base, stride3.y, radius, quant, outlier);
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

// 16x16 data block maps to 16x2 (one warp) thread block
template <typename T, typename EQ, typename FP>
__global__ void parsz::cuda::__kernel::v1_pn::x_lorenzo_2d1l(  //
    EQ*  quant,
    T*   outlier,
    dim3 len3,
    dim3 stride3,
    FP   ebx2,
    T*   xdata)
{
    namespace subr_v0    = parsz::cuda::__device::v0;
    namespace subr_v1_pn = parsz::cuda::__device::v1_pn;

    constexpr auto BLOCK = 16;
    constexpr auto YSEQ  = BLOCK / 2;  // sequentiality in y direction
    static_assert(BLOCK == 16, "In one case, we need BLOCK for 2D == 16");

    __shared__ T intermediate[BLOCK];  // TODO use warp shuffle to eliminate this
    T            thread_private[YSEQ];

    auto gix      = blockIdx.x * BLOCK + threadIdx.x;
    auto giy_base = blockIdx.y * BLOCK + threadIdx.y * YSEQ;  // BDY * YSEQ = BLOCK == 16

    auto get_gid = [&](auto i) { return (giy_base + i) * stride3.y + gix; };

    subr_v1_pn::load_fuse_2d<T, EQ, YSEQ>(quant, outlier, len3.x, gix, len3.y, giy_base, stride3.y, thread_private);
    subr_v0::block_scan_2d<T, EQ, FP, YSEQ>(thread_private, intermediate, ebx2);
    subr_v0::decomp_write_2d<T, YSEQ>(thread_private, len3.x, gix, len3.y, giy_base, stride3.y, xdata);
}

// 16x16 data block maps to 16x2 (one warp) thread block
template <typename T, typename EQ, typename FP>
__global__ void parsz::cuda::__kernel::v0::delta_only::x_lorenzo_2d1l(  //
    EQ*  quant,
    dim3 len3,
    dim3 stride3,
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

    subr_v0::delta_only::load_2d<T, EQ, YSEQ>(quant, len3.x, gix, len3.y, giy_base, stride3.y, thread_private);
    subr_v0::block_scan_2d<T, EQ, FP, YSEQ>(thread_private, intermediate, ebx2);
    subr_v0::decomp_write_2d<T, YSEQ>(thread_private, len3.x, gix, len3.y, giy_base, stride3.y, xdata);
}

// 16x16 data block maps to 16x2 (one warp) thread block
template <typename T, typename EQ, typename FP>
__global__ void parsz::cuda::__kernel::v1_pn::delta_only::x_lorenzo_2d1l(  //
    EQ*  quant,
    dim3 len3,
    dim3 stride3,
    FP   ebx2,
    T*   xdata)
{
    namespace subr_v0    = parsz::cuda::__device::v0;
    namespace subr_v1_pn = parsz::cuda::__device::v1_pn;

    constexpr auto BLOCK = 16;
    constexpr auto YSEQ  = BLOCK / 2;  // sequentiality in y direction
    static_assert(BLOCK == 16, "In one case, we need BLOCK for 2D == 16");

    __shared__ T intermediate[BLOCK];  // TODO use warp shuffle to eliminate this
    T            thread_private[YSEQ];

    auto gix      = blockIdx.x * BLOCK + threadIdx.x;
    auto giy_base = blockIdx.y * BLOCK + threadIdx.y * YSEQ;  // BDY * YSEQ = BLOCK == 16

    auto get_gid = [&](auto i) { return (giy_base + i) * stride3.y + gix; };

    subr_v1_pn::delta_only::load_2d<T, EQ, YSEQ>(quant, len3.x, gix, len3.y, giy_base, stride3.y, thread_private);
    subr_v0::block_scan_2d<T, EQ, FP, YSEQ>(thread_private, intermediate, ebx2);
    subr_v0::decomp_write_2d<T, YSEQ>(thread_private, len3.x, gix, len3.y, giy_base, stride3.y, xdata);
}

template <typename T, typename EQ, typename FP>
__global__ void parsz::cuda::__kernel::v0::obsolete::c_lorenzo_3d1l(
    T*   data,
    dim3 len3,
    dim3 stride3,
    int  radius,
    FP   ebx2_r,
    EQ*  quant,
    T*   outlier)
{
    constexpr auto BLOCK = 8;
    __shared__ T   s[8][8][32];

    auto z = threadIdx.z;

    auto gix      = blockIdx.x * (BLOCK * 4) + threadIdx.x;
    auto giy_base = blockIdx.y * BLOCK;
    auto giz      = blockIdx.z * BLOCK + z;
    auto base_id  = gix + giy_base * stride3.y + giz * stride3.z;

    auto giy = [&](auto y) { return giy_base + y; };
    auto gid = [&](auto y) { return base_id + y * stride3.y; };

    auto load_prequant_3d = [&]() {
        if (gix < len3.x and giz < len3.z) {
            for (auto y = 0; y < BLOCK; y++)
                if (giy(y) < len3.y) s[z][y][threadIdx.x] = round(data[gid(y)] * ebx2_r);  // prequant (fp presence)
        }
        __syncthreads();
    };

    auto quantize_write = [&](T delta, auto x, auto y, auto z, auto gid) {
        bool quantizable = fabs(delta) < radius;
        T    candidate   = delta + radius;
        if (x < len3.x and y < len3.y and z < len3.z) {
            quant[gid]   = quantizable * static_cast<EQ>(candidate);
            outlier[gid] = (not quantizable) * candidate;
        }
    };

    auto x = threadIdx.x % 8;

    auto predict_3d = [&](auto y) {
        T delta = s[z][y][threadIdx.x] -                                               //
                  ((z > 0 and y > 0 and x > 0 ? s[z - 1][y - 1][threadIdx.x - 1] : 0)  // dist=3
                   - (y > 0 and x > 0 ? s[z][y - 1][threadIdx.x - 1] : 0)              // dist=2
                   - (z > 0 and x > 0 ? s[z - 1][y][threadIdx.x - 1] : 0)              //
                   - (z > 0 and y > 0 ? s[z - 1][y - 1][threadIdx.x] : 0)              //
                   + (x > 0 ? s[z][y][threadIdx.x - 1] : 0)                            // dist=1
                   + (y > 0 ? s[z][y - 1][threadIdx.x] : 0)                            //
                   + (z > 0 ? s[z - 1][y][threadIdx.x] : 0));                          //
        return delta;
    };

    ////////////////////////////////////////////////////////////////////////////

    load_prequant_3d();
    for (auto y = 0; y < BLOCK; y++) {
        auto delta = predict_3d(y);
        quantize_write(delta, gix, giy(y), giz, gid(y));
    }
}

template <typename T, typename EQ, typename FP>
__global__ void parsz::cuda::__kernel::v0::c_lorenzo_3d1l(
    T*   data,
    dim3 len3,
    dim3 stride3,
    int  radius,
    FP   ebx2_r,
    EQ*  quant,
    T*   outlier)
{
    constexpr auto BLOCK = 8;
    __shared__ T   s[9][33];
    T              delta[BLOCK + 1] = {0};  // first el = 0

    const auto gix      = blockIdx.x * (BLOCK * 4) + threadIdx.x;
    const auto giy      = blockIdx.y * BLOCK + threadIdx.y;
    const auto giz_base = blockIdx.z * BLOCK;
    const auto base_id  = gix + giy * stride3.y + giz_base * stride3.z;

    auto giz = [&](auto z) { return giz_base + z; };
    auto gid = [&](auto z) { return base_id + z * stride3.z; };

    auto load_prequant_3d = [&]() {
        if (gix < len3.x and giy < len3.y) {
            for (auto z = 0; z < BLOCK; z++)
                if (giz(z) < len3.z) delta[z + 1] = round(data[gid(z)] * ebx2_r);  // prequant (fp presence)
        }
        __syncthreads();
    };

    auto quantize_write = [&](T delta, auto x, auto y, auto z, auto gid) {
        bool quantizable = fabs(delta) < radius;
        T    candidate   = delta + radius;
        if (x < len3.x and y < len3.y and z < len3.z) {
            quant[gid]   = quantizable * static_cast<EQ>(candidate);
            outlier[gid] = (not quantizable) * candidate;
        }
    };

    ////////////////////////////////////////////////////////////////////////////

    /* z-direction, sequential in private buffer
       delta = + (s[z][y][x] - s[z-1][y][x])
               - (s[z][y][x-1] - s[z-1][y][x-1])
               + (s[z][y-1][x-1] - s[z-1][y-1][x-1])
               - (s[z][y-1][x] - s[z-1][y-1][x])

       x-direction, shuffle
       delta = + (s[z][y][x] - s[z][y][x-1])
               - (s[z][y-1][x] - s[z][y-1][x-1])

       y-direction, shmem
       delta = s[z][y][x] - s[z][y-1][x]
     */

    load_prequant_3d();

    for (auto z = BLOCK; z > 0; z--) {
        // z-direction
        delta[z] -= delta[z - 1];

        // x-direction
        auto prev_x = __shfl_up_sync(0xffffffff, delta[z], 1, 8);
        if (threadIdx.x % BLOCK > 0) delta[z] -= prev_x;

        // y-direction, exchange via shmem
        // ghost padding along y
        s[threadIdx.y + 1][threadIdx.x] = delta[z];
        __syncthreads();

        delta[z] -= (threadIdx.y > 0) * s[threadIdx.y][threadIdx.x];

        // now delta[z] is delta
        quantize_write(delta[z], gix, giy, giz(z - 1), gid(z - 1));
    }
}

template <typename T, typename EQ, typename FP>
__global__ void parsz::cuda::__kernel::v0::delta_only::c_lorenzo_3d1l(  //
    T*   data,
    dim3 len3,
    dim3 stride3,
    FP   ebx2_r,
    EQ*  quant)
{
    constexpr auto BLOCK = 8;
    __shared__ T   s[9][33];
    T              delta[BLOCK + 1] = {0};  // first el = 0

    const auto gix      = blockIdx.x * (BLOCK * 4) + threadIdx.x;
    const auto giy      = blockIdx.y * BLOCK + threadIdx.y;
    const auto giz_base = blockIdx.z * BLOCK;
    const auto base_id  = gix + giy * stride3.y + giz_base * stride3.z;

    auto giz = [&](auto z) { return giz_base + z; };
    auto gid = [&](auto z) { return base_id + z * stride3.z; };

    auto load_prequant_3d = [&]() {
        if (gix < len3.x and giy < len3.y) {
            for (auto z = 0; z < BLOCK; z++)
                if (giz(z) < len3.z) delta[z + 1] = round(data[gid(z)] * ebx2_r);  // prequant (fp presence)
        }
        __syncthreads();
    };

    auto quantize_write = [&](T delta, auto x, auto y, auto z, auto gid) {
        if (x < len3.x and y < len3.y and z < len3.z) quant[gid] = static_cast<EQ>(delta);
    };

    ////////////////////////////////////////////////////////////////////////////

    load_prequant_3d();

    for (auto z = BLOCK; z > 0; z--) {
        // z-direction
        delta[z] -= delta[z - 1];

        // x-direction
        auto prev_x = __shfl_up_sync(0xffffffff, delta[z], 1, 8);
        if (threadIdx.x % BLOCK > 0) delta[z] -= prev_x;

        // y-direction, exchange via shmem
        // ghost padding along y
        s[threadIdx.y + 1][threadIdx.x] = delta[z];
        __syncthreads();

        delta[z] -= (threadIdx.y > 0) * s[threadIdx.y][threadIdx.x];

        // now delta[z] is delta
        quantize_write(delta[z], gix, giy, giz(z - 1), gid(z - 1));
    }
}

template <typename T, typename EQ, typename FP>
__global__ void parsz::cuda::__kernel::v1_pn::delta_only::c_lorenzo_3d1l(  //
    T*   data,
    dim3 len3,
    dim3 stride3,
    FP   ebx2_r,
    EQ*  quant)
{
    constexpr auto BYTEWIDTH = sizeof(EQ);

    using UI = EQ;
    using I  = typename parsz::typing::Int<BYTEWIDTH>::T;

    constexpr auto BLOCK = 8;
    __shared__ T   s[9][33];
    T              delta[BLOCK + 1] = {0};  // first el = 0

    const auto gix      = blockIdx.x * (BLOCK * 4) + threadIdx.x;
    const auto giy      = blockIdx.y * BLOCK + threadIdx.y;
    const auto giz_base = blockIdx.z * BLOCK;
    const auto base_id  = gix + giy * stride3.y + giz_base * stride3.z;

    auto giz = [&](auto z) { return giz_base + z; };
    auto gid = [&](auto z) { return base_id + z * stride3.z; };

    auto load_prequant_3d = [&]() {
        if (gix < len3.x and giy < len3.y) {
            for (auto z = 0; z < BLOCK; z++)
                if (giz(z) < len3.z) delta[z + 1] = round(data[gid(z)] * ebx2_r);  // prequant (fp presence)
        }
        __syncthreads();
    };

    auto quantize_write = [&](T delta, auto x, auto y, auto z, auto gid) {
        if (x < len3.x and y < len3.y and z < len3.z) quant[gid] = PN<BYTEWIDTH>::encode(static_cast<I>(delta));
    };

    ////////////////////////////////////////////////////////////////////////////

    load_prequant_3d();

    for (auto z = BLOCK; z > 0; z--) {
        // z-direction
        delta[z] -= delta[z - 1];

        // x-direction
        auto prev_x = __shfl_up_sync(0xffffffff, delta[z], 1, 8);
        if (threadIdx.x % BLOCK > 0) delta[z] -= prev_x;

        // y-direction, exchange via shmem
        // ghost padding along y
        s[threadIdx.y + 1][threadIdx.x] = delta[z];
        __syncthreads();

        delta[z] -= (threadIdx.y > 0) * s[threadIdx.y][threadIdx.x];

        // now delta[z] is delta
        quantize_write(delta[z], gix, giy, giz(z - 1), gid(z - 1));
    }
}

template <typename T, typename EQ, typename FP, typename OutlierDesc>
__global__ void parsz::cuda::__kernel::v0::compaction::c_lorenzo_3d1l(
    T*          data,
    dim3        len3,
    dim3        stride3,
    int         radius,
    FP          ebx2_r,
    EQ*         quant,
    OutlierDesc outlier)
{
    constexpr auto BLOCK = 8;
    __shared__ T   s[9][33];
    T              delta[BLOCK + 1] = {0};  // first el = 0

    const auto gix      = blockIdx.x * (BLOCK * 4) + threadIdx.x;
    const auto giy      = blockIdx.y * BLOCK + threadIdx.y;
    const auto giz_base = blockIdx.z * BLOCK;
    const auto base_id  = gix + giy * stride3.y + giz_base * stride3.z;

    auto giz = [&](auto z) { return giz_base + z; };
    auto gid = [&](auto z) { return base_id + z * stride3.z; };

    auto load_prequant_3d = [&]() {
        if (gix < len3.x and giy < len3.y) {
            for (auto z = 0; z < BLOCK; z++)
                if (giz(z) < len3.z) delta[z + 1] = round(data[gid(z)] * ebx2_r);  // prequant (fp presence)
        }
        __syncthreads();
    };

    auto quantize_compact_write = [&](T delta, auto x, auto y, auto z, auto gid) {
        bool quantizable = fabs(delta) < radius;
        T    candidate   = delta + radius;
        if (x < len3.x and y < len3.y and z < len3.z) {
            quant[gid] = quantizable * static_cast<EQ>(candidate);
            if (not quantizable) {
                auto cur_idx         = atomicAdd(outlier.count, 1);
                outlier.idx[cur_idx] = gid;
                outlier.val[cur_idx] = candidate;
            }
        }
    };

    ////////////////////////////////////////////////////////////////////////////

    load_prequant_3d();

    for (auto z = BLOCK; z > 0; z--) {
        // z-direction
        delta[z] -= delta[z - 1];

        // x-direction
        auto prev_x = __shfl_up_sync(0xffffffff, delta[z], 1, 8);
        if (threadIdx.x % BLOCK > 0) delta[z] -= prev_x;

        // y-direction, exchange via shmem
        // ghost padding along y
        s[threadIdx.y + 1][threadIdx.x] = delta[z];
        __syncthreads();

        delta[z] -= (threadIdx.y > 0) * s[threadIdx.y][threadIdx.x];

        // now delta[z] is delta
        quantize_compact_write(delta[z], gix, giy, giz(z - 1), gid(z - 1));
    }
}

template <typename T, typename EQ, typename FP, typename OutlierDesc>
__global__ void parsz::cuda::__kernel::v1_pn::compaction::c_lorenzo_3d1l(
    T*          data,
    dim3        len3,
    dim3        stride3,
    int         radius,
    FP          ebx2_r,
    EQ*         quant,
    OutlierDesc outlier)
{
    constexpr auto BYTEWIDTH = sizeof(EQ);

    using UI = EQ;
    using I  = typename parsz::typing::Int<BYTEWIDTH>::T;

    constexpr auto BLOCK = 8;
    __shared__ T   s[9][33];
    T              delta[BLOCK + 1] = {0};  // first el = 0

    const auto gix      = blockIdx.x * (BLOCK * 4) + threadIdx.x;
    const auto giy      = blockIdx.y * BLOCK + threadIdx.y;
    const auto giz_base = blockIdx.z * BLOCK;
    const auto base_id  = gix + giy * stride3.y + giz_base * stride3.z;

    auto giz = [&](auto z) { return giz_base + z; };
    auto gid = [&](auto z) { return base_id + z * stride3.z; };

    // TODO move to subroutine.inl
    auto load_prequant_3d = [&]() {
        if (gix < len3.x and giy < len3.y) {
            for (auto z = 0; z < BLOCK; z++)
                if (giz(z) < len3.z) delta[z + 1] = round(data[gid(z)] * ebx2_r);  // prequant (fp presence)
        }
        __syncthreads();
    };

    auto quantize_compact_write = [&](T delta, auto x, auto y, auto z, auto gid) {
        bool quantizable = fabs(delta) < radius;
        UI   UI_delta    = PN<BYTEWIDTH>::encode(static_cast<I>(delta));

        T candidate = delta + radius;
        if (x < len3.x and y < len3.y and z < len3.z) {
            quant[gid] = quantizable * UI_delta;
            if (not quantizable) {
                auto cur_idx         = atomicAdd(outlier.count, 1);
                outlier.idx[cur_idx] = gid;
                outlier.val[cur_idx] = UI_delta;
            }
        }
    };

    ////////////////////////////////////////////////////////////////////////////

    load_prequant_3d();

    for (auto z = BLOCK; z > 0; z--) {
        // z-direction
        delta[z] -= delta[z - 1];

        // x-direction
        auto prev_x = __shfl_up_sync(0xffffffff, delta[z], 1, 8);
        if (threadIdx.x % BLOCK > 0) delta[z] -= prev_x;

        // y-direction, exchange via shmem
        // ghost padding along y
        s[threadIdx.y + 1][threadIdx.x] = delta[z];
        __syncthreads();

        delta[z] -= (threadIdx.y > 0) * s[threadIdx.y][threadIdx.x];

        // now delta[z] is delta
        quantize_compact_write(delta[z], gix, giy, giz(z - 1), gid(z - 1));
    }
}

// 32x8x8 data block maps to 32x1x8 thread block
template <typename T, typename EQ, typename FP>
__global__ void parsz::cuda::__kernel::v0::x_lorenzo_3d1l(  //
    EQ*  quant,
    T*   outlier,
    dim3 len3,
    dim3 stride3,
    int  radius,
    FP   ebx2,
    T*   xdata)
{
    constexpr auto BLOCK = 8;
    constexpr auto YSEQ  = BLOCK;
    static_assert(BLOCK == 8, "In one case, we need BLOCK for 3D == 8");

    __shared__ T intermediate[BLOCK][4][8];
    T            thread_private[YSEQ];

    auto seg_id  = threadIdx.x / 8;
    auto seg_tix = threadIdx.x % 8;

    auto gix      = blockIdx.x * (4 * BLOCK) + threadIdx.x;
    auto giy_base = blockIdx.y * BLOCK;
    auto giy      = [&](auto y) { return giy_base + y; };
    auto giz      = blockIdx.z * BLOCK + threadIdx.z;
    auto gid      = [&](auto y) { return giz * stride3.z + (giy_base + y) * stride3.y + gix; };

    auto load_fuse_3d = [&]() {
    // load to thread-private array (fuse at the same time)
#pragma unroll
        for (auto y = 0; y < YSEQ; y++) {
            if (gix < len3.x and giy_base + y < len3.y and giz < len3.z)
                thread_private[y] = outlier[gid(y)] + static_cast<T>(quant[gid(y)]) - radius;  // fuse
            else
                thread_private[y] = 0;
        }
    };

    auto block_scan_3d = [&]() {
        // partial-sum along y-axis, sequantially
        for (auto y = 1; y < YSEQ; y++) thread_private[y] += thread_private[y - 1];

#pragma unroll
        for (auto i = 0; i < BLOCK; i++) {
            // ND partial-sums along x- and z-axis
            // in-warp shuffle used: in order to perform, it's transposed after X-partial sum
            T val = thread_private[i];

            for (auto dist = 1; dist < BLOCK; dist *= 2) {
                auto addend = __shfl_up_sync(0xffffffff, val, dist, 8);
                if (seg_tix >= dist) val += addend;
            }

            // x-z transpose
            intermediate[threadIdx.z][seg_id][seg_tix] = val;
            __syncthreads();
            val = intermediate[seg_tix][seg_id][threadIdx.z];
            __syncthreads();

            for (auto dist = 1; dist < BLOCK; dist *= 2) {
                auto addend = __shfl_up_sync(0xffffffff, val, dist, 8);
                if (seg_tix >= dist) val += addend;
            }

            intermediate[threadIdx.z][seg_id][seg_tix] = val;
            __syncthreads();
            val = intermediate[seg_tix][seg_id][threadIdx.z];
            __syncthreads();

            thread_private[i] = val;
        }
    };

    auto decomp_write_3d = [&]() {
#pragma unroll
        for (auto y = 0; y < YSEQ; y++)
            if (gix < len3.x and giy(y) < len3.y and giz < len3.z) xdata[gid(y)] = thread_private[y] * ebx2;
    };

    ////////////////////////////////////////////////////////////////////////////
    load_fuse_3d();
    block_scan_3d();
    decomp_write_3d();
}

// 32x8x8 data block maps to 32x1x8 thread block
template <typename T, typename EQ, typename FP>
__global__ void parsz::cuda::__kernel::v1_pn::x_lorenzo_3d1l(  //
    EQ*  quant,
    T*   outlier,
    dim3 len3,
    dim3 stride3,
    FP   ebx2,
    T*   xdata)
{
    constexpr auto BYTEWIDTH = sizeof(EQ);

    using UI = EQ;
    using I  = typename parsz::typing::Int<BYTEWIDTH>::T;

    constexpr auto BLOCK = 8;
    constexpr auto YSEQ  = BLOCK;
    static_assert(BLOCK == 8, "In one case, we need BLOCK for 3D == 8");

    __shared__ T intermediate[BLOCK][4][8];
    T            thread_private[YSEQ];

    auto seg_id  = threadIdx.x / 8;
    auto seg_tix = threadIdx.x % 8;

    auto gix      = blockIdx.x * (4 * BLOCK) + threadIdx.x;
    auto giy_base = blockIdx.y * BLOCK;
    auto giy      = [&](auto y) { return giy_base + y; };
    auto giz      = blockIdx.z * BLOCK + threadIdx.z;
    auto gid      = [&](auto y) { return giz * stride3.z + (giy_base + y) * stride3.y + gix; };

    auto load_fuse_3d = [&]() {
    // load to thread-private array (fuse at the same time)
#pragma unroll
        for (auto y = 0; y < YSEQ; y++) {
            if (gix < len3.x and giy_base + y < len3.y and giz < len3.z)
                thread_private[y] = outlier[gid(y)] + PN<BYTEWIDTH>::decode(quant[gid(y)]);  // fuse
            else
                thread_private[y] = 0;
        }
    };

    auto block_scan_3d = [&]() {
        // partial-sum along y-axis, sequantially
        for (auto y = 1; y < YSEQ; y++) thread_private[y] += thread_private[y - 1];

#pragma unroll
        for (auto i = 0; i < BLOCK; i++) {
            // ND partial-sums along x- and z-axis
            // in-warp shuffle used: in order to perform, it's transposed after X-partial sum
            T val = thread_private[i];

            for (auto dist = 1; dist < BLOCK; dist *= 2) {
                auto addend = __shfl_up_sync(0xffffffff, val, dist, 8);
                if (seg_tix >= dist) val += addend;
            }

            // x-z transpose
            intermediate[threadIdx.z][seg_id][seg_tix] = val;
            __syncthreads();
            val = intermediate[seg_tix][seg_id][threadIdx.z];
            __syncthreads();

            for (auto dist = 1; dist < BLOCK; dist *= 2) {
                auto addend = __shfl_up_sync(0xffffffff, val, dist, 8);
                if (seg_tix >= dist) val += addend;
            }

            intermediate[threadIdx.z][seg_id][seg_tix] = val;
            __syncthreads();
            val = intermediate[seg_tix][seg_id][threadIdx.z];
            __syncthreads();

            thread_private[i] = val;
        }
    };

    auto decomp_write_3d = [&]() {
#pragma unroll
        for (auto y = 0; y < YSEQ; y++)
            if (gix < len3.x and giy(y) < len3.y and giz < len3.z) xdata[gid(y)] = thread_private[y] * ebx2;
    };

    ////////////////////////////////////////////////////////////////////////////
    load_fuse_3d();
    block_scan_3d();
    decomp_write_3d();
}

// 32x8x8 data block maps to 32x1x8 thread block
template <typename T, typename EQ, typename FP>
__global__ void parsz::cuda::__kernel::v0::delta_only::x_lorenzo_3d1l(  //
    EQ*  quant,
    dim3 len3,
    dim3 stride3,
    FP   ebx2,
    T*   xdata)
{
    constexpr auto BLOCK = 8;
    constexpr auto YSEQ  = BLOCK;
    static_assert(BLOCK == 8, "In one case, we need BLOCK for 3D == 8");

    __shared__ T intermediate[BLOCK][4][8];
    T            thread_private[YSEQ];

    auto seg_id  = threadIdx.x / 8;
    auto seg_tix = threadIdx.x % 8;

    auto gix      = blockIdx.x * (4 * BLOCK) + threadIdx.x;
    auto giy_base = blockIdx.y * BLOCK;
    auto giy      = [&](auto y) { return giy_base + y; };
    auto giz      = blockIdx.z * BLOCK + threadIdx.z;
    auto gid      = [&](auto y) { return giz * stride3.z + (giy_base + y) * stride3.y + gix; };

    auto load_3d = [&]() {
    // load to thread-private array (fuse at the same time)
#pragma unroll
        for (auto y = 0; y < YSEQ; y++) {
            if (gix < len3.x and giy_base + y < len3.y and giz < len3.z)
                thread_private[y] = static_cast<T>(quant[gid(y)]);  // fuse
            else
                thread_private[y] = 0;
        }
    };

    auto block_scan_3d = [&]() {
        // partial-sum along y-axis, sequantially
        for (auto y = 1; y < YSEQ; y++) thread_private[y] += thread_private[y - 1];

#pragma unroll
        for (auto i = 0; i < BLOCK; i++) {
            // ND partial-sums along x- and z-axis
            // in-warp shuffle used: in order to perform, it's transposed after X-partial sum
            T val = thread_private[i];

            for (auto dist = 1; dist < BLOCK; dist *= 2) {
                auto addend = __shfl_up_sync(0xffffffff, val, dist, 8);
                if (seg_tix >= dist) val += addend;
            }

            // x-z transpose
            intermediate[threadIdx.z][seg_id][seg_tix] = val;
            __syncthreads();
            val = intermediate[seg_tix][seg_id][threadIdx.z];
            __syncthreads();

            for (auto dist = 1; dist < BLOCK; dist *= 2) {
                auto addend = __shfl_up_sync(0xffffffff, val, dist, 8);
                if (seg_tix >= dist) val += addend;
            }

            intermediate[threadIdx.z][seg_id][seg_tix] = val;
            __syncthreads();
            val = intermediate[seg_tix][seg_id][threadIdx.z];
            __syncthreads();

            thread_private[i] = val;
        }
    };

    auto decomp_write_3d = [&]() {
#pragma unroll
        for (auto y = 0; y < YSEQ; y++)
            if (gix < len3.x and giy(y) < len3.y and giz < len3.z) xdata[gid(y)] = thread_private[y] * ebx2;
    };

    ////////////////////////////////////////////////////////////////////////////
    load_3d();
    block_scan_3d();
    decomp_write_3d();
}
