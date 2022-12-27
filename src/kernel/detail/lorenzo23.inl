/**
 * @file lorenzo23.inl
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-12-22
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "subroutine.inl"

namespace parsz {
namespace cuda {
namespace __kernel {

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

}  // namespace __kernel
}  // namespace cuda
}  // namespace parsz

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
    namespace llfn_v0 = parsz::cuda::__device::v0;

    constexpr auto NTHREAD = BLOCK / SEQ;

    __shared__ struct {
        union {
            T data[BLOCK];
            T outlier[BLOCK];
        };
        EQ quant[BLOCK];
    } shmem;

    T prev{0};
    T thread_scope[SEQ];

    auto id_base = blockIdx.x * BLOCK;

    llfn_v0::load_prequant_1d<T, FP, NTHREAD, SEQ>(data, len3.x, id_base, shmem.data, thread_scope, prev, ebx2_r);
    llfn_v0::pq_with_outlier_1d<T, EQ, SEQ, true>(thread_scope, shmem.quant, shmem.outlier, radius, prev);
    llfn_v0::pq_with_outlier_1d<T, EQ, SEQ, false>(thread_scope, shmem.quant, shmem.outlier, radius);
    llfn_v0::write_1d<EQ, T, NTHREAD, SEQ, false>(shmem.quant, shmem.outlier, len3.x, id_base, quant, outlier);
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
    namespace llfn_v0 = parsz::cuda::__device::v0;
    namespace llfn_v1 = parsz::cuda::__device::v1_pncodec;

    constexpr auto NTHREAD = BLOCK / SEQ;

    __shared__ struct {
        union {
            T data[BLOCK];
            T outlier[BLOCK];
        };
        EQ quant[BLOCK];
    } shmem;

    T prev{0};
    T thread_scope[SEQ];

    auto id_base = blockIdx.x * BLOCK;

    llfn_v0::load_prequant_1d<T, FP, NTHREAD, SEQ>(data, len3.x, id_base, shmem.data, thread_scope, prev, ebx2_r);
    llfn_v1::pq_with_outlier_1d<T, EQ, SEQ, true>(thread_scope, shmem.quant, shmem.outlier, radius, prev);
    llfn_v1::pq_with_outlier_1d<T, EQ, SEQ, false>(thread_scope, shmem.quant, shmem.outlier, radius);
    llfn_v0::write_1d<EQ, T, NTHREAD, SEQ, false>(shmem.quant, shmem.outlier, len3.x, id_base, quant, outlier);
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
    namespace llfn_v0 = parsz::cuda::__device::v0;
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
    } shmem;

    T thread_scope[SEQ];

    auto id_base = blockIdx.x * BLOCK;

    llfn_v0::load_fuse_1d<T, EQ, NTHREAD, SEQ>(quant, outlier, len3.x, id_base, radius, shmem.xdata, thread_scope);
    llfn_v0::blockwide_inclusivescan_1d<T, SEQ, NTHREAD>(
        thread_scope, ebx2, shmem.exchange_in, shmem.exchange_out, shmem.xdata);
    llfn_v0::write_1d<T, T, NTHREAD, SEQ, true>(shmem.xdata, nullptr, len3.x, id_base, xdata, nullptr);
}