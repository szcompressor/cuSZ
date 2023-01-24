/**
 * @file subsub.inl
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2022-12-26
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

namespace parsz {
namespace cuda {
namespace __device {

namespace wave32 {
template <typename T, int SEQ>
__forceinline__ __device__ void intrawarp_inclusivescan_1d(  //
    T private_buffer[SEQ]);

template <typename T, int SEQ, int NTHREAD>
__forceinline__ __device__ void intrablock_exclusivescan_1d(  //
    T           private_buffer[SEQ],
    volatile T* exchange_in,
    volatile T* exchange_out);
}  // namespace wave32

}  // namespace __device
}  // namespace cuda
}  // namespace parsz

template <typename T, int SEQ>
__forceinline__ __device__ void parsz::cuda::__device::wave32::intrawarp_inclusivescan_1d(T private_buffer[SEQ])
{
    for (auto i = 1; i < SEQ; i++) private_buffer[i] += private_buffer[i - 1];
    T addend = private_buffer[SEQ - 1];

    // in-warp shuffle
    for (auto d = 1; d < 32; d *= 2) {
        T n = __shfl_up_sync(0xffffffff, addend, d, 32);
        if (threadIdx.x % 32 >= d) addend += n;
    }
    // exclusive scan
    T prev_addend = __shfl_up_sync(0xffffffff, addend, 1, 32);

    // propagate
    if (threadIdx.x % 32 > 0)
        for (auto i = 0; i < SEQ; i++) private_buffer[i] += prev_addend;
}

template <typename T, int SEQ, int NTHREAD>
__forceinline__ __device__ void parsz::cuda::__device::wave32::intrablock_exclusivescan_1d(
    T           private_buffer[SEQ],
    volatile T* exchange_in,
    volatile T* exchange_out)
{
    constexpr auto NWARP = NTHREAD / 32;
    static_assert(NWARP <= 32, "too big");

    auto warp_id = threadIdx.x / 32;
    auto lane_id = threadIdx.x % 32;

    if (lane_id == 31) exchange_in[warp_id] = private_buffer[SEQ - 1];
    __syncthreads();

    if (NWARP <= 8) {
        if (threadIdx.x == 0) {
            exchange_out[0] = 0;
            for (auto i = 1; i < NWARP; i++) exchange_out[i] = exchange_out[i - 1] + exchange_in[i - 1];
        }
    }
    else if (NWARP <= 32) {
        if (threadIdx.x <= 32) {
            auto addend = exchange_in[threadIdx.x];

            for (auto d = 1; d < 32; d *= 2) {
                T n = __shfl_up_sync(0xffffffff, addend, d, 32);
                if (threadIdx.x >= d) addend += n;
            }
            // exclusive scan
            T prev_addend         = __shfl_up_sync(0xffffffff, addend, 1, 32);
            exchange_out[warp_id] = (warp_id > 0) * prev_addend;
        }
    }
    // else-case handled by static_assert
    __syncthreads();

    // propagate
    auto addend = exchange_out[warp_id];
    for (auto i = 0; i < SEQ; i++) private_buffer[i] += addend;
    __syncthreads();
};
