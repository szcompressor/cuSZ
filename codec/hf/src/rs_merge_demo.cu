#include <cstdio>
#include <memory>

constexpr auto ShuffleTimes = 4u;
constexpr auto NumShards = 1 << ShuffleTimes;

__global__ void GPU_impl()
{
  for (auto sf = ShuffleTimes, stride = 1u; sf > 0; sf--, stride *= 2) {
    auto l = threadIdx.x / (stride * 2) * (stride * 2);
    auto r = l + stride;

    if (threadIdx.x == 0) {
      printf(
          "G  | printed t.id == data.id to move\n"
          "SF | data.group | data.id & id.range\n"
          "---+------------+-------------------\n");
    }
    __syncthreads();

    if (threadIdx.x >= r and threadIdx.x < r + stride) {  //
      printf(
          "%2u | %10u | %6u in [%2u, %2u)\n", sf, threadIdx.x / (stride * 2), threadIdx.x, r,
          r + stride);
    }
    __syncthreads();
    if (threadIdx.x == 0) { printf("\n"); }
    __syncthreads();
  }
}

#define PARFOR1_BLOCK_SHFLMERGE() for (auto thread_idx = 0; thread_idx < NumShards; thread_idx++)

// enumerate states for all "threads"
#define SHFLMERGE_THREAD_STATE_ENUMERATION                                            \
  std::unique_ptr<uint32_t[]> _l = std::make_unique<uint32_t[]>(NumShards);           \
  std::unique_ptr<uint32_t[]> _r = std::make_unique<uint32_t[]>(NumShards);           \
  std::unique_ptr<uint32_t[]> _lbc = std::make_unique<uint32_t[]>(NumShards);         \
  std::unique_ptr<uint32_t[]> _used__units = std::make_unique<uint32_t[]>(NumShards); \
  std::unique_ptr<uint32_t[]> _used___bits = std::make_unique<uint32_t[]>(NumShards); \
  std::unique_ptr<uint32_t[]> _unused_bits = std::make_unique<uint32_t[]>(NumShards); \
  std::unique_ptr<uint32_t[]> _this_point = std::make_unique<uint32_t[]>(NumShards);  \
  std::unique_ptr<uint32_t[]> _lsym = std::make_unique<uint32_t[]>(NumShards);        \
  std::unique_ptr<uint32_t[]> _rsym = std::make_unique<uint32_t[]>(NumShards);

// mimick GPU programming
#define l _l[thread_idx]
#define r _r[thread_idx]
#define lbc _lbc[thread_idx]
#define used__units _used__units[thread_idx]
#define used___bits _used___bits[thread_idx]
#define unused_bits _unused_bits[thread_idx]
#define this_point _this_point[thread_idx]
#define lsym _lsym[thread_idx]
#define rsym _rsym[thread_idx]

void CPU_impl()
{
  for (auto sf = ShuffleTimes, stride = 1u; sf > 0; sf--, stride *= 2) {
    printf(
        "C  | printed t.id == data.id to move\n"
        "SF | data.group | data.id & id.range\n"
        "---+------------+-------------------\n");

    SHFLMERGE_THREAD_STATE_ENUMERATION;

    PARFOR1_BLOCK_SHFLMERGE()
    {
      l = thread_idx / (stride * 2) * (stride * 2);
      r = l + stride;

      if (thread_idx >= r and thread_idx < r + stride) {  //
        printf(
            "%2u | %10u | %6u in [%2u, %2u)\n", sf, thread_idx / (stride * 2), thread_idx, r,
            r + stride);
      }
    }

    printf("\n");
  }
}

int main()
{
  GPU_impl<<<1, (1 << ShuffleTimes)>>>();
  cudaDeviceSynchronize();

  CPU_impl();
}

#undef SHFLMERGE_THREAD_STATE_ENUMERATION
#undef l
#undef r
#undef lbc
#undef used__units
#undef used___bits
#undef unused_bits
#undef this_point
#undef lsym
#undef rsym