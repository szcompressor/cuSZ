#include <bitset>
#include <cstring>
#include <ctime>
#include <iostream>
#include <random>

#include "hf_impl.hh"
#include "kernel/hist.hh"
#include "mem/cxx_backends.h"
#include "rs_merge.hh"

using namespace std;

using u4 = uint32_t;
using u2 = uint16_t;
using T = u2;
using Hf = uint32_t;

constexpr int UnitBytes = sizeof(Hf);
constexpr int UnitBits = UnitBytes * 8;

constexpr int bklen = 256;
constexpr T neutual_val = bklen / 2;

int random_val_gaussian(int stdev = 5)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> distrib{(float)neutual_val, (float)stdev};
  auto val = (int)distrib(gen);

  if (val < neutual_val - 3 * stdev or val > neutual_val + 3 * stdev - 1)
    return neutual_val;
  else
    return val;
}

void bake_input(u2* ori_data, size_t len, float sparsity = 0.1)
{
  for (size_t i = 0; i < len; ++i) ori_data[i] = neutual_val;

  auto stride = (int)(len * sparsity);
  for (auto i = 0; i < len; i += stride) ori_data[i] = random_val_gaussian();
}

#define EQUAL_STR "\033[40;31m = \033[0m"

#define ALLOCATE_ALL(PREFIX)                                                                \
  /* dense output */                                                                        \
  auto PREFIX##_dnout = MAKE_UNIQUE_UNIFIED(Hf, C::ChunkSize);                              \
  auto PREFIX##_dn_bc = MAKE_UNIQUE_UNIFIED(u4, 1);                                         \
  auto PREFIX##_dn_loc = MAKE_UNIQUE_UNIFIED(u4, 1);                                        \
  u4 PREFIX##_loc_inc = 0;                                                                  \
  phf::dense<Hf> PREFIX##_dense = {                                                         \
      PREFIX##_dnout.get(), PREFIX##_dn_bc.get(), PREFIX##_dn_loc.get(), &PREFIX##_loc_inc, \
      C::ChunkSize};                                                                        \
  /* sparse output */                                                                       \
  auto PREFIX##_spval = MAKE_UNIQUE_UNIFIED(T, C::ChunkSize);                               \
  auto PREFIX##_spidx = MAKE_UNIQUE_UNIFIED(u4, C::ChunkSize);                              \
  u4 PREFIX##_spnum = 0;                                                                    \
  phf::sparse<T> PREFIX##_sparse = {PREFIX##_spval.get(), PREFIX##_spidx.get(), &PREFIX##_spnum};

#define PRINT_OUTPUT_WITH_PREFIX(PREFIX, PREFIX_STR)                                              \
  /* dense output */                                                                              \
  auto PREFIX##_num_units = (PREFIX##_dn_bc[0] - 1) / UnitBits + 1;                               \
  printf("\n" PREFIX_STR "\n");                                                                   \
  printf("dense-bit-count" EQUAL_STR "%u\n", PREFIX##_dn_bc[0]);                                  \
  printf("dense-unit-count" EQUAL_STR "%u\n", PREFIX##_num_units);                                \
  printf("dense-output (merged codes in hex):\n");                                                \
  for (auto i = 0u; i < PREFIX##_num_units; i++) {                                                \
    printf("hexcode-%02u" EQUAL_STR, i);                                                          \
    cout << bitset<UnitBytes * 8>(PREFIX##_dnout[i]) << endl;                                     \
  }                                                                                               \
  /* sparse output */                                                                             \
  if (PREFIX##_spnum == 0) { printf(PREFIX_STR "sparse output: no sparse values\n"); }            \
  else {                                                                                          \
    printf(PREFIX_STR "sparse values (breaking points):\n");                                      \
    for (auto i = 0u; i < PREFIX##_spnum; ++i) {                                                  \
      printf(                                                                                     \
          PREFIX_STR "i" EQUAL_STR "%u\t(%u,%u)\n", i, (u4)PREFIX##_spval[i], PREFIX##_spidx[i]); \
    }                                                                                             \
  }

template <size_t Magnitude = 6, size_t ReduceTimes = 3>
int run()
{
  using C = HFReVISIT_config<T, Magnitude, ReduceTimes, Hf>;

  // runtime inputs
  auto ori_data = MAKE_UNIQUE_UNIFIED(T, C::ChunkSize);
  auto decoded_data = MAKE_UNIQUE_UNIFIED(T, C::ChunkSize);
  auto freq = MAKE_UNIQUE_UNIFIED(u4, 256);
  auto bk = MAKE_UNIQUE_UNIFIED(Hf, bklen);
  auto revbk_bytes = phf_reverse_book_bytes(bklen, sizeof(Hf), sizeof(T));
  auto revbk = MAKE_UNIQUE_UNIFIED(u1, revbk_bytes);
  Hf alt_code;
  u4 alt_bitcount;
  bake_input(ori_data.get(), C::ChunkSize, 0.1);
  psz::module::SEQ_histogram_generic<T>(ori_data.get(), C::ChunkSize, freq.get(), bklen, nullptr);
  phf_CPU_build_canonized_codebook_v2<T, Hf>(
      freq.get(), bklen, bk.get(), revbk.get(), revbk_bytes, nullptr);
  phf::make_altcode<Hf>(bk.get(), bklen, ReduceTimes, alt_code, alt_bitcount);
  // print histogram
  for (auto i = 0; i < bklen; i++)
    if (freq[i] != 0) {
      auto pw4 = reinterpret_cast<PW4*>(bk.get() + i);
      cout << "bk-" << i << "\t";
      cout << "bitcount" << EQUAL_STR << pw4->bitcount << "("
           << bitset<PW4::FIELD_BITCOUNT>(pw4->bitcount) << ")\t";
      cout << "hf-bits" << EQUAL_STR << bitset<PW4::FIELD_CODE>(pw4->prefix_code) << "\n";
    }

  ALLOCATE_ALL(CPU);
  ALLOCATE_ALL(GPU);

  // GPU stream
  auto stream = create_stream();

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
  phf::module::CPU_HFReVISIT_encode<T, Magnitude, ReduceTimes, false, Hf>(
      {ori_data.get(), C::ChunkSize}, {bk.get(), bklen, alt_code, alt_bitcount}, CPU_dense,
      CPU_sparse);
  phf::module::GPU_HFReVISIT_encode<T, Magnitude, ReduceTimes, false, Hf>(
      {ori_data.get(), C::ChunkSize}, {bk.get(), bklen, alt_code, alt_bitcount}, GPU_dense,
      GPU_sparse, stream);
  sync_by_stream(stream);
  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  // PRINT_OUTPUT_WITH_PREFIX(CPU, "CPU: ")
  PRINT_OUTPUT_WITH_PREFIX(GPU, "GPU: ")

  phf::cuhip::modules<T, Hf>::GPU_coarse_decode(
      GPU_dense.encoded, revbk.get(), revbk_bytes, GPU_dense.chunk_nbit, GPU_dense.chunk_loc,
      bklen, 1, decoded_data.get(), stream);
  sync_by_stream(stream);

  auto verifid = true;
  for (auto i = 0; i < C::ChunkSize; i++) {
    // printf("ori_data[%d] = %d, decoded_data[%d] = %d\n", i, ori_data[i], i, decoded_data[i]);
    if (ori_data[i] != decoded_data[i]) {
      printf("\n\033[40;31mERROR\033[0m: decoded data is not equal to original data\n");
      verifid = false;
    }
  }

  destroy_stream(stream);

  if (verifid) {
    printf("\n\033[40;32mPASS\033[0m: enc-dec okay.\n");
    return 0;
  }
  else
    return 1;
}

int main() { return run(); }