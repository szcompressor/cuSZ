#ifndef _PORTABLE_UTILS_SYNTH_HH
#define _PORTABLE_UTILS_SYNTH_HH

#include <cstddef>
#include <cstdint>
#include <string>

#include "c_type.h"

using std::string;

namespace _ptb::testutils {

int randint(size_t upper_limit);

template <typename T>
T randfp(T upper = 1.0, T lower = 0.0);

template <typename T>
void rand_array_cpp(T* array, size_t len);

template <typename T>
void rand_array_cu(T* array, size_t len, uint32_t seed = 0x2468);

template <typename T>
void rand_array_dpcpp(T* array, size_t len, uint32_t seed = 0x2468);

struct Synth {
  string   mode;
  double   peak  = 128.0;  // cauchy location
  double   gamma = 2.0;    // cauchy scale: smaller -> higher PMF_1)
  uint32_t max   = 256;    // uniform upper bound (exclusive)
  uint32_t seed  = 43;

  // --synth [spec], e.g., "cauchy:peak=128:gamma=2:seed=43", "uniform:max=256:seed=43"
  static Synth parse(const std::string& spec);

  // fill buf with synthetic symbols
  void fill(void* buf, std::size_t len, _ptb_dtype dt) const;

  double pmf1() const;
  static double pmf1_from(double gamma);
  static double gamma_from(double pmf1);

  static constexpr double sweep_gammas[] = {
      0.500000, 0.363271, 0.254763, 0.162460, 0.079192, 0.039351, 0.007855,
  };
};

}  // namespace _ptb::testutils

#endif  // _PORTABLE_UTILS_SYNTH_HH
