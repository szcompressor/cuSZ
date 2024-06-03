#ifndef B166AEEB_917A_448A_91F6_D0F7A186A36A
#define B166AEEB_917A_448A_91F6_D0F7A186A36A

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <string>
#include <vector>

#include "cusz/type.h"
#include "tehm.hh"
#include "utils/config.hh"

using std::string;
using std::vector;

namespace psz {

template <typename T>
static void print_metrics_cross(
    psz_summary* s, size_t compressed_bytes = 0, bool gpu_checker = false)
{
  auto checker = (not gpu_checker) ? string("(using CPU checker)")
                                   : string("(using GPU checker)");
  auto bytes = (s->len * sizeof(T) * 1.0);

  auto println = [](const char* s, double n1, double n2, double n3,
                    double n4) {
    printf("  %-10s %16.8g %16.8g %16.8g %16.8g\n", s, n1, n2, n3, n4);
  };
  auto printhead = [](const char* s1, const char* s2, const char* s3,
                      const char* s4, const char* s5) {
    printf(
        "  \e[1m\e[31m%-10s %16s %16s %16s %16s\e[0m\n", s1, s2, s3, s4, s5);
  };

  auto is_fp = std::is_same<T, float>::value or std::is_same<T, double>::value
                   ? const_cast<char*>("yes")
                   : const_cast<char*>("no");
  printf("\nquality metrics %s:\n", checker.c_str());

  printhead("", "data-len", "data-byte", "fp-type?", "");
  printf("  %-10s %16zu %16lu %16s\n", "", s->len, sizeof(T), is_fp);

  printhead("", "min", "max", "rng", "std");
  println("origin", s->odata.min, s->odata.max, s->odata.rng, s->odata.std);
  println("eb-lossy", s->xdata.min, s->xdata.max, s->xdata.rng, s->xdata.std);

  printhead("", "abs-val", "abs-idx", "pw-rel", "VS-RNG");
  println(
      "max-error", s->max_err_abs, s->max_err_idx, s->max_err_pwrrel,
      s->max_err_rel);

  printhead("", "CR", "NRMSE", "cross-cor", "PSNR");
  println(
      "metrics", bytes / compressed_bytes, s->score_NRMSE, s->score_coeff,
      s->score_PSNR);

  // printf("\n");
};

static void print_metrics_auto(double* lag1_cor, double* lag2_cor)
{
  auto printhead = [](const char* s1, const char* s2, const char* s3,
                      const char* s4, const char* s5) {
    printf(
        "  \e[1m\e[31m%-10s %16s %16s %16s %16s\e[0m\n", s1, s2, s3, s4, s5);
  };

  printhead("", "lag1-cor", "lag2-cor", "", "");
  printf("  %-10s %16lf %16lf\n", "auto", *lag1_cor, *lag2_cor);
  printf("\n");
};

}  // namespace psz

namespace psz {

using TimeRecordTuple = std::tuple<const char*, double>;
using TimeRecord = std::vector<TimeRecordTuple>;
using timerecord_t = TimeRecord*;

struct TimeRecordViewer {
  static float get_throughput(float milliseconds, size_t bytes)
  {
    auto GiB = 1.0 * 1024 * 1024 * 1024;
    auto seconds = milliseconds * 1e-3;
    return bytes / GiB / seconds;
  }

  static void println_throughput(const char* s, float timer, size_t bytes)
  {
    if (timer == 0.0) return;

    auto t = get_throughput(timer, bytes);
    printf("  %-12s %'12f %'10.2f\n", s, timer, t);
  };

  static void println_throughput_tablehead()
  {
    printf(
        "\n  \e[1m\e[31m%-12s %12s %10s\e[0m\n",  //
        const_cast<char*>("kernel"),              //
        const_cast<char*>("time, ms"),            //
        const_cast<char*>("GiB/s")                //
    );
  }
};

}  // namespace psz

#endif /* B166AEEB_917A_448A_91F6_D0F7A186A36A */
