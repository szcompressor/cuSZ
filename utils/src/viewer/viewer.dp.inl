#include "detail/compare.hh"

namespace psz {

template <typename T, psz_runtime P = THRUST_DPL>
static void pszcxx_evaluate_quality_gpu(
    T* reconstructed, T* origin, size_t len, size_t compressed_bytes = 0)
{
  // cross
  auto stat_x = new psz_statistics;
  psz::dpl::GPU_assess_quality<T>(stat_x, reconstructed, origin, len);
  psz::analysis::print_metrics_cross<T>(stat_x, compressed_bytes, true);

  auto stat_auto_lag1 = new psz_statistics;
  psz::dpl::GPU_assess_quality<T>(stat_auto_lag1, origin, origin + 1, len - 1);
  auto stat_auto_lag2 = new psz_statistics;
  psz::dpl::GPU_assess_quality<T>(stat_auto_lag2, origin, origin + 2, len - 2);

  psz::utils::print_metrics_auto(&stat_auto_lag1->score_coeff, &stat_auto_lag2->score_coeff);

  delete stat_x, delete stat_auto_lag1, delete stat_auto_lag2;
}

template <typename T>
static void psz::analysis::CPU_evaluate_quality_and_print(
    T* _d1, T* _d2, size_t len, size_t compressed_bytes = 0, bool from_device = true)
{
  sycl::device dev_ct1;
  sycl::queue q_ct1(dev_ct1, sycl::property_list{sycl::property::queue::in_order()});
  auto stat = new psz_statistics;
  T* reconstructed;
  T* origin;
  if (not from_device) {
    reconstructed = _d1;
    origin = _d2;
  }
  else {
    printf("allocating tmp space for CPU verification\n");
    auto bytes = sizeof(T) * len;
    reconstructed = (T*)sycl::malloc_host(bytes, q_ct1);
    origin = (T*)sycl::malloc_host(bytes, q_ct1);
    q_ct1.memcpy(reconstructed, _d1, bytes).wait();
    q_ct1.memcpy(origin, _d2, bytes).wait();
  }
  psz::analysis::assess_quality<ONEAPI, T>(stat, reconstructed, origin, len);
  psz::analysis::print_metrics_cross<T>(stat, compressed_bytes, false);

  auto stat_auto_lag1 = new psz_statistics;
  psz::analysis::assess_quality<ONEAPI, T>(stat_auto_lag1, origin, origin + 1, len - 1);
  auto stat_auto_lag2 = new psz_statistics;
  psz::analysis::assess_quality<ONEAPI, T>(stat_auto_lag2, origin, origin + 2, len - 2);

  psz::utils::print_metrics_auto(&stat_auto_lag1->score_coeff, &stat_auto_lag2->score_coeff);

  if (from_device) {
    reconstructed = (T*)sycl::free(reconstructed, q_ct1);
    origin = (T*)sycl::free(origin, q_ct1);
  }

  delete stat, delete stat_auto_lag1, delete stat_auto_lag2;
}

}  // namespace psz
