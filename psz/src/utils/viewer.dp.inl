#include "stat/compare.hh"

namespace psz {

template <typename T, psz_policy P = THRUST>
static void pszcxx_evaluate_quality_gpu(
    T* reconstructed, T* origin, size_t len, size_t compressed_bytes = 0)
{
  // cross
  auto stat_x = new psz_summary;
  psz::dpl::GPU_assess_quality<T>(stat_x, reconstructed, origin, len);
  psz::utils::print_metrics_cross<T>(stat_x, compressed_bytes, true);

  auto stat_auto_lag1 = new psz_summary;
  psz::dpl::GPU_assess_quality<T>(stat_auto_lag1, origin, origin + 1, len - 1);
  auto stat_auto_lag2 = new psz_summary;
  psz::dpl::GPU_assess_quality<T>(stat_auto_lag2, origin, origin + 2, len - 2);

  psz::utils::print_metrics_auto(
      &stat_auto_lag1->score_coeff, &stat_auto_lag2->score_coeff);

  delete stat_x, delete stat_auto_lag1, delete stat_auto_lag2;
}

template <typename T>
static void pszcxx_evaluate_quality_cpu(
    T* _d1, T* _d2, size_t len, size_t compressed_bytes = 0,
    bool from_device = true)
{
  sycl::device dev_ct1;
  sycl::queue q_ct1(
      dev_ct1, sycl::property_list{sycl::property::queue::in_order()});
  auto stat = new psz_summary;
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
  psz::utils::verify_data<T>(stat, reconstructed, origin, len);
  psz::utils::print_metrics_cross<T>(stat, compressed_bytes, false);

  auto stat_auto_lag1 = new psz_summary;
  psz::utils::verify_data<T>(stat_auto_lag1, origin, origin + 1, len - 1);
  auto stat_auto_lag2 = new psz_summary;
  psz::utils::verify_data<T>(stat_auto_lag2, origin, origin + 2, len - 2);

  psz::utils::print_metrics_auto(
      &stat_auto_lag1->score_coeff, &stat_auto_lag2->score_coeff);

  if (from_device) {
    if (reconstructed) sycl::free(reconstructed, q_ct1);
    if (origin) sycl::free(origin, q_ct1);
  }

  delete stat, delete stat_auto_lag1, delete stat_auto_lag2;
}

template <typename T>
static void view(
    psz_header* header, memobj<T>* xdata, memobj<T>* cmp,
    string const& compare)
{
  auto len = psz_utils::uncompressed_len(header);
  auto compressd_bytes = psz_utils::filesize(header);

  auto compare_on_gpu = [&]() {
    cmp->control({MallocHost, Malloc})
        ->file(compare.c_str(), FromFile)
        ->control({H2D});

    pszcxx_evaluate_quality_gpu(
        xdata->dptr(), cmp->dptr(), len, compressd_bytes);
    // cmp->control({FreeHost, Free});
  };

  auto compare_on_cpu = [&]() {
    cmp->control({MallocHost})->file(compare.c_str(), FromFile);
    xdata->control({D2H});
    pszcxx_evaluate_quality_cpu(
        xdata->hptr(), cmp->hptr(), len, compressd_bytes);
    // cmp->control({FreeHost});
  };

  if (compare != "") {
    auto gb = 1.0 * sizeof(T) * len / 1e9;
#warning \
    "[psz::dpcpp::todo] DPL impl of quality assessment not working; revert to CPU version"
    // if (gb < 0.8)
    //   compare_on_gpu();
    // else
    compare_on_cpu();
  }
}
}  // namespace psz
