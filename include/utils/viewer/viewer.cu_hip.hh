#ifndef BF8291FB_FC70_424B_B53C_94C1D8DDAC5A
#define BF8291FB_FC70_424B_B53C_94C1D8DDAC5A

#include "stat/compare/compare.thrust.hh"

namespace psz {

template <typename T>
static void eval_dataquality_gpu(
    T* reconstructed, T* origin, size_t len, size_t compressed_bytes = 0)
{
  // cross
  auto stat_x = new psz_summary;
  psz::thrustgpu_assess_quality<T>(stat_x, reconstructed, origin, len);
  print_metrics_cross<T>(stat_x, compressed_bytes, true);

  auto stat_auto_lag1 = new psz_summary;
  psz::thrustgpu_assess_quality<T>(
      stat_auto_lag1, origin, origin + 1, len - 1);
  auto stat_auto_lag2 = new psz_summary;
  psz::thrustgpu_assess_quality<T>(
      stat_auto_lag2, origin, origin + 2, len - 2);

  print_metrics_auto(
      &stat_auto_lag1->score.coeff, &stat_auto_lag2->score.coeff);

  delete stat_x, delete stat_auto_lag1, delete stat_auto_lag2;
}

template <typename T>
static void eval_dataquality_cpu(
    T* _d1, T* _d2, size_t len, size_t compressed_bytes = 0,
    bool from_device = true)
{
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
    GpuMallocHost(&reconstructed, bytes);
    GpuMallocHost(&origin, bytes);
    GpuMemcpy(reconstructed, _d1, bytes, GpuMemcpyD2H);
    GpuMemcpy(origin, _d2, bytes, GpuMemcpyD2H);
  }
  cusz::verify_data<T>(stat, reconstructed, origin, len);
  print_metrics_cross<T>(stat, compressed_bytes, false);

  auto stat_auto_lag1 = new psz_summary;
  cusz::verify_data<T>(stat_auto_lag1, origin, origin + 1, len - 1);
  auto stat_auto_lag2 = new psz_summary;
  cusz::verify_data<T>(stat_auto_lag2, origin, origin + 2, len - 2);

  print_metrics_auto(
      &stat_auto_lag1->score.coeff, &stat_auto_lag2->score.coeff);

  if (from_device) {
    if (reconstructed) GpuFreeHost(reconstructed);
    if (origin) GpuFreeHost(origin);
  }

  delete stat, delete stat_auto_lag1, delete stat_auto_lag2;
}

template <typename T>
static void view(
    psz_header* header, pszmem_cxx<T>* xdata, pszmem_cxx<T>* cmp,
    string const& compare)
{
  auto len = psz_utils::uncompressed_len(header);
  auto compressd_bytes = psz_utils::filesize(header);

  auto compare_on_gpu = [&]() {
    cmp->control({MallocHost, Malloc})
        ->file(compare.c_str(), FromFile)
        ->control({H2D});

    eval_dataquality_gpu(xdata->dptr(), cmp->dptr(), len, compressd_bytes);
    // cmp->control({FreeHost, Free});
  };

  auto compare_on_cpu = [&]() {
    cmp->control({MallocHost})->file(compare.c_str(), FromFile);
    cmp->control({D2H});
    eval_dataquality_cpu(xdata->hptr(), cmp->hptr(), len, compressd_bytes);
    // cmp->control({FreeHost});
  };

  if (compare != "") {
    auto gb = 1.0 * sizeof(T) * len / 1e9;
    if (gb < 0.8)
      compare_on_gpu();
    else
      compare_on_cpu();
  }
}
}  // namespace psz

#endif /* BF8291FB_FC70_424B_B53C_94C1D8DDAC5A */
