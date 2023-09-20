/**
 * @file viewer.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-04-09
 * @deprecated 0.3.2
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef C6EF99AE_F0D7_485B_ADE4_8F55666CA96C
#define C6EF99AE_F0D7_485B_ADE4_8F55666CA96C

#include <algorithm>
#include <iomanip>

#include "cusz/type.h"
#include "header.h"
#include "mem/memseg_cxx.hh"
#include "port.hh"
#include "stat/compare/compare.thrust.hh"
#include "verify.hh"

namespace psz {

template <typename Data>
static void print_metrics_cross(
    cusz_stats* s, size_t compressed_bytes = 0, bool gpu_checker = false)
{
  auto checker = (not gpu_checker) ? string("(using CPU checker)")
                                   : string("(using GPU checker)");
  auto bytes = (s->len * sizeof(Data) * 1.0);

  auto println = [](const char* s, double n1, double n2, double n3,
                    double n4) {
    printf("  %-10s %16.8g %16.8g %16.8g %16.8g\n", s, n1, n2, n3, n4);
  };
  auto printhead = [](const char* s1, const char* s2, const char* s3,
                      const char* s4, const char* s5) {
    printf(
        "  \e[1m\e[31m%-10s %16s %16s %16s %16s\e[0m\n", s1, s2, s3, s4, s5);
  };

  auto is_fp =
      std::is_same<Data, float>::value or std::is_same<Data, double>::value
          ? const_cast<char*>("yes")
          : const_cast<char*>("no");
  printf("\nquality metrics %s:\n", checker.c_str());

  printhead("", "data-len", "data-byte", "fp-type?", "");
  printf("  %-10s %16zu %16lu %16s\n", "", s->len, sizeof(Data), is_fp);

  printhead("", "min", "max", "rng", "std");
  println("origin", s->odata.min, s->odata.max, s->odata.rng, s->odata.std);
  println("eb-lossy", s->xdata.min, s->xdata.max, s->xdata.rng, s->xdata.std);

  printhead("", "abs-val", "abs-idx", "pw-rel", "VS-RNG");
  println(
      "max-error", s->max_err.abs, s->max_err.idx, s->max_err.pwrrel,
      s->max_err.rel);

  printhead("", "CR", "NRMSE", "cross-cor", "PSNR");
  println(
      "metrics", bytes / compressed_bytes, s->score.NRMSE, s->score.coeff,
      s->score.PSNR);

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

template <typename T>
static void eval_dataquality_gpu(
    T* reconstructed, T* origin, size_t len, size_t compressed_bytes = 0)
{
  // cross
  auto stat_x = new cusz_stats;
  psz::thrustgpu::thrustgpu_assess_quality<T>(
      stat_x, reconstructed, origin, len);
  print_metrics_cross<T>(stat_x, compressed_bytes, true);

  auto stat_auto_lag1 = new cusz_stats;
  psz::thrustgpu::thrustgpu_assess_quality<T>(
      stat_auto_lag1, origin, origin + 1, len - 1);
  auto stat_auto_lag2 = new cusz_stats;
  psz::thrustgpu::thrustgpu_assess_quality<T>(
      stat_auto_lag2, origin, origin + 2, len - 2);

  print_metrics_auto(
      &stat_auto_lag1->score.coeff, &stat_auto_lag2->score.coeff);
}

template <typename T>
static void eval_dataquality_cpu(
    T* _d1, T* _d2, size_t len, size_t compressed_bytes = 0,
    bool from_device = true)
{
  auto stat = new cusz_stats;
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

  auto stat_auto_lag1 = new cusz_stats;
  cusz::verify_data<T>(stat_auto_lag1, origin, origin + 1, len - 1);
  auto stat_auto_lag2 = new cusz_stats;
  cusz::verify_data<T>(stat_auto_lag2, origin, origin + 2, len - 2);

  print_metrics_auto(
      &stat_auto_lag1->score.coeff, &stat_auto_lag2->score.coeff);

  if (from_device) {
    if (reconstructed) GpuFreeHost(reconstructed);
    if (origin) GpuFreeHost(origin);
  }
}

template <typename T>
static void view(
    cusz_header* header, pszmem_cxx<T>* xdata, pszmem_cxx<T>* cmp,
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

namespace cusz {

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

  static double get_total_time(timerecord_t r)
  {
    double total = 0.0;
    std::for_each(r->begin(), r->end(), [&](TimeRecordTuple t) {
      return total += std::get<1>(t);
    });
    return total;
  }

  static void view_compression(
      timerecord_t r, size_t bytes, size_t compressed_bytes = 0)
  {
#warning "[TODO] view_compression is deprecated, use review_compression(...) instead"
    auto report_cr = [&]() {
      auto cr = 1.0 * bytes / compressed_bytes;
      if (compressed_bytes != 0)
        printf("  %-*s %.2f\n", 20, "compression ratio", cr);
    };

    TimeRecord reflow;

    {  // reflow
      TimeRecordTuple book_tuple;

      auto total_time = get_total_time(r);
      auto subtotal_time = total_time;

      for (auto& i : *r) {
        auto item = std::string(std::get<0>(i));
        if (item == "book") {
          book_tuple = i;
          subtotal_time -= std::get<1>(i);
        }
        else {
          reflow.push_back(i);
        }
      }
      reflow.push_back({const_cast<const char*>("(subtotal)"), subtotal_time});
      printf("\e[2m");
      reflow.push_back(book_tuple);
      reflow.push_back({const_cast<const char*>("(total)"), total_time});
      printf("\e[0m");
    }

    printf("\n(c) COMPRESSION REPORT\n");
    report_cr();

    psz_utils::println_throughput_tablehead();
    for (auto& i : reflow)
      psz_utils::println_throughput(std::get<0>(i), std::get<1>(i), bytes);

    printf("\n");
  }

  // [TODO] not a part of TimeRecordViewer
  static void view_cr(pszheader* h)
  {
    // [TODO] put error status
    if (h->dtype != F4 and h->dtype != F8)
      cout << "[psz::log::fatal_error] original length is is zero." << endl;

    auto comp_bytes = [&]() {
      auto END = sizeof(h->entry) / sizeof(h->entry[0]);
      return h->entry[END - 1];
    };

    auto sizeof_T = [&]() { return (h->dtype == F4 ? 4 : 8); };
    auto uncomp_bytes = h->x * h->y * h->z * sizeof_T();
    auto fieldsize = [&](auto FIELD) {
      return h->entry[FIELD + 1] - h->entry[FIELD];
    };
    auto __print = [&](auto str, auto num) {
      cout << "  ";
      cout << std::left;
      cout << std::setw(28) << str;
      cout << std::right;
      cout << std::setw(10) << num;
      cout << '\n';
    };
    auto __print_perc = [&](auto str, auto num) {
      auto perc = num * 100.0 / comp_bytes();
      cout << "  ";
      cout << std::left;
      cout << std::setw(28) << str;
      cout << std::right;
      cout << std::setw(10) << num;
      cout << std::setw(10) << std::setprecision(3) << std::fixed << perc
           << "%\n";
    };
    auto __newline = []() { cout << '\n'; };

    if (comp_bytes() != 0) {
      auto cr = 1.0 * uncomp_bytes / comp_bytes();
      __newline();
      __print("psz::comp::review::CR", cr);
    }
    else {
      cout << "[psz::log::fatal_error] compressed len is zero." << endl;
    }

    __print("original::bytes", uncomp_bytes);
    __print("original::bytes", uncomp_bytes);
    __print("compressed::bytes", comp_bytes());
    __newline();
    __print_perc("compressed::total::bytes", comp_bytes());
    printf("  ------------------------\n");
    __print_perc("compressed::HEADER::bytes", sizeof(pszheader));
    __print_perc("compressed::ANCHOR::bytes", fieldsize(pszheader::ANCHOR));
    __print_perc("compressed::VLE::bytes", fieldsize(pszheader::VLE));
    __print_perc("compressed::SPFMT::bytes", fieldsize(pszheader::SPFMT));
    __newline();
    __print(
        "compressed::ANCHOR:::len", fieldsize(pszheader::ANCHOR) / sizeof_T());
    __print(
        "compressed::OUTLIER:::len",
        fieldsize(pszheader::SPFMT) / (sizeof_T() + sizeof(uint32_t)));
  }

  static void view_timerecord(timerecord_t r, pszheader* h)
  {
    auto sizeof_T = [&]() { return (h->dtype == F4 ? 4 : 8); };
    auto uncomp_bytes = h->x * h->y * h->z * sizeof_T();

    TimeRecord reflow;

    {  // reflow
      TimeRecordTuple book_tuple;

      auto total_time = get_total_time(r);
      auto subtotal_time = total_time;

      for (auto& i : *r) {
        auto item = std::string(std::get<0>(i));
        if (item == "book") {
          book_tuple = i;
          subtotal_time -= std::get<1>(i);
        }
        else {
          reflow.push_back(i);
        }
      }
      reflow.push_back({const_cast<const char*>("(subtotal)"), subtotal_time});
      printf("\e[2m");
      reflow.push_back(book_tuple);
      reflow.push_back({const_cast<const char*>("(total)"), total_time});
      printf("\e[0m");
    }

    psz_utils::println_throughput_tablehead();
    for (auto& i : reflow)
      psz_utils::println_throughput(
          std::get<0>(i), std::get<1>(i), uncomp_bytes);

    printf("\n");
  }

  static void review_compression(timerecord_t r, pszheader* h)
  {
    printf("\n(c) COMPRESSION REPORT\n");
    view_cr(h);
    view_timerecord(r, h);
  }

  static void view_decompression(timerecord_t r, size_t bytes)
  {
    printf("\n(d) deCOMPRESSION REPORT\n");

    auto total_time = get_total_time(r);
    (*r).push_back({const_cast<const char*>("(total)"), total_time});

    psz_utils::println_throughput_tablehead();
    for (auto& i : *r)
      psz_utils::println_throughput(std::get<0>(i), std::get<1>(i), bytes);

    printf("\n");
  }
};

}  // namespace cusz

#endif /* C6EF99AE_F0D7_485B_ADE4_8F55666CA96C */
