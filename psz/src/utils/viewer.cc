#include "utils/viewer.hh"

#include <cstddef>

#include "compressor.hh"
#include "cusz.h"
#include "cusz/type.h"
#include "header.h"
#include "port.hh"
#include "stat/compare.hh"

float get_throughput(float milliseconds, size_t nbyte)
{
  auto GiB = 1.0 * 1024 * 1024 * 1024;
  auto seconds = milliseconds * 1e-3;
  return nbyte / GiB / seconds;
}

void println_throughput(const char* s, float timer, size_t _nbyte)
{
  if (timer == 0.0) return;
  auto t = get_throughput(timer, _nbyte);
  printf("%-12s %'14f %'11.2f\n", s, timer, t);
};

void println_throughput_tablehead()
{
  printf(
      "\n\e[1m\e[31m%-12s %14s %11s\e[0m\n",  //
      const_cast<char*>("kernel"),            //
      const_cast<char*>("time, ms"),          //
      const_cast<char*>("GiB/s")              //
  );
}

double get_total_time(psz::timerecord_t r)
{
  double total = 0.0;
  std::for_each(
      r->begin(), r->end(), [&](psz::TimeRecordTuple t) { return total += std::get<1>(t); });
  return total;
}

void* capi_psz_make_timerecord() { return (void*)new psz::TimeRecord; }

void capi_psz_review_comp_time_breakdown(void* _r, psz_header* h)
{
  auto sizeof_T = [&]() { return (h->dtype == F4 ? 4 : 8); };
  auto uncomp_bytes = h->x * h->y * h->z * sizeof_T();

  auto r = (psz::timerecord_t)_r;

  psz::TimeRecord reflow;

  {  // reflow
    psz::TimeRecordTuple book_tuple;

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

  println_throughput_tablehead();
  for (auto& i : reflow) println_throughput(std::get<0>(i), std::get<1>(i), uncomp_bytes);
}

string const psz_report_query_pred(psz_predtype const p)
{
  const std::unordered_map<psz_predtype const, std::string const> lut = {
      {psz_predtype::Lorenzo, "Lorenzo"},
      {psz_predtype::LorenzoZigZag, "Lrz-ZigZag"},
      {psz_predtype::LorenzoProto, "Lrz-Proto"},
      {psz_predtype::Spline, "Spline"},
  };
  return lut.at(p);
};

string const psz_report_query_hist(psz_histotype const h)
{
  const std::unordered_map<psz_histotype const, std::string const> lut = {
      {psz_histotype::HistogramGeneric, "Hist-generic"},
      {psz_histotype::HistogramSparse, "Hist-sparse"},
      {psz_histotype::NullHistogram, "Hist-(null)"},
  };
  return lut.at(h);
};

string const psz_report_query_codec1(psz_codectype const c)
{
  const std::unordered_map<psz_codectype const, std::string const> lut = {
      {psz_codectype::Huffman, "Huffman"},
      {psz_codectype::FZGPUCodec, "FZGPU-Codec"},
  };
  return lut.at(c);
};

void capi_psz_review_comp_time_from_header(psz_header* h)
{
  printf("\n");
  // [TODO] put error status
  if (h->dtype != F4 and h->dtype != F8)
    cout << "[psz::error] original length is is zero." << endl;

  auto comp_bytes = [&]() {
    auto END = sizeof(h->entry) / sizeof(h->entry[0]);
    return h->entry[END - 1];
  };

  auto sizeof_T = [&]() { return (h->dtype == F4 ? 4 : 8); };
  auto uncomp_bytes = h->x * h->y * h->z * sizeof_T();
  auto fieldsize = [&](auto FIELD) { return h->entry[FIELD + 1] - h->entry[FIELD]; };
  auto __print = [&](auto str, auto num) {
    // cout << "  ";
    cout << std::left;
    cout << std::setw(25) << string(str);
    cout << std::right;
    cout << std::setw(14) << num;
    cout << '\n';
  };
  auto __print_perc = [&](auto str, auto num) {
    auto perc = num * 100.0 / comp_bytes();
    // cout << "  ";
    cout << std::left;
    cout << std::setw(25) << string(str);
    cout << std::right;
    cout << std::setw(14) << num;
    cout << std::setw(10) << std::setprecision(3) << std::fixed << perc << "%\n";
  };
  auto __newline = []() { cout << '\n'; };

  auto n_outlier = fieldsize(PSZHEADER_SPFMT) / (sizeof_T() + sizeof(uint32_t));

  __print("logging::predictor", psz_report_query_pred(h->pred_type));
  __print("logging::histogram", psz_report_query_hist(h->hist_type));
  __print("logging::codec1", psz_report_query_codec1(h->codec1_type));
  __print("logging::max", h->logging_max);
  __print("logging::min", h->logging_min);
  __print("logging::range", h->logging_max - h->logging_min);
  __print("logging::mode", h->mode == Rel ? "Rel" : "Abs");
  __print("logging::input_eb", h->user_input_eb);
  __print("logging::final_eb", h->eb);
  printf("--------------------------------------------------\n");
  if (comp_bytes() != 0) {
    auto cr = 1.0 * uncomp_bytes / comp_bytes();
    __print("data::comp_metric::CR", cr);
  }
  else {
    cout << "[psz::fatal] compressed len is zero." << endl;
  }
  __print("data::original_bytes", uncomp_bytes);
  __print_perc("data::comp_bytes", comp_bytes());
  printf("--------------------------------------------------\n");
  __print_perc("file::header::bytes", fieldsize(PSZHEADER_HEADER));
  __print_perc("file::anchor::bytes", fieldsize(PSZHEADER_ANCHOR));
  __print_perc("file::encoded::bytes", fieldsize(PSZHEADER_ENCODED));
  __print_perc("file::outlier::bytes", fieldsize(PSZHEADER_SPFMT));
  printf("--------------------------------------------------\n");
  __print("file::anchor:::number", fieldsize(PSZHEADER_ANCHOR) / sizeof_T());
  __print("file::outlier:::number", n_outlier);
}

void println_text_v2(string const prefix, string const kw, string const text)
{
  std::string combined = prefix + "::\e[1m\e[31m" + kw + "\e[0m";
  printf("%-*s%*s\n", 36, combined.c_str(), 16, text.c_str());
}

void psz_review_decomp_time_from_header(psz_header* h)
{
  println_text_v2("component", "predictor", psz_report_query_pred(h->pred_type));
  println_text_v2("component", "histogram", psz_report_query_hist(h->hist_type));
  println_text_v2("component", "codec1", psz_report_query_codec1(h->codec1_type));
}

void capi_psz_review_compression(void* r, psz_header* h)
{
  printf("\n(c) COMPRESSION REPORT\n");
  psz_review_comp_time_from_header(h);
  psz_review_comp_time_breakdown((psz::timerecord_t)r, h);
}

void capi_psz_review_decompression(void* r, size_t bytes)
{
  printf(
      "\n\e[1m\e[31m"
      "REPORT::deCOMPRESSION::TIME"
      "\e[0m"
      "\n");

  auto total_time = get_total_time((psz::timerecord_t)r);
  (*(psz::timerecord_t)r).push_back({const_cast<const char*>("(total)"), total_time});

  println_throughput_tablehead();
  for (auto& i : *(psz::timerecord_t)r) println_throughput(std::get<0>(i), std::get<1>(i), bytes);

  printf("\n");
}

// TODO revise name
template <typename T>
void psz::analysis::print_metrics_cross(psz_statistics* s, size_t comp_bytes, bool gpu_checker)
{
  auto checker = (not gpu_checker) ? string("CPU-checker") : string("GPU-checker");

  auto bytes = (s->len * sizeof(T) * 1.0);
  auto is_fp = std::is_floating_point_v<T> ? const_cast<char*>("yes") : const_cast<char*>("no");

  double to_KiB = 1024.0, to_MiB = 1024.0 * to_KiB, to_GiB = 1024.0 * to_MiB;

  bool hlcolor_red = true;
  auto println_v2 = [&](string const prefix, string const kw, double n1,
                        bool rounding_to_4th = false) {
    std::string combined =
        prefix + (hlcolor_red ? "::\e[1m\e[31m" : "::\e[1m\e[34m") + kw + "\e[0m";
    if (not rounding_to_4th)
      printf("%-*s%16.8g\n", 36, combined.c_str(), n1);
    else
      printf("%-*s%16.4f\n", 36, combined.c_str(), n1);
  };

  auto println_segline_solid = []() { printf("---------------------------------------\n"); };
  auto println_segline_dotted = []() { printf(".......................................\n"); };

  string dtype_text = !is_fp ? "non-fp" : sizeof(T) == 4 ? "fp32" : "fp64";

  // TODO (ad hoc) component-checker follow psz_review_decomp_time_from_header
  println_text_v2("component", "checker", checker);
  println_segline_solid();
  println_v2("data", "length", s->len);
  println_text_v2("data", "dtype", dtype_text);
  println_segline_dotted();
  println_v2("data", "original_bytes", bytes);
  println_v2("data", "original_KiB", bytes / to_KiB, true);
  println_v2("data", "original_MiB", bytes / to_MiB, true);
  println_v2("data", "original_GiB", bytes / to_GiB, true);
  println_segline_dotted();
  println_v2("data", "comp_bytes", comp_bytes);
  println_v2("data", "compressed_KiB", comp_bytes / to_KiB, true);
  println_v2("data", "compressed_MiB", comp_bytes / to_MiB, true);
  println_v2("data", "compressed_GiB", comp_bytes / to_GiB, true);
  println_segline_dotted();
  println_v2("data_original", "min", s->odata.min);
  println_v2("data_original", "max", s->odata.max);
  println_v2("data_original", "rng", s->odata.rng);
  println_v2("data_original", "avg", s->odata.avg);
  println_v2("data_original", "std", s->odata.std);
  println_segline_dotted();
  println_v2("data_decompressed", "min", s->xdata.min);
  println_v2("data_decompressed", "max", s->xdata.max);
  println_v2("data_decompressed", "rng", s->xdata.rng);
  println_v2("data_decompressed", "avg", s->xdata.avg);
  println_v2("data_decompressed", "std", s->xdata.std);
  println_segline_solid();
  hlcolor_red = false;
  println_v2("comp_metric", "CR", bytes / comp_bytes);
  println_v2("comp_metric", "bitrate", 32.0 / (bytes / comp_bytes));
  hlcolor_red = true;
  println_v2("comp_metric", "NRMSE", s->score_NRMSE);
  println_v2("comp_metric", "coeff", s->score_coeff);
  hlcolor_red = false;
  println_v2("comp_metric", "PSNR", s->score_PSNR);
  hlcolor_red = true;
  println_segline_dotted();
  println_v2("data_max_error", "index", s->max_err_idx);
  hlcolor_red = false;
  println_v2("data_max_error", "val", s->max_err_abs);
  println_v2("data_max_error", "vs_rng", s->max_err_rel);
  hlcolor_red = true;
  println_segline_dotted();
}

void psz::analysis::print_metrics_auto(double* lag1_cor, double* lag2_cor)
{
  auto println_v2 = [](string const prefix, string const kw, double n1) {
    std::string combined = prefix + "::\e[1m\e[31m" + kw + "\e[0m";
    printf("%-*s%16.8g\n", 36, combined.c_str(), n1);
  };

  println_v2("autocorrelation", "lag1", *lag1_cor);
  println_v2("autocorrelation", "lag2", *lag2_cor);
}

template <typename T>
void psz::analysis::CPU_evaluate_quality_and_print(
    T* _d1, T* _d2, size_t len, size_t comp_bytes, bool from_device)
{
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
    cudaMallocHost(&reconstructed, bytes);
    cudaMallocHost(&origin, bytes);
    cudaMemcpy(reconstructed, _d1, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(origin, _d2, bytes, cudaMemcpyDeviceToHost);
  }
  psz::analysis::assess_quality<SEQ, T>(stat, reconstructed, origin, len);
  psz::analysis::print_metrics_cross<T>(stat, comp_bytes, false);

  auto stat_auto_lag1 = new psz_statistics;
  psz::analysis::assess_quality<SEQ, T>(stat_auto_lag1, origin, origin + 1, len - 1);
  auto stat_auto_lag2 = new psz_statistics;
  psz::analysis::assess_quality<SEQ, T>(stat_auto_lag2, origin, origin + 2, len - 2);

  psz::analysis::print_metrics_auto(&stat_auto_lag1->score_coeff, &stat_auto_lag2->score_coeff);

  if (from_device) {
    if (reconstructed) cudaFreeHost(reconstructed);
    if (origin) cudaFreeHost(origin);
  }

  delete stat, delete stat_auto_lag1, delete stat_auto_lag2;
}

template void psz::analysis::print_metrics_cross<float>(psz_statistics*, size_t, bool);
template void psz::analysis::print_metrics_cross<double>(psz_statistics*, size_t, bool);