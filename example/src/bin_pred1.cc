// Run predictor only: forward predictor + reversed predictor; analysis-only
// path (no lossless coding).
//
// Usage: PROG data_file x y z abs_eb [predictor] [radius] [EXPORT]
//   predictor: spline (default), lrz, lrz-zz, lrz-proto
//   radius:    integer, default 128
//   EXPORT:    if present, dump ectrl as u2 to data_file.pred_<predictor>.ectrl.u2
//              and (for Spline) anchor values to data_file.pred_spline.anchor.f4

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

#include "compressor.hh"
#include "cusz.h"
#include "kernel.hh"
#include "ptb.hh"
#include "test_lib/pred_metrics.hh"

namespace utils = _ptb::utils;
using _ptb::make_view;
using Toggle = psz::Toggle;

template <typename T, Toggle ZigZag>
using GPU_x_lorenzo_nd =
    psz::module::GPU_x_lorenzo_nd<psz::PredictorTyping<T>, psz::PredictorFeature<ZigZag>>;

int main(int argc, char** argv)
{
  if (argc < 6) {
    printf("USAGE: %s data_file x y z abs_eb [predictor] [radius] [EXPORT]\n", argv[0]);
    printf("  predictor: spline (default), lrz, lrz-zz, lrz-proto\n");
    printf("  EXPORT: dump ectrl (and anchor for Spline) to files\n");
    return 1;
  }

  std::string fname(argv[1]);
  size_t x = std::stoul(argv[2]);
  size_t y = std::stoul(argv[3]);
  size_t z = std::stoul(argv[4]);
  double abs_eb = std::stod(argv[5]);
  size_t len = x * y * z;

  std::string pred_name = (argc >= 7) ? argv[6] : "spline";
  int radius = 128;
  bool do_export = false;
  for (int i = 7; i < argc; i++) {
    std::string a(argv[i]);
    if (a == "EXPORT")
      do_export = true;
    else
      radius = std::stoi(a);
  }

  psz_predictor pred_type = psz_predictor::Spline;
  int spline_variant = 0;  // 0 = y25 (2D+3D), 1 = y24 (lean 3D)
  if (pred_name == "lrz" or pred_name == "lorenzo")
    pred_type = psz_predictor::Lorenzo;
  else if (pred_name == "lrz-zz" or pred_name == "lorenzo-zigzag")
    pred_type = psz_predictor::LorenzoZigZag;
  else if (pred_name == "lrz-proto" or pred_name == "lorenzo-proto")
    pred_type = psz_predictor::LorenzoProto;
  else if (pred_name == "spline-y24")
    spline_variant = 1;

  auto h_data = MAKE_UNIQUE_HOST(float, len);
  auto d_data = MAKE_UNIQUE_DEVICE(float, len);
  utils::fromfile_or_die(fname, h_data.get(), len);
  memcpy_allkinds<H2D>(d_data.get(), h_data.get(), len);

  auto d_xdata = MAKE_UNIQUE_DEVICE(float, len);
  memset_device(d_xdata.get(), len);

  auto h_xdata = MAKE_UNIQUE_HOST(float, len);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  psz_len len3{x, y, z};
  auto manager = psz_create_resource_manager(
      F4, {x, y, z}, {pred_type, HistGeneric, HF, CodecNull}, (void*)stream);

  manager->header->rc.eb = abs_eb;
  manager->header->rc.mode = Abs;
  manager->header->rc.radius = radius;
  manager->header->user_input_eb = abs_eb;
  manager->spline_variant = spline_variant;  // 0 = y25, 1 = y24

  using E = uint16_t;
  using M = uint32_t;
  using PPL = psz::compression_pipeline<float, E>;
  using Buf = psz_buf<float, E>;

  auto mem = (Buf*)manager->buf;
  auto h_hist = MAKE_UNIQUE_HOST(uint32_t, manager->bklen);

  auto status = PPL::compress_analysis(manager, mem, d_data.get(), h_hist.get(), (void*)stream);
  if (status != PSZ_SUCCESS) {
    printf("[pred-study] predictor-analysis failed, status=%d\n", status);
    psz_release_resource(manager);
    cudaStreamDestroy(stream);
    return 2;
  }
  cudaStreamSynchronize(stream);

  // reverse predictor to reconstruct and measure quality
  memset_device(d_xdata.get(), len);
  if (manager->header->splen != 0) {
    psz::module::GPU_scatter<float, M>::kernel_v2(
        (_ptb::compact_cell<float, M>*)mem->outlier2_validx_d(), manager->header->splen,
        d_xdata.get(), (void*)stream);
  }

  if (pred_type == psz_predictor::Lorenzo)
    GPU_x_lorenzo_nd<float, Toggle::ZigZag_Off>::kernel(
        make_view(mem->eq_d(), len3), make_view(d_xdata.get(), len3),
        make_view(d_xdata.get(), len3), abs_eb, manager->header->rc.radius, (void*)stream);
  else if (pred_type == psz_predictor::LorenzoZigZag)
    GPU_x_lorenzo_nd<float, Toggle::ZigZag_On>::kernel(
        make_view(mem->eq_d(), len3), make_view(d_xdata.get(), len3),
        make_view(d_xdata.get(), len3), abs_eb, manager->header->rc.radius, (void*)stream);
  else if (pred_type == psz_predictor::LorenzoProto)
    psz::module::GPU_PROTO_x_lorenzo_nd<float, E>::kernel(
        mem->eq_d(), d_xdata.get(), d_xdata.get(), len3, abs_eb * 2, 1 / (abs_eb * 2),
        manager->header->rc.radius, (void*)stream);
  else if (pred_type == psz_predictor::Spline)
    psz::module::GPU_x_spline<float, E>::kernel_v1(
        make_view(mem->anchor_d(), mem->anchor_len3()),
        make_view(mem->eq_d(), mem->eq_len3()),
        make_view(d_xdata.get(), mem->eq_len3()), d_xdata.get(), abs_eb,
        manager->header->rc.radius, manager->header->intp_param, (void*)stream,
        spline_variant == 1 ? SplineVariant::y24 : SplineVariant::y25);

  cudaStreamSynchronize(stream);
  memcpy_allkinds<D2H>(h_xdata.get(), d_xdata.get(), len);

  auto m = psz_test::compute_metrics(
      h_data.get(), h_xdata.get(), len, pred_name, abs_eb, radius, manager->header->splen);
  psz_test::print_human(m);

  if (do_export) {
    auto h_eq = MAKE_UNIQUE_HOST(uint16_t, len);
    memcpy_allkinds<D2H>(h_eq.get(), mem->eq_d(), len);
    std::string eq_out = fname + ".pred_" + pred_name + ".ectrl.u2";
    utils::tofile(eq_out, h_eq.get(), len);
    printf("[pred-study] ectrl written to: %s\n", eq_out.c_str());

    if (pred_type == psz_predictor::Spline) {
      auto anchor_len = mem->anchor_len();
      auto h_anchor = MAKE_UNIQUE_HOST(float, anchor_len);
      memcpy_allkinds<D2H>(h_anchor.get(), mem->anchor_d(), anchor_len);
      std::string anc_out = fname + ".pred_spline.anchor.f4";
      utils::tofile(anc_out, h_anchor.get(), anchor_len);
      printf("[pred-study] anchor(%zu) written to: %s\n", anchor_len, anc_out.c_str());
    }
  }

  psz_release_resource(manager);
  cudaStreamDestroy(stream);
  return 0;
}
