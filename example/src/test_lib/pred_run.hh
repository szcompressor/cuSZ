#ifndef PSZ_TEST_LIB_PRED_RUN_HH
#define PSZ_TEST_LIB_PRED_RUN_HH

#include <cstdio>
#include <memory>
#include <string>

#include "compressor.hh"
#include "cusz.h"
#include "detail/composite.hh"
#include "kernel.hh"
#include "mem/view.hh"
#include "pred_args.hh"
#include "pred_metrics.hh"
#include "utils/io.hh"

using _ptb::make_const_view;
using _ptb::make_view;

namespace psz_test {

namespace _utils = _ptb::utils;
using _Toggle = psz::Toggle;

template <typename T, _Toggle ZigZag>
using GPU_x_lorenzo_nd =
    psz::module::GPU_x_lorenzo_nd<psz::PredictorTyping<T>, psz::PredictorFeature<ZigZag>>;

// Map a predictor name to the cusz `psz_predictor` enum value. The
// `out_spline_v` channel is reserved for the spl-vN side-channel used by
// bin_pred_xv on the spline-evolution branch; on develop it stays 0.
// Returns false on unknown name.
inline bool resolve_predictor(const std::string& name, psz_predictor& out_pred, int& out_spline_v)
{
  out_spline_v = 0;
  if (name == "lrz" or name == "lorenzo") { out_pred = psz_predictor::Lorenzo; }
  else if (name == "lrz-zz" or name == "lorenzo-zigzag") {
    out_pred = psz_predictor::LorenzoZigZag;
  }
  else if (name == "lrz-proto" or name == "lorenzo-proto") {
    out_pred = psz_predictor::LorenzoProto;
  }
  else if (name == "spl" or name == "spline") {
    out_pred = psz_predictor::Spline;
  }
  else if (name == "spl-y24" or name == "spline-y24") {
    out_pred = psz_predictor::Spline;
    out_spline_v = 24;
  }
  else if (name == "spl-y25" or name == "spline-y25") {
    out_pred = psz_predictor::Spline;
    out_spline_v = 25;
  }
  else
    return false;
  return true;
}

class PredRun {
 public:
  using E = uint16_t;
  using M = uint32_t;
  using PPL = psz::compression_pipeline<float, E>;
  using Buf = psz_buf<float, E>;

  // PredRun mutates `args.eb` for rel-mode
  // Downstream readers (bin_pred, bin_pred_xv, pred_xv) see the absolute eb.
  PredArgs args;
  size_t len = 0;
  psz_len len3{0, 0, 0};
  psz_predictor pred_type = psz_predictor::Spline;
  int spline_v_check = 0;

  // -- session state -----
  cudaStream_t stream{};
  psz_resource* manager_ = nullptr;
  Buf* mem = nullptr;
  std::unique_ptr<float, decltype(&cudaFree)> d_data{nullptr, cudaFree};
  std::unique_ptr<float, decltype(&cudaFree)> d_xdata{nullptr, cudaFree};
  std::unique_ptr<float[]> h_data;
  std::unique_ptr<float[]> h_xdata;
  std::unique_ptr<uint32_t[]> h_hist;

  // exit-code carrier (when setup() returns nonzero, this holds the code)
  int exit_code = 0;

  explicit PredRun(const PredArgs& a) : args(a) {}

  ~PredRun()
  {
    if (manager_) psz_release_resource(manager_);
    if (stream) cudaStreamDestroy(stream);
  }

  // -- 1. resolve args, allocate, load data, initialize manager. ---------
  int setup()
  {
    if (!resolve_predictor(args.predictor, pred_type, spline_v_check)) {
      fprintf(stderr, "[pred-study] unknown predictor: %s\n", args.predictor.c_str());
      return 2;
    }

    len = args.total_len();
    len3 = psz_len{args.x, args.y, args.z};

    // host buffers
    h_data = std::unique_ptr<float[]>(new float[len]);
    h_xdata = std::unique_ptr<float[]>(new float[len]);

    // load original data
    _utils::fromfile(args.fname, h_data.get(), len);

    // device buffers (raw via cudaMalloc; unique_ptr<cudaFree> owns them)
    float* p = nullptr;
    cudaMalloc(&p, sizeof(float) * len);
    d_data.reset(p);
    p = nullptr;
    cudaMalloc(&p, sizeof(float) * len);
    d_xdata.reset(p);
    cudaMemcpy(d_data.get(), h_data.get(), sizeof(float) * len, cudaMemcpyHostToDevice);
    cudaMemset(d_xdata.get(), 0, sizeof(float) * len);

    cudaStreamCreate(&stream);

    // rel-mode
    double const user_eb = args.eb;
    if (args.mode == PredArgs::Mode::Rel) {
      double mn = h_data[0], mx = h_data[0];
      for (size_t i = 1; i < len; ++i) {
        double const v = h_data[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      double const rng = mx - mn;
      args.eb = user_eb * rng;
      fprintf(
          stderr,
          "[pred-study] rel-mode: range=%.6e (min=%.6e max=%.6e)  "
          "rel_eb=%.6e -> abs_eb=%.6e\n",
          rng, mn, mx, user_eb, args.eb);
    }

    manager_ = psz_create_resource_manager(
        F4, {args.x, args.y, args.z}, {pred_type, HistogramGeneric, Huffman, NullCodec},
        (void*)stream);

    manager_->header->rc.eb = args.eb;
    manager_->header->rc.mode = (args.mode == PredArgs::Mode::Rel) ? Rel : Abs;
    manager_->header->rc.radius = args.radius;
    manager_->spline_variant = (spline_v_check == 24) ? 1 : 0;  // 1 = y24, 0 = y25
    // Preserve the user's input eb (rel value, if rel-mode) for trace.
    manager_->header->user_input_eb = user_eb;

    mem = (Buf*)manager_->buf;
    mem->set_spline_variant(spline_v_check == 24 ? 1 : 0);  // anchor sizing: BLK8 vs BLK16
    h_hist = std::unique_ptr<uint32_t[]>(new uint32_t[manager_->bklen]);
    return 0;
  }

  psz_resource* mgr() { return manager_; }
  const psz_resource* mgr() const { return manager_; }

  // -- 2. forward predictor (compress_analysis path). --------------------
  int compress()
  {
    auto status = PPL::compress_analysis(mgr(), mem, d_data.get(), h_hist.get(), (void*)stream);
    if (status != PSZ_SUCCESS) {
      fprintf(stderr, "[pred-study] predictor-analysis failed, status=%d\n", status);
      return 2;
    }
    cudaStreamSynchronize(stream);
    return 0;
  }

  // -- 3. reverse predictor on v1's eq layout. ------------------------
  void reconstruct_v1_path()
  {
    cudaMemset(d_xdata.get(), 0, sizeof(float) * len);
    if (mgr()->header->splen != 0) {
      psz::module::GPU_scatter<float, M>::kernel_v2(
          (_ptb::compact_cell<float, M>*)mem->outlier2_validx_d(), mgr()->header->splen,
          d_xdata.get(), (void*)stream);
    }
    if (pred_type == psz_predictor::Lorenzo)
      GPU_x_lorenzo_nd<float, _Toggle::ZigZag_Off>::kernel(
          make_view(mem->eq_d(), len3), make_view(d_xdata.get(), len3),
          make_view(d_xdata.get(), len3), args.eb, mgr()->header->rc.radius, (void*)stream);
    else if (pred_type == psz_predictor::LorenzoZigZag)
      GPU_x_lorenzo_nd<float, _Toggle::ZigZag_On>::kernel(
          make_view(mem->eq_d(), len3), make_view(d_xdata.get(), len3),
          make_view(d_xdata.get(), len3), args.eb, mgr()->header->rc.radius, (void*)stream);
    else if (pred_type == psz_predictor::LorenzoProto)
      psz::module::GPU_PROTO_x_lorenzo_nd<float, E>::kernel(
          mem->eq_d(), d_xdata.get(), d_xdata.get(), len3, args.eb * 2, 1 / (args.eb * 2),
          mgr()->header->rc.radius, (void*)stream);
    else if (pred_type == psz_predictor::Spline)
      psz::module::GPU_x_spline<float, E>::kernel_v1(
          make_view(mem->anchor_d(), mem->anchor_len3()),
          make_view(mem->eq_d(), mem->eq_len3()),
          make_view(d_xdata.get(), mem->eq_len3()), d_xdata.get(), args.eb,
          mgr()->header->rc.radius, mgr()->header->intp_param, (void*)stream,
          spline_v_check == 24 ? SplineVariant::y24 : SplineVariant::y25);
    cudaStreamSynchronize(stream);
    cudaMemcpy(h_xdata.get(), d_xdata.get(), sizeof(float) * len, cudaMemcpyDeviceToHost);
  }

  // -- 4. build PredMetrics from h_data (input) vs h_xdata (reconstructed)
  PredMetrics compute_metrics() const
  {
    return psz_test::compute_metrics(
        h_data.get(), h_xdata.get(), len, args.predictor, args.eb, args.radius,
        mgr()->header->splen);
  }
};

}  // namespace psz_test

#endif
