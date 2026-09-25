#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include "compressor.hh"
#include "context_impl.h"
#include "cusz/type.h"

namespace nb = nanobind;
using nb::literals::operator""_a;

NB_MODULE(_psz_comp, m)
{
  // exported by name, not by value, so this stays in sync with cusz/type.h
  nb::enum_<psz_dtype>(m, "Dtype").value("F4", F4).value("F8", F8);

  nb::enum_<psz_predictor>(m, "Predictor")
      .value("Lorenzo", Lorenzo)
      .value("LorenzoZigZag", LorenzoZigZag)
      .value("SplineY25", SplineY25)
      .value("SplineY24", SplineY24);

  nb::enum_<psz_codec>(m, "Codec")
      .value("HF", HF)
      .value("HF_r2", HF_r2)
      .value("HFR", HFR)
      .value("HFR_V2", HFR_V2)
      .value("HFR_V3", HFR_V3)
      .value("HFR_V4", HFR_V4)
      .value("HFR_PBKC", HFR_PBKC)
      .value("HFR_PBKGO", HFR_PBKGO)
      .value("HFR_PBKF", HFR_PBKF)
      .value("LC_TCMS", LC_TCMS)
      .value("LC_DRH", LC_DRH)
      .value("LC_BITR", LC_BITR)
      .value("LC_RTR", LC_RTR)
      .value("FZG", FZG)
      .value("CodecNull", CodecNull);

  // what the CLI's "_" / "*" / "default" resolve to, taken from cusz/type.h
  m.attr("DEFAULT_PREDICTOR") = DEFAULT_PREDICTOR;
  m.attr("DEFAULT_CODEC") = DEFAULT_CODEC;

  nb::class_<psz_ppl>(m, "Pipeline", "psz_ppl from the CLI's --pipeline syntax")
      .def_ro("predictor", &psz_ppl::predictor)
      .def_ro("codec1", &psz_ppl::codec1)
      .def_ro("codec2", &psz_ppl::codec2);

  m.def(
      "pszctx_pipeline_from_name",
      [](std::string const& spec) {
        psz_ppl ppl{};
        std::string err;
        if (auto const e = pszctx_pipeline_from_name(spec.c_str(), &ppl)) err = e;
        return std::make_tuple(err, ppl);
      },
      "spec"_a);

  // psz_header stays opaque: .len and .dtype size and type psz_decompress_init's output.
  nb::class_<psz_header>(m, "Header")
      .def_prop_ro(
          "len", [](psz_header const& h) { return std::make_tuple(h.len.x, h.len.y, h.len.z); })
      .def_prop_ro("dtype", [](psz_header const& h) { return h.dtype; });

  nb::class_<psz_data_summary>(m, "DataSummary")
      .def_ro("min", &psz_data_summary::min)
      .def_ro("max", &psz_data_summary::max)
      .def_ro("rng", &psz_data_summary::rng)
      .def_ro("std", &psz_data_summary::std)
      .def_ro("avg", &psz_data_summary::avg);

  nb::class_<psz::pybind::Quality>(m, "Quality")
      .def_ro("psnr", &psz::pybind::Quality::psnr)
      .def_ro("mse", &psz::pybind::Quality::mse)
      .def_ro("nrmse", &psz::pybind::Quality::nrmse)
      .def_ro("coeff", &psz::pybind::Quality::coeff)
      .def_ro("max_err_abs", &psz::pybind::Quality::max_err_abs)
      .def_ro("max_err_rel", &psz::pybind::Quality::max_err_rel)
      .def_ro("max_err_pwrrel", &psz::pybind::Quality::max_err_pwrrel)
      .def_ro("max_err_idx", &psz::pybind::Quality::max_err_idx)
      .def_ro("origin_min", &psz::pybind::Quality::origin_min)
      .def_ro("origin_max", &psz::pybind::Quality::origin_max)
      .def_ro("origin_rng", &psz::pybind::Quality::origin_rng)
      .def_ro("origin_std", &psz::pybind::Quality::origin_std)
      .def_ro("origin_avg", &psz::pybind::Quality::origin_avg)
      .def_ro("len", &psz::pybind::Quality::len);

  // one module-level function per cusz.h entry point, in cusz.h order
  m.def(
      "psz_compress_init", &psz::pybind::psz_compress_init, "dtype"_a, "x"_a, "y"_a, "z"_a,
      "stream"_a = 0);
  m.def(
      "psz_compress_init_3stage", &psz::pybind::psz_compress_init_3stage, "dtype"_a, "x"_a, "y"_a,
      "z"_a, "stream"_a = 0);
  m.def("psz_decompress_init", &psz::pybind::psz_decompress_init, "header"_a, "stream"_a = 0);
  m.def("psz_free", &psz::pybind::psz_free, "ctx"_a);
  m.def("psz_last_error", &psz::pybind::psz_last_error);
  m.def("psz_error_string", &psz_error_string, "e"_a);

  m.def("psz_compress_extrema_float", &psz::pybind::psz_compress_extrema_float, "ctx"_a, "d_in"_a);
  m.def(
      "psz_compress_extrema_double", &psz::pybind::psz_compress_extrema_double, "ctx"_a, "d_in"_a);

  m.def(
      "psz_compress_process_float", &psz::pybind::psz_compress_process_float, "ctx"_a,
      "pipeline"_a, "eb"_a, "d_in"_a);
  m.def(
      "psz_compress_process_double", &psz::pybind::psz_compress_process_double, "ctx"_a,
      "pipeline"_a, "eb"_a, "d_in"_a);

  m.def("psz_compress_archive", &psz::pybind::psz_compress_archive, "ctx"_a);
  m.def("psz_compress_reset", &psz::pybind::psz_compress_reset, "ctx"_a);

  m.def(
      "psz_decompress_process_float", &psz::pybind::psz_decompress_process_float, "ctx"_a,
      "d_in"_a, "in_bytes"_a, "out"_a);
  m.def(
      "psz_decompress_process_double", &psz::pybind::psz_decompress_process_double, "ctx"_a,
      "d_in"_a, "in_bytes"_a, "out"_a);
  m.def("psz_decompress_reset", &psz::pybind::psz_decompress_reset, "ctx"_a);

  m.def(
      "psz_assess_quality_float", &psz::pybind::psz_assess_quality_float, "d_reconst"_a,
      "d_origin"_a, "len"_a);
  m.def(
      "psz_assess_quality_double", &psz::pybind::psz_assess_quality_double, "d_reconst"_a,
      "d_origin"_a, "len"_a);

  m.def(
      "psz_print_concise_quality", &psz::pybind::psz_print_concise_quality, "header"_a,
      "quality"_a, "comp_bytes"_a);

  m.def("psz_review_compression", &psz::pybind::psz_review_compression, "header"_a);
  m.def(
      "psz_review_compression_verbose", &psz::pybind::psz_review_compression_verbose, "header"_a);
  m.def("psz_review_decompression", &psz::pybind::psz_review_decompression, "header"_a);
  m.def(
      "psz_review_decompression_verbose", &psz::pybind::psz_review_decompression_verbose,
      "header"_a);

  m.def("header_from_archive", &psz::pybind::header_from_archive, "d_archive"_a);
}
