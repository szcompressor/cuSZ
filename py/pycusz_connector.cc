#include <nanobind/nanobind.h>

#include "../psz/src/libcusz.cc"

namespace nb = nanobind;

nb::object PYCONNECTOR_create_resource_manager(
    psz_dtype t, uint32_t x, uint32_t y, uint32_t z, uintptr_t stream_ptr)
{
  void* stream = reinterpret_cast<void*>(stream_ptr);  // Convert integer to pointer
  psz_resource* res = CAPI_psz_create_resource_manager(t, x, y, z, stream);

  if (not res) throw std::runtime_error("Failed to create resource manager");

  return nb::cast(res, nb::rv_policy::reference);  // Return as a raw pointer
}

// for shorter name of `psz_create_resource_manager_from_header`
nb::object PYCONNECTOR_create_resource_manager_from_header(
    psz_header* header, uintptr_t stream_ptr)
{
  void* stream = reinterpret_cast<void*>(stream_ptr);  // Convert integer to pointer
  psz_resource* res = CAPI_psz_create_resource_manager_from_header(header, stream);

  if (not res) throw std::runtime_error("Failed to create resource manager");

  return nb::cast(res, nb::rv_policy::reference);  // Return as a raw pointer
}

// for shorter name of `psz_modify_resource_manager_from_header`
void PYCONNECTOR_modify_resource_manager_from_header(psz_resource* manager, psz_header* header)
{
  CAPI_psz_modify_resource_manager_from_header(manager, header);
}

// for shorter name of `psz_release_resource`
int PYCONNECTOR_release_resource(psz_resource* manager) { return psz_release_resource(manager); }

// for shorter name of `psz_compress_float`
int PYCONNECTOR_compress_float(
    psz_resource* manager, psz_rc rc, uintptr_t d_data_ptr, psz_header* compressed_metadata,
    uintptr_t dptr_compressed_ptr, uintptr_t compressed_bytes_ptr)
{
  float* IN_d_data = reinterpret_cast<float*>(d_data_ptr);
  uint8_t** OUT_dptr_compressed = reinterpret_cast<uint8_t**>(dptr_compressed_ptr);
  size_t* OUT_compressed_bytes = reinterpret_cast<size_t*>(compressed_bytes_ptr);
  return CAPI_psz_compress_float(
      manager, rc, IN_d_data, compressed_metadata, OUT_dptr_compressed, OUT_compressed_bytes);
}

// for shorter name of `psz_compress_double`
int PYCONNECTOR_compress_double(
    psz_resource* manager, psz_rc rc, uintptr_t d_data_ptr, psz_header* compressed_metadata,
    uintptr_t dptr_compressed_ptr, uintptr_t compressed_bytes_ptr)
{
  double* IN_d_data = reinterpret_cast<double*>(d_data_ptr);
  uint8_t** OUT_dptr_compressed = reinterpret_cast<uint8_t**>(dptr_compressed_ptr);
  size_t* OUT_compressed_bytes = reinterpret_cast<size_t*>(compressed_bytes_ptr);
  return CAPI_psz_compress_double(
      manager, rc, IN_d_data, compressed_metadata, OUT_dptr_compressed, OUT_compressed_bytes);
}

// for shorter name of `psz_decompress_float`
int PYCONNECTOR_decompress_float(
    psz_resource* manager, uintptr_t d_compressed_ptr, size_t compressed_len,
    uintptr_t d_decompressed_ptr)
{
  uint8_t* IN_d_compressed = reinterpret_cast<uint8_t*>(d_compressed_ptr);
  float* OUT_d_decompressed = reinterpret_cast<float*>(d_decompressed_ptr);
  return CAPI_psz_decompress_float(manager, IN_d_compressed, compressed_len, OUT_d_decompressed);
}

// for shorter name of `psz_decompress_double`
int PYCONNECTOR_decompress_double(
    psz_resource* manager, uintptr_t d_compressed_ptr, size_t compressed_len,
    uintptr_t d_decompressed_ptr)
{
  uint8_t* IN_d_compressed = reinterpret_cast<uint8_t*>(d_compressed_ptr);
  double* OUT_d_decompressed = reinterpret_cast<double*>(d_decompressed_ptr);
  return CAPI_psz_decompress_double(manager, IN_d_compressed, compressed_len, OUT_d_decompressed);
}

// build nanobind module
NB_MODULE(pycusz_connector, m)
{
  nb::enum_<_portable_dtype>(m, "portable_dtype")
      .value("F4", _portable_dtype::F4)
      .value("F8", _portable_dtype::F8)
      .value("U1", _portable_dtype::U1)
      .value("U2", _portable_dtype::U2)
      .value("U4", _portable_dtype::U4)
      .value("U8", _portable_dtype::U8)
      .value("I1", _portable_dtype::I1)
      .value("I2", _portable_dtype::I2)
      .value("I4", _portable_dtype::I4)
      .value("I8", _portable_dtype::I8)
      .value("ULL", _portable_dtype::ULL)
      .export_values();
  m.attr("psz_dtype") = m.attr("portable_dtype");

  nb::enum_<psz_mode>(m, "psz_mode")
      .value("Abs", psz_mode::Abs)
      .value("Rel", psz_mode::Rel)
      .export_values();

  nb::enum_<psz_predtype>(m, "psz_predtype")
      .value("Lorenzo", psz_predtype::Lorenzo)
      .value("LorenzoZigZag", psz_predtype::LorenzoZigZag)
      .value("LorenzoProto", psz_predtype::LorenzoProto)
      .value("Spline", psz_predtype::Spline)
      .export_values();

  nb::enum_<psz_codectype>(m, "psz_codectype")
      .value("Huffman", psz_codectype::Huffman)
      .value("HuffmanRevisit", psz_codectype::HuffmanRevisit)
      .value("FZGPUCodec", psz_codectype::FZGPUCodec)
      .value("RunLength", psz_codectype::RunLength)
      .value("NullCodec", psz_codectype::NullCodec)
      .export_values();

  nb::enum_<psz_histotype>(m, "psz_histotype")
      .value("HistogramGeneric", psz_histotype::HistogramGeneric)
      .value("HistogramSparse", psz_histotype::HistogramSparse)
      .value("NullHistogram", psz_histotype::NullHistogram)
      .export_values();

  nb::class_<psz_resource>(m, "psz_resource").def(nb::init<>());
  nb::class_<psz_header>(m, "psz_header").def(nb::init<>());

  //   nb::class_<psz_runtime_config>(m, "psz_runtime_config").def(nb::init<>());
  nb::class_<psz_runtime_config>(m, "psz_runtime_config")
      .def(nb::init<>())
      .def_rw("predictor", &psz_runtime_config::predictor)
      .def_rw("hist", &psz_runtime_config::hist)
      .def_rw("codec1", &psz_runtime_config::codec1)
      .def_rw("_future_codec2", &psz_runtime_config::_future_codec2)
      .def_rw("mode", &psz_runtime_config::mode)
      .def_rw("eb", &psz_runtime_config::eb)
      .def_rw("_future_radius", &psz_runtime_config::_future_radius);

  m.def(
      "PYCONNECTOR_create_resource_manager", &PYCONNECTOR_create_resource_manager,
      "Create a psz_resource manager");
  m.def(
      "PYCONNECTOR_create_resource_manager_from_header",
      &PYCONNECTOR_create_resource_manager_from_header, "Create a resource manager from header");
  m.def(
      "PYCONNECTOR_modify_resource_manager_from_header",
      &PYCONNECTOR_modify_resource_manager_from_header,
      "Modify an existing resource manager from header");
  m.def(
      "PYCONNECTOR_release_resource", &PYCONNECTOR_release_resource,
      "Release the resource manager");

  m.def("PYCONNECTOR_compress_float", &PYCONNECTOR_compress_float, "Compress float data");
  m.def("PYCONNECTOR_compress_double", &PYCONNECTOR_compress_double, "Compress double data");
  m.def("PYCONNECTOR_decompress_float", &PYCONNECTOR_decompress_float, "Decompress float data");
  m.def("PYCONNECTOR_decompress_double", &PYCONNECTOR_decompress_double, "Decompress double data");

  m.attr("DEFAULT_PREDICTOR") = psz_predtype::Lorenzo;
  m.attr("DEFAULT_HISTOGRAM") = psz_histotype::HistogramGeneric;
  m.attr("DEFAULT_CODEC") = psz_codectype::Huffman;
  m.attr("NULL_HISTOGRAM") = psz_histotype::NullHistogram;
  m.attr("NULL_CODEC") = psz_codectype::NullCodec;
}
