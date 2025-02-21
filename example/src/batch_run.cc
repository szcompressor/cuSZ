#include <cuda_runtime.h>

#include <sstream>
#include <vector>

#include "api_v2.h"
#include "ex_utils2.hh"
#include "mem/cxx_backends.h"
#include "stat/compare.hh"
#include "utils/io.hh"

namespace utils = _portable::utils;

using T = float;

const auto mode = Abs;  // set compression mode
const string mode_str("abs");
const string eb_str("3e0");
const auto eb = 3.0f;  // set error bound
const auto width = 5;

GPU_unique_hptr<T[]> h_uncomp;
GPU_unique_dptr<T[]> d_uncomp;
GPU_unique_hptr<T[]> h_decomp;
GPU_unique_dptr<T[]> d_decomp;
GPU_unique_dptr<T[]> d_compressed;

int main(int argc, char** argv)
{
  Arguments args = parse_arguments(argc, argv);

  const size_t len = args.x * args.y * args.z;
  const size_t oribytes = sizeof(T) * len;

  auto file_names = construct_file_names(
      args.fname_prefix, args.fname_suffix, args.from_number, args.to_number, width);

  psz_header header;

  auto d_uncomp = MAKE_UNIQUE_DEVICE(T, len);
  auto h_uncomp = MAKE_UNIQUE_HOST(T, len);
  auto d_decomp = MAKE_UNIQUE_DEVICE(T, len);
  auto h_decomp = MAKE_UNIQUE_HOST(T, len);

  auto d_compressed = MAKE_UNIQUE_DEVICE(uint8_t, oribytes);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  uint8_t* p_compressed;
  size_t comp_len;
  psz_len3 uncomp_len3 = {args.x, args.y, args.z};
  psz_len3 decomp_len3 = uncomp_len3;

  psz_resource* m = psz_create_resource_manager(F4, args.x, args.y, args.z, stream);
  m->cli = new psz_cli_config;  // TODO mix use the cli and "resource manager"
  if (args.codec_type == Huffman) {
    cout << "using Huffman" << endl;
    m->cli->dump_hist = true;
  }
  else {
    cout << "using FZGPUCodec" << endl;
  }
  // m->cli->dump_quantcode = true;

  for (const auto& fname : file_names) {
    cout << "\e[34mFNAME\t" + fname + "\e[0m" << endl;
    strcpy(m->cli->file_input, fname.c_str());
    strcpy(m->cli->char_mode, mode_str.c_str());
    strcpy(m->cli->char_meta_eb, eb_str.c_str());

    utils::fromfile(fname, h_uncomp.get(), len);
    memcpy_allkinds<H2D>(d_uncomp.get(), h_uncomp.get(), len);

    {  // compresion
      psz_compress_float(
          m, {Lorenzo, DEFAULT_HISTOGRAM, args.codec_type, NULL_CODEC, mode, eb, args.radius},
          d_uncomp.get(), &header, &p_compressed, &comp_len);
      //   psz_review_compression(comp_timerecord, &header);

      memcpy_allkinds<D2D>(d_compressed.get(), p_compressed, comp_len);
    }

    {  // decompression
      auto comp_len = pszheader_filesize(&header);
      psz_decompress_float(m, d_compressed.get(), comp_len, d_decomp.get());
    }

    {  // evaulation
      auto comp_len = pszheader_filesize(&header);
      //   psz_review_decompression(decomp_timerecord, oribytes);
      auto s = new psz_statistics;
      psz::cuhip::GPU_assess_quality(s, d_uncomp.get(), d_decomp.get(), len);
      printf(
          "R\t%u\t"
          "CR\t%lf\t"
          "PSNR\t%lf\t"
          "NRMSE\t%lf\t"
          "MAX.ABS.EB\t%lf\t"
          "MAX.REL.EB\t%lf\n",
          args.radius, len * sizeof(T) * 1.0 / comp_len, s->score_PSNR, s->score_NRMSE,
          s->max_err_abs, s->max_err_rel);
    }

    // !!!! TODO (root cause?) otherwise wrong in evaluation
    memset_device(d_decomp.get(), len, 0);
  }

  psz_release_resource(m);
  cudaStreamDestroy(stream);

  return 0;
}