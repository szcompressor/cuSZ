#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "cusz.h"
#include "ex_utils2.hh"

using std::cout;
using std::endl;
using std::string;

using T = float;

const auto mode = Abs;  // set compression mode
const auto eb = 3.0f;   // set error bound
const auto width = 5;

int main(int argc, char** argv)
{
  Arguments args = parse_arguments(argc, argv);

  const size_t len = args.x * args.y * args.z;
  const size_t oribytes = sizeof(T) * len;

  auto file_names = construct_file_names(
      args.fname_prefix, args.fname_suffix, args.from_number, args.to_number, width);

  psz_header header;

  std::vector<T> h_uncomp(len);
  T *d_uncomp, *d_decomp;
  uint8_t* d_compressed;
  cudaMalloc(&d_uncomp, oribytes);
  cudaMalloc(&d_decomp, oribytes);
  cudaMalloc(&d_compressed, oribytes);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  uint8_t* p_compressed;
  size_t comp_len;

  psz_ppl const ppl{Lorenzo, DEFAULT_HISTOGRAM, args.codec_type, CodecNull};
  psz_ctx* m = psz_compress_init(F4, {args.x, args.y, args.z}, stream);
  if (args.codec_type == HF)
    cout << "using Huffman" << endl;
  else
    cout << "using FZGPUCodec" << endl;

  for (const auto& fname : file_names) {
    cout << "\e[34mFNAME\t" + fname + "\e[0m" << endl;

    if (not std::ifstream(fname, std::ios::binary).read((char*)h_uncomp.data(), oribytes)) {
      cout << "cannot read " << fname << endl;
      continue;
    }
    cudaMemcpy(d_uncomp, h_uncomp.data(), oribytes, cudaMemcpyHostToDevice);

    {  // compresion
      auto abs_eb = eb;
      if (mode == Rel) abs_eb *= psz_compress_extrema_float(m, d_uncomp).rng;
      psz_compress_reset(m);
      psz_compress_process_float(m, ppl, abs_eb, d_uncomp);
      psz_compress_archive(m, &header, &p_compressed, &comp_len);
      //   psz_review_compression(&header);

      cudaMemcpy(d_compressed, p_compressed, comp_len, cudaMemcpyDeviceToDevice);
    }

    {  // decompression
      auto comp_len = pszheader_filesize(&header);
      psz_ctx* x = psz_decompress_init(&header, stream);
      psz_decompress_process_float(x, d_compressed, comp_len, d_decomp);
      psz_free(x);
    }

    {  // evaulation
      auto comp_len = pszheader_filesize(&header);

      psz_stats s{};
      psz_assess_quality_float(&s, d_decomp, d_uncomp, len);
      psz_print_concise_quality(&header, &s, comp_len);
    }

    // !!!! TODO (root cause?) otherwise wrong in evaluation
    cudaMemset(d_decomp, 0, oribytes);
  }

  psz_free(m);
  cudaFree(d_uncomp);
  cudaFree(d_decomp);
  cudaFree(d_compressed);
  cudaStreamDestroy(stream);

  return 0;
}
