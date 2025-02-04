#include <cuda_runtime.h>

#include <iomanip>
#include <sstream>
#include <vector>

#include "cusz.h"
#include "stat/compare.hh"
#include "utils/io.hh"  // io::read_binary_to_array

namespace utils = _portable::utils;

using T = float;

const auto mode = Abs;  // set compression mode
const string mode_str("abs");
const string eb_str("3e0");
const auto eb = 3.0f;  // set error bound
const auto width = 5;

T *d_decomp, *h_decomp;
T *d_uncomp, *h_uncomp;
void* comp_timerecord;
void* decomp_timerecord;

struct Arguments {
  std::string fname_prefix;
  std::string fname_suffix;
  int from_number;
  int to_number;
  psz_codectype codec_type{Huffman};
  size_t x;
  size_t y;
  size_t z;
  size_t radius;
};

void print_help()
{
  std::cout << "usage: batch_run [options]\n"
            << "options:\n"
            << "  --fname-prefix PREFIX   file name prefix\n"
            << "  --fname-suffix SUFFIX   file name suffix\n"
            << "  --from NUMBER           start number\n"
            << "  --to NUMBER             end number\n"
            << "  -x NUMBER               dim x\n"
            << "  -y NUMBER               dim y\n"
            << "  -z NUMBER               dim z\n"
            << "  -r NUMBER               radius\n"
            << "  --help                  this message\n";
}

Arguments parse_arguments(int argc, char* argv[])
{
  Arguments args;
  bool fname_prefix_set = false;
  bool fname_suffix_set = false;
  bool from_number_set = false;
  bool to_number_set = false;
  bool x_set = false;
  bool y_set = false;
  bool z_set = false;
  bool radius_set = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--fname-prefix" and i + 1 < argc) {
      args.fname_prefix = argv[++i];
      fname_prefix_set = true;
    }
    else if (arg == "--fname-suffix" and i + 1 < argc) {
      args.fname_suffix = argv[++i];
      fname_suffix_set = true;
    }
    else if (arg == "--from" and i + 1 < argc) {
      args.from_number = atoi(argv[++i]);
      from_number_set = true;
    }
    else if (arg == "--to" and i + 1 < argc) {
      args.to_number = atoi(argv[++i]);
      to_number_set = true;
    }
    else if (arg == "-x" and i + 1 < argc) {
      args.x = atoi(argv[++i]);
      x_set = true;
    }
    else if (arg == "-y" and i + 1 < argc) {
      args.y = atoi(argv[++i]);
      y_set = true;
    }
    else if (arg == "-z" and i + 1 < argc) {
      args.z = atoi(argv[++i]);
      z_set = true;
    }
    else if (arg == "-r" and i + 1 < argc) {
      args.radius = atoi(argv[++i]);
      radius_set = true;
    }
    else if (arg == "--codec" and i + 1 < argc) {
      args.codec_type = string(argv[++i]) == "fzg" ? FZGPUCodec : Huffman;
    }
    else if (arg == "--help") {
      print_help();
      exit(0);
    }
    else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      print_help();
      exit(1);
    }
  }

  if (not fname_prefix_set or not fname_suffix_set or not from_number_set or not to_number_set or
      not x_set or not y_set or not z_set) {
    std::cerr << "Error: Missing required arguments." << std::endl;
    print_help();
    exit(1);
  }

  return args;
}

std::vector<std::string> construct_file_names(
    const std::string& prefix, const std::string& suffix, int start, int end, int width)
{
  std::vector<std::string> file_names;
  for (int i = start; i < end; ++i) {
    std::ostringstream oss;

    oss << prefix << "." << std::setw(width) << std::setfill('0') << i << "." + suffix;
    file_names.push_back(oss.str());
  }
  return file_names;
}

int main(int argc, char** argv)
{
  Arguments args = parse_arguments(argc, argv);

  const size_t len = args.x * args.y * args.z;
  const size_t oribytes = sizeof(T) * len;

  auto file_names = construct_file_names(
      args.fname_prefix, args.fname_suffix, args.from_number, args.to_number, width);

  psz_header header;
  uint8_t* compressed;
  cudaMalloc(&compressed, oribytes);

  cudaMalloc(&d_uncomp, oribytes), cudaMallocHost(&h_uncomp, oribytes);
  cudaMalloc(&d_decomp, oribytes), cudaMallocHost(&h_decomp, oribytes);

  comp_timerecord = psz_make_timerecord();
  decomp_timerecord = psz_make_timerecord();

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  uint8_t* p_compressed;
  size_t comp_len;
  psz_len3 uncomp_len3 = {args.x, args.y, args.z};
  psz_len3 decomp_len3 = uncomp_len3;

  psz_compressor* cor;
  if (args.codec_type == Huffman) {
    cout << "using Huffman" << endl;
    cor = psz_create(F4, uncomp_len3, Lorenzo, args.radius, Huffman);
    cor->ctx->cli->dump_hist = true;
  }
  else {
    cout << "using FZGPUCodec" << endl;
    cor = psz_create(F4, uncomp_len3, LorenzoZigZag, args.radius, FZGPUCodec);
  }
  // cor->ctx->cli->dump_quantcode = true;

  for (const auto& fname : file_names) {
    cout << "\e[34mFNAME\t" + fname + "\e[0m" << endl;
    strcpy(cor->ctx->cli->file_input, fname.c_str());
    strcpy(cor->ctx->cli->char_mode, mode_str.c_str());
    strcpy(cor->ctx->cli->char_meta_eb, eb_str.c_str());

    utils::fromfile(fname, h_uncomp, len);
    cudaMemcpy(d_uncomp, h_uncomp, oribytes, cudaMemcpyHostToDevice);

    {  // compresion
      psz_compress(
          cor, d_uncomp, uncomp_len3, eb, mode, &p_compressed, &comp_len, &header, comp_timerecord,
          stream);
      //   psz_review_compression(comp_timerecord, &header);

      cudaMemcpy(compressed, p_compressed, comp_len, cudaMemcpyDeviceToDevice);
    }

    {  // decompression
      auto comp_len = pszheader_filesize(&header);
      psz_decompress(cor, compressed, comp_len, d_decomp, decomp_len3, decomp_timerecord, stream);
    }

    {  // evaulation
      auto comp_len = pszheader_filesize(&header);
      //   psz_review_decompression(decomp_timerecord, oribytes);
      auto s = new psz_statistics;
      psz::cuhip::GPU_assess_quality(s, d_uncomp, d_decomp, len);
      printf(
          "CR\t%lf\t"
          "PSNR\t%lf\t"
          "NRMSE\t%lf\n",
          len * sizeof(T) * 1.0 / comp_len, s->score_PSNR, s->score_NRMSE);
    }

    capi_psz_clear_buffer(cor);

    cudaMemset(d_decomp, 0, oribytes);  // !!!! TODO (root cause?) otherwise wrong in evaluation
  }

  psz_release(cor);

  // clean up
  cudaFree(compressed);
  cudaFree(d_uncomp);
  cudaFree(d_decomp);
  cudaFreeHost(h_decomp);
  cudaFreeHost(h_decomp);

  cudaStreamDestroy(stream);

  return 0;
}