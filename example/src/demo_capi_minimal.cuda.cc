#include <cuda_runtime.h>

#include "cusz.h"
#include "utils/io.hh"  // io::read_binary_to_array for demo

template <typename T>
void f(std::string fname)
{
  auto len = 3600 * 1800;

  uint8_t* ptr_compressed;
  size_t comp_len;

  T *d_uncomp, *h_uncomp;
  T *d_decomp, *h_decomp;

  auto oribytes = sizeof(T) * len;
  cudaMalloc(&d_uncomp, oribytes), cudaMallocHost(&h_uncomp, oribytes);
  cudaMalloc(&d_decomp, oribytes), cudaMallocHost(&h_decomp, oribytes);

  io::read_binary_to_array(fname, h_uncomp, len);
  cudaMemcpy(d_uncomp, h_uncomp, oribytes, cudaMemcpyHostToDevice);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  auto mode = Rel;   // set compression mode
  auto eb = 2.4e-4;  // set error bound

  pszcompressor* comp = psz_create(
      /* dtype */ F4, /* predictor */ Lorenzo, /* quantizer radius */ 512,
      /* codec */ Huffman, eb, mode);

  auto uncomp_len = psz_len3{3600, 1800, 1};  // x, y, z
  auto decomp_len = uncomp_len;

  pszheader header;

  /* compression */
  {
    psz_compress_init(comp, uncomp_len);
    psz_compress(
        comp, d_uncomp, uncomp_len, &ptr_compressed, &comp_len, &header,
        nullptr, stream);
  }

  uint8_t* ext_comp;
  cudaMalloc(&ext_comp, comp_len);
  cudaMemcpy(ext_comp, ptr_compressed, comp_len, cudaMemcpyDeviceToDevice);

  /* decompression */
  {
    psz_decompress_init(comp, &header);
    psz_decompress(
        comp, ext_comp, comp_len, d_decomp, decomp_len, nullptr, stream);
  }

  /* release */
  psz_release(comp);

  /*
  Clean up CUDA stuff.
  */
  cudaFree(ext_comp);
  cudaFree(d_uncomp), cudaFreeHost(h_uncomp);
  cudaFree(d_decomp), cudaFreeHost(h_decomp);
  cudaStreamDestroy(stream);
}

int main(int argc, char** argv)
{
  if (argc < 2) {
    printf("PROG /path/to/cesm-3600x1800\n");
    exit(0);
  }

  f<float>(std::string(argv[1]));
  return 0;
}