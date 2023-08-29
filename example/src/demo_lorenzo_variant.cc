/**
 * @file demo_lorenzo_variant.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-09-29
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "busyheader.hh"
#include "kernel/l23.hh"
#include "port.hh"
#include "utils/err.hh"
#include "utils/io.hh"
#include "utils/print_arr.hh"
#include "utils/viewer.hh"

using std::cerr;
using std::cout;
using std::endl;

template <typename DeltaT = uint16_t>
int f(
    std::string fname, size_t x, size_t y, size_t z, double eb,
    size_t start = 10000)
{
  float* h_data;
  float* data;
  float* xdata;
  bool* signum;
  DeltaT* delta;

  dim3 len3 = dim3(x, y, z);
  dim3 stride3 = dim3(1, x, x * y);
  size_t len = x * y * z;

  GpuMallocHost(&h_data, len * sizeof(float));

  GpuMalloc(&data, len * sizeof(float));
  GpuMalloc(&xdata, len * sizeof(float));
  GpuMalloc(&signum, len * sizeof(bool));
  GpuMalloc(&delta, len * sizeof(DeltaT));

  io::read_binary_to_array<float>(fname, h_data, len);
  GpuMemcpy(data, h_data, len * sizeof(float), GpuMemcpyHostToDevice);

  /* a casual peek */
  printf("peeking data, 20 elements\n");
  psz::peek_device_data<float>(data, 100);

  GpuStreamT stream;
  GpuStreamCreate(&stream);

  float time_comp;

  asz::experimental::psz_comp_l21var<float, DeltaT, float>(
      data, len3, eb, delta, signum, &time_comp, stream);

  {
    printf("signum\n");
    psz::peek_device_data<int8_t>((int8_t*)signum, 100);

    printf("delta\n");
    psz::peek_device_data<DeltaT>(delta, 100);
  }

  cout << "comp time\t" << time_comp << endl;

  float time_decomp;
  asz::experimental::psz_decomp_l21var<float, DeltaT, float>(
      delta, signum, len3, eb, xdata, &time_decomp, stream);

  cout << "decomp time\t" << time_decomp << endl;

  {
    printf("xdata\n");
    psz::peek_device_data<float>(xdata, 100);
  }

  psz::eval_dataquality_gpu(xdata, data, len);

  GpuFreeHost(h_data);
  GpuFree(data);
  GpuFree(xdata);
  GpuFree(signum);
  GpuFree(delta);

  GpuStreamDestroy(stream);

  return 0;
}

int main(int argc, char** argv)
{
  if (argc < 5) {
    cout << "                       default: ui16" << endl;
    cout << "                       ui8,ui16,ui32" << endl;
    cout << "PROG fname x y z [eb] [delta type] [print offset]" << endl;
    cout << "0    1     2 3 4 [5]  [6]          [7]" << endl;

    return 1;
  }

  auto fname = std::string(argv[1]);
  auto x = atoi(argv[2]);
  auto y = atoi(argv[3]);
  auto z = atoi(argv[4]);

  double eb = 1e-4;
  std::string delta_type = "ui16";
  size_t print_start = 10000;

  if (argc >= 6) eb = atof(argv[5]);

  if (argc >= 7) delta_type = std::string(argv[6]);

  if (argc >= 8) print_start = atoi(argv[7]);

  if (delta_type == "ui8")
    f<uint8_t>(fname, x, y, z, eb, print_start);
  else if (delta_type == "ui16")
    f<uint16_t>(fname, x, y, z, eb, print_start);
  else if (delta_type == "ui32")
    f<uint32_t>(fname, x, y, z, eb, print_start);

  return 0;
}