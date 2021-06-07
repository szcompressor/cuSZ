#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>

#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>

#include "../src/kernel/prototype_spline2.cuh"
#include "../src/utils/io.hh"

using std::cout;

using Input  = float;
using Output = unsigned short;

Input*  data;
Output* error_control;

unsigned int   dimx, dimy, dimz, dimx_pad, dimy_pad, dimz_pad;
unsigned int   len, len_padded;
dim3           dim3d, dim3d_pad, stride3d, stride3d_pad;
constexpr auto Block = 8;

auto eb             = 1.0f;
auto eb_r           = 1.0f;
auto ebx2           = 2.0f;
auto ebx2_r         = 0.5f;
auto radius         = 512;
auto get_npartition = [](auto size, auto subsize) { return (size + subsize - 1) / subsize; };

std::string fname;

// void test_lorenzo3dc(int n = 1)
// {
//     cout << "testing lorenzo3d\n";
//     auto stridey = dimx, stridez = dimx * dimy;

//     auto dim_block = dim3(32, 1, 8);
//     auto dim_grid  = dim3(
//         get_npartition(dimx, 32),  //
//         get_npartition(dimy, 8),   //
//         get_npartition(dimz, 8)    //
//     );

//     cudaMallocManaged((void**)&error_control, len * sizeof(Input));
//     cudaMemset(data, 0x00, len * sizeof(Input));

//     // for (auto i = 0; i < n; i++) {
//     // cout << "3Dc " << i << '\n';
//     kernel::c_lorenzo_3d1l_v1_32x8x8data_mapto_32x1x8<Input, Output><<<dim_grid, dim_block>>>  //
//         (data, error_control, dimx, dimy, dimz, stridey, stridez, radius, ebx2_r);
//     cudaDeviceSynchronize();

//     io::WriteArrayToBinary(fname + ".lorenzo", error_control, len);

//     cudaFree(error_control);
//     // }
// }

template <typename T, bool PrintFP = false, bool Padding = true>
void print_block_from_CPU(T* data, int radius = 512)
{
    cout << "dimxpad: " << dimx_pad << "\tdimypad: " << dimy_pad << '\n';

    for (auto z = 0; z < (Block + (int)Padding); z++) {
        printf("\nprint from CPU, z=%d\n", z);
        printf("    ");
        for (auto i = 0; i < 33; i++) printf("%3d", i);
        printf("\n");

        for (auto y = 0; y < (Block + (int)Padding); y++) {
            printf("y=%d ", y);

            for (auto x = 0; x < (4 * Block + (int)Padding); x++) {  //
                auto gid = x + y * dimx_pad + z * dimx_pad * dimy_pad;
                if CONSTEXPR (PrintFP) { printf("%.2e\t", data[gid]); }
                else {
                    auto c = (int)data[gid] - radius;
                    if (c == 0)
                        printf("%3c", '.');
                    else {
                        if (abs(c) >= 10)
                            printf("%3c", '*');
                        else
                            printf("%3d", c);
                    }
                }
            }
            printf("\n");
        }
    }
    printf("\nCPU print end\n\n");
}

void test_spline3dc()
{
    cout << "testing spline3d\n";
    auto nblockx = get_npartition(dimx, Block * 4);
    auto nblocky = get_npartition(dimy, Block);
    auto nblockz = get_npartition(dimz, Block);

    dimx_pad = nblockx * Block * 4;
    dimy_pad = nblocky * Block;
    dimz_pad = nblockz * Block;

    auto len_padded = dimx_pad * dimy_pad * dimz_pad;
    // +(nblockx + 1) * (nblocky + 1) * (nblockz + 1);

    std::cout << "len padded: " << len_padded << '\n';
    std::cout << "len: " << len << '\n';
    std::cout << "len padded/len: " << 1.0 * len_padded / len << '\n';
    printf("dim and dimpad: (%d, %d, %d), (%d, %d, %d)\n", dimx, dimy, dimz, dimx_pad, dimy_pad, dimz_pad);

    dim3d        = dim3(dimx, dimy, dimz);
    stride3d     = dim3(1, dimx, dimx * dimy);
    dim3d_pad    = dim3(dimx_pad, dimy_pad, dimz_pad);
    stride3d_pad = dim3(1, dimx_pad, dimx_pad * dimy_pad);

    cudaMallocManaged((void**)&error_control, len_padded * sizeof(Output));
    cudaMemset(error_control, 0x00, len_padded * sizeof(Output));

    kernel::spline3d_infprecis_32x8x8data<Input*, Output*, float, 256, false, false>
        <<<dim3(nblockx, nblocky, nblockz), dim3(256, 1, 1)>>>  //
        // <<<dim3(1, 1, 1), dim3(256, 1, 1)>>>  //
        (data, dim3d, stride3d, error_control, dim3d_pad, stride3d_pad, eb_r, ebx2, radius);
    cudaDeviceSynchronize();

    // print_block_from_CPU<Input, true>(data);
    print_block_from_CPU<Output, false, true>(error_control);

    auto hist = new int[radius * 2]();

    for (auto i = 0; i < len_padded; i++) { hist[error_control[i]]++; }
    for (auto i = 0; i < radius * 2; i++) {
        if (hist[i] != 0) std::cout << i << '\t' << hist[i] << '\n';
    }

    io::WriteArrayToBinary(fname + ".spline", error_control, len_padded);

    cudaFree(error_control);
}

int main(int argc, char** argv)
{
    // auto           dimx = 512, dimy = 512, dimz = 512;
    dimz = 449, dimy = 449, dimx = 235;
    len = dimx * dimy * dimz;

    if (argc < 2) {
        struct passwd* pw      = getpwuid(getuid());
        const char*    homedir = pw->pw_dir;
        fname                  = std::string(homedir) + std::string("/rtm-data/snapshot-2815.f32");
    }
    else if (argc == 2)
        fname = std::string(argv[1]);

    cudaDeviceReset();

    cudaMallocManaged((void**)&data, len * sizeof(Input));
    cudaMemset(data, 0x00, len * sizeof(Input));

    std::cout << "opening " << fname << std::endl;
    io::ReadBinaryToArray(fname, data, len);

    thrust::device_ptr<Input> g_ptr      = thrust::device_pointer_cast(data);
    auto                      max_el_loc = thrust::max_element(g_ptr, g_ptr + len);  // excluding padded
    auto                      min_el_loc = thrust::min_element(g_ptr, g_ptr + len);  // excluding padded
    // auto   max_el_loc = std::max_element(data, data + len);
    // auto   min_el_loc = std::min_element(data, data + len);
    double max_value = *max_el_loc;
    double min_value = *min_el_loc;
    double rng       = max_value - min_value;

    std::cout << "range: " << rng << '\n';

    auto eb = 1e-2;
    eb *= rng;
    eb_r   = 1 / eb;
    ebx2   = eb * 2;
    ebx2_r = 1 / ebx2;

    test_spline3dc();
    // test_lorenzo3dc();

    cudaFree(data);

    return 0;
}
