/**
 * @file ex_spline.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-06-06
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

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
#include "../src/utils/verify.hh"
#include "../src/utils/verify_gpu.cuh"

using std::cout;

using Data  = float;
using Quant = unsigned short;

Data*  data;
Data*  xdata;
Data*  anchor;
Quant* errctrl;

bool print_hist = false;

unsigned int   dimx, dimy, dimz;
unsigned int   dimx_pad, dimy_pad, dimz_pad;
unsigned int   nblockx, nblocky, nblockz;
unsigned int   in_range_nanchorx, in_range_nanchory, in_range_nanchorz;
unsigned int   nanchorx, nanchory, nanchorz;
unsigned int   len, len_padded, len_anchor;
dim3           dim3d, dim3d_pad, stride3d, stride3d_pad, anchor_dim3, anchor_stride3;
constexpr auto Block = 8;

auto eb             = 1.0f;
auto eb_r           = 1.0f;
auto ebx2           = 2.0f;
auto ebx2_r         = 0.5f;
auto radius         = 512;
auto get_npart      = [](auto size, auto subsize) { return (size + subsize - 1) / subsize; };
auto get_npart_pad1 = [](auto size, auto subsize) { return (size + subsize - 2) / subsize; };

std::string fname;

// void test_lorenzo3dc(int n = 1)
// {
//     cout << "testing lorenzo3d\n";
//     auto stridey = dimx, stridez = dimx * dimy;

//     auto dim_block = dim3(32, 1, 8);
//     auto dim_grid  = dim3(
//         get_npart(dimx, 32),  //
//         get_npart(dimy, 8),   //
//         get_npart(dimz, 8)    //
//     );

//     cudaMallocManaged((void**)&errctrl, len * sizeof(Data));
//     cudaMemset(data, 0x00, len * sizeof(Data));

//     // for (auto i = 0; i < n; i++) {
//     // cout << "3Dc " << i << '\n';
//     kernel::c_lorenzo_3d1l_v1_32x8x8data_mapto_32x1x8<Data, Quant><<<dim_grid, dim_block>>>  //
//         (data, errctrl, dimx, dimy, dimz, stridey, stridez, radius, ebx2_r);
//     cudaDeviceSynchronize();

//     io::WriteArrayToBinary(fname + ".lorenzo", errctrl, len);

//     cudaFree(errctrl);
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
    printf("\nCPU print end\n\n\n");

    printf("print *sv: \"x y z val\"\n");

    for (auto z = 0; z < (Block + (int)Padding); z++) {
        for (auto y = 0; y < (Block + (int)Padding); y++) {
            for (auto x = 0; x < (4 * Block + (int)Padding); x++) {  //
                auto gid = x + y * dimx_pad + z * dimx_pad * dimy_pad;
                auto c   = (int)data[gid] - radius;
                if (c != 0) printf("%d %d %d %d\n", x, y, z, c);
            }
        }
    }
    printf("\n");
}

void test_spline3dc()
{
    cout << "testing spline3d\n";

    {  //
        nblockx    = get_npart(dimx, Block * 4);
        nblocky    = get_npart(dimy, Block);
        nblockz    = get_npart(dimz, Block);
        dimx_pad   = nblockx * 32;  // 235 -> 256
        dimy_pad   = nblocky * 8;   // 449 -> 456
        dimz_pad   = nblockz * 8;   // 449 -> 456
        len_padded = dimx_pad * dimy_pad * dimz_pad;

        // TODO: an alternative
        // auto nblockx = get_npart_pad1(dimx, Block * 4);
        // auto nblocky = get_npart_pad1(dimy, Block);
        // auto nblockz = get_npart_pad1(dimz, Block);

        dim3d        = dim3(dimx, dimy, dimz);
        stride3d     = dim3(1, dimx, dimx * dimy);
        dim3d_pad    = dim3(dimx_pad, dimy_pad, dimz_pad);
        stride3d_pad = dim3(1, dimx_pad, dimx_pad * dimy_pad);

        std::cout << "len: " << len << '\n';
        std::cout << "len padded: " << len_padded << '\n';
        std::cout << "len padded/len: " << 1.0 * len_padded / len << '\n';
        printf("dim and dimpad: (%d, %d, %d), (%d, %d, %d)\n", dimx, dimy, dimz, dimx_pad, dimy_pad, dimz_pad);
    }

    {  // anchor point

        in_range_nanchorx = int(dimx / Block);
        in_range_nanchory = int(dimy / Block);
        in_range_nanchorz = int(dimz / Block);
        nanchorx          = in_range_nanchorx + 1;
        nanchory          = in_range_nanchory + 1;
        nanchorz          = in_range_nanchorz + 1;
        len_anchor        = nanchorx * nanchory * nanchorz;

        anchor_dim3    = dim3(nanchorx, nanchory, nanchorz);
        anchor_stride3 = dim3(1, nanchorx, nanchorx * nanchory);

        std::cout << "len anchor: " << len_anchor << '\n';
        printf("len anchor xyz: (%d, %d, %d)\n", nanchorx, nanchory, nanchorz);
        std::cout << "len anchor: " << len_anchor << '\n';

        // end of block
    }

    // launch kernel of handling anchor

    cudaMallocManaged((void**)&anchor, len_anchor * sizeof(Data));
    cudaMemset(anchor, 0x00, len_anchor * sizeof(Data));
    {  //

        // TODO spline3d_handle_4x1x1anchors
        // kernel::spline3d_handle_4x1x1anchors<Data, 128, 8, 8, 8, Gather>
        //     <<<dim3(get_npart(nblockx + 1, 8), get_npart(nblocky + 1, 8), get_npart(nblockz + 1, 8)),  //
        //        dim3(128, 1, 1)>>>                                                                      //
        //     (data, dim3d, stride3d, anchor,                                                            //
        //      dim3(nblockx + 1, nblocky + 1, nblockz + 1),                                              //
        //      dim3(1, nblockx + 1, (nblockx + 1) * (nblocky + 1))                                       //
        //     );

        for (auto iz = 0, z = 0; z < dimz; iz++, z += 8) {
            for (auto iy = 0, y = 0; y < dimy; iy++, y += 8) {
                for (auto ix = 0, x = 0; x < dimx; ix++, x += 8) {
                    auto data_id      = x + y * stride3d.y + z * stride3d.z;
                    auto anchor_id    = ix + iy * anchor_stride3.y + iz * anchor_stride3.z;
                    anchor[anchor_id] = data[data_id];
                }
            }
        }
        /*
        // print
        for (auto z = 0; z < nanchorz; z++) {
            for (auto y = 0; y < nanchory; y++) {
                for (auto x = 0; x < nanchorx; x++) {
                    auto anchor_id = x + y * anchor_stride3.y + z * anchor_stride3.z;
                    printf("(%d,%d,%d): %4.2e\t", x * 8, y * 8, z * 8, anchor[anchor_id]);
                }
                cout << '\n';
            }
            cout << "\n\n";
        }
         */
        // end of block
    }

    cudaMallocManaged((void**)&errctrl, len_padded * sizeof(Quant));
    cudaMemset(errctrl, 0x00, len_padded * sizeof(Quant));
    {  // launch kernel of pred-quant

        for (auto i = 0; i < 20; i++) {
            kernel::c_spline3d_infprecis_32x8x8data<Data*, Quant*, float, 256, false, false>
                <<<dim3(nblockx, nblocky, nblockz), dim3(256, 1, 1)>>>  //
                (data, dim3d, stride3d,                                 //
                 errctrl, dim3d_pad, stride3d_pad,                      //
                 eb_r, ebx2, radius);
            cudaDeviceSynchronize();
        }
    }

    {  // verification
       // print_block_from_CPU<Data, true>(data);

        // print_block_from_CPU<Quant, false, true>(errctrl);
    }

    auto hist = new int[radius * 2]();

    if (print_hist) {
        for (auto i = 0; i < len_padded; i++) { hist[errctrl[i]]++; }
        for (auto i = 0; i < radius * 2; i++) {
            if (hist[i] != 0) std::cout << i << '\t' << hist[i] << '\n';
        }
    }

    io::WriteArrayToBinary(fname + ".spline", errctrl, len_padded);

    cudaMallocManaged((void**)&xdata, len * sizeof(Data));
    cudaMemset(xdata, 0x00, len * sizeof(Data));
    {
        for (auto i = 0; i < 20; i++) {
            kernel::x_spline3d_infprecis_32x8x8data<Quant*, Data*, float, 256, false>
                <<<dim3(nblockx, nblocky, nblockz), dim3(256, 1, 1)>>>  //
                (errctrl, dim3d_pad, stride3d_pad,                      //
                 anchor, anchor_dim3, anchor_stride3,                   //
                 xdata, dim3d, stride3d,                                //
                 eb_r, ebx2, radius);
            cudaDeviceSynchronize();
        }
    }

    {  // verification
        auto verified_okay = true;
        for (auto i = 0; i < len; i++) {
            auto err = fabs(data[i] - xdata[i]);

            if (err > eb) {
                printf("overbound first at idx: %d, data:%4.2e, xdata: %4.2e, exiting\n", i, data[i], xdata[i]);
                break;
            }
        }

        cout << '\n';
        if (verified_okay)
            printf(">> PASSED error boundness check.\n");
        else
            printf("** FAILED error boundness check.\n.");

        stat_t stat_gpu;
        VerifyDataGPU(&stat_gpu, xdata, data, len);
        analysis::PrintMetrics<Data>(&stat_gpu, false, eb, 0, 1, false, true);

        stat_t stat;
        analysis::VerifyData<Data>(&stat, xdata, data, len);
        analysis::PrintMetrics<Data>(&stat, false, eb, 0, 1, true, false);
        // printf("data[max-err-idx]: %f\n", data[stat.max_abserr_index]);
        // printf("xdata[max-err-idx]: %f\n", xdata[stat.max_abserr_index]);
    }

    cudaFree(errctrl);
    cudaFree(anchor);
}

int main(int argc, char** argv)
{
    // auto           dimx = 512, dimy = 512, dimz = 512;
    dimz = 449, dimy = 449, dimx = 235;
    len     = dimx * dimy * dimz;
    auto eb = 1e-2;

    if (argc < 3) {
        std::cout << "<prog> <file> <eb> <hist>" << '\n';
        std::cout << "e.g. \"./spline ${HOME}/rtm-data/snapshot-2815.f32 1e-2\"" << '\n';
        std::cout << '\n';
        struct passwd* pw      = getpwuid(getuid());
        const char*    homedir = pw->pw_dir;
        fname                  = std::string(homedir) + std::string("/rtm-data/snapshot-2815.f32");
    }
    else if (argc == 3) {
        fname = std::string(argv[1]);
        eb    = atof(argv[2]);
    }
    else if (argc == 4) {
        fname      = std::string(argv[1]);
        eb         = atof(argv[2]);
        print_hist = std::string(argv[3]) == "hist";
    }

    cudaDeviceReset();

    cudaMallocManaged((void**)&data, len * sizeof(Data));
    cudaMemset(data, 0x00, len * sizeof(Data));

    std::cout << "opening " << fname << std::endl;
    io::ReadBinaryToArray(fname, data, len);

    thrust::device_ptr<Data> g_ptr      = thrust::device_pointer_cast(data);
    auto                     max_el_loc = thrust::max_element(g_ptr, g_ptr + len);  // excluding padded
    auto                     min_el_loc = thrust::min_element(g_ptr, g_ptr + len);  // excluding padded
    // auto   max_el_loc = std::max_element(data, data + len);
    // auto   min_el_loc = std::min_element(data, data + len);
    double max_value = *max_el_loc;
    double min_value = *min_el_loc;
    double rng       = max_value - min_value;

    std::cout << "range: " << rng << '\n';
    std::cout << "input eb: " << eb << '\n';

    eb *= rng;
    eb_r   = 1 / eb;
    ebx2   = eb * 2;
    ebx2_r = 1 / ebx2;

    test_spline3dc();
    // test_lorenzo3dc();

    cudaFree(data);

    return 0;
}
