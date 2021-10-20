/**
 * @file ex_spline3_demo1.cu
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

#include "../src/common.hh"
#include "../src/kernel/spline3.cuh"
#include "../src/utils.hh"
#include "../src/wrapper/interp_spline3.cuh"

using std::cout;

using T = float;
// using E = unsigned short;
using E = float;

bool print_fullhist = false;
bool write_quant    = false;

constexpr unsigned int dimz = 449, dimy = 449, dimx = 235;
constexpr unsigned int len    = dimx * dimy * dimz;
constexpr auto         BLOCK  = 8;
constexpr auto         radius = 512;

auto get_npart = [](auto size, auto subsize) { return (size + subsize - 1) / subsize; };

std::string fname;

/*
template <typename T>
void c_gather_anchor_cpu(
    T*           data,
    unsigned int dimx,
    unsigned int dimy,
    unsigned int dimz,
    dim3         leap,
    T*           anchor,
    dim3         anchor_leap)
{
    for (auto iz = 0, z = 0; z < dimz; iz++, z += 8) {
        for (auto iy = 0, y = 0; y < dimy; iy++, y += 8) {
            for (auto ix = 0, x = 0; x < dimx; ix++, x += 8) {
                auto data_id      = x + y * leap.y + z * leap.z;
                auto anchor_id    = ix + iy * anchor_leap.y + iz * anchor_leap.z;
                anchor[anchor_id] = data[data_id];
            }
        }
    }
}

template <typename T, bool PRINT_FP = false, bool PADDING = true>
void print_block_from_CPU(T* data, int radius = 512)
{
    cout << "dimxpad: " << dimx_aligned << "\tdimypad: " << dimy_aligned << '\n';

    for (auto z = 0; z < (BLOCK + (int)PADDING); z++) {
        printf("\nprint from CPU, z=%d\n", z);
        printf("    ");
        for (auto i = 0; i < 33; i++) printf("%3d", i);
        printf("\n");

        for (auto y = 0; y < (BLOCK + (int)PADDING); y++) {
            printf("y=%d ", y);

            for (auto x = 0; x < (4 * BLOCK + (int)PADDING); x++) {  //
                auto gid = x + y * dimx_aligned + z * dimx_aligned * dimy_aligned;
                if CONSTEXPR (PRINT_FP) { printf("%.2e\t", data[gid]); }
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

    for (auto z = 0; z < (BLOCK + (int)PADDING); z++) {
        for (auto y = 0; y < (BLOCK + (int)PADDING); y++) {
            for (auto x = 0; x < (4 * BLOCK + (int)PADDING); x++) {  //
                auto gid = x + y * dimx_aligned + z * dimx_aligned * dimy_aligned;
                auto c   = (int)data[gid] - radius;
                if (c != 0) printf("%d %d %d %d\n", x, y, z, c);
            }
        }
    }
    printf("\n");
}
*/

void prescan(T* data, unsigned int len, double& max_value, double& min_value, double& rng)
{
    thrust::device_ptr<T> g_ptr      = thrust::device_pointer_cast(data);
    auto                  max_el_loc = thrust::max_element(g_ptr, g_ptr + len);  // excluding padded
    auto                  min_el_loc = thrust::min_element(g_ptr, g_ptr + len);  // excluding padded

    max_value = *max_el_loc;
    min_value = *min_el_loc;
    rng       = max_value - min_value;
}

void test_spline3d_wrapped(double _eb)
{
    constexpr auto MODE = cuszDEV::TEST;
    constexpr auto LOC  = cuszLOC::UNIFIED;

    Capsule<T, true> data(len);
    Capsule<T, true> xdata(len);

    data.alloc<MODE, LOC>();
    xdata.alloc<MODE, LOC>();

    data.from_fs_to<LOC>(fname);

    double max_value, min_value, rng;
    prescan(data.get<LOC>(), len, max_value, min_value, rng);

    double eb     = _eb * rng;
    double eb_r   = 1 / eb;
    double ebx2   = eb * 2;
    double ebx2_r = 1 / ebx2;

    std::cout << "wrapped:\n";
    std::cout << "opening " << fname << std::endl;
    std::cout << "input eb: " << _eb << '\n';
    std::cout << "range: " << rng << '\n';
    std::cout << "r2r eb: " << eb << '\n';

    cusz::Spline3<T, E, float> predictor(dim3(dimx, dimy, dimz), eb, 512);

    std::cout << "predictor.get_anchor_len() = " << predictor.get_anchor_len() << '\n';
    std::cout << "predictor.get_quant_len() = " << predictor.get_quant_len() << '\n';

    Capsule<T, true> anchor(predictor.get_anchor_len());
    Capsule<E, true> errctrl(predictor.get_quant_len());
    anchor.alloc<MODE, LOC>();
    errctrl.alloc<MODE, LOC>();

    predictor.construct(data.get<LOC>(), anchor.get<LOC>(), errctrl.get<LOC>());

    // {
    //     auto hist = new int[radius * 2]();
    //     for (auto i = 0; i < predictor.get_quant_len(); i++) hist[(int)errctrl.get<LOC>()[i]]++;
    //     for (auto i = 0; i < radius * 2; i++)
    //         if (hist[i] != 0) std::cout << i << '\t' << hist[i] << '\n';

    //     delete[] hist;
    // }

    predictor.reconstruct(anchor.get<LOC>(), errctrl.get<LOC>(), xdata.get<LOC>());

    data.from_fs_to<LOC>(fname);
    stat_t stat;
    analysis::verify_data<T>(&stat, xdata.get<LOC>(), data.get<LOC>(), len);
    analysis::print_data_quality_metrics<T>(&stat, 0, false);

    errctrl.free<cuszDEV::DEV, LOC>();
    anchor.free<MODE, LOC>();
    data.free<MODE, LOC>();
    xdata.free<MODE, LOC>();
}

void test_spline3d_proto(double _eb)
{
    std::cout << "prototype:\n";
    std::cout << "input eb: " << _eb << '\n';

    T* data;
    T* xdata;
    T* anchor;
    E* errctrl;

    unsigned int dimx_aligned, dimy_aligned, dimz_aligned;
    unsigned int nblockx, nblocky, nblockz;
    unsigned int in_range_nanchorx, in_range_nanchory, in_range_nanchorz;
    unsigned int nanchorx, nanchory, nanchorz;
    unsigned int len_aligned, len_anchor;
    dim3         size, size_aligned, leap, leap_aligned, anchor_size, anchor_leap;

    double max_value, min_value, rng;

    cudaMallocManaged((void**)&data, len * sizeof(T));
    cudaMemset(data, 0x00, len * sizeof(T));

    std::cout << "opening " << fname << std::endl;
    io::read_binary_to_array(fname, data, len);

    prescan(data, len, max_value, min_value, rng);

    auto eb     = _eb *= rng;
    auto eb_r   = 1 / eb;
    auto ebx2   = eb * 2;
    auto ebx2_r = 1 / ebx2;

    std::cout << "range: " << rng << '\n';
    std::cout << "r2r eb: " << eb << '\n';

    {  //
        nblockx      = get_npart(dimx, BLOCK * 4);
        nblocky      = get_npart(dimy, BLOCK);
        nblockz      = get_npart(dimz, BLOCK);
        dimx_aligned = nblockx * 32;  // 235 -> 256
        dimy_aligned = nblocky * 8;   // 449 -> 456
        dimz_aligned = nblockz * 8;   // 449 -> 456
        len_aligned  = dimx_aligned * dimy_aligned * dimz_aligned;

        // TODO: an alternative
        // auto nblockx = get_npart_aligned1(dimx, BLOCK * 4);
        // auto nblocky = get_npart_aligned1(dimy, BLOCK);
        // auto nblockz = get_npart_aligned1(dimz, BLOCK);

        size         = dim3(dimx, dimy, dimz);
        leap         = dim3(1, dimx, dimx * dimy);
        size_aligned = dim3(dimx_aligned, dimy_aligned, dimz_aligned);
        leap_aligned = dim3(1, dimx_aligned, dimx_aligned * dimy_aligned);

        std::cout << "len: " << len << '\n';
        std::cout << "len padded: " << len_aligned << '\n';
        std::cout << "len padded/len: " << 1.0 * len_aligned / len << '\n';
        printf(
            "dim and dim-aligned: (%d, %d, %d), (%d, %d, %d)\n", dimx, dimy, dimz, dimx_aligned, dimy_aligned,
            dimz_aligned);
    }

    {  // anchor point

        in_range_nanchorx = int(dimx / BLOCK);
        in_range_nanchory = int(dimy / BLOCK);
        in_range_nanchorz = int(dimz / BLOCK);
        nanchorx          = in_range_nanchorx + 1;
        nanchory          = in_range_nanchory + 1;
        nanchorz          = in_range_nanchorz + 1;
        len_anchor        = nanchorx * nanchory * nanchorz;

        anchor_size = dim3(nanchorx, nanchory, nanchorz);
        anchor_leap = dim3(1, nanchorx, nanchorx * nanchory);

        std::cout << "len anchor: " << len_anchor << '\n';
        printf("len anchor xyz: (%d, %d, %d)\n", nanchorx, nanchory, nanchorz);

        // end of block
    }

    // launch kernel of handling anchor

    cudaMallocManaged((void**)&anchor, len_anchor * sizeof(T));
    cudaMemset(anchor, 0x00, len_anchor * sizeof(T));

    cudaMallocManaged((void**)&errctrl, len_aligned * sizeof(E));
    cudaMemset(errctrl, 0x00, len_aligned * sizeof(E));
    {  // launch kernel of pred-quant

        for (auto i = 0; i < 20; i++) {
            cusz::c_spline3d_infprecis_32x8x8data<T*, E*, float, 256, false>
                <<<dim3(nblockx, nblocky, nblockz), dim3(256, 1, 1)>>>  //
                (data, size, leap,                                      //
                 errctrl, size_aligned, leap_aligned,                   //
                 anchor, anchor_leap,                                   //
                 eb_r, ebx2, radius);
            cudaDeviceSynchronize();
        }
    }

    {
        // verification
        // print_block_from_CPU<T, true>(data);
        // print_block_from_CPU<E, false, true>(errctrl);
    }

    {
        std::cout << "calculating sparsity part\n";
        auto count = 0;
        for (auto i = 0; i < len_aligned; i++) {
            auto code = errctrl[i];
            if (code != radius) count++;
        }
        double percent     = 1 - (count * 1.0) / len_aligned;
        double sparsity_cr = 1 / (1 - percent) / 2;
        printf("non-zero offset count:\t%d\tpercentage:\t%.8lf%%\tCR:\t%.4lf\n", count, percent * 100, sparsity_cr);
    }

    // auto hist = new int[radius * 2]();
    // if (print_fullhist) {
    //     for (auto i = 0; i < len_aligned; i++) { hist[(int)errctrl[i]]++; }
    //     for (auto i = 0; i < radius * 2; i++) {
    //         if (hist[i] != 0) std::cout << i << '\t' << hist[i] << '\n';
    //     }
    // }

    if (write_quant) io::write_array_to_binary(fname + ".spline", errctrl, len_aligned);

    cudaMallocManaged((void**)&xdata, len * sizeof(T));
    cudaMemset(xdata, 0x00, len * sizeof(T));
    {
        for (auto i = 0; i < 20; i++) {
            cusz::x_spline3d_infprecis_32x8x8data<E*, T*, float, 256>
                <<<dim3(nblockx, nblocky, nblockz), dim3(256, 1, 1)>>>  //
                (errctrl, size_aligned, leap_aligned,                   //
                 anchor, anchor_size, anchor_leap,                      //
                 xdata, size, leap,                                     //
                 eb_r, ebx2, radius);
            cudaDeviceSynchronize();
        }
    }

    {  // verification
        auto verified_okay = true;
        for (auto i = 0; i < len; i++) {
            auto err = fabs(data[i] - xdata[i]);

            if (err > eb) {
                verified_okay = false;
                printf("overbound first at idx: %d, data:%4.2e, xdata: %4.2e, exiting\n", i, data[i], xdata[i]);
                break;
            }
        }

        cout << '\n';
        if (verified_okay)
            printf(">> PASSED error boundness check.\n");
        else
            printf("** FAILED error boundness check.\n.");

        // stat_t stat_gpu;
        // verify_data_GPU(&stat_gpu, xdata, data, len);
        // analysis::print_data_quality_metrics<T>(&stat_gpu, false, eb, 0, 1, false, true);

        stat_t stat;
        analysis::verify_data<T>(&stat, xdata, data, len);
        analysis::print_data_quality_metrics<T>(&stat, 0, false);

        // printf("data[max-err-idx]: %f\n", data[stat.max_abserr_index]);
        // printf("xdata[max-err-idx]: %f\n", xdata[stat.max_abserr_index]);
    }

    cudaFree(errctrl);
    cudaFree(anchor);
    cudaFree(data);
}

int main(int argc, char** argv)
{
    double eb = 1e-2;

    if (argc < 3) {
        std::cout << "<prog> <file> <eb> <hist>" << '\n';
        std::cout << "e.g. \"./spline ${HOME}/Develop/dev-env-cusz/rtm-data/snapshot-2815.f32 1e-2\"" << '\n';
        std::cout << '\n';
        struct passwd* pw      = getpwuid(getuid());
        const char*    homedir = pw->pw_dir;
        fname                  = std::string(homedir) + std::string("/Develop/dev-env-cusz/rtm-data/snapshot-2815.f32");
    }
    else if (argc == 3) {
        fname = std::string(argv[1]);
        eb    = atof(argv[2]);
    }
    else if (argc == 4) {
        fname          = std::string(argv[1]);
        eb             = atof(argv[2]);
        print_fullhist = std::string(argv[3]) == "hist";
    }
    else if (argc == 5) {
        fname          = std::string(argv[1]);
        eb             = atof(argv[2]);
        print_fullhist = std::string(argv[3]) == "hist";
        write_quant    = std::string(argv[4]) == "write.quant";
    }

    cudaDeviceReset();

    test_spline3d_proto(eb);

    test_spline3d_wrapped(eb);

    return 0;
}
