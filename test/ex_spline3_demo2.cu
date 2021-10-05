/**
 * @file ex_spline3_demo2.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-10-02
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
#include "../src/wrapper/csr10.cuh"
#include "../src/wrapper/csr11.cuh"
#include "../src/wrapper/interp_spline3.cuh"

using std::cout;

using T = float;
// using E    = unsigned short;
using BYTE = uint8_t;
using E    = float;

using REDUCER = cusz::OutlierHandler11<E>;

bool print_fullhist = false;
bool write_quant    = false;

constexpr unsigned int dimz = 449, dimy = 449, dimx = 235;
constexpr unsigned int len    = dimx * dimy * dimz;
constexpr auto         BLOCK  = 8;
constexpr auto         radius = 512;

auto get_npart = [](auto size, auto subsize) { return (size + subsize - 1) / subsize; };

std::string fname;

void prescan(T* data, unsigned int len, double& max_value, double& min_value, double& rng)
{
    thrust::device_ptr<T> g_ptr      = thrust::device_pointer_cast(data);
    auto                  max_el_loc = thrust::max_element(g_ptr, g_ptr + len);  // excluding padded
    auto                  min_el_loc = thrust::min_element(g_ptr, g_ptr + len);  // excluding padded

    max_value = *max_el_loc;
    min_value = *min_el_loc;
    rng       = max_value - min_value;
}

void test_spline3d_predictor_reducer(double _eb)
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

    cout << "wrapped:\n";
    cout << "opening " << fname << '\n';
    cout << "input eb: " << _eb << '\n';
    cout << "range: " << rng << '\n';
    cout << "r2r eb: " << eb << '\n';

    cusz::Spline3<T, E, float> predictor(dim3(dimx, dimy, dimz), eb, 512);

    cout << "predictor.get_anchor_len() = " << predictor.get_anchor_len() << '\n';
    cout << "predictor.get_quant_len() = " << predictor.get_quant_len() << '\n';
    cout << "(in use) predictor.get_quant_footprint() = " << predictor.get_quant_footprint() << '\n';

    Capsule<T, true> anchor(predictor.get_anchor_len());
    Capsule<E, true> errctrl(predictor.get_quant_footprint());
    anchor.alloc<MODE, LOC>();
    errctrl.alloc<MODE, LOC>();

    predictor.construct(data.get<LOC>(), anchor.get<LOC>(), errctrl.get<LOC>());

    // =============================================================================
    // end of compression
    // =============================================================================

    /*
        {
            auto hist = new int[radius * 2]();
            for (auto i = 0; i < predictor.get_quant_len(); i++) hist[(int)errctrl.get<LOC>()[i]]++;
            for (auto i = 0; i < radius * 2; i++)
                if (hist[i] != 0) cout << i << '\t' << hist[i] << '\n';

            delete[] hist;
        }
     */

    Capsule<BYTE, false> sp_use;  // TODO compatibility issue

    REDUCER csr_comp(predictor.get_quant_footprint());

    uint32_t sp_dump_nbyte;
    int      nnz;

    cout << "init csr nbyte: "                              //
         << SparseMethodSetup::get_init_csr_nbyte<E, int>(  //
                predictor.get_quant_footprint())            //
         << '\n';

    sp_use
        .set_len(                                           //
            SparseMethodSetup::get_init_csr_nbyte<E, int>(  //
                predictor.get_quant_footprint()))
        .alloc<MODE, cuszLOC::HOST_DEVICE>();

    auto init_sp_use  = sp_use.dptr;
    auto exact_sp_use = sp_use.hptr;

    csr_comp.gather(         //
        errctrl.get<LOC>(),  //
        // sp_use.get<cuszLOC::DEVICE>(),  //
        // sp_use.get<cuszLOC::HOST>(),    //
        init_sp_use, exact_sp_use,
        sp_dump_nbyte,  //
        nnz);

    // need to change
    sp_use.host2device();

    cudaMemset(errctrl.get<LOC>(), 0x00, errctrl.nbyte());

    REDUCER csr_decomp(predictor.get_quant_footprint(), nnz);

    csr_decomp.scatter(exact_sp_use, errctrl.get<LOC>());
    sp_use.free<MODE, cuszLOC::HOST_DEVICE>();
    // =============================================================================
    // start of decompression
    // =============================================================================
    predictor.reconstruct(anchor.get<LOC>(), errctrl.get<LOC>(), xdata.get<LOC>());

    data.from_fs_to<LOC>(fname);  // reload to verify

    stat_t stat;
    analysis::verify_data<T>(&stat, xdata.get<LOC>(), data.get<LOC>(), len);
    analysis::print_data_quality_metrics<T>(&stat, 0, false);

    errctrl.free<MODE, LOC>();
    anchor.free<MODE, LOC>();
    data.free<MODE, LOC>();
    xdata.free<MODE, LOC>();
}

int main(int argc, char** argv)
{
    double eb = 1e-2;

    if (argc < 3) {
        cout << "<prog> <file> <eb> <hist>" << '\n';
        cout << "e.g. \"./spline ${HOME}/Develop/dev-env-cusz/rtm-data/snapshot-2815.f32 1e-2\"" << '\n';
        cout << '\n';
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
        write_quant    = std::string(argv[4]) == "write.errctrl";
    }

    cudaDeviceReset();

    test_spline3d_predictor_reducer(eb);

    return 0;
}
