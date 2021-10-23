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

#include "ex_spline3_common.cuh"

#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>

using std::cout;

using T = float;
// using E = unsigned short;
using E = float;

bool print_fullhist = false;
bool write_quant    = false;

constexpr unsigned int dimz = 449, dimy = 449, dimx = 235;
constexpr unsigned int len = dimx * dimy * dimz;
// constexpr auto         BLOCK  = 8;
// constexpr auto         radius = 512;

std::string fname;

void test_spline3d_wrapped(double _eb)
{
    // constexpr auto MODE = cuszDEV::TEST;
    constexpr auto LOC = cuszLOC::UNIFIED;

    Capsule<T, true> data(len);
    Capsule<T, true> xdata(len);

    data.alloc<LOC>().from_fs_to<LOC>(fname);
    xdata.alloc<LOC>();

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
    anchor.alloc<LOC>();
    errctrl.alloc<LOC>();

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

    errctrl.free<LOC>();
    anchor.free<LOC>();
    data.free<LOC>();
    xdata.free<LOC>();
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

    cout << "--------------------------------------------------------------------------------" << endl;
    cout << "                              everything POD                                    " << endl;
    cout << "--------------------------------------------------------------------------------" << endl;
    test_spline3d_proto(fname, eb);

    cout << "--------------------------------------------------------------------------------" << endl;
    cout << "                              using Capsule<T>, OOD                            " << endl;
    cout << "--------------------------------------------------------------------------------" << endl;
    test_spline3d_wrapped(eb);

    cout << "--------------------------------------------------------------------------------" << endl;
    cout << "                              without SpReducer, OOD                            " << endl;
    cout << "--------------------------------------------------------------------------------" << endl;
    TestSpline3Wrapped t1(fname, eb);
    t1.run_test();

    cout << "--------------------------------------------------------------------------------" << endl;
    cout << "                              with SpReducer, OOD                               " << endl;
    cout << "--------------------------------------------------------------------------------" << endl;
    TestSpline3Wrapped t2(fname, eb);
    t2.run_test2();

    return 0;
}
