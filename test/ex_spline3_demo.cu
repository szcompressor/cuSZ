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
using E = float;

bool print_fullhist = false;
bool write_quant    = false;

constexpr unsigned int dimz = 449, dimy = 449, dimx = 235;
constexpr unsigned int len = dimx * dimy * dimz;

std::string fname;

void demo(double eb)
{
    // data preparation
    Capsule<T> data(len, "original data");
    data  //
        .alloc<cusz::LOC::HOST_DEVICE>()
        .from_fs_to<cusz::LOC::HOST>(fname)
        .host2device();
    auto rng = data.prescan().get_rng();

    TestSpline3Wrapped<10> compressor(data.dptr, dim3(dimx, dimy, dimz), eb * rng);

    // set external space
    Capsule<uint8_t> spdump;                                  // specify length after .compress() use some init size
    spdump.set_len(len * 4).alloc<cusz::LOC::HOST_DEVICE>();  // or from_existing_on<EXEC_SPACE>(...);
    Capsule<float> anchordump(compressor.get_anchor_len());
    anchordump.alloc<cusz::LOC::HOST_DEVICE>();

    // compression
    // ----------------------------------------
    compressor.compress2();
    auto nnz = compressor.get_nnz();
    spdump.set_len(compressor.get_exact_spdump_nbyte());
    compressor.export_after_compress2(spdump.dptr, anchordump.dptr);
    // ----------------------------------------

    Capsule<T> xdata(len, "decompressed data");
    xdata.alloc<cusz::LOC::DEVICE>();

    // decompression
    // ----------------------------------------
    compressor.decompress2(xdata.dptr, spdump.dptr, nnz, anchordump.dptr);
    compressor.data_analysis(data.hptr);
    // ----------------------------------------
}

int main(int argc, char** argv)
{
    double eb = 1e-2;

    if (argc < 2) {
        // not specifying file or eb
        std::cout << "<prog> <file> <eb>" << '\n';
        std::cout << "e.g. \"./spline ${HOME}/nvme/dev-env-cusz/rtm-data/snapshot-2815.f32 1e-2\"" << '\n';
        std::cout << '\n';

        struct passwd* pw      = getpwuid(getuid());
        const char*    homedir = pw->pw_dir;

        fname = std::string(homedir) + std::string("/nvme/dev-env-cusz/rtm-data/snapshot-2815.f32");
    }
    else if (argc < 3) {
        // specified file but not eb
        fname = std::string(argv[1]);
    }
    else if (argc == 3) {
        fname = std::string(argv[1]);
        eb    = atof(argv[2]);
    }

    cudaDeviceReset();

    demo(eb);

    return 0;
}
