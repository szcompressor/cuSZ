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

#include "../src/sp_path.cuh"
#include "ex_common.cuh"

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

template <class Compressor = SparsityAwarePath::DefaultCompressor>
void demo(double eb)
{
    /********/  // data preparation
    /********/ Capsule<T> data(len, "original data");
    /********/ data  //
        .alloc<cusz::LOC::HOST_DEVICE>()
        .from_fs_to<cusz::LOC::HOST>(fname)
        .host2device();
    /********/ auto rng = data.prescan().get_rng();

    // declare ancillary structs
    auto header         = new cuszHEADER;
    auto resuable_space = new SpArchive(header, len, 10);  // assume 1/10 sparse

    // declare reusable_compressor
    Compressor compressor(data.dptr, dim3(dimx, dimy, dimz), eb * rng);

    // compression
    // ----------------------------------------
    compressor.compress();

    (*resuable_space)  //
        .set_nnz(compressor.get_nnz())
        .set_anchor_len(compressor.get_anchor_len());
    compressor.export_after_compress(  //
        resuable_space->get_spdump(),  //
        resuable_space->get_anchor());
    // ----------------------------------------

    /********/  // on-demand setup of decompressed data
    /********/ Capsule<T> xdata(len, "decompressed data");
    /********/ xdata.alloc<cusz::LOC::DEVICE>();

    // decompression
    // ----------------------------------------
    compressor.decompress(
        xdata.dptr,                    //
        resuable_space->get_spdump(),  //
        resuable_space->get_nnz(),     //
        resuable_space->get_anchor()   //
    );
    compressor.data_analysis(data.hptr);
    // ----------------------------------------

    /********/  // clean up
    /********/ data.free<cusz::LOC::HOST_DEVICE>();
    /********/ xdata.free<cusz::LOC::DEVICE>();
}

int main(int argc, char** argv)
{
    double eb = 1e-2;

    if (argc < 2) {
        // not specifying file or eb
        std::cout << "<prog> <file> <eb>" << '\n';
        std::cout << "e.g. \"./splinedemo ${HOME}/nvme/dev-env-cusz/rtm-data/snapshot-2815.f32 1e-2\"" << '\n';
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

    cout << "SparsityAwarePath::DefaultCompressor" << endl;
    demo<SparsityAwarePath::DefaultCompressor>(eb);

    cout << "Lorenzo (dryrun)" << endl;
    exp_sppath_lorenzo_quality(len, fname, eb);

    return 0;
}
