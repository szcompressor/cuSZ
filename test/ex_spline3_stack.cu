/**
 * @file ex_spline3_stack.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-11-23
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include "../src/sp_path.cuh"

#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>
#include <iomanip>
#include <sstream>

using std::cout;

using T = float;
using E = float;

bool print_fullhist = false;
bool write_quant    = false;

constexpr unsigned int dimz = 449, dimy = 449, dimx = 235;
// constexpr unsigned int len = dimx * dimy * dimz;

std::string fname;

int main(int argc, char** argv)
{
    double eb = 3e-3 * 4e-03;

    cudaDeviceReset();

    std::vector<std::string> filelist;

    auto path = "/home/jtian/nvme/dev-env-cusz/aramco-data/";

    for (auto i = 600; i <= 2800; i += 20) {
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(4) << i;
        filelist.push_back(path + ss.str() + ".f32");
        // cout << "adding " << ss.str() + ".f32 to list\n";
    }
    SparsityAwarePath::DefaultCompressor compressor(dim3(dimx, dimy, dimz));

    compressor.exp_stack_absmode(filelist, eb, path, false, false, true);

    return 0;
}
