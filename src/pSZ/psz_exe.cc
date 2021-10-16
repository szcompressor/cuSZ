/**
 * @file psz_exe.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.1.4
 * @date 2020-02-13
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cstring>
#include <string>
#include <vector>

#include "../common.hh"
#include "../datasets.hh"
#include "../utils/verify.hh"
#include "psz_workflow.hh"

namespace fm = psz::FineMassiveSimulation;

const size_t DICT_SIZE = 1024;
// const size_t DICT_SIZE = 4096;
// #if defined(_1D)
// const size_t BLK = 256;
// #elif defined(_2D)
// const size_t BLK = 16;
// #elif defined(_3D)
// const size_t BLK = 8;
// #endif

int main(int argc, char** argv)
{
    std::string eb_mode, dataset, datum_path;
    bool        if_blocking, if_dualquant;
    double      mantissa, exponent;

#if defined(_1D)
    cout << "\e[46mThis program is working for 1D datasets.\e[0m" << endl;
#elif defined(_2D)
    cout << "\e[46mThis program is working for 2D datasets.\e[0m" << endl;
#elif defined(_3D)
    cout << "\e[46mThis program is working for 3D datasets.\e[0m" << endl;
#endif

    if (argc != 8) {
        cout << "./<program> <abs|rel2range OR r2r> <mantissa> <exponent> <if blocking> <if dualquant> <dataset> "
                "<datum_path path>"
             << endl;
        cout << "supported dimension and datasets" << endl;
        cout << "\t1D\t./psz1d r2r 1.23 -4.56 <noblk|yesblk> <nodq|dq> <hacc> /path/to/vx.f32" << endl;
        cout << "\t2D\t./psz2d r2r 1.23 -4.56 <noblk|yesblk> <nodq|dq> <cesm> /path/to/CLDHGH_1_1800_3600.f32" << endl;
        cout << "\t3D\t./psz3d r2r 1.23 -4.56 <noblk|yesblk> <nodq|dq> <hurricane|nyx|qmc|qmcpre> "
                "/path/to/CLOUDf48.bin.f32"
             << endl;
        exit(0);
    }
    else {
        eb_mode      = std::string(argv[1]);
        mantissa     = std::stod(argv[2]);
        exponent     = std::stod(argv[3]);
        if_blocking  = std::string(argv[4]) == "yesblk";
        if_dualquant = std::string(argv[5]) == "dq";
        dataset      = std::string(argv[6]);
        datum_path   = std::string(argv[7]);
    }

    for_each(argv, argv + 8, [](auto i) { cout << i << " "; });
    cout << endl;
    auto eb_config = new config_t(DICT_SIZE, mantissa, exponent);
    auto dims_L16  = InitializeDemoDims(dataset, DICT_SIZE);
    printf("%-20s%s\n", "filename", datum_path.c_str());
    printf("%-20s%lu\n", "filesize", dims_L16[LEN] * sizeof(float));
    if (eb_mode == "r2r") {  // as of C++ 14, string is directly comparable?
        double value_range = GetDatumValueRange<float>(datum_path, dims_L16[LEN]);
        eb_config->ChangeToRelativeMode(value_range);
    }
    eb_config->debug();
    //    size_t c_byteSize;
    size_t num_outlier = 0;  // for calculating compression ratio

    // cout << "block size:\t" << BLK << endl;
    auto ebs_L4 = InitializeErrorBoundFamily(eb_config);
    fm::cx_sim<float, int>(datum_path, dims_L16, ebs_L4, num_outlier, if_dualquant, if_blocking, true);
}
