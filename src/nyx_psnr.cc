//
// Created by JianNan Tian on 2/29/20.
//

#include <iostream>
#include <string>

#include "__io.hh"
#include "verify.hh"

using Analysis::psnr;

int main() {
    std::string path("/Users/jtian/WorkSpace/SDRbench/05_NYX_512x512x512/");
    auto        orin = io::ReadBinaryFile<float>(path + "baryon_density.dat", 512 * 512 * 512);

    auto asz14x = io::ReadBinaryFile<float>(path + "baryon_density.dat.sz.out", 512 * 512 * 512);
    auto psz14x = io::ReadBinaryFile<float>(path + "baryon_density.dat.psz.sz14.out", 512 * 512 * 512);

    auto asz14_prederr = io::ReadBinaryFile<float>(path + "baryon_density.dat.asz14.prederr", 512 * 512 * 512);
    auto asz14_xerr    = io::ReadBinaryFile<float>(path + "baryon_density.dat.asz14.xerr", 512 * 512 * 512);

    auto psz14_prederr = io::ReadBinaryFile<float>(path + "baryon_density.dat.psz.sz14.prederr", 512 * 512 * 512);
    auto psz14_xerr    = io::ReadBinaryFile<float>(path + "baryon_density.dat.psz.sz14.xerr", 512 * 512 * 512);

    cout << endl;
    //    for (size_t i = 2; i < 3; i++) {
    //        for (size_t j = 0; j < 512; j++) {
    //            for (size_t k = 0; k < 512; k++) {
    //                if (j > 50 and j < 60 and k > 100 and k < 110) {
    //                    size_t gid = k + j * 512 + i * 512 * 512;
    //                    cout << "origin:\t" << orin[gid] << "\n";
    //
    //                    cout << "asz_14x\t" << asz14x[gid] << "\t ";
    //                    cout << "asz_prede\t" << asz14_prederr[gid] << "\t";
    //                    cout << "asz_xerr\t" << asz14_xerr[gid] << "\n";
    //
    //                    cout << "psz_14x\t" << psz14x[gid] << "\t";
    //                    cout << "psz_prede\t" << psz14_prederr[gid] << "\t";
    //                    cout << "psz_xerr\t" << psz14_xerr[gid] << "\n";
    //                    cout << endl;
    //                }
    //            }
    //        }
    //    }


    print(asz14x)

//    for (size_t i = 0; i < 512; i++) {
//        psnr(asz14x + i * 512 * 512, orin + i * 512 * 512, 512 * 512);
//        psnr(psz14x + i * 512 * 512, orin + i * 512 * 512, 512 * 512);
//        cout << endl;
//    }
}
