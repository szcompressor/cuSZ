//
// Created by JianNan Tian on 9/12/19.
//

#if not defined(LOGGING_HH)
#define LOGGING_HH

#include <iomanip>
#include <iostream>
using std::cout;
using std::endl;

namespace logging {

template <typename T>
static void throughput(size_t const& n_els, double const& totalTime_huffman, double const& totalTime_zstd, std::string const& name) {
    std::cout << std::fixed << std::showpoint;
    std::cout << std::setprecision(4);
    cout << name << "\t"
         << "\33[43m"                                                                                                //
         << std::right << std::setw(12) << static_cast<double>(n_els * sizeof(T)) / totalTime_huffman / 1024 / 1024  //
         << " MiB/s\33[0m\t"                                                                                         //
         << std::right << std::setw(12) << static_cast<double>(n_els * sizeof(T)) / totalTime_zstd / 1024 / 1024     //
         << " MiB/s" << endl;
}

static void timeElapsed(double const& portionOfTime, double const& totalTime_huffman, double const& totalTime_zstd, std::string const& name) {
    std::cout << std::fixed << std::showpoint;
    std::cout << std::setprecision(4);
    cout << name                                   //
         << "\t" << std::setw(8) << portionOfTime  //
         << "\t" << std::setw(8) << portionOfTime / totalTime_huffman * 100 << "%\t"
         << "\t" << std::setw(8) << portionOfTime / totalTime_zstd * 100 << "%" << endl;
}

static void timeElapsed(double const& portionOfTime, std::string const& name) {
    std::cout << std::fixed << std::showpoint;
    std::cout << std::setprecision(4);
    cout << name << "\t" << portionOfTime << endl;
}

}  // namespace logging

#endif  // LOGGING_HH
