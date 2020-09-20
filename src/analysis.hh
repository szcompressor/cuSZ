#ifndef ANALYSIS_HH
#define ANALYSIS_HH

/**
 * @file analysis.hh
 * @author Jiannan Tian
 * @brief Analysis and visualization of datum.
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-02-09
 *
 * @copyright Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <vector>

using std::cerr;
using std::cout;
using std::endl;
using std::tuple;

namespace Analysis {
template <typename T>
tuple<double, double, double> getStat(T* __d, size_t l, bool print = false)
{
    double Min = *std::min_element(__d, __d + l);
    double Max = *std::max_element(__d, __d + l);
    double sum = std::accumulate(__d, __d + l, 0);
    double rng = Max - Min;
    double avg = sum / l;
    if (print) {
        cout << "rng: " << rng << endl;
        cout << "min: " << Min << endl;
        cout << "max: " << Max << endl;
        cout << "avg: " << avg << endl;
    }
    return std::make_tuple(Max, Min, rng);
}

template <typename T>
tuple<double, double, double> getStat(std::vector<T> __d, bool print = false)
{
    double Min = *std::min_element(__d.begin(), __d.end());
    double Max = *std::max_element(__d.begin(), __d.end());
    double sum = std::accumulate(__d.begin(), __d.end(), 0);
    double rng = Max - Min;
    double avg = sum / __d.size();
    if (print) {
        cout << "rng: " << rng << endl;
        cout << "min: " << Min << endl;
        cout << "max: " << Max << endl;
        cout << "avg: " << avg << endl;
    }
    return std::make_tuple(Max, Min, rng);
}

template <typename T>
void getEntropy(T* code, size_t l, size_t cap = 1024)
{
    if (cap == 0) {
        cerr << "wrong cap" << endl;
        exit(1);
    }
    auto arr = new size_t[cap]();
    for (size_t i = 0; i < l; i++) arr[code[i]]++;
    std::vector<double> raw(arr, arr + cap);
    std::vector<double> frequencies;
    std::copy_if(raw.begin(), raw.end(), std::back_inserter(frequencies), [](double& e) { return e != 0; });
    double entropy = 0;
    for (auto freq : frequencies) {
        //        cout << -(freq / l) * log2(freq / l) << endl;
        entropy += -(freq / l) * log2(freq / l);
    }

    cout << "entropy:\t" << entropy << endl;
    delete[] arr;
}

// TODO automatically omit bins that are less than 1%
template <typename T>
void histogram(
    const std::string& tag,
    T*                 __d_POD,
    size_t             l,
    size_t             __bins                  = 16,
    bool               log_freq                = false,
    double             override_min            = 0,
    double             override_max            = 0,
    bool               eliminate_zeros         = false,
    bool               use_scientific_notation = true)
{
    std::vector<T> __d(__d_POD, __d_POD + l);
    std::vector<T> __d_nonzero;
    //    std::vector<size_t> arr;
    //    arr.reserve(__bins);
    //    for (size_t i = 0; i< __bins; i++) arr.push_back(0);
    auto arr = new size_t[__bins]();

    if (eliminate_zeros) {
        std::copy_if(__d.begin(), __d.end(), std::back_inserter(__d_nonzero), [](int i) { return i != 0; });
    }
    double Min = *std::min_element(__d.begin(), __d.end());
    double Max = *std::max_element(__d.begin(), __d.end());
    //    double sum = std::accumulate(__d.begin(), __d.end(), 0);
    double rng = Max - Min;
    //    double avg = sum / l;

    cout << "\e[7m[[" << tag << "]]\e[0m";
    if (override_max > override_min) {
        cout << "zoom into " << override_min << "--" << override_max << endl;
        std::tie(Max, Min, rng) = std::make_tuple(override_max, override_min, override_max - override_min);
    }
    double step = rng / __bins;
    for (size_t i = 0; i < l; i++) arr[static_cast<size_t>((__d[i] - Min) / step)]++;
    std::vector<size_t> __viz(arr, arr + __bins);
    //    std::vector<size_t> __viz(arr);

    // visualization
    printf("\tbins:\t%zu\tbin_width:\t%lf\n", __bins, step);
    //    printf("count:\t%zu\tmin:\t%lf\tmax:\t%lf\trng:\t%lf\n", l, Min, Max, rng);
    cout << "count:\t" << l << "\t";
    cout << "min:\t" << Min << "\t";
    cout << "max:\t" << Max << "\t";
    cout << "rng:\t" << rng << endl;

    if (log_freq) {
        cout << "using log_freq" << endl;
        std::for_each(__viz.begin(), __viz.end(), [](size_t& n) { n = log2(n); });
    }

    size_t longest     = *std::max_element(__viz.begin(), __viz.end());
    size_t bar_str_len = 64;  // scale according to the longest
    std::for_each(__viz.begin(), __viz.end(), [&](size_t& n) {
        n = static_cast<size_t>(n / static_cast<double>(longest) * bar_str_len);
    });

    for (size_t i = 0; i < __bins; i++) {
        // normalize to width
        cout << "|"
             << "\33[43m";

        for (size_t j = 0; j < bar_str_len + 1; j++) {
            if (j < __viz[i])
                cout << "-";
            else if (j == __viz[i])
                cout << "\33[0m"
                     << "+";
            else
                cout << " ";
        }
        cout.precision(2);
        cout << "    ";
        if (use_scientific_notation) cout << std::scientific;
        cout << Min + i * step << " -- " << Min + (i + 1) * step;
        cout << "  ";
        cout << std::setw((int)log10(l) + 2);
        cout << arr[i];
        cout << "   ";
        cout << std::defaultfloat << std::setw(5) << arr[i] / static_cast<double>(l) * 100 << "%" << endl;
    }
    cout << endl;
    //    delete[] arr;
}

}  // namespace Analysis

#endif
