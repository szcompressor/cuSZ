// 200209

#ifndef ANALYSIS_HH
#define ANALYSIS_HH

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
tuple<double, double, double> getStat(T* __d, size_t __len, bool print = false) {
    double __min = *std::min_element(__d, __d + __len);
    double __max = *std::max_element(__d, __d + __len);
    double __sum = std::accumulate(__d, __d + __len, 0);
    double __rng = __max - __min;
    double __avg = __sum / __len;
    if (print) {
        cout << "rng: " << __rng << endl;
        cout << "min: " << __min << endl;
        cout << "max: " << __max << endl;
        cout << "avg: " << __avg << endl;
    }
    return std::make_tuple(__max, __min, __rng);
}

template <typename T>
tuple<double, double, double> getStat(std::vector<T> __d, bool print = false) {
    double __min = *std::min_element(__d.begin(), __d.end());
    double __max = *std::max_element(__d.begin(), __d.end());
    double __sum = std::accumulate(__d.begin(), __d.end(), 0);
    double __rng = __max - __min;
    double __avg = __sum / __d.size();
    if (print) {
        cout << "rng: " << __rng << endl;
        cout << "min: " << __min << endl;
        cout << "max: " << __max << endl;
        cout << "avg: " << __avg << endl;
    }
    return std::make_tuple(__max, __min, __rng);
}

template <typename T>
void getEntropy(T* __code, size_t __len, size_t __cap = 1024) {
    if (__cap == 0) {
        cerr << "wrong cap" << endl;
        exit(1);
    }
    auto __a = new size_t[__cap]();
    for (size_t i = 0; i < __len; i++) __a[__code[i]]++;
    std::vector<double> __raw(__a, __a + __cap);
    std::vector<double> frequencies;
    std::copy_if(__raw.begin(), __raw.end(), std::back_inserter(frequencies), [](double& e) { return e != 0; });
    double entropy = 0;
    for (auto freq : frequencies) {
        //        cout << -(freq / __len) * log2(freq / __len) << endl;
        entropy += -(freq / __len) * log2(freq / __len);
    }

    cout << "entropy:\t" << entropy << endl;
    delete[] __a;
}

// TODO automatically omit bins that are less than 1%
template <typename T>
void histogram(const std::string& tag,
               T*                 __d_POD,
               size_t             __len,
               size_t             __bins                  = 16,
               bool               log_freq                = false,
               double             override_min            = 0,
               double             override_max            = 0,
               bool               eliminate_zeros         = false,
               bool               use_scientific_notation = true) {
    std::vector<T> __d(__d_POD, __d_POD + __len);
    std::vector<T> __d_nonzero;
//    std::vector<size_t> __a;
//    __a.reserve(__bins);
//    for (size_t i = 0; i< __bins; i++) __a.push_back(0);
    auto           __a = new size_t[__bins]();

    if (eliminate_zeros) {
        std::copy_if(__d.begin(), __d.end(), std::back_inserter(__d_nonzero), [](int i) { return i != 0; });
    }
    double __min = *std::min_element(__d.begin(), __d.end());
    double __max = *std::max_element(__d.begin(), __d.end());
//    double __sum = std::accumulate(__d.begin(), __d.end(), 0);
    double __rng = __max - __min;
//    double __avg = __sum / __len;

    cout << "\e[7m[[" << tag << "]]\e[0m";
    if (override_max > override_min) {
        cout << "zoom into " << override_min << "--" << override_max << endl;
        std::tie(__max, __min, __rng) = std::make_tuple(override_max, override_min, override_max - override_min);
    }
    double step = __rng / __bins;
    for (size_t i = 0; i < __len; i++) __a[static_cast<size_t>((__d[i] - __min) / step)]++;
    std::vector<size_t> __viz(__a, __a + __bins);
//    std::vector<size_t> __viz(__a);

    // visualization
    printf("\tbins:\t%zu\tbin_width:\t%lf\n", __bins, step);
//    printf("count:\t%zu\tmin:\t%lf\tmax:\t%lf\trng:\t%lf\n", __len, __min, __max, __rng);
    cout << "count:\t" << __len << "\t";
    cout << "min:\t" << __min << "\t";
    cout << "max:\t" << __max << "\t";
    cout << "rng:\t" << __rng << endl;

    if (log_freq) {
        cout << "using log_freq" << endl;
        std::for_each(__viz.begin(), __viz.end(), [](size_t& n) { n = log2(n); });
    }

    size_t longest     = *std::max_element(__viz.begin(), __viz.end());
    size_t bar_str_len = 64;  // scale according to the longest
    std::for_each(__viz.begin(), __viz.end(), [&](size_t& n) { n = static_cast<size_t>(n / static_cast<double>(longest) * bar_str_len); });

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
        cout << __min + i * step << " -- " << __min + (i + 1) * step;
        cout << "  ";
        cout << std::setw((int)log10(__len) + 2);
        cout << __a[i];
        cout << "   ";
        cout << std::defaultfloat << std::setw(5) << __a[i] / static_cast<double>(__len) * 100 << "%" << endl;
    }
    cout << endl;
//    delete[] __a;
}

}  // namespace Analysis

#endif
