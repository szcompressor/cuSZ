#ifndef UTILS_VIS_STAT_HH
#define UTILS_VIS_STAT_HH

/**
 * @file vis_stat.hh
 * @author Jiannan Tian
 * @brief Analysis and visualization of datum.
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-02-09
 *
 * @copyright (C) 2020 by Washington State University, The University of
 * Alabama, Argonne National Laboratory See LICENSE in top-level directory
 *
 */

#include <iomanip>
#include <tuple>
#include <unordered_map>

#include "busyheader.hh"
#include "utils/vis_stat.hh"

using std::tuple;

template <typename T>
double get_entropy(T* code, size_t l, size_t cap)
{
  if (cap == 0) {
    cerr << "wrong cap" << endl;
    exit(-1);
  }
  auto arr = new size_t[cap]();
  for (size_t i = 0; i < l; i++) arr[code[i]]++;
  std::vector<double> raw(arr, arr + cap);
  std::vector<double> frequencies;
  std::copy_if(
      raw.begin(), raw.end(), std::back_inserter(frequencies),
      [](double& e) { return e != 0; });
  double entropy = 0;
  for (auto freq : frequencies) {
    entropy += -(freq * 1.0 / l) * log2(freq * 1.0 / l);
  }

  //    cout << "entropy:\t" << entropy << endl;
  delete[] arr;
  return entropy;
}

// TODO automatically omit bins that are less than 1%
template <typename T>
void visualize_histogram(
    const std::string& tag, T* _d_POD, size_t l, size_t _bins, bool log_freq,
    double override_min, double override_max, bool eliminate_zeros,
    bool use_scientific_notation)
{
  std::vector<T> _d(_d_POD, _d_POD + l);
  std::vector<T> _d_nonzero;
  //    std::vector<size_t> arr;
  //    arr.reserve(_bins);
  //    for (size_t i = 0; i< _bins; i++) arr.push_back(0);
  auto arr = new size_t[_bins]();

  if (eliminate_zeros) {
    std::copy_if(
        _d.begin(), _d.end(), std::back_inserter(_d_nonzero),
        [](int i) { return i != 0; });
  }
  double min_v = *std::min_element(_d.begin(), _d.end());
  double max_v = *std::max_element(_d.begin(), _d.end());
  //    double sum = std::accumulate(_d.begin(), _d.end(), 0);
  double rng = max_v - min_v;
  //    double avg = sum / l;

  cout << "\e[7m[[" << tag << "]]\e[0m";
  if (override_max > override_min) {
    cout << "zoom into " << override_min << "--" << override_max << endl;
    std::tie(max_v, min_v, rng) = std::make_tuple(
        override_max, override_min, override_max - override_min);
  }
  double step = rng / _bins;
  for (size_t i = 0; i < l; i++)
    arr[static_cast<size_t>((_d[i] - min_v) / step)]++;
  std::vector<size_t> _viz(arr, arr + _bins);
  //    std::vector<size_t> _viz(arr);

  // visualization
  printf("\tbins:\t%zu\tbin_width:\t%lf\n", _bins, step);
  //    printf("count:\t%zu\tmin:\t%lf\tmax:\t%lf\trng:\t%lf\n", l, min_v,
  //    max_v, rng);
  cout << "count:\t" << l << "\t";
  cout << "min:\t" << min_v << "\t";
  cout << "max:\t" << max_v << "\t";
  cout << "rng:\t" << rng << endl;

  if (log_freq) {
    cout << "using log_freq" << endl;
    std::for_each(_viz.begin(), _viz.end(), [](size_t& n) { n = log2(n); });
  }

  size_t longest = *std::max_element(_viz.begin(), _viz.end());
  size_t bar_str_len = 64;  // scale according to the longest
  std::for_each(_viz.begin(), _viz.end(), [&](size_t& n) {
    n = static_cast<size_t>(n / static_cast<double>(longest) * bar_str_len);
  });

  for (size_t i = 0; i < _bins; i++) {
    // normalize to width
    cout << "|"
         << "\33[43m";

    for (size_t j = 0; j < bar_str_len + 1; j++) {
      if (j < _viz[i])
        cout << "-";
      else if (j == _viz[i])
        cout << "\33[0m"
             << "+";
      else
        cout << " ";
    }
    cout.precision(2);
    cout << "    ";
    if (use_scientific_notation) cout << std::scientific;
    cout << min_v + i * step << " -- " << min_v + (i + 1) * step;
    cout << "  ";
    cout << std::setw((int)log10(l) + 2);
    cout << arr[i];
    cout << "   ";
    cout << std::defaultfloat << std::setw(5)
         << arr[i] / static_cast<double>(l) * 100 << "%" << endl;
  }
  cout << endl;
  //    delete[] arr;
}

#endif
