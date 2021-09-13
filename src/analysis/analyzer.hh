/**
 * @file analyzer.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-03-26
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef ANALYSIS_ANALYZER_HH
#define ANALYSIS_ANALYZER_HH

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <algorithm>
#include <numeric>

#include "../kernel/codec_huffman.cuh"
#include "../kernel/hist.cuh"
#include "../utils/timer.hh"
#include "../wrapper/huffman_enc_dec.cuh"

#if __cplusplus >= 201703L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

enum class ExecutionPolicy { host, cuda_device };
enum class AnalyzerMethod { thrust, cuda_native, stl };

class Analyzer {
    typedef struct ExtremaResult {
        double max_val;
        double min_val;
        double rng;
        double seconds;
    } extrema_result_t;

    typedef struct Compressibility {
        size_t len;
        struct {
            double       entropy;
            unsigned int most_likely_freq;
            double       most_likely_prob;
            double       dropout_equiv_bitlen_2x() const { return 64 * (1 - most_likely_prob); }
            double       dropout_equiv_bitlen_1_5x() const { return 48 * (1 - most_likely_prob); }
        } hist;
        struct {
            double redundancy_lower_bound;
            double avg_bitlen_lower_bound;
            double redundancy_upper_bound;
            double avg_bitlen_upper_bound;
        } huffman_est;
        struct {
            double shortest_bitlen;
            double avg_bitlen;
        } huffman_stat;
    } c13y_t;

    c13y_t compressibility;

   public:
    Analyzer()  = default;
    ~Analyzer() = default;

    template <typename Data, ExecutionPolicy policy, AnalyzerMethod method>
    extrema_result_t GetMaxMinRng(Data* d_data, size_t len)
    {
        if CONSTEXPR (policy == ExecutionPolicy::cuda_device and method == AnalyzerMethod::thrust) {
            auto time_0 = hires::now();
            // ------------------------------------------------------------
            thrust::device_ptr<Data> g_ptr = thrust::device_pointer_cast(d_data);

            auto max_el_loc = thrust::max_element(g_ptr, g_ptr + len);  // excluding padded
            auto min_el_loc = thrust::min_element(g_ptr, g_ptr + len);  // excluding padded

            double max_value = *max_el_loc;
            double min_value = *min_el_loc;
            double rng       = max_value - min_value;
            // ------------------------------------------------------------
            auto time_1 = hires::now();

            return extrema_result_t{max_value, min_value, rng, static_cast<duration_t>(time_1 - time_0).count()};
        }
        else {
            throw std::runtime_error("Analyzer::GetMaxMinRng() Other policy and method not implemented.");
        }
    }

    template <typename UInt, ExecutionPolicy policy, AnalyzerMethod method>
    Analyzer& Histogram(UInt* data, size_t data_len, unsigned int* freq, size_t num_bins)
    {
        // TODO static check UInt
        if CONSTEXPR (policy == ExecutionPolicy::cuda_device and method == AnalyzerMethod::cuda_native) {
            float dummy;
            wrapper::get_frequency(data, data_len, freq, num_bins, dummy);
        }
        else {
            // TODO static check
            throw std::runtime_error("Analyzer::Histogram() using other policy or method not implemented.");
        }
        return *this;
    }

    Analyzer& EstimateFromHistogram(unsigned int* h_freq, size_t dict_size)
    {
        auto   len              = std::accumulate(h_freq, h_freq + dict_size, 0u);  // excluding outlier
        auto   most_likely_freq = *std::max_element(h_freq, h_freq + dict_size);
        double most_likely_prob = (1.0 * most_likely_freq) / (1.0 * len);
        double entropy          = 0.0;
        for (auto i = 0; i < dict_size; i++) {
            double p = h_freq[i] / (1.0 * len);
            if (p != 0) entropy += -std::log2(p) * p;
        }
        double redundancy_lower_bound     = 1 - (-std::log2(most_likely_prob) * most_likely_prob -
                                             std::log2(1 - most_likely_prob) * (1 - most_likely_prob));
        double redundancy_upper_bound     = most_likely_prob + 0.086;  // [Gallager 78]
        double est_avg_bitlen_lower_bound = entropy + redundancy_lower_bound;
        double est_avg_bitlen_upper_bound = entropy + redundancy_upper_bound;

        // dropout
        // auto equiv_bitlen_dropout_2x   = 64 * (1 - most_likely_prob);
        // auto equiv_bitlen_dropout_1_5x = 48 * (1 - most_likely_prob);

        // record
        compressibility.len                                = len;
        compressibility.hist.entropy                       = entropy;
        compressibility.hist.most_likely_freq              = most_likely_freq;
        compressibility.hist.most_likely_prob              = most_likely_prob;
        compressibility.huffman_est.redundancy_lower_bound = redundancy_lower_bound;
        compressibility.huffman_est.redundancy_upper_bound = redundancy_upper_bound;
        compressibility.huffman_est.avg_bitlen_lower_bound = est_avg_bitlen_lower_bound;
        compressibility.huffman_est.avg_bitlen_upper_bound = est_avg_bitlen_upper_bound;

        return *this;
    };

    template <typename Huff>
    Analyzer& GetHuffmanCodebookStat(const unsigned int* h_freq, const Huff* h_codebook, size_t len, size_t num_bins)
    {
        // real-bitlen, for reference only, not part of workflow
        std::vector<Huff>         v_canon_cb(h_codebook, h_codebook + num_bins);
        std::vector<unsigned int> v_freq(h_freq, h_freq + num_bins);

        // TODO somewhere explicitly state that null codeword is of length 0xff
        std::sort(v_canon_cb.begin(), v_canon_cb.end(), [](Huff& a, Huff& b) {
            auto a_bits = reinterpret_cast<struct PackedWord<Huff>*>(&a)->bits;
            auto b_bits = reinterpret_cast<struct PackedWord<Huff>*>(&b)->bits;
            return a_bits < b_bits;
        });
        std::sort(v_freq.begin(), v_freq.end(), std::greater<Huff>());

        double real_avg_bitlen = 0.0;
        for (auto i = 0; i < num_bins; i++) {
            if (v_freq[i] != 0) {
                auto bits = reinterpret_cast<struct PackedWord<Huff>*>(&v_canon_cb[i])->bits;
                real_avg_bitlen += v_freq[i] * bits;
            }
        }
        real_avg_bitlen /= len;

        compressibility.huffman_stat.avg_bitlen = real_avg_bitlen;
        compressibility.huffman_stat.shortest_bitlen =
            reinterpret_cast<struct PackedWord<Huff>*>(&v_canon_cb.at(0))->bits;

        return *this;
    }

    Analyzer& PrintCompressibilityInfo(
        bool   print_huffman_stat  = false,
        bool   print_dropout       = false,
        double equiv_origin_bitlen = 32)
    {
        cout << "\n\e[31m";  // extra linebreak on start

        cout << "* Derived from histogram:" << '\n';
        cout << "  - len (freq sum):\t" << compressibility.len << '\n';
        cout << "  - entropy H(X):\t" << compressibility.hist.entropy << '\n';
        cout << "  - most likely freq:\t" << compressibility.hist.most_likely_freq << '\n';
        cout << "  - most likely prob (p1):\t" << compressibility.hist.most_likely_prob << '\n';
        cout << '\n';

        if (compressibility.hist.most_likely_prob < 0.4) {
            cout << "* The probability of the most likely symbol < 0.4, go recoding (Huffman)." << '\n';
            cout << "* Compressibility lower bound is for reference only." << '\n';
            cout << "  - est. redundancy upper bound (arbitrary p1):\t"
                 << compressibility.huffman_est.redundancy_upper_bound << '\n';
            cout << "  - est. avg.bitlen upper bound (arbitrary p1):\t"
                 << compressibility.huffman_est.avg_bitlen_upper_bound << '\n';
            cout << "  - est. CR lower bound (arbitrary p1):\t"
                 << equiv_origin_bitlen / compressibility.huffman_est.avg_bitlen_upper_bound << '\n';
            cout << '\n';
        }
        else {
            cout << "* Compressibility upper bound is determined by the lower bound of average bitlength." << '\n';
            cout << "  - est. redundancy lower bound (p1 > 0.4):\t"
                 << compressibility.huffman_est.redundancy_lower_bound << '\n';
            cout << "  - est. avg.bitlen lower bound (p1 > 0.4):\t"
                 << compressibility.huffman_est.avg_bitlen_lower_bound << '\n';
            cout << "  - est. CR upper bound (arbitrary p1):\t"
                 << equiv_origin_bitlen / compressibility.huffman_est.avg_bitlen_lower_bound << '\n';
            cout << '\n';

            cout << "* Compressibility lower bound is for reference only." << '\n';
            cout << "  - est. redundancy upper bound (arbitrary p1):\t"
                 << compressibility.huffman_est.redundancy_upper_bound << '\n';
            cout << "  - est. avg.bitlen upper bound (arbitrary p1):\t"
                 << compressibility.huffman_est.avg_bitlen_upper_bound << '\n';
            cout << "  - est. CR lower bound (arbitrary p1):\t"
                 << equiv_origin_bitlen / compressibility.huffman_est.avg_bitlen_upper_bound << '\n';
            cout << '\n';

            if (print_dropout) {
                auto dropout_equiv_bitlen_2x   = compressibility.hist.dropout_equiv_bitlen_2x();
                auto dropout_equiv_bitlen_1_5x = compressibility.hist.dropout_equiv_bitlen_1_5x();
                // TODO determine path, print log
                cout << "* Considering dropout:" << '\n';
                cout << "  - dropout at 1.0x metadata overhead" << '\n';
                cout << "    | equiv.bitlen:\t" << dropout_equiv_bitlen_2x << '\n';
                cout << "    | reduction rate:\t" << (equiv_origin_bitlen / dropout_equiv_bitlen_2x) << '\n';
                cout << "    | bitlen_dropout <= bitlen_enc?\t"
                     << (dropout_equiv_bitlen_2x <= compressibility.huffman_est.avg_bitlen_lower_bound) << '\n';
                cout << "  - dropout at 0.5x metadata overhead" << '\n';
                cout << "    | equiv.bitlen:\t" << dropout_equiv_bitlen_1_5x << '\n';
                cout << "    | reduction rate (fp32):\t" << (equiv_origin_bitlen / dropout_equiv_bitlen_1_5x) << '\n';
                cout << "    | bitlen_dropout <= bitlen_enc?\t"
                     << (dropout_equiv_bitlen_1_5x <= compressibility.huffman_est.avg_bitlen_lower_bound) << '\n';
                cout << '\n';
            }
        }

        if (print_huffman_stat) {
            cout << "* From Huffman codebook:" << '\n';
            cout << "  - avg. bitlen:\t" << compressibility.huffman_stat.avg_bitlen << '\n';
            cout << "  - shortest bitlen:\t" << compressibility.huffman_stat.shortest_bitlen << '\n';
            cout << '\n';
        }
        cout << "\e[0m";

        return *this;
    }
};

#endif