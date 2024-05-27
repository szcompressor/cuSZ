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

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>

#include "busyheader.hh"
#include "hf/hfcodec.hh"
#include "utils/timer.hh"

using std::cout;

#if __cplusplus >= 201703L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

// TODO remove
enum class ExecutionPolicy { host, cuda_device };
enum class AnalyzerMethod { thrust, cuda_native, stl };

class Analyzer {
    typedef struct ExtremaResult {
        double max_val, min_val, rng;
        double seconds;
    } extrema_result_t;

    typedef struct Compressibility {
        size_t len;
        struct {
            double       entropy;
            unsigned int top1_freq;
            double       top1_prob;
            double       dropout_equiv_bitlen_2x() const { return 64 * (1 - top1_prob); }
            double       dropout_equiv_bitlen_1_5x() const { return 48 * (1 - top1_prob); }
        } hist;
        struct {
            double r_lowerbound;
            double avgb_lowerbound;
            double r_upperbound;
            double avgb_upperbound;
        } huffman_theory;
        struct {
            double min_bitlen;
            double avgb;
        } huffman_stat;
    } theory_t;

    theory_t theory;

   public:
    Analyzer()  = default;
    ~Analyzer() = default;

    // TODO execution policy
    template <typename T, ExecutionPolicy policy = ExecutionPolicy::host>
    static std::vector<T> percentile100(T* in, size_t len)
    {
        std::vector<T> res;
        auto           step = int(ceil(len / 100));

        if CONSTEXPR (policy == ExecutionPolicy::cuda_device) {
            // caveat: no residence check
            thrust::sort(thrust::device, in, in + len);
            T* htmp;
            GpuMallocHost(&htmp, sizeof(T) * len);
            GpuMemcpy(htmp, in, sizeof(T) * len, GpuMemcpyD2H);
            for (auto i = 0; i < len; i += step) {  //
                res.push_back(htmp[i]);
            }
            res.push_back(htmp[len - 1]);
            GpuFreeHost(htmp);
        }
        else {  // fallback
            std::sort(in, in + len);
            for (auto i = 0; i < len; i += step) {  //
                res.push_back(in[i]);
            }
            res.push_back(in[len - 1]);
        }

        return res;
    }

    template <typename Data, ExecutionPolicy policy, AnalyzerMethod method>
    static extrema_result_t get_maxmin_rng(Data* d_data, size_t len)
    {
        if CONSTEXPR (policy == ExecutionPolicy::cuda_device and method == AnalyzerMethod::thrust) {
            auto t0 = hires::now();
            // ------------------------------------------------------------
            thrust::device_ptr<Data> g_ptr = thrust::device_pointer_cast(d_data);

            auto max_el_loc = thrust::max_element(g_ptr, g_ptr + len);  // excluding padded
            auto min_el_loc = thrust::min_element(g_ptr, g_ptr + len);  // excluding padded

            double max_val = *max_el_loc;
            double min_val = *min_el_loc;
            double rng     = max_val - min_val;
            // ------------------------------------------------------------
            auto t1 = hires::now();

            return extrema_result_t{max_val, min_val, rng, static_cast<duration_t>(t1 - t0).count()};
        }
        else {
            throw std::runtime_error("Analyzer::get_maxmin_rng() Other policy and method not implemented.");
        }
    }

    template <typename UInt, ExecutionPolicy policy, AnalyzerMethod method>
    static void get_histogram(UInt* data, size_t data_len, unsigned int* freq, size_t num_bins)
    {
        // TODO static check UInt
        if CONSTEXPR (policy == ExecutionPolicy::cuda_device and method == AnalyzerMethod::cuda_native) {
            float dummy;
            launch_histogram(data, data_len, freq, num_bins, dummy);
        }
        else {
            // TODO static check
            throw std::runtime_error("Analyzer::get_histogram() using other policy or method not implemented.");
        }
    }

    Analyzer& estimate_compressibility_from_histogram(unsigned int* h_freq, size_t dict_size)
    {
        auto   len       = std::accumulate(h_freq, h_freq + dict_size, 0u);  // excluding outlier
        auto   top1_freq = *std::max_element(h_freq, h_freq + dict_size);
        double top1_prob = (1.0 * top1_freq) / (1.0 * len);
        double entropy   = 0.0;
        for (auto i = 0; i < dict_size; i++) {
            double p = h_freq[i] / (1.0 * len);
            if (p != 0) entropy += -std::log2(p) * p;
        }
        double r_lowerbound    = 1 - (-std::log2(top1_prob) * top1_prob - std::log2(1 - top1_prob) * (1 - top1_prob));
        double r_upperbound    = top1_prob + 0.086;  // [Gallager 78]
        double avgb_lowerbound = entropy + r_lowerbound;
        double avgb_upperbound = entropy + r_upperbound;

        // dropout
        // auto equiv_bitlen_dropout_2x   = 64 * (1 - top1_prob);
        // auto equiv_bitlen_dropout_1_5x = 48 * (1 - top1_prob);

        // record
        theory.len                            = len;
        theory.hist.entropy                   = entropy;
        theory.hist.top1_freq                 = top1_freq;
        theory.hist.top1_prob                 = top1_prob;
        theory.huffman_theory.r_lowerbound    = r_lowerbound;
        theory.huffman_theory.r_upperbound    = r_upperbound;
        theory.huffman_theory.avgb_lowerbound = avgb_lowerbound;
        theory.huffman_theory.avgb_upperbound = avgb_upperbound;

        return *this;
    };

    template <typename Huff>
    Analyzer&
    get_stat_from_huffman_book(const unsigned int* h_freq, const Huff* h_codebook, size_t len, size_t num_bins)
    {
        // real-bitlen, for reference only, not part of workflow
        std::vector<Huff>         v_canon_cb(h_codebook, h_codebook + num_bins);
        std::vector<unsigned int> v_freq(h_freq, h_freq + num_bins);

        // TODO somewhere explicitly state that null codeword is of length 0xff
        std::sort(v_canon_cb.begin(), v_canon_cb.end(), [](Huff& a, Huff& b) {
            auto a_bits = reinterpret_cast<struct PackedWordByWidth<sizeof(Huff)>*>(&a)->bits;
            auto b_bits = reinterpret_cast<struct PackedWordByWidth<sizeof(Huff)>*>(&b)->bits;
            return a_bits < b_bits;
        });
        std::sort(v_freq.begin(), v_freq.end(), std::greater<Huff>());

        double real_avgb = 0.0;
        for (auto i = 0; i < num_bins; i++) {
            if (v_freq[i] != 0) {
                auto bits = reinterpret_cast<struct PackedWordByWidth<sizeof(Huff)>*>(&v_canon_cb[i])->bits;
                real_avgb += v_freq[i] * bits;
            }
        }
        real_avgb /= len;

        theory.huffman_stat.avgb = real_avgb;
        theory.huffman_stat.min_bitlen =
            reinterpret_cast<struct PackedWordByWidth<sizeof(Huff)>*>(&v_canon_cb.at(0))->bits;

        return *this;
    }

    Analyzer&
    print_compressibility(bool print_huffman_stat = false, bool print_dropout = false, double equiv_origin_bitlen = 32)
    {
        cout << "\n\e[31m";  // extra linebreak on start

        cout << "* Derived from histogram:" << '\n';
        cout << "  - len (freq sum):\t" << theory.len << '\n';
        cout << "  - entropy H(X):\t" << theory.hist.entropy << '\n';
        cout << "  - most likely freq:\t" << theory.hist.top1_freq << '\n';
        cout << "  - most likely prob (p1):\t" << theory.hist.top1_prob << '\n';
        cout << '\n';

        if (theory.hist.top1_prob < 0.4) {
            cout << "* The probability of the most likely symbol < 0.4, go recoding (Huffman)." << '\n';
            cout << "* Compressibility lower bound is for reference only." << '\n';
            cout << "  - est. redundancy upper bound (arbitrary p1):\t" << theory.huffman_theory.r_upperbound << '\n';
            cout << "  - est. avg.bitlen upper bound (arbitrary p1):\t" << theory.huffman_theory.avgb_upperbound
                 << '\n';
            cout << "  - est. CR lower bound (arbitrary p1):\t"
                 << equiv_origin_bitlen / theory.huffman_theory.avgb_upperbound << '\n';
            cout << '\n';
        }
        else {
            cout << "* Compressibility upper bound is determined by the lower bound of average bitlength." << '\n';
            cout << "  - est. redundancy lower bound (p1 > 0.4):\t" << theory.huffman_theory.r_lowerbound << '\n';
            cout << "  - est. avg.bitlen lower bound (p1 > 0.4):\t" << theory.huffman_theory.avgb_lowerbound << '\n';
            cout << "  - est. CR upper bound (arbitrary p1):\t"
                 << equiv_origin_bitlen / theory.huffman_theory.avgb_lowerbound << '\n';
            cout << '\n';

            cout << "* Compressibility lower bound is for reference only." << '\n';
            cout << "  - est. redundancy upper bound (arbitrary p1):\t" << theory.huffman_theory.r_upperbound << '\n';
            cout << "  - est. avg.bitlen upper bound (arbitrary p1):\t" << theory.huffman_theory.avgb_upperbound
                 << '\n';
            cout << "  - est. CR lower bound (arbitrary p1):\t"
                 << equiv_origin_bitlen / theory.huffman_theory.avgb_upperbound << '\n';
            cout << '\n';

            if (print_dropout) {
                auto dropout_equiv_bitlen_2x   = theory.hist.dropout_equiv_bitlen_2x();
                auto dropout_equiv_bitlen_1_5x = theory.hist.dropout_equiv_bitlen_1_5x();
                // TODO determine path, print log
                cout << "* Considering dropout:" << '\n';
                cout << "  - dropout at 1.0x metadata overhead" << '\n';
                cout << "    | equiv.bitlen:\t" << dropout_equiv_bitlen_2x << '\n';
                cout << "    | reduction rate:\t" << (equiv_origin_bitlen / dropout_equiv_bitlen_2x) << '\n';
                cout << "    | bitlen_dropout <= bitlen_enc?\t"
                     << (dropout_equiv_bitlen_2x <= theory.huffman_theory.avgb_lowerbound) << '\n';
                cout << "  - dropout at 0.5x metadata overhead" << '\n';
                cout << "    | equiv.bitlen:\t" << dropout_equiv_bitlen_1_5x << '\n';
                cout << "    | reduction rate (fp32):\t" << (equiv_origin_bitlen / dropout_equiv_bitlen_1_5x) << '\n';
                cout << "    | bitlen_dropout <= bitlen_enc?\t"
                     << (dropout_equiv_bitlen_1_5x <= theory.huffman_theory.avgb_lowerbound) << '\n';
                cout << '\n';
            }
        }

        if (print_huffman_stat) {
            cout << "* From Huffman codebook:" << '\n';
            cout << "  - avg. bitlen:\t" << theory.huffman_stat.avgb << '\n';
            cout << "  - shortest bitlen:\t" << theory.huffman_stat.min_bitlen << '\n';
            cout << '\n';
        }
        cout << "\e[0m";

        return *this;
    }
};

#endif
