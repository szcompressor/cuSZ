/**
 * @file configs.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-09-26
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_COMMON_CONFIGS_HH
#define CUSZ_COMMON_CONFIGS_HH

#include <cxxabi.h>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "definition.hh"

#if __cplusplus >= 201703L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

struct Reinterpret1DTo2D {
    template <typename T>
    static T get_square_size(T len)
    {  //
        return static_cast<T>(ceil(sqrt(len)));
    }
};

struct Align {
    template <cusz::ALIGNDATA ad = cusz::ALIGNDATA::NONE>
    static size_t get_aligned_datalen(size_t len)
    {
        if CONSTEXPR (ad == cusz::ALIGNDATA::NONE) return len;
        if CONSTEXPR (ad == cusz::ALIGNDATA::SQUARE_MATRIX) {
            auto m = Reinterpret1DTo2D::get_square_size(len);
            return m * m;
        }
    }

    static const int DEFAULT_ALIGN_NBYTE = 128;

    template <int NUM>
    static inline bool is_aligned_at(const void* ptr)
    {  //
        return reinterpret_cast<uintptr_t>(ptr) % NUM == 0;
    };

    template <typename T, int NUM = DEFAULT_ALIGN_NBYTE>
    static size_t get_aligned_nbyte(size_t len)
    {
        return ((sizeof(T) * len - 1) / NUM + 1) * NUM;
    }
};

// sparsity rate is less that 5%
struct SparseMethodSetup {
    // "Density" denotes the degree of non-zeros (nz).
    static constexpr float default_density  = 0.25;                 // ratio of nonzeros (R_nz)
    static constexpr float default_sparsity = 1 - default_density;  // ratio of zeros, 1 - R_nz

    static constexpr int default_density_factor = 4;  // ratio of nonzeros (R_nz)

    template <typename T, typename M = int>
    static uint32_t get_csr_nbyte(uint32_t len, uint32_t nnz)
    {
        auto m     = Reinterpret1DTo2D::get_square_size(len);
        auto nbyte = sizeof(M) * (m + 1) + sizeof(M) * nnz + sizeof(T) * nnz;
        return nbyte;
    }
};

struct HuffmanHelper {
    // deprecated
    // template <typename SYM, typename BOOK>
    // static uint32_t get_revbook_nbyte(int dict_size)
    // {
    //     constexpr auto TYPE_BITCOUNT = sizeof(BOOK) * 8;
    //     return sizeof(BOOK) * (2 * TYPE_BITCOUNT) + sizeof(SYM) * dict_size;
    // }

    static const int BLOCK_DIM_ENCODE  = 256;
    static const int BLOCK_DIM_DEFLATE = 256;

    static const int ENC_SEQUENTIALITY = 4;  // empirical
    static const int DEFLATE_CONSTANT  = 4;  // TODO -> deflate_chunk_constant
};

struct StringHelper {
    static std::string nnz_percentage(uint32_t nnz, uint32_t data_len)
    {
        return "(" + std::to_string(nnz / 1.0 / data_len * 100) + "%)";
    }
};

struct ConfigHelper {
    static uint32_t predictor_lookup(std::string name)
    {
        const std::unordered_map<std::string, uint32_t> lut = {
            {"lorenzo", 0}, {"lorenzoii", 1}, {"spline3", 2}  //
        };
        if (lut.find(name) != lut.end()) throw std::runtime_error("no such predictor as " + name);
        return lut.at(name);
    }

    static uint32_t codec_lookup(std::string name)
    {
        const std::unordered_map<std::string, uint32_t> lut = {
            {"huffman-coarse", 0}  //
        };
        if (lut.find(name) != lut.end()) throw std::runtime_error("no such codec as " + name);
        return lut.at(name);
    }

    static uint32_t spreducer_lookup(std::string name)
    {
        const std::unordered_map<std::string, uint32_t> lut = {
            {"csr11", 0}, {"spgs", 1}  //
        };
        if (lut.find(name) != lut.end()) throw std::runtime_error("no such codec as " + name);
        return lut.at(name);
    }

    static std::string get_default_predictor() { return "lorenzo"; }
    static std::string get_default_spreducer() { return "csr11"; }
    static std::string get_default_codec() { return "huffman-coarse"; }
    static std::string get_default_cuszmode() { return "r2r"; }
    static std::string get_default_dtype() { return "f32"; }

    static bool check_predictor(const std::string& val, bool fatal = false)
    {
        auto legal = (val == "lorenzo") or (val == "spline3");
        if (not legal) {
            if (fatal)
                throw std::runtime_error("`predictor` must be \"lorenzo\" or \"spline3\".");
            else
                printf("fallback to the default \"%s\".", get_default_predictor().c_str());
        }
        return legal;
    }

    static bool check_codec(const std::string& val, bool fatal = false)
    {
        auto legal = (val == "huffman-coarse");
        if (not legal) {
            if (fatal)
                throw std::runtime_error("`codec` must be \"huffman-coarse\".");
            else
                printf("fallback to the default \"%s\".", get_default_codec().c_str());
        }
        return legal;
    }

    static bool check_spreducer(const std::string& val, bool fatal = false)
    {
        auto legal = (val == "csr11") or (val == "rle");
        if (not legal) {
            if (fatal)
                throw std::runtime_error("`codec` must be \"csr11\" or \"rle\".");
            else
                printf("fallback to the default \"%s\".", get_default_codec().c_str());
        }
        return legal;
    }

    static bool check_cuszmode(const std::string& val, bool fatal = false)
    {
        auto legal = (val == "r2r") or (val == "abs");
        if (not legal) {
            if (fatal)
                throw std::runtime_error("`mode` must be \"r2r\" or \"abs\".");
            else
                printf("fallback to the default \"%s\".", get_default_cuszmode().c_str());
        }
        return legal;
    }

    static bool check_dtype(const std::string& val, bool fatal = false)
    {
        auto legal = (val == "f32");
        // auto legal = (val == "f32") or (val == "f64");
        if (not legal) {
            if (fatal)
                throw std::runtime_error("`dtype` must be \"f32\".");
            else
                printf("fallback to the default \"%s\".", get_default_dtype().c_str());
        }
        return legal;
    }

    static void parse_length_literal(const char* str, std::vector<std::string>& dims)
    {
        std::stringstream data_len_ss(str);
        auto              data_len_literal = data_len_ss.str();
        char              delimiter        = ',';

        bool use_charx = data_len_literal.find('x') != std::string::npos;
        bool use_comma = data_len_literal.find(',') != std::string::npos;
        bool delim_ok  = use_comma or use_charx;

        if (use_charx)
            delimiter = 'x';
        else if (use_comma)
            delimiter = ',';

        while (data_len_ss.good()) {
            std::string substr;
            std::getline(data_len_ss, substr, delimiter);
            dims.push_back(substr);
        }

        if (dims.size() != 1 and (not delim_ok))
            throw std::runtime_error("data-size literal must be delimited by \'x\' or \',\'.");

        // TODO check if a good number (size==1) using regex
    }

    static size_t get_filesize(std::string fname)
    {
        std::ifstream in(fname.c_str(), std::ifstream::ate | std::ifstream::binary);
        return in.tellg();
    }

    template <typename T1, typename T2>
    static size_t get_npart(T1 size, T2 subsize)
    {
        static_assert(
            std::numeric_limits<T1>::is_integer and std::numeric_limits<T2>::is_integer,
            "[get_npart] must be plain interger types.");

        return (size + subsize - 1) / subsize;
    }
};

struct CompareHelper {
    template <typename TRIO>
    static bool eq(TRIO a, TRIO b)
    {  //
        return (a.x == b.x) and (a.y == b.y) and (a.z == b.z);
    };
};

struct ReportHelper {
    static float get_throughput(float milliseconds, size_t nbyte)
    {
        auto GiB     = 1.0 * 1024 * 1024 * 1024;
        auto seconds = milliseconds * 1e-3;
        return nbyte / GiB / seconds;
    }

    static void print_throughput_line(const char* s, float timer, size_t _nbyte)
    {
        if (timer == 0.0) return;

        auto t = get_throughput(timer, _nbyte);
        printf("  %-12s %'12f %'10.2f\n", s, timer, t);
    };

    static void print_throughput_tablehead()
    {
        printf(
            "\n  \e[1m\e[31m%-12s %12s %10s\e[0m\n",  //
            const_cast<char*>("kernel"),              //
            const_cast<char*>("time, ms"),            //
            const_cast<char*>("GiB/s")                //
        );
    }

    static void print_datasegment_tablehead()
    {
        printf(
            "\ndata segments:\n  \e[1m\e[31m%-18s\t%12s\t%15s\t%15s\e[0m\n",  //
            const_cast<char*>("name"),                                        //
            const_cast<char*>("nbyte"),                                       //
            const_cast<char*>("start"),                                       //
            const_cast<char*>("end"));
    }

    static std::string demangle(const char* name)
    {
        int   status = -4;
        char* res    = abi::__cxa_demangle(name, nullptr, nullptr, &status);

        const char* const demangled_name = (status == 0) ? res : name;
        std::string       ret_val(demangled_name);
        free(res);
        return ret_val;
    };
};

#endif
