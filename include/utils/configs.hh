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

#include <cuda_runtime.h>
#include <cxxabi.h>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../common/definition.hh"
#include "../header.h"

#if __cplusplus >= 201703L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

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

    static uint32_t spcodec_lookup(std::string name)
    {
        const std::unordered_map<std::string, uint32_t> lut = {
            {"spmat", 0}, {"spvec", 1}  //
        };
        if (lut.find(name) != lut.end()) throw std::runtime_error("no such codec as " + name);
        return lut.at(name);
    }

    static std::string get_default_predictor() { return "lorenzo"; }
    static std::string get_default_spcodec() { return "csr11"; }
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

    static bool check_spcodec(const std::string& val, bool fatal = false)
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

    static bool check_opt_in_list(std::string const& opt, std::vector<std::string> vs)
    {
        for (auto& i : vs) {
            if (opt == i) return true;
        }
        return false;
    }

    static void parse_length_literal(const char* str, std::vector<std::string>& dims)
    {
        std::stringstream data_len_ss(str);
        auto              data_len_literal = data_len_ss.str();
        char              delimiter        = 'x';

        while (data_len_ss.good()) {
            std::string substr;
            std::getline(data_len_ss, substr, delimiter);
            dims.push_back(substr);
        }
    }

    static size_t get_filesize(std::string fname)
    {
        std::ifstream in(fname.c_str(), std::ifstream::ate | std::ifstream::binary);
        return in.tellg();
    }

    static size_t get_filesize(cusz_header* h)
    {
        auto END = sizeof(h->entry) / sizeof(h->entry[0]);
        return h->entry[END - 1];
    }

    static size_t get_uncompressed_len(cusz_header* h) { return h->x * h->y * h->z; }

    template <typename T1, typename T2>
    static size_t get_npart(T1 size, T2 subsize)
    {
        static_assert(
            std::numeric_limits<T1>::is_integer and std::numeric_limits<T2>::is_integer,
            "[get_npart] must be plain interger types.");

        return (size + subsize - 1) / subsize;
    }

    // #ifdef __CUDACC__
    static int get_ndim(dim3 len3)
    {
        auto ndim = 3;
        if (len3.z == 1) ndim = 2;
        if (len3.z == 1 and len3.y == 1) ndim = 1;
        return ndim;
    }

    static dim3 get_pardeg3(dim3 len3, dim3 sublen3)
    {
        return dim3(
            get_npart(len3.x, sublen3.x),  //
            get_npart(len3.y, sublen3.y),  //
            get_npart(len3.z, sublen3.z));
    }

    template <typename T>
    static dim3 get_pardeg3(dim3 len3, T sublen3[3])
    {
        return dim3(
            get_npart(len3.x, sublen3[0]),  //
            get_npart(len3.y, sublen3[1]),  //
            get_npart(len3.z, sublen3[2]));
    }

    template <typename T>
    static dim3 multiply_dim3(dim3 a, T b[3])
    {
        return dim3(a.x * b[0], a.y * b[1], a.z * b[2]);
    }

    static dim3 multiply_dim3(dim3 a, dim3 b)
    {  //
        return dim3(a.x * b.x, a.y * b.y, a.z * b.z);
    }

    static size_t get_serialized_len(dim3 a) { return a.x * a.y * a.z; }

    static dim3 get_leap(dim3 len3) { return dim3(1, len3.x, len3.x * len3.y); }

    // #endif

    template <typename T>
    static size_t get_serialized_len(T a[3])
    {  //
        return a[0] * a[1] * a[2];
    }
};

struct CompareHelper {
    template <typename TRIO>
    static bool eq(TRIO a, TRIO b)
    {
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

    static void println_throughput(const char* s, float timer, size_t _nbyte)
    {
        if (timer == 0.0) return;
        auto t = get_throughput(timer, _nbyte);
        printf("  %-12s %'12f %'10.2f\n", s, timer, t);
    };

    static void println_throughput_tablehead()
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
