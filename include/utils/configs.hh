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

// #include <cuda_runtime.h>
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
#include "../cusz/type.h"
#include "../header.h"

#if __cplusplus >= 201703L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

struct HuffmanHelper {
    static const int BLOCK_DIM_ENCODE  = 256;
    static const int BLOCK_DIM_DEFLATE = 256;

    static const int ENC_SEQUENTIALITY = 4;  // empirical
    static const int DEFLATE_CONSTANT  = 4;  // TODO -> deflate_chunk_constant
};

struct psz_utils {
    static std::string nnz_percentage(uint32_t nnz, uint32_t data_len)
    {
        return "(" + std::to_string(nnz / 1.0 / data_len * 100) + "%)";
    }

    static void check_cuszmode(const std::string& val)
    {
        auto legal = (val == "r2r") or (val == "abs");
        if (not legal) throw std::runtime_error("`mode` must be \"r2r\" or \"abs\".");
    }

    static bool check_dtype(const std::string& val, bool delay_failure = true)
    {
        auto legal = (val == "f32") or (val == "f4");
        // auto legal = (val == "f32") or (val == "f64");
        if (not legal)
            if (not delay_failure) throw std::runtime_error("Only `f32`/`f4` is supported temporarily.");

        return legal;
    }

    static bool check_dtype(const cusz_dtype& val, bool delay_failure = true)
    {
        auto legal = (val == F4);
        if (not legal)
            if (not delay_failure) throw std::runtime_error("Only `f32` is supported temporarily.");

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

    template <typename TRIO>
    static bool eq(TRIO a, TRIO b)
    {
        return (a.x == b.x) and (a.y == b.y) and (a.z == b.z);
    };

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
