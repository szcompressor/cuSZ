#ifndef TYPE_TRAIT_HH
#define TYPE_TRAIT_HH

/**
 * @file type_trait.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.1.1
 * @date 2020-09-23
 * (create) 2020-09-23, (rev) 2021-09-17
 *
 * @copyright (C) 2020 by Washington State University, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cxxabi.h>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "types.hh"

// clang-format off

template <bool FP, int DWIDTH> struct DataTrait;
//template <> struct DataTrait<true, 1> {typedef FP4 type;}; // placeholder for Cartesian expansion
//template <> struct DataTrait<true, 2> {typedef FP8 type;};
template <> struct DataTrait<true, 4> { typedef FP4 type; };
template <> struct DataTrait<true, 8> { typedef FP8 type; };
//template <> struct DataTrait<false, 1> {typedef FP4 type;}; // future use
//template <> struct DataTrait<false, 2> {typedef FP8 type;};
//template <> struct DataTrait<false, 4> {typedef FP4 type;};
//template <> struct DataTrait<false, 8> {typedef FP8 type;};

template <int ndim> struct MetadataTrait;
template <> struct MetadataTrait<1>     { static const int Block = 256; static const int Sequentiality = 8; };
template <> struct MetadataTrait<0x101> { static const int Block = 128; };
template <> struct MetadataTrait<0x201> { static const int Block = 64;  };
template <> struct MetadataTrait<2>     { static const int Block = 16; static const int YSequentiality = 8;};
template <> struct MetadataTrait<3>     { static const int Block = 8;  static const int YSequentiality = 8;};

template <int QWIDTH> struct QuantTrait;
template <> struct QuantTrait<1> { typedef UI1 type; };
template <> struct QuantTrait<2> { typedef UI2 type; };
template <> struct QuantTrait<4> { typedef UI4 type; };

template <int EWIDTH> struct ErrCtrlTrait;
template <> struct ErrCtrlTrait<1> { typedef UI1 type; };
template <> struct ErrCtrlTrait<2> { typedef UI2 type; };
template <> struct ErrCtrlTrait<4> { typedef UI4 type; };

// TODO where there is atomicOps should be with static<unsigned long long int>(some_uint64_array)

template <int HWIDTH> struct HuffTrait;
template <> struct HuffTrait<4> { typedef UI4 type; };
template <> struct HuffTrait<8> { typedef UI8 type; };

template <int RWIDTH> struct ReducerTrait;
template <> struct ReducerTrait<4> { typedef UI4 type; };
template <> struct ReducerTrait<8> { typedef UI8 type; };

template <int MWIDTH> struct MetadataTrait;
template <> struct MetadataTrait<4> { typedef uint32_t type; };
template <> struct MetadataTrait<8> { typedef size_t   type; };

// sparsity rate is less that 5%
struct SparseMethodSetup { static const int factor = 20; };

// clang-format on

struct Reinterpret1DTo2D {
    static uint32_t get_square_size(uint32_t len)
    {  //
        return static_cast<uint32_t>(ceil(sqrt(len)));
    }
};

struct HuffmanHelper {
    template <typename SYM, typename BOOK>
    static uint32_t get_revbook_nbyte(int dict_size)
    {
        constexpr auto TYPE_BITCOUNT = sizeof(BOOK) * 8;
        return sizeof(BOOK) * (2 * TYPE_BITCOUNT) + sizeof(SYM) * dict_size;
    }

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
        const std::unordered_map<std::string, uint32_t> lut =  //
            {{"lorenzo", 0}, {"lorenzoii", 1}, {"spline3", 2}};
        if (lut.find(name) != lut.end()) throw std::runtime_error("no such predictor as " + name);
        return lut.at(name);
    }

    static uint32_t reducer_lookup(std::string name)
    {
        const std::unordered_map<std::string, uint32_t> lut =  //
            {{"huffman", 0}, {"csr", 1}, {"rle", 2}};
        if (lut.find(name) != lut.end()) throw std::runtime_error("no such reducer as " + name);
        return lut.at(name);
    }

    template <typename CONFIG>
    static void set_eb_series(double eb, CONFIG& config)
    {
        config.eb     = eb;
        config.ebx2   = eb * 2;
        config.ebx2_r = 1 / (eb * 2);
        config.eb_r   = 1 / eb;
    }

    template <typename DST, typename SRC>
    static void deep_copy_config_items(DST* c1, SRC* c2)
    {
        c1->x         = c2->x;
        c1->y         = c2->y;
        c1->z         = c2->z;
        c1->w         = c2->w;
        c1->ndim      = c2->ndim;
        c1->eb        = c2->eb;
        c1->data_len  = c2->data_len;
        c1->quant_len = c2->quant_len;
        c1->radius    = c2->radius;
        c1->dict_size = c2->dict_size;

        c1->quant_nbyte = c2->quant_nbyte;
        c1->huff_nbyte  = c2->huff_nbyte;

        c1->nnz_outlier = c2->nnz_outlier;

        c1->huffman_chunk     = c2->huffman_chunk;
        c1->huffman_num_uints = c2->huffman_num_uints;

        c1->to_skip.huffman = c2->to_skip.huffman;
    }

    static std::string get_default_predictor() { return "lorenzo"; }
    static std::string get_default_reducer() { return "huffman"; }
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

    static bool check_reducer(const std::string& val, bool fatal = false)
    {
        auto legal = (val == "huffman") or (val == "csr") or (val == "rle");
        if (not legal) {
            if (fatal)
                throw std::runtime_error("`reducer` must be \"huffman\", \"csr\" or \"rle\".");
            else
                printf("fallback to the default \"%s\".", get_default_reducer().c_str());
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
    };
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
        auto t = get_throughput(timer, _nbyte);
        printf("  %-18s\t%'12f\t%'15f\n", s, timer, t);
    };

    static void print_throughput_tablehead(const char* name)
    {
        printf(
            "\n%s throughput report (ms, 1e-3 sec):\n"
            "  \e[1m\e[31m%-18s\t%12s\t%15s\e[0m\n",  //
            name,                                     //
            const_cast<char*>("kernel"),              //
            const_cast<char*>("milliseconds"),        //
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
