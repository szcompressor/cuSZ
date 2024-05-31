/**
 * @file config.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-09-19
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

// #include <cuda_runtime.h>

#ifndef C5FC9A7F_B78D_4E1E_AC2A_3E18232CEEE3
#define C5FC9A7F_B78D_4E1E_AC2A_3E18232CEEE3

#include <cxxabi.h>

#include <fstream>
#include <regex>
#include <unordered_map>

#include "busyheader.hh"
#include "cusz/type.h"
#include "format.hh"
#include "header.h"

#if __cplusplus >= 201703L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

using std::cerr;
using std::endl;

using ss_t = std::stringstream;
using map_t = std::unordered_map<std::string, std::string>;
using str_list = std::vector<std::string>;

struct psz_helper {
  static unsigned int str2int(const char* s)
  {
    char* end;
    auto res = std::strtol(s, &end, 10);
    if (*end) {
      const char* notif = "invalid option value, non-convertible part: ";
      cerr << LOG_ERR << notif << "\e[1m" << s << "\e[0m" << endl;
    }
    return res;
  }

  static unsigned int str2int(std::string s) { return str2int(s.c_str()); }

  static double str2fp(const char* s)
  {
    char* end;
    auto res = std::strtod(s, &end);
    if (*end) {
      const char* notif = "invalid option value, non-convertible part: ";
      cerr << LOG_ERR << notif << "\e[1m" << end << "\e[0m" << endl;
    }
    return res;
  }

  static double str2fp(std::string s) { return str2fp(s.c_str()); }

  static bool is_kv_pair(std::string s)
  {
    return s.find("=") != std::string::npos;
  }

  static std::pair<std::string, std::string> separate_kv(std::string& s)
  {
    std::string delimiter = "=";

    if (s.find(delimiter) == std::string::npos)
      throw std::runtime_error(
          "\e[1mnot a correct key-value syntax, must be \"opt=value\"\e[0m");

    std::string k = s.substr(0, s.find(delimiter));
    std::string v =
        s.substr(s.find(delimiter) + delimiter.length(), std::string::npos);

    return std::make_pair(k, v);
  }

  static void parse_strlist_as_kv(const char* in_str, map_t& kv_list)
  {
    ss_t ss(in_str);
    while (ss.good()) {
      std::string tmp;
      std::getline(ss, tmp, ',');
      kv_list.insert(separate_kv(tmp));
    }
  }

  static void parse_strlist(const char* in_str, str_list& list)
  {
    ss_t ss(in_str);
    while (ss.good()) {
      std::string tmp;
      std::getline(ss, tmp, ',');
      list.push_back(tmp);
    }
  }

  static std::pair<std::string, bool> parse_kv_onoff(std::string in_str)
  {
    auto kv_literal = "(.*?)=(on|ON|off|OFF)";
    std::regex kv_pattern(kv_literal);
    std::regex onoff_pattern("on|ON|off|OFF");

    bool onoff = false;
    std::string k, v;

    std::smatch kv_match;
    if (std::regex_match(in_str, kv_match, kv_pattern)) {
      // the 1st match: whole string
      // the 2nd: k, the 3rd: v
      if (kv_match.size() == 3) {
        k = kv_match[1].str(), v = kv_match[2].str();

        std::smatch v_match;
        if (std::regex_match(v, v_match, onoff_pattern)) {  //
          onoff = (v == "on") or (v == "ON");
        }
        else {
          throw std::runtime_error("not legal (k=v)-syntax");
        }
      }
    }
    return std::make_pair(k, onoff);
  }

  static std::string doc_format(const std::string& s)
  {
    std::regex gray("%(.*?)%");
    std::string gray_text("\e[37m$1\e[0m");

    std::regex bful("@(.*?)@");
    std::string bful_text("\e[1m\e[4m$1\e[0m");
    std::regex bf("\\*(.*?)\\*");
    std::string bf_text("\e[1m$1\e[0m");
    std::regex ul(R"(_((\w|-|\d|\.)+?)_)");
    std::string ul_text("\e[4m$1\e[0m");
    std::regex red(R"(\^\^(.*?)\^\^)");
    std::string red_text("\e[31m$1\e[0m");

    auto a = std::regex_replace(s, bful, bful_text);
    auto b = std::regex_replace(a, bf, bf_text);
    auto c = std::regex_replace(b, ul, ul_text);
    auto d = std::regex_replace(c, red, red_text);
    auto e = std::regex_replace(d, gray, gray_text);

    return e;
  }
};

struct psz_utils {
  static std::string nnz_percentage(uint32_t nnz, uint32_t data_len)
  {
    return "(" + std::to_string(nnz / 1.0 / data_len * 100) + "%)";
  }

  static void check_cuszmode(const std::string& val)
  {
    auto legal = (val == "r2r" or val == "rel") or (val == "abs");
    if (not legal)
      throw std::runtime_error("`mode` must be \"r2r\" or \"abs\".");
  }

  static bool check_dtype(const std::string& val, bool delay_failure = true)
  {
    auto legal = (val == "f32") or (val == "f4");
    // auto legal = (val == "f32") or (val == "f64");
    if (not legal)
      if (not delay_failure)
        throw std::runtime_error("Only `f32`/`f4` is supported temporarily.");

    return legal;
  }

  static bool check_dtype(const psz_dtype& val, bool delay_failure = true)
  {
    auto legal = (val == F4);
    if (not legal)
      if (not delay_failure)
        throw std::runtime_error("Only `f32` is supported temporarily.");

    return legal;
  }

  static bool check_opt_in_list(
      std::string const& opt, std::vector<std::string> vs)
  {
    for (auto& i : vs) {
      if (opt == i) return true;
    }
    return false;
  }

  static void parse_length_literal(
      const char* str, std::vector<std::string>& dims)
  {
    std::stringstream data_len_ss(str);
    auto data_len_literal = data_len_ss.str();
    char delimiter = 'x';

    while (data_len_ss.good()) {
      std::string substr;
      std::getline(data_len_ss, substr, delimiter);
      dims.push_back(substr);
    }
  }

  static size_t filesize(std::string fname)
  {
    std::ifstream in(
        fname.c_str(), std::ifstream::ate | std::ifstream::binary);
    return in.tellg();
  }

  static size_t filesize(psz_header* h)
  {
    auto END = sizeof(h->entry) / sizeof(h->entry[0]);
    return h->entry[END - 1];
  }

  static size_t uncompressed_len(psz_header* h) { return h->x * h->y * h->z; }

  template <typename T1, typename T2>
  static size_t get_npart(T1 size, T2 subsize)
  {
    static_assert(
        std::numeric_limits<T1>::is_integer and
            std::numeric_limits<T2>::is_integer,
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
    auto GiB = 1.0 * 1024 * 1024 * 1024;
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
    int status = -4;
    char* res = abi::__cxa_demangle(name, nullptr, nullptr, &status);

    const char* const demangled_name = (status == 0) ? res : name;
    std::string ret_val(demangled_name);
    free(res);
    return ret_val;
  };
};

#endif /* C5FC9A7F_B78D_4E1E_AC2A_3E18232CEEE3 */
