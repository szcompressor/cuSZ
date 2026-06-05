#ifndef _PORTABLE_DETAIL_KV_PARSE_HH
#define _PORTABLE_DETAIL_KV_PARSE_HH

#ifdef __CUDACC__
#error "detail/kv_parse.hh must not be included from CUDA translation units"
#endif

#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "c_type.h"
#include "cxx_type.hh"
#include "detail/str2num.hh"

namespace _portable::detail {

// ---- dimension parsing --------------------------------------------------

using _portable::xyz_t;

inline void parse_strlist(const char* in_str, std::vector<std::string>& out)
{
  std::stringstream ss(in_str);
  std::string       tmp;
  while (std::getline(ss, tmp, ','))
    if (not tmp.empty()) out.push_back(tmp);
}

// Parse dimension string "NxMxK", "N,M,K", "N*M*K", "N-M-K", "NmMmK", or
// plain "N".  Returns {len, ndim} where ndim is the token count in the string.
inline xyz_t parse_xyz(const char* str)
{
  if (not str or not *str) throw std::runtime_error("empty dimension string");

  std::string              s(str);
  std::vector<std::string> tokens;

  for (char delim : {'x', ',', '*', '-', 'm'}) {
    if (s.find(delim) == std::string::npos) continue;
    std::stringstream ss(s);
    std::string       tok;
    while (std::getline(ss, tok, delim))
      if (not tok.empty()) tokens.push_back(tok);
    break;
  }

  if (tokens.empty()) tokens.push_back(s);  // 1-D: no delimiter found

  int ndim = static_cast<int>(tokens.size());
  if (ndim > 3) throw std::runtime_error("dimension string has more than 3 components");

  _portable_len3 len{1, 1, 1};
  auto           at = [&](int i) -> size_t {
    auto v = str_to_int(tokens[i].c_str());
    if (not v) throw std::runtime_error("non-integer in dimension: " + tokens[i]);
    return static_cast<size_t>(*v);
  };

  if (ndim >= 1) len.x = at(0);
  if (ndim >= 2) len.y = at(1);
  if (ndim >= 3) len.z = at(2);

  return {len, ndim};
}

// Parse in z,y,x (math/slowest-to-fastest) order.
inline xyz_t parse_zyx(const char* str)
{
  auto r = parse_xyz(str);
  if (r.ndim == 3)
    std::swap(r.len.x, r.len.z);
  else if (r.ndim == 2)
    std::swap(r.len.x, r.len.y);
  return r;
}

// ---- key-value parsing --------------------------------------------------

inline bool is_kv_pair(const std::string& s) { return s.find('=') != std::string::npos; }

inline std::pair<std::string, std::string> separate_kv(const std::string& s)
{
  auto pos = s.find('=');
  if (pos == std::string::npos)
    throw std::runtime_error("not a correct key-value syntax, must be \"opt=value\"");
  return {s.substr(0, pos), s.substr(pos + 1)};
}

inline void parse_strlist_as_kv(
    const char* in_str, std::unordered_map<std::string, std::string>& kv)
{
  std::stringstream ss(in_str);
  std::string       tmp;
  while (std::getline(ss, tmp, ','))
    if (not tmp.empty()) kv.insert(separate_kv(tmp));
}

// Parse "key=(on|ON|off|OFF)" -> {key, bool}.  Returns {"", false} on no match.
inline std::pair<std::string, bool> parse_kv_onoff(const std::string& s)
{
  std::regex  pat(R"((.*?)=(on|ON|off|OFF))");
  std::smatch m;
  if (not std::regex_match(s, m, pat)) return {"", false};
  return {m[1].str(), m[2].str() == "on" or m[2].str() == "ON"};
}

}  // namespace _portable::detail

#endif  // _PORTABLE_DETAIL_KV_PARSE_HH
