#ifndef _PORTABLE_UTILS_FORMAT_HH
#define _PORTABLE_UTILS_FORMAT_HH

// Jiannan Tian
// (created) 2020-04-27 (update) 2020-09-20...2024-12-22

#include <iostream>
#include <regex>
#include <sstream>
#include <string>

namespace _portable::utils {

#define LOG_NULL "      "
#define LOG_INFO "  ::  "
#define LOG_ERR " ERR  "
#define LOG_WARN "WARN  "
#define LOG_DBG " dbg  "
#define LOG_EXCEPTION "  !!  "

// https://stackoverflow.com/a/26080768/8740097  CC BY-SA 3.0
template <typename T>
void build(std::ostream& o, T t)
{
  o << t << " ";
}

template <typename T, typename... Args>
void build(std::ostream& o, T t, Args... args)  // recursive variadic function
{
  build(o, t);
  build(o, args...);
}

template <typename... Args>
void LOGGING(const std::string& log_head, Args... args)
{
  std::ostringstream oss;
  oss << log_head;
  build(oss, args...);

  oss.seekp(0, std::ios::end);
  std::stringstream::pos_type offset = oss.tellp();
  if (log_head == LOG_DBG) { std::cout << "\e[2m"; }  // dbg
  std::cout << oss.str() << std::endl;                // print content
  if (log_head == LOG_DBG) std::cout << "\e[0m";      // finish printing dbg
}

// Terminal markup → ANSI escape codes.
// Syntax: @bold+ul@  *bold*  _underline_  ^^red^^  %gray%
// Spaces are intentionally excluded from _underline_ to avoid false positives
// on underscores in identifiers. Nesting is not supported.
inline std::string doc_format(const std::string& s)
{
  static const std::regex bful(R"(@(.*?)@)");
  static const std::regex bf(R"(\*(.*?)\*)");
  static const std::regex ul(R"(_((\w|-|\d|\.)+?)_)");
  static const std::regex red(R"(\^\^(.*?)\^\^)");
  static const std::regex gray(R"(%(.*?)%)");

  static const std::string s_bful("\e[1m\e[4m$1\e[0m");
  static const std::string s_bf("\e[1m$1\e[0m");
  static const std::string s_ul("\e[4m$1\e[0m");
  static const std::string s_red("\e[31m$1\e[0m");
  static const std::string s_gray("\e[37m$1\e[0m");

  auto a = std::regex_replace(s, bful, s_bful);
  auto b = std::regex_replace(a, bf,   s_bf);
  auto c = std::regex_replace(b, ul,   s_ul);
  auto d = std::regex_replace(c, red,  s_red);
  return   std::regex_replace(d, gray, s_gray);
}

}  // namespace _portable::utils

#endif /* _PORTABLE_UTILS_FORMAT_HH */
