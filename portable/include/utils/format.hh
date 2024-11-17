#ifndef _PORTABLE_UTILS_FORMAT_HH
#define _PORTABLE_UTILS_FORMAT_HH

// Jiannan Tian
// (created) 2020-04-27 (update) 2020-09-20...2024-12-22

#include <iostream>
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

}  // namespace _portable::utils

#endif /* _PORTABLE_UTILS_FORMAT_HH */
