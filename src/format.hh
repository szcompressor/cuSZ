//
// Created by jtian on 4/27/20.
//

#ifndef FORMAT_HH
#define FORMAT_HH

#include <iostream>
#include <sstream>
#include <string>

using std::string;

extern const string log_null;
extern const string log_err;
extern const string log_dbg;
extern const string log_info;
extern const string log_warn;

namespace cusz {
namespace log {

// https://stackoverflow.com/a/26080768/8740097
template <typename T>
void build(std::ostream& o, T t);

template <typename T, typename... Args>
void build(std::ostream& o, T t, Args... args);

template <typename... Args>
void print(string log_head, Args... args);

}  // namespace log
}  // namespace cusz

#endif  // FORMAT_HH
