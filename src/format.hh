/**
 * @file format.hh
 * @author Jiannan Tian
 * @brief Formatting for log print (header).
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-04-27
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

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
void logall(string log_head, Args... args)
{
    std::ostringstream oss;
    oss << log_head;
    build(oss, args...);

    oss.seekp(0, std::ios::end);
    std::stringstream::pos_type offset = oss.tellp();

    // print dbg
    if (log_head == log_dbg) std::cout << "\e[2m";

    // print progress
    if (static_cast<int>(offset) <= 80 and log_head == log_info) {
        oss << std::string(80 - log_head.size() - offset, '.');  // +9, ad hoc for log_*
        oss << " [ok]";
    }

    // print content
    std::cout << oss.str() << std::endl;

    // finish priting dbg
    if (log_head == log_dbg) std::cout << "\e[0m";
}

#endif  // FORMAT_HH
