#ifndef UTILS_FORMAT_HH
#define UTILS_FORMAT_HH

/**
 * @file format.hh
 * @author Jiannan Tian
 * @brief Formatting for log print (header).
 * @version 0.2
 * @date 2020-09-20
 * Created on 2020-04-27
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <iostream>
#include <sstream>
#include <string>

using std::string;

const string log_null = "      ";
const string log_err  = " ERR  ";
const string log_dbg  = " dbg  ";
const string log_info = "info  ";
const string log_warn = "WARN  ";
const string log_exp  = " exp  ";

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
void LogAll(const string& log_head, Args... args)
{
    std::ostringstream oss;
    oss << log_head;
    build(oss, args...);

    oss.seekp(0, std::ios::end);
    std::stringstream::pos_type offset = oss.tellp();
    if (log_head == log_dbg) { std::cout << "\e[2m"; }  // dbg

    // print progress
    if (log_head == log_info) {
        if (static_cast<int>(offset) + log_head.size() <= 80)
            oss << std::string(80 - log_head.size() - offset, '.');  // +9, ad hoc for log_*
        else
            oss << std::string(8, '.');  // +9, ad hoc for log_*
        oss << " [ok]";
    }
    std::cout << oss.str() << std::endl;            // print content
    if (log_head == log_dbg) std::cout << "\e[0m";  // finish printing dbg
}

#endif  // FORMAT_HH
