//
// Created by jtian on 4/27/20.
//

#include "format.hh"
#include <iostream>
#include <sstream>
#include <string>

using std::string;

const string log_null = "       ";
const string log_err  = "\e[31m[ERR]\e[0m  ";
const string log_dbg  = "\e[34m[dbg]\e[0m  ";
const string log_info = "\e[32m[info]\e[0m ";
const string log_warn = "\e[31m[WARN]\e[0m ";

// https://stackoverflow.com/a/26080768/8740097
template <typename T>
void cusz::log::build(std::ostream& o, T t)
{
    o << t << std::endl;
}

// template void cusz::log::build<int>(std::ostream& o, int t);
// template void cusz::log::build<T>(std::ostream& o, T t);

template <typename T, typename... Args>
void cusz::log::build(std::ostream& o, T t, Args... args)  // recursive variadic function
{
    cusz::log::build(o, t);
    cusz::log::build(o, args...);
}

template <typename... Args>
void cusz::log::print(string log_head, Args... args)
{
    std::ostringstream oss;
    cusz::log::build(oss, args...);
    std::cout << log_head << oss.str();
}
