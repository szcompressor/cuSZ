//
// Created by jtian on 4/27/20.
//

#ifndef FORMAT_HH
#define FORMAT_HH

#include <string>

using std::string;

extern const string log_null = "       ";
extern const string log_err  = "\e[31m[ERR]\e[0m  ";
extern const string log_dbg  = "\e[34m[dbg]\e[0m  ";
extern const string log_info = "\e[32m[info]\e[0m ";
extern const string log_warn = "\e[31m[WARN]\e[0m ";

#endif  // FORMAT_HH
