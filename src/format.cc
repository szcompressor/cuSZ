/**
 * @file format.cc
 * @author Jiannan Tian
 * @brief Formatting for log print.
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-04-27
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include "format.hh"
#include <iostream>
#include <sstream>
#include <string>

using std::string;

// const string log_null = "       ";
// const string log_err  = "\e[31m[ERR]\e[0m  ";
// const string log_dbg  = "\e[34m[dbg]\e[0m  ";
// const string log_info = "\e[32m[info]\e[0m ";
// const string log_warn = "\e[31m[WARN]\e[0m ";

// TODO change the following to const char[]
const string log_null = "      ";
const string log_err  = " ERR  ";
const string log_dbg  = " dbg  ";
const string log_info = "info  ";
const string log_warn = "WARN  ";
