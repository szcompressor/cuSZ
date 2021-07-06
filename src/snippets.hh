/**
 * @file snippets.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-03-26
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef SNIPPETS_HH
#define SNIPPETS_HH

#include <cxxabi.h>
#include <string>

using std::string;

string demangle(const char* name);

#endif