/**
 * @file snippets.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-03-26
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include "snippets.hh"

// TODO g++ and clang++ use mangled type_id name, add macro
// https://stackoverflow.com/a/4541470/8740097
string demangle(const char* name)
{
    int   status = -4;
    char* res    = abi::__cxa_demangle(name, nullptr, nullptr, &status);

    const char* const demangled_name = (status == 0) ? res : name;
    string            ret_val(demangled_name);
    free(res);
    return ret_val;
};