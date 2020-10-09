/**
 * @file query.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.1.2
 * @date 2020-10-05
 *
 * (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 *
 */

#ifndef QUERY_HH
#define QUERY_HH

//#include <cuda_runtime.h>
#include <array>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using std::cerr;
using std::cout;
using std::endl;
using std::string;

std::string ExecShellCommand(const char* cmd)
{
    std::array<char, 128>                    buffer;
    std::string                              result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) { throw std::runtime_error("popen() failed!"); }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) { result += buffer.data(); }
    return result;
}

std::vector<std::string> GetMachineProperties()
{
    std::vector<std::string> v;

    auto cpuinfo = ExecShellCommand(  //
        std::string("cat /proc/cpuinfo "
                    "| grep \"model name\" "
                    "| head -n 1 "
                    "| awk -F': ' '{print $NF}'")
            .c_str());
    cout << "cpu model\t" << cpuinfo;

    auto meminfo = ExecShellCommand(  //
        std::string("cat /proc/meminfo"
                    "| grep \"MemTotal\" "
                    "| awk -F' ' '{print $2\" \"$3}'")
            .c_str());

    cout << "memory size\t" << meminfo;

    auto endianness = ExecShellCommand(  //
        std::string("lscpu "
                    "| grep Endian "
                    "| awk -F'  ' '{print $NF}'")
            .c_str());

    cout << "byte order\t" << endianness;
    return v;
}

#endif
