/**
 * @file query.hh
 * @author Jiannan Tian
 * @brief query machine information
 * @version 0.1.3
 * @date 2020-10-05
 *
 * @copyright (C) 2020 by Washington State University, Argonne National
 * Laboratory See LICENSE in top-level directory
 *
 */

#ifndef QUERY_HH
#define QUERY_HH

#include "busyheader.hh"
#include "cusz/type.h"
#include "port.hh"
#include "query_dev.cu_hip.hh"

struct Diagnostics {
  static std::string ExecShellCommand(const char* cmd)
  {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) { throw std::runtime_error("popen() failed!"); }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
      result += buffer.data();
    }
    return result;
  }

  static void GetMachineProperties()
  {
    std::vector<std::string> v;
    std::cout << "host information: " << std::endl;

    auto cpuinfo = ExecShellCommand(  //
        std::string("cat /proc/cpuinfo "
                    "| grep \"model name\" "
                    "| head -n 1 "
                    "| awk -F': ' '{print $NF}'")
            .c_str());
    std::cout << "  cpu model\t" << cpuinfo;

    auto meminfo = ExecShellCommand(  //
        std::string("cat /proc/meminfo"
                    "| grep \"MemTotal\" "
                    "| awk -F' ' '{print $2\" \"$3}'")
            .c_str());

    std::cout << "  memory size\t" << meminfo;

    auto endianness = ExecShellCommand(  //
        std::string("lscpu "
                    "| grep Endian "
                    "| awk -F'  ' '{print $NF}'")
            .c_str());

    std::cout << "  byte order\t" << endianness;
    printf("\n");
  }
};

#endif
