// Author: Jiannan Tian
// query machine information

#ifndef E8CDEF97_5136_45C6_A6F2_3FECD549F8A4
#define E8CDEF97_5136_45C6_A6F2_3FECD549F8A4

#include <memory>
#include <vector>

#include "cusz/type.h"

#if defined(_WIN32)
#define popen _popen
#define pclose _pclose
#endif

struct cpu_diagnostics {
  static std::string exec_shellcmd(const char* cmd)
  {
    std::array<char, 128> buffer;
    std::string result;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
#pragma GCC diagnostic pop
    if (!pipe) { throw std::runtime_error("popen() failed!"); }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
      result += buffer.data();
    }
    return result;
  }

  static void get_cpu_properties()
  {
    std::vector<std::string> v;
    std::cout << "host information: " << std::endl;

    auto cpuinfo = exec_shellcmd(  //
        std::string("cat /proc/cpuinfo "
                    "| grep \"model name\" "
                    "| head -n 1 "
                    "| awk -F': ' '{print $NF}'")
            .c_str());
    std::cout << "  cpu model\t" << cpuinfo;

    auto meminfo = exec_shellcmd(  //
        std::string("cat /proc/meminfo"
                    "| grep \"MemTotal\" "
                    "| awk -F' ' '{print $2\" \"$3}'")
            .c_str());

    std::cout << "  memory size\t" << meminfo;

    auto endianness = exec_shellcmd(  //
        std::string("lscpu "
                    "| grep Endian "
                    "| awk -F'  ' '{print $NF}'")
            .c_str());

    std::cout << "  byte order\t" << endianness;
    printf("\n");
  }
};

#endif /* E8CDEF97_5136_45C6_A6F2_3FECD549F8A4 */
