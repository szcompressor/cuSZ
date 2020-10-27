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

#include <cuda_runtime.h>

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

void GetDeviceProperty()
{
    int         num_dev  = 0;
    cudaError_t error_id = cudaGetDeviceCount(&num_dev);

    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", static_cast<int>(error_id), cudaGetErrorString(error_id));
        exit(EXIT_FAILURE);
    }
    if (num_dev == 0) { printf("NO CUDA device detected.\n"); }
    int dev, driver_ver = 0, runtime_ver = 0;

    for (dev = 0; dev < num_dev; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp dev_prop;
        cudaGetDeviceProperties(&dev_prop, dev);
        printf("device #%d, %s: \n", dev, dev_prop.name);

        cudaDriverGetVersion(&driver_ver);
        cudaRuntimeGetVersion(&runtime_ver);
        printf(
            "  driver/runtime\t%d.%d/%d.%d\n", driver_ver / 1000, (driver_ver % 100) / 10, runtime_ver / 1000,
            (runtime_ver % 100) / 10);
        printf("  compute capability:\t%d.%d\n", dev_prop.major, dev_prop.minor);
        printf("  global memory:\t%.0f MiB\n", static_cast<float>(dev_prop.totalGlobalMem / 1048576.0f));
        printf("  constant memory:\t%zu bytes\n", dev_prop.totalConstMem);
        printf("  shared mem per block:\t%zu bytes\n", dev_prop.sharedMemPerBlock);
        printf("  shared mem per SM:\t%zu bytes\n", dev_prop.sharedMemPerMultiprocessor);
        printf("  registers per block:\t%d\n", dev_prop.regsPerBlock);
    }
    printf("\n");
}

void GetMachineProperties()
{
    std::vector<std::string> v;
    cout << "host informaton: " << endl;

    auto cpuinfo = ExecShellCommand(  //
        std::string("cat /proc/cpuinfo "
                    "| grep \"model name\" "
                    "| head -n 1 "
                    "| awk -F': ' '{print $NF}'")
            .c_str());
    cout << "  cpu model\t" << cpuinfo;

    auto meminfo = ExecShellCommand(  //
        std::string("cat /proc/meminfo"
                    "| grep \"MemTotal\" "
                    "| awk -F' ' '{print $2\" \"$3}'")
            .c_str());

    cout << "  memory size\t" << meminfo;

    auto endianness = ExecShellCommand(  //
        std::string("lscpu "
                    "| grep Endian "
                    "| awk -F'  ' '{print $NF}'")
            .c_str());

    cout << "  byte order\t" << endianness;
    printf("\n");
}

#endif
