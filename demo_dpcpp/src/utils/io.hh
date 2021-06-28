#ifndef UTILS_IO_HH
#define UTILS_IO_HH

/**
 * @file io.hh
 * @author Jiannan Tian
 * @brief Read and write binary.
 * @version 0.2
 * @date 2020-09-20
 * Created on 2019-08-27
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <fstream>
#include <iostream>

namespace io {

template <typename T>
T* ReadBinaryToNewArray(const std::string& name_on_fs, size_t dtype_len)
{
    std::ifstream ifs(name_on_fs.c_str(), std::ios::binary | std::ios::in);
    if (not ifs.is_open()) {
        std::cerr << "fail to open " << name_on_fs << std::endl;
        exit(1);
        // return;
    }
    auto _a = new T[dtype_len]();
    ifs.read(reinterpret_cast<char*>(_a), std::streamsize(dtype_len * sizeof(T)));
    ifs.close();
    return _a;
}

template <typename T>
void ReadBinaryToArray(const std::string& name_on_fs, T* _a, size_t dtype_len)
{
    std::ifstream ifs(name_on_fs.c_str(), std::ios::binary | std::ios::in);
    if (not ifs.is_open()) {
        std::cerr << "fail to open " << name_on_fs << std::endl;
        exit(1);
        // return;
    }
    // auto _a = new T[dtype_len]();
    ifs.read(reinterpret_cast<char*>(_a), std::streamsize(dtype_len * sizeof(T)));
    ifs.close();
}

template <typename T>
void WriteArrayToBinary(const std::string& name_on_fs, T* const _a, size_t const dtype_len)
{
    std::ofstream ofs(name_on_fs.c_str(), std::ios::binary | std::ios::out);
    if (not ofs.is_open()) return;
    ofs.write(reinterpret_cast<const char*>(_a), std::streamsize(dtype_len * sizeof(T)));
    ofs.close();
}

}  // namespace io

#endif  // IO_HH
