/**
 * @file capsule.hh
 * @author Jiannan Tian
 * @brief Simple data analysis (header)
 * @version 0.2.3
 * @date 2020-11-03
 * (create) 2020-11-03 (rev1) 2021-03-24 (rev2) 2021-09-08
 * @deprecated 0.3.2
 *
 * @copyright (C) 2020 by Washington State University, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef CAPSULE_HH
#define CAPSULE_HH

#if __cplusplus >= 201703L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

#include <cuda_runtime.h>
#include <driver_types.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "../stat/compare_gpu.hh"
// #include "../utils/io.hh"
#include "../utils/timer.hh"
#include "definition.hh"

template <typename T>
class Capsule {
   private:
    // variables
    struct {
        bool hptr{false}, dptr{false}, uniptr{false};
    } allocation_status;

    T *_dptr{nullptr}, *_hptr{nullptr}, *_uniptr{nullptr};

    uint32_t _len{0};

    std::string name;

    // logging setup; standalone
    const std::string LOG_NULL      = "      ";
    const std::string LOG_INFO      = "  ::  ";
    const std::string LOG_ERR       = " ERR  ";
    const std::string LOG_WARN      = "WARN  ";
    const std::string LOG_DBG       = " dbg  ";
    const std::string LOG_EXCEPTION = "  !!  ";

    // https://stackoverflow.com/a/26080768/8740097  CC BY-SA 3.0
    template <typename S>
    void build_string(std::ostream& o, S t)
    {
        o << t << " ";
    }

    template <typename S, typename... Args>
    void build_string(std::ostream& o, S t, Args... args)  // recursive variadic function
    {
        build_string(o, t);
        build_string(o, args...);
    }

    template <typename... Args>
    void LOGGING(const std::string& log_head, Args... args)
    {
        std::ostringstream oss;
        oss << log_head;
        build_string(oss, args...);

        oss.seekp(0, std::ios::end);
        std::stringstream::pos_type offset = oss.tellp();
        if (log_head == LOG_DBG) { std::cout << "\e[2m"; }  // dbg
        std::cout << oss.str() << std::endl;                // print content
        if (log_head == LOG_DBG) std::cout << "\e[0m";      // finish printing dbg
    }

    // IO
    int fs2mem(const char* fname, void* array, size_t num_els)
    {
        auto bytes = sizeof(T) * num_els;

        std::ifstream ifs(fname, std::ios::binary | std::ios::in);
        if (not ifs.is_open()) {
            std::cerr << "fail to open " << fname << std::endl;
            return -1;
        }
        ifs.read(reinterpret_cast<char*>(array), std::streamsize(bytes));
        ifs.close();

        return 0;
    }

    int mem2fs(const char* fname, void* array, size_t num_els)
    {
        auto bytes = sizeof(type) * num_els;

        std::ofstream ofs(fname, std::ios::binary | std::ios::out);
        if (not ofs.is_open()) {
            std::cerr << "fail to open " << fname << std::endl;
            return -1;
        }

        ofs.write(reinterpret_cast<const char*>(array), std::streamsize(bytes));
        ofs.close();

        return 0;
    }

    std::string ERRSTR_BUILDER(std::string func, std::string msg)
    {
        return "[Capsule(\"" + name + "\")::" + func + "] " + msg;
    }

    void check_len(std::string funcname)
    {
        if (_len == 0) throw std::runtime_error("[Capsule(\"" + name + "\")::" + funcname + "] " + "len == 0");
    }

    std::string ERROR_UNDEFINED_BEHAVIOR(std::string func, std::string msg = "undefined behavior")
    {  //
        return ERRSTR_BUILDER(func, "undefined behavior");
    }

   public:
    using type = T;

    // TODO rule of n
    // constructor
    Capsule() = default;
    Capsule(uint32_t len, const std::string _str = std::string("<unnamed>")) : _len(len), name(_str) {}
    Capsule(const std::string _str) : name(_str){};

    ~Capsule()
    {
        // Becasue _hptr can be obtained externally, and could be non-pinned, cudaFreeHost may not work properly.
        // if (allocation_status.hptr) cudaFreeHost(_hptr);

        if (allocation_status.dptr) cudaFree(_dptr);
        if (allocation_status.uniptr) cudaFree(_uniptr);
    }

    // getter
    T*&      dptr() { return _dptr; }
    T*&      hptr() { return _hptr; }
    T*&      uniptr() { return _uniptr; }
    T&       dptr(uint32_t i) { return _dptr[i]; }
    T&       hptr(uint32_t i) { return _hptr[i]; }
    T&       uniptr(uint32_t i) { return _uniptr[i]; }
    uint32_t len() const { return _len; }

    // setter
    Capsule& set_hptr(T* ptr)
    {
        _hptr                  = ptr;
        allocation_status.hptr = true;
        return *this;
    }
    Capsule& set_dptr(T* ptr)
    {
        _dptr                  = ptr;
        allocation_status.dptr = true;
        return *this;
    }
    Capsule& set_uniptr(T* ptr)
    {
        _uniptr                  = ptr;
        allocation_status.uniptr = true;
        return *this;
    }

    // debug
    void debug()
    {
        printf("Capsule debugging information\n");
        printf("  name   : %s\n", name.c_str());
        printf("  len    : %u\n", len());
        printf("  hptr   : %s\n", allocation_status.hptr ? "set" : "not set");
        printf("  dptr   : %s\n", allocation_status.dptr ? "set" : "not set");
        printf("  uniptr : %s\n", allocation_status.uniptr ? "set" : "not set");
    }

    // for debugging
    Capsule& set_name(std::string _str)
    {
        name = _str;
        return *this;
    }

    // variable len
    Capsule& set_len(uint32_t len)
    {
        if (len <= 0) throw std::runtime_error("length must be greater than 0");
        _len = len;
        return *this;
    }

    // IO
    Capsule& fromfile(std::string fname, double* time = nullptr)
    {
        if (not _hptr) throw std::runtime_error(ERRSTR_BUILDER("fromfile", "_hptr not set"));
        if (_len == 0) throw std::runtime_error(ERRSTR_BUILDER("fromfile", "len == 0"));

        auto a = hires::now();
        fs2mem(fname.c_str(), _hptr, _len);
        auto z = hires::now();

        if (time) *time = static_cast<duration_t>(z - a).count();

        return *this;
    }

    Capsule& tofile(std::string fname, double* time = nullptr)
    {
        if (not _hptr) { throw std::runtime_error(ERRSTR_BUILDER("tofile", "_hptr not set")); }
        if (_len == 0) throw std::runtime_error(ERRSTR_BUILDER("tofile", "len == 0"));

        auto a = hires::now();
        mem2fs(fname.c_str(), _hptr, _len);
        auto z = hires::now();

        if (time) *time = static_cast<duration_t>(z - a).count();

        return *this;
    }

    uint32_t nbyte() const { return _len * sizeof(T); }

    // memcpy h2d, synchronous
    Capsule& host2device()
    {
        check_len("host2device");

        cudaMemcpy(_dptr, _hptr, nbyte(), cudaMemcpyHostToDevice);
        return *this;
    }
    // memcpy d2h, synchronous
    Capsule& device2host()
    {
        check_len("device2host");

        cudaMemcpy(_hptr, _dptr, nbyte(), cudaMemcpyDeviceToHost);
        return *this;
    }
    // memcpy h2d, asynchronous
    Capsule& host2device_async(cudaStream_t stream)
    {
        check_len("host2device_async");

        cudaMemcpyAsync(_dptr, _hptr, nbyte(), cudaMemcpyHostToDevice, stream);
        return *this;
    }
    // memcpy d2h, asynchronous
    Capsule& device2host_async(cudaStream_t stream)
    {
        check_len("device2host_async");

        cudaMemcpyAsync(_hptr, _dptr, nbyte(), cudaMemcpyDeviceToHost, stream);
        return *this;
    }
    // shorthand
    Capsule& h2d() { return host2device(); }
    Capsule& d2h() { return device2host(); }
    Capsule& async_h2d(cudaStream_t stream) { return host2device_async(stream); }
    Capsule& async_d2h(cudaStream_t stream) { return device2host_async(stream); }

    // cudaMalloc wrapper
    Capsule& malloc(bool do_memset = true, uint8_t memset_val = 0)
    {
        check_len("malloc");

        if (allocation_status.dptr)
            LOGGING(LOG_WARN, "already allocated on device");
        else {
            cudaMalloc(&_dptr, nbyte());
            cudaMemset(_dptr, memset_val, nbyte());
            allocation_status.dptr = true;
        }
        return *this;
    }
    // cudaMallocHost wrapper, pinned
    Capsule& mallochost(bool do_memset = true, uint8_t memset_val = 0)
    {
        check_len("mallochost");

        if (allocation_status.hptr)
            LOGGING(LOG_WARN, "already allocated on host");
        else {
            cudaMallocHost(&_hptr, nbyte());
            memset(_hptr, memset_val, nbyte());
            allocation_status.hptr = true;
        }
        return *this;
    }
    // cudaMallocManaged wrapper
    Capsule& mallocmanaged(bool do_memset = true, uint8_t memset_val = 0)
    {
        check_len("mallocmanaged");

        if (allocation_status.uniptr)
            LOGGING(LOG_WARN, "already allocated as unified");
        else {
            cudaMallocManaged(&_uniptr, nbyte());
            cudaMemset(_uniptr, memset_val, nbyte());
            allocation_status.uniptr = true;
        }
        return *this;
    }
    // cudaFree wrapper
    Capsule& free()
    {
        if (not _dptr) throw std::runtime_error(ERRSTR_BUILDER("free", "_dptr is null"));
        cudaFree(_dptr);
        allocation_status.dptr = false;
        return *this;
    }
    // cudaFreeHost wrapper
    Capsule& freehost()
    {
        if (not _hptr) throw std::runtime_error(ERRSTR_BUILDER("free", "_hptr is null"));
        cudaFreeHost(_hptr);
        allocation_status.hptr = false;
        return *this;
    }
    // cudaFree wrapper, but for unified memory
    Capsule& freemanaged()
    {
        if (not _uniptr) throw std::runtime_error(ERRSTR_BUILDER("free", "_uniptr is null"));
        cudaFree(_uniptr);
        allocation_status.uniptr = false;
        return *this;
    }

   private:
    double maxval, minval, rng;

   public:
    double get_maxval() { return maxval; }
    double get_minval() { return minval; }
    double get_rng() { return rng; }

    // data scan
    Capsule& prescan(double& max_value, double& min_value, double& rng)
    {
        // may not work for _uniptr
        T result[4];
        parsz::thrustgpu_get_extrema_rawptr<T>(_dptr, _len, result);

        min_value = result[0];
        max_value = result[1];
        rng       = max_value - min_value;

        return *this;
    }
    // data scan
    Capsule& prescan()
    {
        prescan(maxval, minval, rng);
        return *this;
    }
};

#endif
