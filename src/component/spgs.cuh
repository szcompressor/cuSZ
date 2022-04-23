/**
 * @file spgs.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-12-01
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_COMPONENT_SPGS_CUH
#define CUSZ_COMPONENT_SPGS_CUH

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <iostream>
#include "common.hh"
#include "utils/timer.hh"

using std::cout;

template <typename T>
struct is_outlier {
    __host__ __device__ bool operator()(T x) { return x != 0; }
};

namespace cusz {

template <typename T = float>
class spGS {
   public:
    using Origin = T;

   private:
    unsigned int len;
    int*         idx;
    T*           val;
    float        milliseconds{0.0};

    int nnz;

    struct out_of_scope {
        int radius;

        __host__ __device__ bool operator()(T x)
        {
            auto xx = (x >= 0) ? x : -x;
            return xx >= radius;
        }
    };

    unsigned int __to128byte(unsigned int offset) { return ((offset - 1) / 128 + 1) * 128; }

   public:
    float get_time_elapsed() const { return milliseconds; }

   public:
    uint32_t get_total_nbyte(uint32_t len, int nnz) { return sizeof(int) * nnz + sizeof(T) * nnz; }

    void
    encode(T* in, uint32_t in_len, int* nullarray, int*& out_idx, T*& out_val, int& out_nnz, unsigned int& dump_nbyte);

    template <cusz::LOC FROM = cusz::LOC::DEVICE, cusz::LOC TO = cusz::LOC::HOST>
    spGS& consolidate(uint8_t* dst);

    void decode(int*& in_idx, T*& in_val, int nnz, T* out);

    void decode(uint8_t* _pool, int nnz, T* out, uint32_t out_len);
};

}  // namespace cusz

#endif
