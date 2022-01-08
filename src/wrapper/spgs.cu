/**
 * @file spgs.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-12-01
 * (created) 202-11-12 (rev.1) 2021-12-01
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include "../utils.hh"
#include "spgs.cuh"

namespace cusz {

template <typename T>
template <cusz::LOC FROM, cusz::LOC TO>
spGS<T>& spGS<T>::consolidate(uint8_t* dst)
{
    constexpr auto direction = CopyDirection<FROM, TO>::direction;
    auto           nbyte_idx = nnz * sizeof(int);
    auto           nbyte_val = nnz * sizeof(T);
    // index first
    CHECK_CUDA(cudaMemcpy(dst, /*       */ idx, nbyte_idx, direction));
    CHECK_CUDA(cudaMemcpy(dst + nbyte_idx, val, nbyte_val, direction));

    return *this;
}

template <typename T>
void spGS<T>::gather(
    T*            in,
    uint32_t      in_len,
    int*          nullarray,
    int*&         out_idx,
    T*&           out_val,
    int&          out_nnz,
    unsigned int& dump_nbyte)
{
    this->idx = out_idx;
    this->val = out_val;

    {  // phase 1: count nnz
        cuda_timer_t t;
        t.timer_start();
        out_nnz = thrust::count_if(thrust::device, in, in + in_len, [] __device__(const T& x) { return x != 0; });
        t.timer_end();
        milliseconds = t.get_time_elapsed();
    }
    // TODO improve
    this->nnz = out_nnz;

    // phase 2: gather
    thrust::counting_iterator<int> zero(0);
    using Tuple = thrust::tuple<T, int>;

    auto zipped_in      = thrust::make_zip_iterator(thrust::make_tuple(in, zero));
    auto zipped_in_end  = thrust::make_zip_iterator(thrust::make_tuple(in + in_len, zero + in_len));
    auto zipped_out     = thrust::make_zip_iterator(thrust::make_tuple(out_val, out_idx));
    auto zipped_out_end = thrust::make_zip_iterator(thrust::make_tuple(out_val + out_nnz, out_idx + out_nnz));

    {
        cuda_timer_t t;
        t.timer_start();
        thrust::copy_if(thrust::device, zipped_in, zipped_in_end, zipped_out, [] __host__ __device__(const Tuple& t) {
            return thrust::get<0>(t) != 0;
        });
        t.timer_end();
        milliseconds += t.get_time_elapsed();
    }

    dump_nbyte = (sizeof(int) + sizeof(T)) * out_nnz;
}

template <typename T>
void spGS<T>::scatter(int*& in_idx, T*& in_val, int nnz, T* out)
{
    cuda_timer_t t;
    t.timer_start();
    thrust::scatter(thrust::device, in_val, in_val + nnz, in_idx, out);
    t.timer_end();
    milliseconds = t.get_time_elapsed();
}

template <typename T>
void spGS<T>::scatter(uint8_t* _pool, int nnz, T* out, uint32_t out_len)
{
    auto nbyte_idx = nnz * sizeof(int);
    auto in_idx    = reinterpret_cast<int*>(_pool);
    auto in_val    = reinterpret_cast<T*>(_pool + (nbyte_idx));

    scatter(in_idx, in_val, nnz, out);
}

}  // namespace cusz

#define SPGS_TYPE cusz::spGS<float>

template class SPGS_TYPE;

template SPGS_TYPE& SPGS_TYPE::consolidate<cusz::LOC::HOST, cusz::LOC::HOST>(uint8_t*);
template SPGS_TYPE& SPGS_TYPE::consolidate<cusz::LOC::HOST, cusz::LOC::DEVICE>(uint8_t*);
template SPGS_TYPE& SPGS_TYPE::consolidate<cusz::LOC::DEVICE, cusz::LOC::HOST>(uint8_t*);
template SPGS_TYPE& SPGS_TYPE::consolidate<cusz::LOC::DEVICE, cusz::LOC::DEVICE>(uint8_t*);
