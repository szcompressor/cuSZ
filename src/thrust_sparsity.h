/**
 * @file thrust_gather_scatter.cuh
 * @author Jiannan Tian
 * @brief Thrust based gather-scatter methods, header only.
 * @version 0.1.4
 * @date 2020-11-12
 *
 * @copyright (C) 2020 by Washington State University, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <iostream>
#include <utility>

using std::cout;
using std::endl;

#if __cplusplus >= 201402L

#include "utils/cuda_mem.cuh"

template <typename Data, typename Index>
std::tuple<Data*, Index*> ThrustGatherDualQuantOutlier(Data* d_dense, const size_t len, unsigned int& nnz)
{
    using tup = thrust::tuple<Data, Index>;

    thrust::device_ptr<Data> p_dense = thrust::device_pointer_cast(d_dense);

    nnz = thrust::count_if(thrust::device, p_dense, p_dense + len, [] __device__(const Data& a) { return a != 0.0f; });

    // TODO thrust copy to host, https://stackoverflow.com/a/43983021/8740097
    auto d_sparse = mem::CreateCUDASpace<Data>(nnz);
    auto d_idxmap = mem::CreateCUDASpace<Index>(nnz);

    thrust::device_ptr<Data>  p_sparse = thrust::device_pointer_cast(d_sparse);
    thrust::device_ptr<Index> p_idxmap = thrust::device_pointer_cast(d_idxmap);

    thrust::counting_iterator<Index> idx_first(0);
    // thrust::counting_iterator<Index> idx_last = idx_first + len;
    // clang-format off
    auto in_begin  = thrust::make_zip_iterator(thrust::make_tuple(p_dense,        idx_first)      );
    auto in_end    = thrust::make_zip_iterator(thrust::make_tuple(p_dense  + len, idx_first + len));
    auto out_begin = thrust::make_zip_iterator(thrust::make_tuple(p_sparse,       p_idxmap)       );
    auto out_end   = thrust::make_zip_iterator(thrust::make_tuple(p_sparse + nnz, p_idxmap  + nnz));
    // clang-format on

    thrust::copy_if(
        thrust::device, in_begin, in_end, out_begin, [] __device__(const tup& t) { return thrust::get<0>(t) != 0.0f; });

    auto h_sparse = mem::CreateHostSpaceAndMemcpyFromDevice(d_sparse, nnz);
    auto h_idxmap = mem::CreateHostSpaceAndMemcpyFromDevice(d_idxmap, nnz);

    return std::make_pair(h_sparse, h_idxmap);
}

template <typename Data, typename Index>
void ThrustScatterDualQuantOutlier(Data* d_dense, size_t len, unsigned int& nnz, Data* d_sparse, Index* d_idxmap)
{
    thrust::device_ptr<Data>  p_dense  = thrust::device_pointer_cast(d_dense);
    thrust::device_ptr<Data>  p_sparse = thrust::device_pointer_cast(d_sparse);
    thrust::device_ptr<Index> p_idxmap = thrust::device_pointer_cast(d_idxmap);
    // dense[idxmap[i]] = sparse[i]
    // len(sparse) == len(idxmap), much less than (<<) dense
    thrust::scatter(p_sparse, p_sparse + nnz, p_idxmap, p_dense);
}

#endif