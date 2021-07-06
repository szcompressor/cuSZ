/**
 * @file handle_sparsity.h
 * @author Jiannan Tian
 * @brief (header) A high-level sparsity handling wrapper.
 * @version 0.3
 * @date 2021-06-17
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_WRAPPER_HANDLE_SPARSITY_H
#define CUSZ_WRAPPER_HANDLE_SPARSITY_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>

// put it where functions are called

// auto matrixified_len1  = [](auto size) {
//     static_assert(std::numeric_limits<decltype(size)>::is_integer, "[matrixify] must be plain interger types.");
//     return = static_cast<size_t>(ceil(sqrt(len)));
// }

template <typename Data = float>
struct OutlierDescriptor {
    int*  ondev_row_ptr{nullptr};
    int*  ondev_col_idx{nullptr};
    Data* ondev_csr_val{nullptr};

    unsigned int m{0};
    unsigned int nnz{0};
    unsigned int bytelen_rowptr{0};
    unsigned int bytelen_colidx{0};
    unsigned int bytelen_values{0};
    unsigned int bytelen_total{0};

    OutlierDescriptor(unsigned int _len) { this->m = static_cast<size_t>(ceil(sqrt(_len))); }

    /********************************************************************************
     * use after nnz is known
     ********************************************************************************/
    void configure(int nnz)
    {
        this->nnz      = nnz;
        bytelen_rowptr = sizeof(unsigned int) * (m + 1);
        bytelen_colidx = sizeof(unsigned int) * nnz;
        bytelen_values = sizeof(Data) * nnz;
        bytelen_total  = bytelen_rowptr + bytelen_colidx + bytelen_values;
    }

    /********************************************************************************
     * so many input args just to decouple from ANY header arrangement
     ********************************************************************************/
    void compress_archive(
        uint8_t*      start,
        unsigned int& nnz,
        unsigned int& bytelen_rowptr,
        unsigned int& bytelen_colidx,
        unsigned int& bytelen_values,
        unsigned int& bytelen_total)
    {
        nnz            = this->nnz;
        bytelen_rowptr = this->bytelen_rowptr;
        bytelen_colidx = this->bytelen_colidx;
        bytelen_values = this->bytelen_values;
        bytelen_total  = this->bytelen_total;

        cudaMemcpy(
            start,  //
            this->ondev_row_ptr, bytelen_rowptr, cudaMemcpyDeviceToHost);
        cudaMemcpy(
            start + bytelen_rowptr,  //
            this->ondev_col_idx, bytelen_colidx, cudaMemcpyDeviceToHost);
        cudaMemcpy(
            start + bytelen_rowptr + bytelen_colidx,  //
            this->ondev_csr_val, bytelen_values, cudaMemcpyDeviceToHost);
    }

    void decompress_extract(
        uint8_t*     onhost_start,
        uint8_t*     ondev_start,
        unsigned int nnz,
        unsigned int bytelen_rowptr,
        unsigned int bytelen_colidx,
        unsigned int bytelen_values,
        unsigned int bytelen_total,
        bool         memcpy_h2d)
    {
        if (memcpy_h2d) cudaMemcpy(ondev_start, onhost_start, bytelen_total, cudaMemcpyHostToDevice);
        this->ondev_row_ptr = reinterpret_cast<int*>(ondev_start);
        this->ondev_col_idx = reinterpret_cast<int*>(ondev_start + bytelen_rowptr);
        this->ondev_csr_val = reinterpret_cast<Data*>(ondev_start + bytelen_rowptr + bytelen_colidx);
    };

    void destroy_devspace()
    {
        if (this->ondev_row_ptr) cudaFree(this->ondev_row_ptr);
        if (this->ondev_col_idx) cudaFree(this->ondev_col_idx);
        if (this->ondev_csr_val) cudaFree(this->ondev_csr_val);
    }
};

void compress_gather_CUDA10(struct OutlierDescriptor<float>* outlier_desc, float* ondev_outlier);
void decompress_scatter_CUDA10(struct OutlierDescriptor<float>* outlier_desc, float* ondev_outlier);

#endif