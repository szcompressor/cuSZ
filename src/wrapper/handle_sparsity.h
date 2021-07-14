/**
 * @file handle_sparsity.h
 * @author Jiannan Tian
 * @brief (header) A high-level sparsity handling wrapper. Gather/scatter method to handle cuSZ prediction outlier.
 * @version 0.3
 * @date 2021-07-08
 * (created) 2020-09-10 (rev1) 2021-06-17 (rev2) 2021-07-08
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_WRAPPER_HANDLE_SPARSITY_H
#define CUSZ_WRAPPER_HANDLE_SPARSITY_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

template <typename Data = float>
struct OutlierDescriptor {
    struct {
        uint8_t* ptr;

        struct {
            int*  rowptr;
            int*  colidx;
            Data* values;
        } entry;

        struct {
            unsigned int rowptr;
            unsigned int colidx;
            unsigned int values;
        } offset;

    } pool;

    struct {
        unsigned int rowptr;
        unsigned int colidx;
        unsigned int values;
        unsigned int total;
    } bytelen;

    unsigned int m{0};
    unsigned int dummy_nnz{0};
    unsigned int nnz{0};

    /********************************************************************************
     * compression use
     ********************************************************************************/
    OutlierDescriptor(unsigned int _len)
    {  //
        this->m = static_cast<size_t>(ceil(sqrt(_len)));
    }

    /**
     * @brief nnz is not necessarity the real one.
     *
     * @param try_nnz
     * @return unsigned int
     */
    unsigned int compress_query_pool_bytelen(unsigned int try_nnz)
    {
        pool.offset.rowptr = 0;
        pool.offset.colidx = sizeof(int) * (m + 1);
        pool.offset.values = sizeof(int) * (m + 1) + sizeof(int) * try_nnz;

        return sizeof(int) * (m + 1)      // rowptr
               + sizeof(int) * try_nnz    // colidx
               + sizeof(Data) * try_nnz;  // values
    }

    /**
     * @brief set up memory pool
     *
     * @param _pool
     * @param try_nnz
     */
    void compress_configure_pool(uint8_t* _pool, unsigned int try_nnz)
    {
        if (not _pool) throw std::runtime_error("Memory pool is no allocated.");
        pool.ptr          = _pool;
        pool.entry.rowptr = reinterpret_cast<int*>(pool.ptr + pool.offset.rowptr);
        pool.entry.colidx = reinterpret_cast<int*>(pool.ptr + pool.offset.colidx);
        pool.entry.values = reinterpret_cast<Data*>(pool.ptr + pool.offset.values);
    }

    // TODO handle nnz == 0 otherwise
    unsigned int compress_query_csr_bytelen() const
    {
        return sizeof(int) * (m + 1)  // rowptr
               + sizeof(int) * nnz    // colidx
               + sizeof(Data) * nnz;  // values
    }

    /**
     * @brief use when the real nnz is known
     *
     * @param nnz
     */
    void compress_configure_with_nnz(int nnz)
    {
        this->nnz      = nnz;
        bytelen.rowptr = sizeof(int) * (m + 1);
        bytelen.colidx = sizeof(int) * nnz;
        bytelen.values = sizeof(Data) * nnz;
        bytelen.total  = bytelen.rowptr + bytelen.colidx + bytelen.values;
    }

    void compress_archive_outlier(uint8_t* archive, int& nnz)
    {
        nnz = this->nnz;

        // clang-format off
        cudaMemcpy(archive + 0,                               pool.entry.rowptr, bytelen.rowptr, cudaMemcpyDeviceToHost);
        cudaMemcpy(archive + bytelen.rowptr,                  pool.entry.colidx, bytelen.colidx, cudaMemcpyDeviceToHost);
        cudaMemcpy(archive + bytelen.rowptr + bytelen.colidx, pool.entry.values, bytelen.values, cudaMemcpyDeviceToHost);
        // clang-format on
    }

    /********************************************************************************
     * decompression use
     ********************************************************************************/
    OutlierDescriptor(unsigned int _len, unsigned int _nnz)
    {  //
        this->m   = static_cast<size_t>(ceil(sqrt(_len)));
        this->nnz = _nnz;

        bytelen.rowptr = sizeof(int) * (this->m + 1);
        bytelen.colidx = sizeof(int) * this->nnz;
        bytelen.values = sizeof(Data) * this->nnz;
        bytelen.total  = bytelen.rowptr + bytelen.colidx + bytelen.values;
    }

    void decompress_extract_outlier(uint8_t* _pool)
    {
        pool.offset.rowptr = 0;
        pool.offset.colidx = bytelen.rowptr;
        pool.offset.values = bytelen.rowptr + bytelen.colidx;

        pool.ptr          = _pool;
        pool.entry.rowptr = reinterpret_cast<int*>(pool.ptr + pool.offset.rowptr);
        pool.entry.colidx = reinterpret_cast<int*>(pool.ptr + pool.offset.colidx);
        pool.entry.values = reinterpret_cast<Data*>(pool.ptr + pool.offset.values);
    };
};

void compress_gather_CUDA10(struct OutlierDescriptor<float>*, float*, float&);
void decompress_scatter_CUDA10(struct OutlierDescriptor<float>*, float*, float&);

#endif