/**
 * @file sparse_op.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-03-22
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef SPARSE_HH
#define SPARSE_HH

// std headers
#include <cstdlib>
#include <iostream>
#include <stdexcept>

// CUDA headers
#include <cuda_runtime_api.h>
#include <cusparse.h>

#include "../type_trait.hh"

using std::cerr;
using std::cout;
using std::endl;

#if __cplusplus >= 201703L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

enum class cuSPARSEver { cuda10_or_earlier, cuda11_onward };
enum class sparseMethod { as_matrix, as_vector };

struct cusparseTypeAlias {
    // consistent with CUSPARSE_INDEX_32I: 32-bit signed integer [1, 2^31 - 1]
    // using IdxSprs             = int;
    static const auto IdxType = CUSPARSE_INDEX_32I;
    static const auto IdxBase = CUSPARSE_INDEX_BASE_ZERO;
    static const auto MatType = CUSPARSE_MATRIX_TYPE_GENERAL;
    // static const auto MatMajor = CUSPARSE_DIRECTION_ROW;
    static const auto MatOrder = CUSPARSE_ORDER_ROW;
};

template <typename T>
struct cusparseBinding;

// change to std::enable_if
template <>
struct cusparseBinding<float> {
    static const auto ValType = CUDA_R_32F;
};
template <>
struct cusparseBinding<double> {
    static const auto ValType = CUDA_R_64F;
};

template <typename T>
struct DenseMatrix {
    T*           mat;
    unsigned int num_rows;
    unsigned int num_cols;
    unsigned int ld() const { return num_cols; }
    unsigned int len() const { return num_rows * num_cols; }

    DenseMatrix(unsigned int num_rows_, unsigned int num_cols_) : num_rows(num_rows_), num_cols(num_cols_) {}
};

template <typename T>
struct CompressedSparseRow {
    int* offsets{};
    struct {
        unsigned int num_rows;
        unsigned int num_cols;
    } dn_size;
    unsigned int num_offsets() const { return dn_size.num_rows + 1; }

    T*   values{};
    int* columns{};
    union {
        int64_t nnz;
        int64_t values;
        int64_t columns;
    } sp_size;

    CompressedSparseRow(unsigned num_rows_, unsigned int num_cols_) : dn_size({num_rows_, num_cols_}) {}
    CompressedSparseRow(unsigned num_rows_, unsigned int num_cols_, int64_t nnz_) : dn_size({num_rows_, num_cols_})
    {
        sp_size.nnz = nnz_;
    }
};

// mostly device only class (excl. export)
template <typename T>
class SparseOps {
    using CSR = struct CompressedSparseRow<T>;
    using MAT = struct DenseMatrix<T>;

   private:
    // metadata and data
    MAT*            mat;
    CSR*            csr;
    Index<3>::idx_t trio_bytelen{};
    size_t          total_bytelen{};

    struct {
        void*  space;
        size_t size;
    } buffer;

    // runtime
    cusparseHandle_t     handle{nullptr};
    cusparseDnMatDescr_t dn_descri{};
    cusparseSpMatDescr_t sp_descri{};

   public:
    SparseOps(MAT* mat_, CSR* csr_)
    {
        mat = mat_;
        csr = csr_;
    }

    ~SparseOps();

    size_t get_total_bytelen() const { return total_bytelen; }

    template <cuSPARSEver ver>
    SparseOps& Gather();

    template <cuSPARSEver ver>
    SparseOps& Scatter();

    SparseOps& ExportCSR(uint8_t* start_addr);
    SparseOps& ImportCSR(uint8_t* start_addr, Index<3>::idx_t trio);
};

template <typename T>
template <cuSPARSEver ver>
SparseOps<T>& SparseOps<T>::Gather()
{
    if (ver == cuSPARSEver::cuda11_onward) {
        cusparseCreate(&handle);
        cusparseCreateDnMat(
            &dn_descri, mat->num_rows, mat->num_cols, mat->ld(), mat->mat, cusparseBinding<T>::ValType,
            cusparseTypeAlias::MatOrder);

        cusparseCreateCsr(
            &sp_descri, csr->dn_size.num_rows, csr->dn_size.num_cols, 0 /*nnz placeholder*/, csr->offsets,
            nullptr /*csr->columns placeholder*/, nullptr /*csr->values placeholder*/, cusparseTypeAlias::IdxType,
            cusparseTypeAlias::IdxType, cusparseTypeAlias::IdxBase, cusparseBinding<T>::ValType);

//        cout << "num rows: " << mat->num_rows << '\n';
//        cout << "num cols: " << mat->num_cols << '\n';
//        cout << "ld(): " << mat->ld() << '\n';
//        cout << "dn_size.num_rows: " << csr->dn_size.num_rows << '\n';
//        cout << "dn_size.num_cols: " << csr->dn_size.num_cols << '\n';

        // allocate buffer (if needed) //
        cusparseDenseToSparse_bufferSize(
            handle, dn_descri, sp_descri, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &buffer.size);
        cudaMalloc(&buffer.space, buffer.size);

//        cout << "buffersize: " << buffer.size << '\n';

        // execute conversion //
        cusparseDenseToSparse_analysis(handle, dn_descri, sp_descri, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, buffer.space);

        // get nnz //
        int64_t num_rows_tmp, num_cols_tmp, nnz_tmp;
        cusparseSpMatGetSize(sp_descri, &num_rows_tmp, &num_cols_tmp, &nnz_tmp);
        csr->sp_size.nnz = nnz_tmp;

        cout << "nnz: " << csr->sp_size.nnz << endl;

        // allocate CSR column indices and values //
        cudaMalloc((void**)&csr->columns, csr->sp_size.columns * sizeof(int));  // TODO when to release?
        cudaMalloc((void**)&csr->values, csr->sp_size.values * sizeof(T));      // TODO when to release?

        // reset offsets, column indices, and values pointers //
        cusparseCsrSetPointers(sp_descri, csr->offsets, csr->columns, csr->values);

        // execute Sparse to Dense conversion //
        cusparseDenseToSparse_convert(handle, dn_descri, sp_descri, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, buffer.space);
    }
    else {
        cerr << "not implemented" << endl;
    }

    // record metadata to export //
    trio_bytelen._0 = csr->num_offsets() * sizeof(int);  // TODO use aliased typename
    trio_bytelen._1 = csr->sp_size.columns * sizeof(int);
    trio_bytelen._2 = csr->sp_size.values * sizeof(T);
    total_bytelen   = trio_bytelen._0 + trio_bytelen._1 + trio_bytelen._2;

    cout << "trio_bytelen._0\t" << trio_bytelen._0 << '\n';
    cout << "trio_bytelen._1\t" << trio_bytelen._1 << '\n';
    cout << "trio_bytelen._2\t" << trio_bytelen._2 << '\n';

    return *this;
}

template <typename T>
template <cuSPARSEver ver>
SparseOps<T>& SparseOps<T>::Scatter()
{
    if (csr->values == nullptr or csr->columns == nullptr or csr->offsets == nullptr)
        throw std::runtime_error("One or more of 3 CSR datasets are null.");
    if (csr->sp_size.nnz == 0) throw std::runtime_error("NNZ cannot be 0.");

    if (ver == cuSPARSEver::cuda11_onward) {
        cusparseCreate(&handle);
        cusparseCreateDnMat(
            &dn_descri, mat->num_rows, mat->num_cols, mat->ld(), mat->mat, cusparseBinding<T>::ValType,
            cusparseTypeAlias::MatOrder);

        cusparseCreateCsr(
            &sp_descri, csr->dn_size.num_rows, csr->dn_size.num_cols, csr->sp_size.nnz, csr->offsets, csr->columns,
            csr->values, cusparseTypeAlias::IdxType, cusparseTypeAlias::IdxType, cusparseTypeAlias::IdxBase,
            cusparseBinding<T>::ValType);

        // allocate buffer (if needed) //
        cusparseDenseToSparse_bufferSize(
            handle, dn_descri, sp_descri, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &buffer.size);
        cudaMalloc(&buffer.space, buffer.size);
        // execute Sparse to Dense conversion //
        cusparseSparseToDense(handle, sp_descri, dn_descri, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, buffer.space);
    }
    else {
        cerr << "not implemented" << endl;
    }
    return *this;
}
template <typename T>
SparseOps<T>::~SparseOps()
{
    cusparseDestroySpMat(sp_descri);
    cusparseDestroyDnMat(dn_descri);
    cusparseDestroy(handle);

    cudaFree(buffer.space);
}

template <typename T>
SparseOps<T>& SparseOps<T>::ExportCSR(uint8_t* start_addr)
{
    // clang-format off
    cudaMemcpy(start_addr,                   csr->offsets, trio_bytelen._0, cudaMemcpyDeviceToHost);
    cudaMemcpy(start_addr + trio_bytelen._0, csr->columns, trio_bytelen._1, cudaMemcpyDeviceToHost);
    cudaMemcpy(start_addr + trio_bytelen._1, csr->values,  trio_bytelen._2, cudaMemcpyDeviceToHost);
    // clang-format on
    return *this;
}
template <typename T>
SparseOps<T>& SparseOps<T>::ImportCSR(uint8_t* start_addr, Index<3>::idx_t trio)
{
    trio_bytelen = trio;

    // TODO can move in batch
    // clang-format off
    cudaMemcpy(csr->offsets, start_addr,                   trio_bytelen._0, cudaMemcpyHostToDevice);
    cudaMemcpy(csr->columns, start_addr + trio_bytelen._0, trio_bytelen._1, cudaMemcpyHostToDevice);
    cudaMemcpy(csr->values,  start_addr + trio_bytelen._1, trio_bytelen._2, cudaMemcpyHostToDevice);
    // clang-format on

    return *this;
}

#endif
