//
// Created by Jiannan Tian on 3/24/21.
//

#include <iostream>

using std::cout;
using std::endl;

#include "common/capsule.hh"
#include "ood/sparse_op.hh"

void TestSp2Dn()
{
    int   num_rows         = 5;
    int   num_cols         = 4;
    int   nnz              = 11;
    int   ld               = num_cols;
    int   dense_size       = ld * num_rows;
    int   h_csr_offsets[]  = {0, 3, 4, 7, 9, 11};
    int   h_csr_columns[]  = {0, 2, 3, 1, 0, 2, 3, 1, 3, 1, 2};
    float h_csr_values[]   = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
    float h_dense[]        = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                       0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float h_dense_result[] = {1.0f, 0.0f, 2.0f, 3.0f, 0.0f, 4.0f, 0.0f, 0.0f,  5.0f,  0.0f,
                              6.0f, 7.0f, 0.0f, 8.0f, 0.0f, 9.0f, 0.0f, 10.0f, 11.0f, 0.0f};

    DataPack<float> dn("dense");
    DataPack<float> sp_csr_vals("csr vals");
    DataPack<int>   sp_csr_cols("csr cols");
    DataPack<int>   sp_csr_offsets("csr offsets");

    struct CompressedSparseRow<float> csr(num_rows, num_cols, nnz);
    sp_csr_offsets.SetLen(csr.num_offsets()).SetHostSpace(h_csr_offsets).AllocDeviceSpace().Move<transfer::h2d>();
    csr.offsets = sp_csr_offsets.dptr();  // must be after csr.AllocDeviceSpace

    sp_csr_cols.SetLen(nnz).SetHostSpace(h_csr_columns).AllocDeviceSpace().Move<transfer::h2d>();
    sp_csr_vals.SetLen(nnz).SetHostSpace(h_csr_values).AllocDeviceSpace().Move<transfer::h2d>();
    dn.SetLen(dense_size).SetHostSpace(h_dense).AllocDeviceSpace();

    csr.offsets = sp_csr_offsets.dptr();
    csr.values  = sp_csr_vals.dptr();
    csr.columns = sp_csr_cols.dptr();

    struct DenseMatrix<float> mat(num_rows, num_cols);
    mat.mat = dn.dptr();

    SparseOps<float> op(&mat, &csr);

    op.Scatter<cuSPARSEver::cuda11_onward>();

    dn.Move<transfer::d2h>();

    int correct = 1;
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            if (h_dense[i * ld + j] != h_dense_result[i * ld + j]) {
                correct = 0;
                break;
            }
        }
    }
    if (correct)
        printf("sparse2dense test PASSED\n");
    else
        printf("sparse2dense test FAILED: wrong result\n");
}

void TestDn2Sp()
{
    // sample data
    int num_rows   = 5;
    int num_cols   = 4;
    int nnz        = 11;
    int ld         = num_cols;
    int dense_size = ld * num_rows;

    float h_dense[]              = {1.0f, 0.0f, 2.0f, 3.0f, 0.0f, 4.0f, 0.0f, 0.0f,  5.0f,  0.0f,
                       6.0f, 7.0f, 0.0f, 8.0f, 0.0f, 9.0f, 0.0f, 10.0f, 11.0f, 0.0f};
    int   h_csr_offsets[]        = {0, 0, 0, 0, 0, 0};
    int   h_csr_columns[]        = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    float h_csr_values[]         = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int   h_csr_offsets_result[] = {0, 3, 4, 7, 9, 11};
    int   h_csr_columns_result[] = {0, 2, 3, 1, 0, 2, 3, 1, 3, 1, 2};
    float h_csr_values_result[]  = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};

    DataPack<float> dn("dense");
    DataPack<float> sp_csr_vals("csr vals");
    DataPack<int>   sp_csr_cols("csr cols");
    DataPack<int>   sp_csr_offsets("csr offsets");

    dn.SetLen(dense_size).SetHostSpace(h_dense).AllocDeviceSpace().Move<transfer::h2d>();

    struct CompressedSparseRow<float> csr(num_rows, num_cols);
    sp_csr_offsets.SetLen(csr.num_offsets()).SetHostSpace(h_csr_offsets).AllocDeviceSpace();
    csr.offsets = sp_csr_offsets.dptr();  // must be after csr.AllocDeviceSpace

    struct DenseMatrix<float> mat(num_rows, num_cols);
    mat.mat = dn.dptr();

    SparseOps<float> op(&mat, &csr);

    op.Gather<cuSPARSEver::cuda11_onward>();

    sp_csr_offsets.Move<transfer::d2h>();
    sp_csr_vals.SetLen(csr.sp_size.nnz).SetDeviceSpace(csr.values).SetHostSpace(h_csr_values).Move<transfer::d2h>();
    sp_csr_cols.SetLen(csr.sp_size.nnz).SetDeviceSpace(csr.columns).SetHostSpace(h_csr_columns).Move<transfer::d2h>();

    if (nnz != csr.sp_size.nnz) throw std::runtime_error("NNZ not equal");

    int correct = 1;
    for (int i = 0; i < num_rows + 1; i++) {
        if (h_csr_offsets[i] != h_csr_offsets_result[i]) {
            correct = 0;
            break;
        }
    }
    for (int i = 0; i < nnz; i++) {
        if (h_csr_columns[i] != h_csr_columns_result[i]) {
            correct = 0;
            break;
        }
    }
    for (int i = 0; i < nnz; i++) {
        if (h_csr_values[i] != h_csr_values_result[i]) {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("dense2sparse test PASSED\n");
    else
        printf("dense2sparse test FAILED: wrong result\n");
}

int main()
{
    TestDn2Sp();
    TestSp2Dn();

    return 0;
}