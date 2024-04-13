#ifndef PSZ_WORKFLOW_HH
#define PSZ_WORKFLOW_HH

/**
 * @file psz_workflow.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.1.0
 * @date 2020-02-11
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include "../analysis.hh"
#include "../utils/io.hh"
#include "../utils/verify.hh"
#include "psz_14.hh"
#include "psz_14blocked.hh"
#include "psz_dualquant.hh"

const int LOCAL_B_1d = 32;
const int LOCAL_B_2d = 16;
const int LOCAL_B_3d = 8;

namespace PdQ  = psz::dualquant;
namespace PQRs = psz::sz1_4;
namespace PQRb = psz::sz1_4_blocked;

namespace psz {

namespace FineMassiveSimulation {

template <typename Data, typename Quant>
void cx_sim(
    std::string&        finame,  //
    size_t const* const dims,
    double const* const eb_variants,
    size_t&             num_outlier,
    bool                fine_massive = false,
    bool                blocked      = false,
    bool                show_histo   = false)
{
    size_t len = dims[LEN];

    auto data     = io::read_binary_to_new_array<Data>(finame, len);
    auto data_cmp = io::read_binary_to_new_array<Data>(finame, len);

    Data* pred_err = nullptr;
    Data* comp_err = nullptr;
#ifdef PRED_COMP_ERR
    pred_err = new T[len]();
    comp_err = new T[len]();
#endif

    auto xdata   = new Data[len]();
    auto outlier = new Data[len]();
    auto code    = new Quant[len]();

    if (fine_massive)
        cout << "\e[46musing (blocked) dualquant\e[0m" << endl;
    else {
        cout << (blocked ? "\e[46musing blocked sz14\e[0m" : "\e[46musing non-blocked sz14\e[0m") << endl;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // start of compression
    ////////////////////////////////////////////////////////////////////////////////
    // TODO omp version
    if (dims[nDIM] == 1) {
        if (blocked) {
#pragma omp parallel for
            for (size_t b0 = 0; b0 < dims[nBLK0]; b0++) {
                if (fine_massive)
                    PdQ::c_lorenzo_1d1l<Data, Quant, LOCAL_B_1d>(
                        data, outlier, code, dims, eb_variants, pred_err, comp_err, b0);
                else
                    PQRb::c_lorenzo_1d1l<Data, Quant, LOCAL_B_1d>(
                        data, outlier, code, dims, eb_variants, pred_err, comp_err, b0);
            }
        }
        else {
            PQRs::c_lorenzo_1d1l<Data, Quant, LOCAL_B_1d>(data, outlier, code, dims, eb_variants, pred_err, comp_err);
        }
    }
    else if (dims[nDIM] == 2) {
        if (blocked) {
#pragma omp parallel for
            for (size_t b1 = 0; b1 < dims[nBLK1]; b1++) {
                for (size_t b0 = 0; b0 < dims[nBLK0]; b0++) {
                    if (fine_massive)
                        PdQ::c_lorenzo_2d1l<Data, Quant, LOCAL_B_2d>(
                            data, outlier, code, dims, eb_variants, pred_err, comp_err, b0, b1);
                    else
                        PQRb::c_lorenzo_2d1l<Data, Quant, LOCAL_B_2d>(
                            data, outlier, code, dims, eb_variants, pred_err, comp_err, b0, b1);
                }
            }
        }
        else {
            PQRs::c_lorenzo_2d1l<Data, Quant, LOCAL_B_2d>(data, outlier, code, dims, eb_variants, pred_err, comp_err);
        }
    }
    else if (dims[nDIM] == 3) {
        if (blocked) {
#pragma omp parallel for
            for (size_t b2 = 0; b2 < dims[nBLK2]; b2++) {
                for (size_t b1 = 0; b1 < dims[nBLK1]; b1++) {
                    for (size_t b0 = 0; b0 < dims[nBLK0]; b0++) {
                        if (fine_massive)
                            PdQ::c_lorenzo_3d1l<Data, Quant, LOCAL_B_3d>(
                                data, outlier, code, dims, eb_variants, pred_err, comp_err, b0, b1, b2);
                        else
                            PQRb::c_lorenzo_3d1l<Data, Quant, LOCAL_B_3d>(
                                data, outlier, code, dims, eb_variants, pred_err, comp_err, b0, b1, b2);
                    }
                }
            }
        }
        else {
            PQRs::c_lorenzo_3d1l<Data, Quant, LOCAL_B_3d>(data, outlier, code, dims, eb_variants, pred_err, comp_err);
        }
    }

    //    io::write_binary_file(code, len, new string("/Users/jtian/WorkSpace/cuSZ/src/CLDMED.bincode"));

    if (show_histo) { Analysis::histogram<int>(std::string("bincode/quant.code"), code, len, 8); }
    Analysis::GetEntropy(code, len, 1024);
#ifdef PRED_COMP_ERR
    Analysis::histogram<T>(std::string("pred.error"), pred_err, len, 8);  // TODO when changing to 8, seg fault
    Analysis::histogram<T>(std::string("comp.error"), comp_err, len, 16);
#endif

    for_each(outlier, outlier + len, [&](Data& n) { num_outlier += n == 0 ? 0 : 1; });

    ////////////////////////////////////////////////////////////////////////////////
    // start of decompression
    ////////////////////////////////////////////////////////////////////////////////
    if (dims[nDIM] == 1) {
        if (blocked) {
#pragma omp parallel for
            for (size_t b0 = 0; b0 < dims[nBLK0]; b0++) {
                if (fine_massive)
                    PdQ::x_lorenzo_1d1l<Data, Quant, LOCAL_B_1d>(xdata, outlier, code, dims, eb_variants[EBx2], b0);
                else
                    PQRb::x_lorenzo_1d1l<Data, Quant, LOCAL_B_1d>(
                        xdata, outlier, code, dims, eb_variants, b0);  // TODO __2EB
            }
        }
        else {
            PQRs::x_lorenzo_1d1l<Data, Quant, LOCAL_B_1d>(xdata, outlier, code, dims, eb_variants);  // TODO __2EB
        }
    }
    if (dims[nDIM] == 2) {
        if (blocked) {
#pragma omp parallel for
            for (size_t b1 = 0; b1 < dims[nBLK1]; b1++) {
                for (size_t b0 = 0; b0 < dims[nBLK0]; b0++) {
                    if (fine_massive)
                        PdQ::x_lorenzo_2d1l<Data, Quant, LOCAL_B_2d>(
                            xdata, outlier, code, dims, eb_variants[EBx2], b0, b1);
                    else
                        PQRb::x_lorenzo_2d1l<Data, Quant, LOCAL_B_2d>(
                            xdata, outlier, code, dims, eb_variants, b0, b1);  // TODO __2EB
                }
            }
        }
        else {
            PQRs::x_lorenzo_2d1l<Data, Quant, LOCAL_B_2d>(xdata, outlier, code, dims, eb_variants);  // TODO __2EB
        }
    }
    else if (dims[nDIM] == 3) {
        if (blocked) {
#pragma omp parallel for
            for (size_t b2 = 0; b2 < dims[nBLK2]; b2++) {
                for (size_t b1 = 0; b1 < dims[nBLK1]; b1++) {
                    for (size_t b0 = 0; b0 < dims[nBLK0]; b0++) {
                        if (fine_massive)
                            PdQ::x_lorenzo_3d1l<Data, Quant, LOCAL_B_3d>(
                                xdata, outlier, code, dims, eb_variants[EBx2], b0, b1, b2);
                        else
                            PQRb::x_lorenzo_3d1l<Data, Quant, LOCAL_B_3d>(
                                xdata, outlier, code, dims, eb_variants, b0, b1, b2);  // TODO __2EB
                    }
                }
            }
        }
        else {
            PQRs::x_lorenzo_3d1l<Data, Quant, LOCAL_B_3d>(xdata, outlier, code, dims, eb_variants);  // TODO __2EB
        }
    }

    if (show_histo) {
        Analysis::histogram(std::string("original datum"), data_cmp, len, 16);
        Analysis::histogram(std::string("reconstructed datum"), xdata, len, 16);
    }

    cout << "\e[46mnum.outlier:\t" << num_outlier << "\e[0m" << endl;
    cout << setprecision(5) << "error bound: " << eb_variants[EB] << endl;

    if (fine_massive) { io::write_array_to_binary(finame + ".psz.cusz.out", xdata, len); }
    else if (blocked and (not fine_massive)) {
        io::write_array_to_binary(finame + ".psz.sz14blocked.out", xdata, len);
    }
    else if (!blocked and (not fine_massive)) {
        io::write_array_to_binary(finame + ".psz.sz14.out", xdata, len);
        io::write_array_to_binary(finame + ".psz.sz14.prederr", pred_err, len);
        io::write_array_to_binary(finame + ".psz.sz14.xerr", comp_err, len);
    }

    Stat stat;
    cusz::verify_data(&stat, xdata, data_cmp, len);
    psz::print_metrics_cross<Data>(&stat);
}

}  // namespace FineMassiveSimulation
}  // namespace psz

#endif
