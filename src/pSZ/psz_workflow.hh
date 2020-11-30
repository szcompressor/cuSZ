// 200211

#ifndef PSZ_WORKFLOW_HH
#define PSZ_WORKFLOW_HH

#include "analysis.hh"
// #include "constants.hh"
#include "io.hh"
#include "psz_14.hh"
#include "psz_14blocked.hh"
#include "psz_dualquant.hh"
#include "verify.hh"

const int LOCAL_B_1d = 32;
const int LOCAL_B_2d = 16;
const int LOCAL_B_3d = 8;

namespace PdQ  = pSZ::PredictionDualQuantization;
namespace PQRs = pSZ::PredictionQuantizationReconstructionSingleton;
namespace PQRb = pSZ::PredictionQuantizationReconstructionBlocked;

namespace AnalysisNamespace = analysis;

namespace pSZ {

namespace FineMassiveSimulation {

namespace __loop {
}

template <typename T, typename Q>
void cx_sim(
    std::string&        finame,  //
    size_t const* const dims_L16,
    double const* const ebs_L4,
    size_t&             num_outlier,
    bool                fine_massive = false,
    bool                blocked      = false,
    bool                show_histo   = false)
{
    size_t len = dims_L16[LEN];

    auto data     = io::ReadBinaryFile<T>(finame, len);
    auto data_cmp = io::ReadBinaryFile<T>(finame, len);

    T* pred_err = nullptr;
    T* comp_err = nullptr;
#ifdef PRED_COMP_ERR
    pred_err = new T[len]();
    comp_err = new T[len]();
#endif

    auto xdata   = new T[len]();
    auto outlier = new T[len]();
    auto code    = new Q[len]();

    if (fine_massive)
        cout << "\e[46musing (blocked) dualquant\e[0m" << endl;
    else {
        cout << (blocked == true ? "\e[46musing blocked sz14\e[0m" : "\e[46musing non-blocked sz14\e[0m") << endl;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // start of compression
    ////////////////////////////////////////////////////////////////////////////////
    // TODO omp version
    if (dims_L16[nDIM] == 1) {
        if (blocked) {
#pragma omp parallel for
            for (size_t b0 = 0; b0 < dims_L16[nBLK0]; b0++) {
                if (fine_massive)
                    PdQ::c_lorenzo_1d1l<T, Q, LOCAL_B_1d>(data, outlier, code, dims_L16, ebs_L4, pred_err, comp_err, b0);
                else
                    PQRb::c_lorenzo_1d1l<T, Q, LOCAL_B_1d>(data, outlier, code, dims_L16, ebs_L4, pred_err, comp_err, b0);
            }
        }
        else {
            PQRs::c_lorenzo_1d1l<T, Q, LOCAL_B_1d>(data, outlier, code, dims_L16, ebs_L4, pred_err, comp_err);
        }
    }
    else if (dims_L16[nDIM] == 2) {
        if (blocked) {
#pragma omp parallel for
            for (size_t b1 = 0; b1 < dims_L16[nBLK1]; b1++) {
                for (size_t b0 = 0; b0 < dims_L16[nBLK0]; b0++) {
                    if (fine_massive)
                        PdQ::c_lorenzo_2d1l<T, Q, LOCAL_B_2d>(data, outlier, code, dims_L16, ebs_L4, pred_err, comp_err, b0, b1);
                    else
                        PQRb::c_lorenzo_2d1l<T, Q, LOCAL_B_2d>(data, outlier, code, dims_L16, ebs_L4, pred_err, comp_err, b0, b1);
                }
            }
        }
        else {
            PQRs::c_lorenzo_2d1l<T, Q, LOCAL_B_2d>(data, outlier, code, dims_L16, ebs_L4, pred_err, comp_err);
        }
    }
    else if (dims_L16[nDIM] == 3) {
        if (blocked) {
#pragma omp parallel for
            for (size_t b2 = 0; b2 < dims_L16[nBLK2]; b2++) {
                for (size_t b1 = 0; b1 < dims_L16[nBLK1]; b1++) {
                    for (size_t b0 = 0; b0 < dims_L16[nBLK0]; b0++) {
                        if (fine_massive)
                            PdQ::c_lorenzo_3d1l<T, Q, LOCAL_B_3d>(data, outlier, code, dims_L16, ebs_L4, pred_err, comp_err, b0, b1, b2);
                        else
                            PQRb::c_lorenzo_3d1l<T, Q, LOCAL_B_3d>(data, outlier, code, dims_L16, ebs_L4, pred_err, comp_err, b0, b1, b2);
                    }
                }
            }
        }
        else {
            PQRs::c_lorenzo_3d1l<T, Q, LOCAL_B_3d>(data, outlier, code, dims_L16, ebs_L4, pred_err, comp_err);
        }
    }

    //    io::write_binary_file(code, len, new string("/Users/jtian/WorkSpace/cuSZ/src/CLDMED.bincode"));

    if (show_histo) {
        Analysis::histogram<int>(std::string("bincode/quant.code"), code, len, 8);
    }
    Analysis::getEntropy(code, len, 1024);
#ifdef PRED_COMP_ERR
    Analysis::histogram<T>(std::string("pred.error"), pred_err, len, 8);  // TODO when changing to 8, seg fault
    Analysis::histogram<T>(std::string("comp.error"), comp_err, len, 16);
#endif

    for_each(outlier, outlier + len, [&](T& n) { num_outlier += n == 0 ? 0 : 1; });

    ////////////////////////////////////////////////////////////////////////////////
    // start of decompression
    ////////////////////////////////////////////////////////////////////////////////
    if (dims_L16[nDIM] == 1) {
        if (blocked) {
#pragma omp parallel for
            for (size_t b0 = 0; b0 < dims_L16[nBLK0]; b0++) {
                if (fine_massive)
                    PdQ::x_lorenzo_1d1l<T, Q, LOCAL_B_1d>(xdata, outlier, code, dims_L16, ebs_L4[EBx2], b0);
                else
                    PQRb::x_lorenzo_1d1l<T, Q, LOCAL_B_1d>(xdata, outlier, code, dims_L16, ebs_L4, b0);  // TODO __2EB
            }
        }
        else {
            PQRs::x_lorenzo_1d1l<T, Q, LOCAL_B_1d>(xdata, outlier, code, dims_L16, ebs_L4);  // TODO __2EB
        }
    }
    if (dims_L16[nDIM] == 2) {
        if (blocked) {
#pragma omp parallel for
            for (size_t b1 = 0; b1 < dims_L16[nBLK1]; b1++) {
                for (size_t b0 = 0; b0 < dims_L16[nBLK0]; b0++) {
                    if (fine_massive)
                        PdQ::x_lorenzo_2d1l<T, Q, LOCAL_B_2d>(xdata, outlier, code, dims_L16, ebs_L4[EBx2], b0, b1);
                    else
                        PQRb::x_lorenzo_2d1l<T, Q, LOCAL_B_2d>(xdata, outlier, code, dims_L16, ebs_L4, b0, b1);  // TODO __2EB
                }
            }
        }
        else {
            PQRs::x_lorenzo_2d1l<T, Q, LOCAL_B_2d>(xdata, outlier, code, dims_L16, ebs_L4);  // TODO __2EB
        }
    }
    else if (dims_L16[nDIM] == 3) {
        if (blocked) {
#pragma omp parallel for
            for (size_t b2 = 0; b2 < dims_L16[nBLK2]; b2++) {
                for (size_t b1 = 0; b1 < dims_L16[nBLK1]; b1++) {
                    for (size_t b0 = 0; b0 < dims_L16[nBLK0]; b0++) {
                        if (fine_massive)
                            PdQ::x_lorenzo_3d1l<T, Q, LOCAL_B_3d>(xdata, outlier, code, dims_L16, ebs_L4[EBx2], b0, b1, b2);
                        else
                            PQRb::x_lorenzo_3d1l<T, Q, LOCAL_B_3d>(xdata, outlier, code, dims_L16, ebs_L4, b0, b1, b2);  // TODO __2EB
                    }
                }
            }
        }
        else {
            PQRs::x_lorenzo_3d1l<T, Q, LOCAL_B_3d>(xdata, outlier, code, dims_L16, ebs_L4);  // TODO __2EB
        }
    }

    if (show_histo) {
        Analysis::histogram(std::string("original datum"), data_cmp, len, 16);
        Analysis::histogram(std::string("reconstructed datum"), xdata, len, 16);
    }

    cout << "\e[46mnum.outlier:\t" << num_outlier << "\e[0m" << endl;
    cout << setprecision(5) << "error bound: " << ebs_L4[EB] << endl;

    if (fine_massive) {
        io::WriteBinaryFile(xdata, len, new string(finame + ".psz.cusz.out"));
    }
    else if (blocked == true and fine_massive == false) {
        io::WriteBinaryFile(xdata, len, new string(finame + ".psz.sz14blocked.out"));
    }
    else if (blocked == false and fine_massive == false) {
        io::WriteBinaryFile(xdata, len, new string(finame + ".psz.sz14.out"));
        io::WriteBinaryFile(pred_err, len, new string(finame + ".psz.sz14.prederr"));
        io::WriteBinaryFile(comp_err, len, new string(finame + ".psz.sz14.xerr"));
    }
    AnalysisNamespace::VerifyData(xdata, data_cmp, len, 1);
}

}  // namespace FineMassiveSimulation
}  // namespace pSZ

#endif
