/**
 * @file ex_spline3_stack.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-11-23
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include "../src/sp_path.cuh"

#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>
#include <iomanip>
#include <sstream>

using std::cout;

using T = float;
using E = float;

bool print_fullhist = false;
bool write_quant    = false;

constexpr unsigned int dimz = 449, dimy = 449, dimx = 235;
constexpr unsigned int len = dimx * dimy * dimz;

const static auto DEVICE      = cusz::LOC::DEVICE;
const static auto HOST        = cusz::LOC::HOST;
const static auto HOST_DEVICE = cusz::LOC::HOST_DEVICE;

std::string fname;

void exp_stack_absmode_lorenzo(
    std::vector<std::string> filelist,  //
    double                   _eb,
    std::string              out_prefix,
    bool                     each_analysis = true)
{
    Capsule<T> data(len);
    Capsule<T> xdata(len);

    // stack img
    struct {
        Capsule<T> ori;
        Capsule<T> rec;
    } img;

    img.ori.set_len(len).template alloc<HOST>();
    img.rec.set_len(len).template alloc<HOST>();
    data.template alloc<HOST>();
    xdata.template alloc<HOST>();

    for (auto f : filelist) {
        cout << "(Lorenzo) processing " << f << "\n";

        data.template from_fs_to<HOST>(f);
        memcpy(xdata.template get<HOST>(), data.template get<HOST>(), sizeof(T) * len);

        auto rng    = 1.0;
        auto r2r_eb = _eb * rng;
        auto ebx2   = r2r_eb * 2;
        auto ebx2_r = 1 / ebx2;

        std::for_each(
            xdata.template get<HOST>(),                     //
            xdata.template get<HOST>() + len,               //
            [&](T& el) { el = round(el * ebx2_r) * ebx2; }  //
        );

        if (each_analysis) {
            stat_t stat;
            analysis::verify_data<T>(&stat, data.template get<HOST>(), xdata.template get<HOST>(), len);
            analysis::print_data_quality_metrics<T>(&stat, 0, false);
        }

        // stack original data
        thrust::plus<T> op;
        thrust::transform(
            thrust::host,                                                      //
            img.ori.template get<HOST>(), img.ori.template get<HOST>() + len,  // input 1
            data.template get<HOST>(),                                         // input 2
            img.ori.template get<HOST>(),                                      // output
            op);

        // stack reconstructed data
        thrust::transform(
            thrust::host,                                                      //
            img.rec.template get<HOST>(), img.rec.template get<HOST>() + len,  // input 1
            xdata.template get<HOST>(),                                        // input 2
            img.rec.template get<HOST>(),                                      // output
            op);
    }

    // img.ori.template to_fs_from<HOST>(out_prefix + "stack_ori.raw");
    img.rec.template to_fs_from<HOST>(out_prefix + "stack_rec_lorenzo.raw");

    thrust::minus<T> op_diff;
    // do diff of stack images
    thrust::transform(
        thrust::host,                                                      //
        img.rec.template get<HOST>(), img.rec.template get<HOST>() + len,  // input 1
        img.ori.template get<HOST>(),                                      // input 2
        img.rec.template get<HOST>(),                                      // output
        op_diff);
    img.rec.template to_fs_from<HOST>(out_prefix + "stack_diff_lorenzo.raw");
}

int main(int argc, char** argv)
{
    double eb = 3e-3 * 4e-02;

    cudaDeviceReset();

    std::vector<std::string> filelist;

    auto path = "/home/jtian/nvme/dev-env-cusz/aramco-data/";

    for (auto i = 600; i <= 2800; i += 20) {
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(4) << i;
        filelist.push_back(path + ss.str() + ".f32");
        // cout << "adding " << ss.str() + ".f32 to list\n";
    }
    SparsityAwarePath::DefaultCompressor compressor(dim3(dimx, dimy, dimz));

    // compressor.exp_stack_absmode(filelist, eb, path, false, true, false);

    exp_stack_absmode_lorenzo(filelist, eb, path, true);

    return 0;
}
