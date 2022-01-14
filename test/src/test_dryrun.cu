/**
 * @file test_dryrun.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-12-23
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include "base_compressor.cuh"

#include "wrapper/extrap_lorenzo.cuh"
using DefaultCompressor = BaseCompressor<cusz::PredictorLorenzo<float, uint16_t, float>>;

#if __has_include("wrapper/interp_spline3.cuh")
#include("wrapper/interp_spline3.cuh")
using SPCompressor = BaseCompressor<cusz::Spline3<float, float, float>>;
#endif

template <class DryrunCompressor>
void dryrun(string fname, int x, int y, int z, bool r2r)
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    DryrunCompressor dryrun;
    cuda_timer_t     timer;

    cout << "\ngeneric dryrun" << '\n';
    dryrun.init_generic_dryrun(dim3(x, y, z));
    timer.timer_start(stream);
    {
        dryrun.generic_dryrun(fname, 1e-4, 512, r2r, stream);
    }
    timer.timer_end(stream);
    printf("generic_dryrun: %lf seconds.\n", timer.get_time_elapsed());
    dryrun.destroy_generic_dryrun();

    cout << "\ndualquant dryrun" << '\n';
    dryrun.init_dualquant_dryrun(dim3(x, y, z));

    timer.timer_start(stream);
    {
        dryrun.dualquant_dryrun(fname, 1e-4, r2r, stream);
    }
    timer.timer_end(stream);
    printf("dualquant_dryrun: %lf seconds.\n", timer.get_time_elapsed());
    dryrun.destroy_dualquant_dryrun();

    cudaStreamDestroy(stream);
}

int main(int argc, char** argv)
{
    if (argc < 7) {
        cout << "./program <filename> <x> <y> <z> <r2r|abs> <predictor id>";
        exit(0);
    }

    auto fname = std::string(argv[1]);
    auto x     = atoi(argv[2]);
    auto y     = atoi(argv[3]);
    auto z     = atoi(argv[4]);
    auto r2r   = std::string(argv[5]) == "r2r";
    auto pid   = atoi(argv[6]);

    if (pid == 0) dryrun<DefaultCompressor>(fname, x, y, z, r2r);
#if __has_include("wrapper/interp_spline3.cuh")
    if (pid == 1) dryrun<SPCompressor>(fname, x, y, z, r2r);
#endif

    return 0;
}