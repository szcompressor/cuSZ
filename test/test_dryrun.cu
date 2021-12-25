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

#include "../src/base_cusz.cuh"
#include "../src/wrapper/extrap_lorenzo.cuh"

using Compressor = BaseCompressor<cusz::PredictorLorenzo<float, uint16_t, float>>;

int main(int argc, char** argv)
{
    if (argc < 6) {
        cout << "./program <filename> <x> <y> <z> <r2r|abs>";
        exit(0);
    }

    auto fname = std::string(argv[1]);
    auto x     = atoi(argv[2]);
    auto y     = atoi(argv[3]);
    auto z     = atoi(argv[4]);
    auto r2r   = std::string(argv[5]) == "r2r";

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    Compressor   dryrun;
    host_timer_t timer;

    cout << "\ngeneric dryrun" << '\n';
    dryrun.init_generic_dryrun(dim3(x, y, z));
    timer.timer_start();
    {
        dryrun.generic_dryrun(fname, 1e-4, 512, r2r, stream);
    }
    timer.timer_end();
    printf("generic_dryrun: %lf seconds.\n", timer.get_time_elapsed());
    dryrun.destroy_generic_dryrun();

    cout << "\ndualquant dryrun" << '\n';
    dryrun.init_dualquant_dryrun(dim3(x, y, z));

    timer.timer_start();
    {
        dryrun.dualquant_dryrun(fname, 1e-4, r2r, stream);
    }
    timer.timer_end();
    printf("dualquant_dryrun: %lf seconds.\n", timer.get_time_elapsed());
    dryrun.destroy_dualquant_dryrun();

    cudaStreamDestroy(stream);

    return 0;
}