/**
 * @file cusz-cli.cu
 * @author Jiannan Tian
 * @brief Driver program of cuSZ.
 * @version 0.1
 * @date 2020-09-20
 * (created) 2019-12-30 (rev) 2022-02-20
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include "cli/cli.cuh"

int main(int argc, char** argv)
{
    auto ctx = new cuszCTX(argc, argv);

    if (ctx->verbose) {
        Diagnostics::GetMachineProperties();
        GpuDiagnostics::GetDeviceProperty();
    }

    cusz::CLI<float> cusz_cli;
    cusz_cli.dispatch(ctx);
}
