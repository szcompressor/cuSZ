/**
 * @file tgpu.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-31
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include <cuda_runtime.h>
#include "busyheader.hh"
#include <unistd.h>
#include "utils/timer.hh"

__global__ void dummy()
{
    for (auto i = 0; i < 10; i++) double a = 1 + 2;
}

bool f()
{
    // psztime* t1 = psz_timer_create(CUDA, NULL);
    // psztime* t2 = psz_timer_create(CUDA, NULL);

    // {
    //     psz_timer_start(t1);
    //     for (auto i = 0; i < 20; i++) dummy<<<1, 1>>>();
    //     cudaDeviceSynchronize();
    //     psz_timer_end(t1);
    // }
    // {
    //     float* a = NULL;
    //     float* b = NULL;

    //     psz_timer_start(t2);
    //     cudaMalloc(&a, 100000);
    //     cudaMalloc(&b, 100000);
    //     cudaMemcpy(b, a, 100000, cudaMemcpyDeviceToDevice);
    //     cudaFree(a);
    //     cudaFree(b);
    //     psz_timer_end(t2);
    // }

    // double s1 = psz_time_elapsed(t1);
    // double s2 = psz_time_elapsed(t2);

    // psz_timer_destroy(t1);
    // psz_timer_destroy(t2);

    // printf("s1: %lf, s2: %lf\n", s1, s2);

    // return s2 > 0 && s1 > 0;
    return true;
}

int main(int argc, char** argv)
{
    bool all_pass = true;

    all_pass = all_pass && f();
    all_pass = all_pass && f();
    all_pass = all_pass && f();
    all_pass = all_pass && f();

    if (all_pass)
        return 0;
    else
        return -1;
}