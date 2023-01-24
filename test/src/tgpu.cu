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
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "utils/timer.h"

__global__ void dummy()
{
    for (auto i = 0; i < 10; i++) double a = 1 + 2;
}

bool f()
{
    // asz_timer* t1 = asz_timer_create(CUDA, NULL);
    // asz_timer* t2 = asz_timer_create(CUDA, NULL);

    // {
    //     asz_timer_start(t1);
    //     for (auto i = 0; i < 20; i++) dummy<<<1, 1>>>();
    //     cudaDeviceSynchronize();
    //     asz_timer_end(t1);
    // }
    // {
    //     float* a = NULL;
    //     float* b = NULL;

    //     asz_timer_start(t2);
    //     cudaMalloc(&a, 100000);
    //     cudaMalloc(&b, 100000);
    //     cudaMemcpy(b, a, 100000, cudaMemcpyDeviceToDevice);
    //     cudaFree(a);
    //     cudaFree(b);
    //     asz_timer_end(t2);
    // }

    // double s1 = asz_time_elapsed(t1);
    // double s2 = asz_time_elapsed(t2);

    // asz_timer_destroy(t1);
    // asz_timer_destroy(t2);

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