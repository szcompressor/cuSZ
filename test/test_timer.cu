#include "../src/utils/timer.hh"

#include <stdio.h>
#include <iostream>

using namespace std;

__global__ void df(double a, double b)
{
    //
    printf("test device, %u\n", threadIdx.x);
}

void hf(double a, double b)
{
    //
    printf("test host\n");
}

int main()
{
    auto t1 = TimeThisFunction(hf, 2.0, 3.0);
    cout << t1 << endl;

    auto t2 = TimeThisCUDAFunction(df, kernelcfg{dim3(1), dim3(64), 0, nullptr}, 2.0, 3.0);
    cout << t2 << endl;

    return 0;
}
