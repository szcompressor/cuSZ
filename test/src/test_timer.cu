#include "../src/utils/timer.hh"

#include <stdio.h>
#include <iostream>

using namespace std;

class A {
   public:
    A() = default;
    static void f() {}
    static void g() {}
};

__global__ void df(double a, double b)
{
    //
    printf("test device; thread id: %u\n", threadIdx.x);
}

void hf(double a, double b)
{
    //
    printf("test host\n");
}

void hf2()
{
    //
    printf("test host\n");
}

int main()
{
    auto t1 = TimeThisRoutine(hf, 2.0, 3.0);
    cout << t1 << endl;

    t1 = TimeThisRoutine(hf2);
    cout << t1 << endl;

    auto t2 = TimeThisCUDARoutine(df, kernelcfg{dim3(1), dim3(64), 0, nullptr}, 2.0, 3.0);
    cout << t2 << endl;

    A a;

    host_timer_t timer;
    timer.timer_start();
    a.f();
    timer.timer_end();
    cout << timer.get_time_elapsed() << endl;

    timer.timer_start();
    A::g();
    timer.timer_end();
    cout << timer.get_time_elapsed() << endl;

    return 0;
}
