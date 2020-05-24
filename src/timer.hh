//
// Created by JianNan Tian on 2019-08-26.
//

#ifndef TIMER_HH
#define TIMER_HH

#include <chrono>

using std::cerr;
using std::cout;
using std::endl;

using hires = std::chrono::high_resolution_clock;
typedef std::chrono::duration<double>                               duration_t;
typedef std::chrono::time_point<std::chrono::high_resolution_clock> hires_clock_t;

#endif  // TIMER_HH
