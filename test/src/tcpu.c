/**
 * @file tcpu.c
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-31
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "utils/timer.hh"

bool f(double sleep_time)
{
    char buf[10];
    sprintf(buf, "%f", sleep_time);

    char cmd[40];
    strcpy(cmd, "sleep ");
    strcat(cmd, buf);

    psz_cputimer* t = psz_cputimer_create();

    psz_cputimer_start(t);
    int status = system(cmd);
    psz_cputimer_end(t);
    double second = psz_cputime_elapsed(t);

    psz_cputimer_destroy(t);

    printf("sleep time: %f, recorded time: %f\n", (float)sleep_time, second);
    return second >= sleep_time;
}

int main(int argc, char** argv)
{
    bool all_pass = true;

    all_pass = all_pass && f(0.0001);
    all_pass = all_pass && f(0.001);
    all_pass = all_pass && f(0.01);
    all_pass = all_pass && f(0.1);

    if (all_pass)
        return 0;
    else
        return -1;
}