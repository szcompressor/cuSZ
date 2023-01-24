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
#include "utils/timer.h"

bool f(double sleep_time)
{
    char buf[10];
    sprintf(buf, "%f", sleep_time);

    char cmd[40];
    strcpy(cmd, "sleep ");
    strcat(cmd, buf);

    asz_cputimer* t = asz_cputimer_create();

    asz_cputimer_start(t);
    int status = system(cmd);
    asz_cputimer_end(t);
    double second = asz_cputime_elapsed(t);

    asz_cputimer_destroy(t);

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