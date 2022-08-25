/**
 * @file rand.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-08-25
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include <iostream>
#include <random>

int randint(size_t upper_limit)
{
    std::random_device dev;
    std::mt19937       gen(dev());

    std::uniform_int_distribution<> distribute(0, upper_limit);

    return distribute(gen);
}