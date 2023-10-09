/**
 * @file debug.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-06-10
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef AAD97A4D_258B_450D_B6D1_9AD5D180424D
#define AAD97A4D_258B_450D_B6D1_9AD5D180424D

#include <cstdio>

#include "header.h"

enum class DebugType { Header, HfHeader };

struct psz_debug {
  static void __header(const psz_header* h)
  {
    printf("header::{x, y, z, w}\t{%u, %u, %u, %u}\n", h->x, h->y, h->z, h->w);
    printf("header::{radius, eb}\t{%u, %lf}\n", h->radius, h->eb);
    printf("\n");
  }
};

#endif /* AAD97A4D_258B_450D_B6D1_9AD5D180424D */
