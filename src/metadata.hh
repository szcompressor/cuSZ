#ifndef CUSZ_METADATA_HH
#define CUSZ_METADATA_HH

/**
 * @file metadata.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.1.1
 * @date 2020-09-22
 *
 * @copyright (C) 2020 by Washington State University, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

template <int ndim>
struct MetadataTrait;

// clang-format off
template <> struct MetadataTrait<1>     { static const int Block = 256; };
template <> struct MetadataTrait<0x101> { static const int Block = 128; };
template <> struct MetadataTrait<0x201> { static const int Block = 64;  };
template <> struct MetadataTrait<2>     { static const int Block = 16;  };
template <> struct MetadataTrait<3>     { static const int Block = 8;   };
// clang-format on

#endif