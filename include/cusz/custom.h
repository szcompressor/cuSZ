/**
 * @file compress.h
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-04-30
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "type.h"

cusz_custom_predictor     cusz_default_predictor();
cusz_custom_codec         cusz_default_codec();
cusz_custom_huffman_codec cusz_default_huffman_codec();
cusz_custom_spcodec       cusz_default_spcodec();
cusz_custom_framework*    cusz_default_framework();

#ifdef __cplusplus
}
#endif
