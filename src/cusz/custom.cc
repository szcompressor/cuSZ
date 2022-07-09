/**
 * @file custom.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-04-30
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include "cusz/custom.h"

extern "C" {

cusz_custom_predictor     cusz_default_predictor() { return {LorenzoI, false, false}; }
cusz_custom_quantization  cusz_default_quantization() { return {512, false}; }
cusz_custom_codec         cusz_default_codec() { return {Huffman, true, 0.5}; }
cusz_custom_huffman_codec cusz_default_huffman_codec() { return {Canonical, Device, Coarse, 1024, 768}; }
cusz_custom_spcodec       cusz_default_spcodec() { return {SparseMat, 0.2}; }
cusz_custom_framework*    cusz_default_framework()
{
    return new cusz_custom_framework{
        FP32,  // placeholder; set in another function call
        Auto, cusz_default_predictor(), cusz_default_quantization(), cusz_default_codec(),
        // cusz_default_spcodec(),
        cusz_default_huffman_codec()};
}

void cusz_set_datatype(cusz_custom_framework* config, cusz_datatype datatype) { config->datatype = datatype; }
void cusz_set_pipelinetype(cusz_custom_framework* config, cusz_pipelinetype pipeline) { config->pipeline = pipeline; }

// end of extern C
}
