#!/bin/bash

LOG_FILE=cuSZ-0.1.1-HACC-on-Summit.txt
AWK_CMD1="awk -F'(' '{print \$1}'"
AWK_CMD2="awk -F'<' '{print \$1}'"
AWK_CMD3="awk '{print \$6\"\t\"\$7\" \"\$8}'" 
SUPER_AWK="${AWK_CMD1} | ${AWK_CMD2} | ${AWK_CMD3}"
#echo "${SUPER_AWK}"

echo "zip, dual-quant kernel:"
eval "cat ${LOG_FILE} | grep c_lorenzo_ | ${SUPER_AWK}"
echo

echo "zip, Huffman codebook:"
eval "cat ${LOG_FILE} | grep p2013Histogram |  ${SUPER_AWK}"
# eval "cat ${LOG_FILE} | grep parHuff:: |  ${SUPER_AWK}"
eval "cat ${LOG_FILE} | grep GPU_ |  ${SUPER_AWK}"
eval "cat ${LOG_FILE} | grep thrust::cuda_cub:: |  ${SUPER_AWK}"
echo

echo "zip, Huffman encoding:"
eval "cat ${LOG_FILE} | grep EncodeFixedLen |  ${SUPER_AWK}"
eval "cat ${LOG_FILE} | grep Deflate |  ${SUPER_AWK}"
echo

echo "zip, gather outlier:"
eval "cat ${LOG_FILE} | grep cusparseIinclusive | ${SUPER_AWK}"
eval "cat ${LOG_FILE} | grep prune_dense_core | ${SUPER_AWK}"
echo

echo "unzip, Huffman decoding:"
eval "cat ${LOG_FILE} | grep Decode | ${SUPER_AWK}"
echo

echo "unzip, scatter outliers:"
eval "cat ${LOG_FILE} | grep cusparseZeroOutForCSR_kernel | ${SUPER_AWK}"
eval "cat ${LOG_FILE} | grep cusparseCsrToDense_kernel | ${SUPER_AWK}"
echo 

echo "unzip, reversed dual-quant kernel:"
eval "cat ${LOG_FILE} | grep x_lorenzo_ | ${SUPER_AWK}"
echo

