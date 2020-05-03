#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description='listing description')

parser.add_argument('--switch', help='')
parser.add_argument('-i',
                    '--input-file',
                    help='specify the input file',
                    required=True)
parser.add_argument('-t',
                    '--data-type',
                    help='specify the data type',
                    required=True)
parser.add_argument('-c',
                    '--compress',
                    help='to compress',
                    action='store_true')
parser.add_argument('-x',
                    '--deompress',
                    help='to decompress',
                    action='store_true')
parser.add_argument('-d',
                    '--dims',
                    nargs='+',
                    type=int,
                    help='specify dimensions, e.g., \"1800 3600\"' +
                    ' for \"...-1800x3600.f32\" dataset')
parser.add_argument('-s',
                    '-subdims',
                    help='specify block dimensions (subdims)' +
                    ' in accordance to the dimension')
parser.add_argument('-a', '--abs', help='absolute mode')
parser.add_argument('-r', '--rel', help='relative mode')
parser.add_argument('-p', '--pointwise', help='pointwise mode')
parser.add_argument('-C',
                    '--cap',
                    help='quantization bin capacity',
                    type=int,
                    default=32768)
parser.add_argument('-e',
                    '--eb',
                    help='error bound in base-10 exponent, e.g., \"-3\"',
                    type=int,
                    default=-3)
parser.add_argument('--eb-bin',
                    help='error bound in base-2 exponent, e.g., \"-10\"')
parser.add_argument(
    '--eb-abs',
    help='error bound in base-10 absulute number, e.g., \"15\"',
)
parser.add_argument('--native-seq', action='store_true', help='native CPU run')
parser.add_argument('-u',
                    '--native-cuda',
                    help='offloading to GPU, running with native CUDA')
parser.add_argument('--kokkos-seq', help='(Kokkos) CPU sequantial run')
parser.add_argument('--kokkos-omp', help='(Kokkos) omp parallelization')
parser.add_argument('--kokkos-cuda', help='(Kokkos) offloading to GPU')
parser.add_argument('--raja-seq', help='(RAJA) CPU sequantial run')
parser.add_argument('--raja-simd', help='(RAJA) compiler hinted SIMD')
parser.add_argument('--raja-omp', help='(RAJA) omp parallelization')
parser.add_argument('--raja-omp-gpu', help='(RAJA) omp-GPU parallelization')
parser.add_argument('--raja-cuda', help='(RAJA) offloading to GPU')
parser.add_argument('-P', '--print-metadata', help='print metadata')

args = parser.parse_args()

for _, value in parser.parse_args()._get_kwargs():
    if value is not None:
        print(value)
