#!/bin/bash
echo bsub -P csc338 -nnodes 1 -W 60 -alloc_flags gpumps -Is $SHELL
bsub -P csc338 -nnodes 1 -W 60 -alloc_flags gpumps -Is $SHELL
