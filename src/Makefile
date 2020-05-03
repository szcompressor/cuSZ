
all: cusz 

CPP_STD = -std=c++11


# Issue 1: UA PantaRrhei does not support compute_75
DEPLOY=-gencode=arch=compute_61,code=sm_61 \
       -gencode=arch=compute_70,code=sm_70
#       -gencode=arch=compute_75,code=sm_75 \
#       -gencode=arch=compute_75,code=compute_75
STRICT_CHECK=-Xcompiler -Wall
PTX_VERBOSE=-Xptxas -O3,-v

SRC=cusz_exe.cu cusz_workflow.cuh cusz_dualquant.cuh huffman_workflow.cuh types.hh SDRB.hh argparse.hh verify.hh

cusz: $(SRC)
	nvcc $(CPP_STD) $(DEPLOY) -lcusparse -rdc=true -O3 -o cusz cusz_exe.cu

huff: gfx_huffman.cu huffman_codec.cuh
	nvcc $(CPP_STD) $(DEPLOY) gfx_huffman.cu -O3 -o huff

.PHONY: clean

clean:
	rm -f cusz huff
	rm -f *.b8 *.b8.*
	rm -f *.b16 *.b16.*
	rm -f *.b32 *.b32.*

