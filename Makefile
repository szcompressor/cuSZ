#STRICT_CHECK=-Xcompiler -Wall
#PTX_VERBOSE=-Xptxas -O3,-v


#CXX       := clang++ -fPIE
CXX       := g++
NVCC      := nvcc
STD       := -std=c++11
CCFLAGS   := $(STD) -O3 -g
NVCCFLAGS := $(STD) -O3 -g --expt-relaxed-constexpr
HOST_DBG  := -O0 -g
CUDA_DBG  := -O0 -G -g
SRC_DIR   := src
OBJ_DIR   := src
BIN_DIR   := bin

GPU_VOLTA := -gencode=arch=compute_70,code=sm_70
GPU_TURING:= -gencode=arch=compute_75,code=sm_75
DEPLOY    := $(GPU_VOLTA) #$(GPU_TURING)

CCFLAGS   := $(STD) -O3
NVCCFLAGS := $(STD) $(DEPLOY)

CCFILES   := $(wildcard $(SRC_DIR)/*.cc)

MAIN      := $(SRC_DIR)/cusz.cu
CUFILES2  := $(SRC_DIR)/cusz_workflow.cu $(SRC_DIR)/cusz_dualquant.cu
CUFILES3  := $(SRC_DIR)/canonical.cu $(SRC_DIR)/par_merge.cu $(SRC_DIR)/par_huffman.cu
CUFILES1  := $(filter-out $(MAIN) $(CUFILES3) $(CUFILES2), $(wildcard $(SRC_DIR)/*.cu))

CCOBJS    := $(CCFILES:$(SRC_DIR)/%.cc=$(OBJ_DIR)/%.o)
CUOBJS1   := $(CUFILES1:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
CUOBJS2   := $(CUFILES2:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
CUOBJS3   := $(CUFILES3:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

CUOBJS    := $(CUOBJS1) $(CUOBJS2) $(CUOBJS3)
OBJS      := $(CCOBJS) $(CUOBJS)

$(CUOBJS1): NVCCFLAGS +=
$(CUOBJS2): NVCCFLAGS += -rdc=true
$(CUOBJS3): NVCCFLAGS += $(DEPLOY) -rdc=true

all: ; @$(MAKE) cusz -j

################################################################################

HUFF_DIR   := $(SRC_DIR)/huffre

_DEPS_ARG  := $(SRC_DIR)/argparse.o
_DEPS_MEM  := $(SRC_DIR)/cuda_mem.o
_DEPS_HIST := $(SRC_DIR)/histogram.o $(SRC_DIR)/huffman_workflow.o $(SRC_DIR)/format.o $(SRC_DIR)/canonical.o $(SRC_DIR)/huffman.o -rdc=true
_DEPS_OLDENC := $(SRC_DIR)/huffman_codec.o
DEPS_HUFF := $(_DEPS_MEM) $(_DEPS_HIST) $(_DEPS_OLDENC) $(_DEPS_ARG)

huff: $(HUFF_DIR)/huff.cu $(SRC_DIR)/argparse.cc
	$(NVCC) $(NVCCFLAGS) $(DEPS_HUFF) $(HUFF_DIR)/huff.cu -o huff
# nvcc huff.cu ../argparse.o ../constants.o ../cuda_mem.o ../huffman_workflow.o ../types.o ../format.o ../histogram.o ../huffman.o ../canonical.o ../huffman_codec.o -gencode=arch=compute_75,code=sm_75

huffredemo: $(HUFF_DIR)/reduce_move_merge.cuh $(HUFF_DIR)/huffre-demo.cu
	$(NVCC) $(NVCCFLAGS) $(DEPS_HUFF) $(HUFF_DIR)/huffre-demo.cu -o huffre-demo

huffretime: $(HUFF_DIR)/reduce_move_merge.cuh $(HUFF_DIR)/huffre-demo.cu
	$(NVCC) $(NVCCFLAGS) $(DEPS_HUFF) $(HUFF_DIR)/huffre-demo.cu -o huffre-reduce1  -DREDUCE1TIME
	$(NVCC) $(NVCCFLAGS) $(DEPS_HUFF) $(HUFF_DIR)/huffre-demo.cu -o huffre-reduce12 -DREDUCE12TIME
	$(NVCC) $(NVCCFLAGS) $(DEPS_HUFF) $(HUFF_DIR)/huffre-demo.cu -o huffre-allmerge -DALLMERGETIME 

huffredbg: $(HUFF_DIR)/reduce_move_merge.cuh $(HUFF_DIR)/huffre-demo.cu
	$(NVCC) $(NVCCFLAGS) $(CUDA_DBG) $(DEPS_HUFF) $(HUFF_DIR)/huffre-demo.cu -o huffre-dbg -DDBG0,DBG1,DBG2

install: bin/cusz
	cp bin/cusz /usr/local/bin

cusz: $(OBJS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -lcusparse $(DEPLOY) $(MAIN) -rdc=true $^ -o $(BIN_DIR)/$@
$(BIN_DIR):
	mkdir $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc | $(OBJ_DIR)
	$(CXX) $(CCFLAGS) -c $< -o $@

$(CUOBJS): $(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJS)
