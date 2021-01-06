#STRICT_CHECK=-Xcompiler -Wall
#PTX_VERBOSE=-Xptxas -O3,-v


#CXX       := clang++ -fPIE
PWD       := $(shell pwd)
CXX       := g++
NVCC      := nvcc
STD       := -std=c++14
HOST_DBG  := -O0 -g
CUDA_DBG  := -O0 -G -g
SRC_DIR   := src
OBJ_DIR   := src
BIN_DIR   := bin

NVCOMP_DIR        := external/nvcomp
NVCOMP_INCLUDE_DIR:= $(NVCOMP_DIR)/build/include
NVCOMP_LIB_DIR    := $(NVCOMP_DIR)/build/lib
NVCOMP_STATIC_LIB := $(NVCOMP_LIB_DIR)/libnvcomp.a

GTEST_DIR         := external/googletest
GTEST_INCLUDE_DIR := $(GTEST_DIR)/googletest/include
GTEST_LIB_DIR     := $(GTEST_DIR)/build/lib
GTEST_STATIC_LIB  := $(GTEST_LIB_DIR)/libgtest.a

GPU_PASCAL:= -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61
GPU_VOLTA := -gencode=arch=compute_70,code=sm_70
GPU_TURING:= -gencode=arch=compute_75,code=sm_75
GPU_AMPERE:= -gencode=arch=compute_80,code=sm_80
DEPLOY    := $(GPU_PASCAL) $(GPU_VOLTA)

CUDA_MAJV := $(shell nvcc --version | grep "release" | \
               awk '{print $$6}' | cut -c2- | cut -d. -f1)

ifeq ($(shell test $(CUDA_MAJV) -ge 10; echo $$?), 0)
  DEPLOY += $(GPU_TURING)
endif

ifeq ($(shell test $(CUDA_MAJV) -ge 11; echo $$?), 0)
  DEPLOY += $(GPU_AMPERE)
endif

CCFLAGS   := $(STD) -O3
NVCCFLAGS := $(STD) $(DEPLOY) --expt-relaxed-constexpr --expt-extended-lambda

CC_FILE_OMP:=$(SRC_DIR)/analysis_utils.cc
CC_FILE   := $(filter-out $(CC_FILE_OMP), $(wildcard $(SRC_DIR)/*.cc))

MAIN      := $(SRC_DIR)/cusz.cu
CU_FILE_2 := $(SRC_DIR)/cusz_interface.cu $(SRC_DIR)/dualquant.cu
CU_FILE_3 := $(SRC_DIR)/par_merge.cu $(SRC_DIR)/par_huffman.cu
CU_FILE_4 := $(SRC_DIR)/huff_interface.cu
CU_FILE_1 := $(filter-out $(MAIN) $(CU_FILE_3) $(CU_FILE_4) $(CU_FILE_2), $(wildcard $(SRC_DIR)/*.cu))

CC_OBJ_OMP:= $(CC_FILE_OMP:$(SRC_DIR)/%.cc=$(OBJ_DIR)/%.o)
CC_OBJ    := $(CC_FILE:$(SRC_DIR)/%.cc=$(OBJ_DIR)/%.o)

CU_OBJ_1  := $(CU_FILE_1:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
CU_OBJ_2  := $(CU_FILE_2:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
CU_OBJ_3  := $(CU_FILE_3:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
CU_OBJ_4  := $(CU_FILE_4:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

CU_OBJ    := $(CU_OBJ_1) $(CU_OBJ_2) $(CU_OBJ_3) $(CU_OBJ_4)
OBJ_ALL   := $(CC_OBJ) $(CC_OBJ_OMP) $(CU_OBJ)

$(CC_OBJ_OMP): CCFLAGS += -fopenmp
# $(CU_OBJ_1): NVCCFLAGS +=
$(CU_OBJ_2): NVCCFLAGS += -rdc=true
$(CU_OBJ_3): NVCCFLAGS += -rdc=true
$(CU_OBJ_4): NVCCFLAGS += -I $(NVCOMP_INCLUDE_DIR)/ 
	#-L $(NVCOMP_LIB_DIR)/

all: ; @$(MAKE) cusz -j

################################################################################

_DEPS_ARG  := $(SRC_DIR)/argparse.o
_DEPS_MEM  := $(SRC_DIR)/cuda_mem.o
_DEPS_HIST := $(SRC_DIR)/hist.o $(SRC_DIR)/huff_interface.o $(SRC_DIR)/format.o -rdc=true
_DEPS_OLDENC := $(SRC_DIR)/huff_codec.o $(SRC_DIR)/par_huffman.o $(SRC_DIR)/par_huffman_sortbyfreq.o $(SRC_DIR)/par_merge.o
DEPS_HUFF := $(_DEPS_MEM) $(_DEPS_HIST) $(_DEPS_OLDENC) $(_DEPS_ARG)

install: bin/cusz
	cp bin/cusz /usr/local/bin

cusz: $(NVCOMP_STATIC_LIB) $(OBJ_ALL) $(GTEST_STATIC_LIB) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(MAIN) \
		-lgomp -lcusparse -lpthread \
		-rdc=true \
		$(NVCOMP_STATIC_LIB) \
		$(GTEST_STATIC_LIB) -I$(GTEST_INCLUDE_DIR)/ \
		$^ -o $(BIN_DIR)/$@

$(BIN_DIR):
	mkdir $@

libnvcomp: $(NVCOMP_STATIC_LIB)
libgtest:  $(GTEST_STATIC_LIB)

patch_nvcomp:
	patch external/nvcomp/src/CMakeLists.txt external/patch.nvcomp-1.1

$(NVCOMP_STATIC_LIB): patch_nvcomp
	cmake \
	    -DCUB_DIR=$(PWD)/external/cub \
	    -DCMAKE_C_COMPILER=$(shell which gcc) \
	    -DCMAKE_CXX_COMPILER=$(shell which g++) \
	    -S ${NVCOMP_DIR} -B ${NVCOMP_DIR}/build  && \
	make -j -C ${NVCOMP_DIR}/build

$(GTEST_STATIC_LIB):
	cmake \
	    -DCMAKE_C_COMPILER=$(shell which gcc) \
	    -DCMAKE_CXX_COMPILER=$(shell which g++) \
	    -S ${GTEST_DIR} -B ${GTEST_DIR}/build  && \
	make -j -C ${GTEST_DIR}/build

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc $(NVCOMP_LIB_DIR)/libnvcomp.a | $(OBJ_DIR)
	$(CXX)  $(CCFLAGS) -c $< -o $@

$(CU_OBJ): $(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(NVCOMP_LIB_DIR)/libnvcomp.a | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJ_ALL)
