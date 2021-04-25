#title           :tpl.Makefile
#description     :This snippet is used to generate Makefile.
#author          :Jiannan Tian
#copyright       :(C) 2021 by Washington State University, Argonne National Laboratory
#license         :See LICENSE in top-level directory
#date            :2021-01-10
#version         :0.2
#==============================================================================

#### presets
## --profile or -pg
##		Instrument generated code/executable for use by gprof (Linux only).
## --generate-line-info or -lineinfo
##		Generate line-number information for device code.
## --device-debug or -G
##		Generate debug information for device code. Turns off all optimizations.
##		Don't use for profiling; use -lineinfo instead.

#### compiler
CC       := g++ -std=c++14
NVCC     := nvcc -std=c++14
CC_FLAGS :=
NV_FLAGS := --expt-relaxed-constexpr --expt-extended-lambda

CUDA_VER      := $(shell nvcc --version | grep "release" | awk '{print $$5}' | cut -d, -f1)

#### external libs
NVCOMP_DIR        := external/nvcomp
NVCOMP_INCLUDE_DIR:= $(NVCOMP_DIR)/build/include
NVCOMP_LIB_DIR    := $(NVCOMP_DIR)/build/lib
NVCOMP_STATIC_LIB := $(NVCOMP_LIB_DIR)/libnvcomp.a

GTEST_DIR         := external/googletest
GTEST_INCLUDE_DIR := $(GTEST_DIR)/googletest/include
GTEST_LIB_DIR     := $(GTEST_DIR)/build/lib
GTEST_STATIC_LIB  := $(GTEST_LIB_DIR)/libgtest.a

#### filesystem
SRC_DIR     := src
OBJ_DIR     := src
BIN_DIR     := bin

#### source code classification
# CCsrc_omp :=$(SRC_DIR)/analysis_utils.cc
# CCsrc     := $(filter-out $(CCsrc_omp), $(wildcard $(SRC_DIR)/*.cc))
CCsrc     := $(wildcard $(SRC_DIR)/*.cc)

MAIN      := $(SRC_DIR)/cusz.cu

CUsrc4    := $(SRC_DIR)/huff_interface.cu
CUsrc3    := $(SRC_DIR)/par_merge.cu $(SRC_DIR)/par_huffman.cu
CUsrc2    := $(SRC_DIR)/cusz_interface.cu
CUsrc1    := $(filter-out $(MAIN) $(CUsrc3) $(CUsrc2), $(wildcard $(SRC_DIR)/*.cu))

# CCo_omp   := $(CCsrc_omp:$(SRC_DIR)/%.cc=$(OBJ_DIR)/%.o)
CCo       := $(CCsrc:$(SRC_DIR)/%.cc=$(OBJ_DIR)/%.o)

CUo1      := $(CUsrc1:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
CUo2      := $(CUsrc2:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
CUo3      := $(CUsrc3:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
CUo4      := $(CUsrc4:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

CUo       := $(CUo1) $(CUo2) $(CUo3)
# OBJ_all   := $(CCo) $(CCo_omp) $(CUo)
OBJ_all   := $(CCo) $(CUo)

# $(CCo_omp): CC_FLAGS += -fopenmp
$(CUo2): NV_FLAGS += -rdc=true
$(CUo3): NV_FLAGS += -rdc=true
$(CUo4): NV_FLAGS += -I $(NVCOMP_INCLUDE_DIR)/ 

CC_FLAGS  += ####
NV_FLAGS  += ####
HW_TARGET += ####

$(info CUDA $(CUDA_VER) is in use.)

target_proxy: $(eval NV_FLAGS = $(NV_FLAGS) $(HW_TARGET))
target_proxy: ; @$(MAKE) cusz -j

#### compilation commands
all: target_proxy

#cusz: $(OBJ_all) | $(BIN_DIR)
#	$(NVCC) $(NV_FLAGS) -lgomp -lcusparse $(MAIN) -rdc=true $^ -o $(BIN_DIR)/$@

cusz: $(OBJ_all) $(NVCOMP_STATIC_LIB)  $(GTEST_STATIC_LIB) | $(BIN_DIR)
	$(NVCC) $(NV_FLAGS) $(MAIN) \
		-lcusparse -lpthread \
		-rdc=true \
		$(NVCOMP_STATIC_LIB) \
		$(GTEST_STATIC_LIB) -I$(GTEST_INCLUDE_DIR)/ \
		$^ -o $(BIN_DIR)/$@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc | $(OBJ_DIR)
	$(CC) $(CC_FLAGS) -c $< -o $@

$(CUo): $(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(NVCOMP_STATIC_LIB) | $(OBJ_DIR)
	$(NVCC) $(NV_FLAGS) -c $< -o $@

$(BIN_DIR):
	mkdir $@

install: bin/cusz
	cp bin/cusz /usr/local/bin

.PHONY: clean
clean:
	$(RM) $(OBJ_all)
