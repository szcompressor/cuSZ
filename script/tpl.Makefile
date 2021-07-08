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
API_DIR     := src/wrapper

MAIN        := $(SRC_DIR)/cusz.cu

ALL_OBJ  = $(SRC_DIR)/argparse.o $(SRC_DIR)/cusz_interface.o $(API_DIR)/deprecated_lossless_huffman.o $(API_DIR)/par_huffman.o $(API_DIR)/extrap_lorenzo.o $(API_DIR)/handle_sparsity.o
CC_FLAGS  += ####
NV_FLAGS  += ####
HW_TARGET += ####

$(info CUDA $(CUDA_VER) is in use.)

all: target_proxy

target_proxy: $(eval NV_FLAGS = $(NV_FLAGS) $(HW_TARGET))
target_proxy: ; @$(MAKE) cusz -j

argparse: $(SRC_DIR)/argparse.cc
	$(CC) $(CC_FLAGS) $(SRC_DIR)/argparse.cc -c -o $(SRC_DIR)/argparse.o

submain: $(SRC_DIR)/cusz_interface.cu
	$(NVCC) $(NV_FLAGS) -c $< -o $(SRC_DIR)/cusz_interface.o

api_sparsity: $(API_DIR)/handle_sparsity.cu
	$(NVCC) $(NV_FLAGS) -c $< -o $(API_DIR)/handle_sparsity.o

extrap_lorenzo: $(API_DIR)/extrap_lorenzo.cu
	$(NVCC) $(NV_FLAGS) -c $< -o $(API_DIR)/extrap_lorenzo.o

api_huffman: $(API_DIR)/deprecated_lossless_huffman.cu
	$(NVCC) $(NV_FLAGS) -c $< -o $(API_DIR)/deprecated_lossless_huffman.o -rdc=true

api_par_huff: $(API_DIR)/par_huffman.cu
	$(NVCC) $(NV_FLAGS) -c $< -o $(API_DIR)/par_huffman.o -rdc=true

objs: submain api_huffman api_sparsity api_par_huff argparse extrap_lorenzo 

cusz: objs | $(BIN_DIR)
	$(NVCC) $(NV_FLAGS) $(MAIN) -lcusparse -lpthread -rdc=true $(ALL_OBJ) -o $(BIN_DIR)/cusz

$(BIN_DIR):
	mkdir $@

install: bin/cusz
	mkdir $(HOME)/bin
	cp bin/cusz $(HOME)/bin

.PHONY: clean
clean:
	$(RM) $(ALL_OBJ)
