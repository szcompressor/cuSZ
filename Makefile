#STRICT_CHECK=-Xcompiler -Wall
#PTX_VERBOSE=-Xptxas -O3,-v

#CXX       := clang++ -fPIE
CXX       := g++
NVCC      := nvcc
STD       := -std=c++11
CCFLAGS   := $(STD) -O3 -g
NVCCFLAGS := $(STD) -O3 -g --expt-relaxed-constexpr
SRC_DIR   := src
OBJ_DIR   := src
BIN_DIR   := bin

GPU_P1000 := -gencode=arch=compute_61,code=sm_61
GPU_V100  := -gencode=arch=compute_70,code=sm_70
DEPLOY    := $(GPU_P1000) $(GPU_V100)

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
