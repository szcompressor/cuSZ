#STRICT_CHECK=-Xcompiler -Wall
#PTX_VERBOSE=-Xptxas -O3,-v


#CXX       := clang++ -fPIE
CXX       := g++
NVCC      := nvcc
STD       := -std=c++11
HOST_DBG  := -O0 -g
CUDA_DBG  := -O0 -G -g
SRC_DIR   := src
OBJ_DIR   := src
BIN_DIR   := bin

GPU_VOLTA := -gencode=arch=compute_70,code=sm_70
GPU_TURING:= -gencode=arch=compute_75,code=sm_75
GPU_AMPERE:= -gencode=arch=compute_80,code=sm_80
DEPLOY    := $(GPU_VOLTA) $(GPU_TURING)

CCFLAGS   := $(STD) -O3
NVCCFLAGS := $(STD) $(DEPLOY)

CCFILES   := $(wildcard $(SRC_DIR)/*.cc)

MAIN      := $(SRC_DIR)/cusz.cu
CUFILES2  := $(SRC_DIR)/cusz_workflow.cu $(SRC_DIR)/cusz_dualquant.cu
CUFILES3  := $(SRC_DIR)/canonical.cu
CUFILES1  := $(filter-out $(MAIN) $(CUFILES3) $(CUFILES2), $(wildcard $(SRC_DIR)/*.cu))

CCOBJS    := $(CCFILES:$(SRC_DIR)/%.cc=$(OBJ_DIR)/%.o)
CUOBJS1   := $(CUFILES1:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
CUOBJS2   := $(CUFILES2:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
CUOBJS3   := $(CUFILES3:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

CUOBJS    := $(CUOBJS1) $(CUOBJS2) $(CUOBJS3)
OBJS      := $(CCOBJS) $(CUOBJS)

# $(CUOBJS1): NVCCFLAGS +=
$(CUOBJS2): NVCCFLAGS += -rdc=true
$(CUOBJS3): NVCCFLAGS += -rdc=true

all: ; @$(MAKE) cusz -j

install: bin/cusz
	cp bin/cusz /usr/local/bin

cusz: $(OBJS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -lcusparse $(MAIN) -rdc=true $^ -o $(BIN_DIR)/$@
$(BIN_DIR):
	mkdir $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc | $(OBJ_DIR)
	$(CXX)  $(CCFLAGS) -c $< -o $@

$(CUOBJS): $(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJS)
