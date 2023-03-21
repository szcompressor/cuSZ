all: l23
l23: v0

# SHARED_SOURCE = test_l23_utils.hh lorenzo23.inl subroutine.inl subsub.inl
NVCC=nvcc -ccbin g++-9
COMPILE_SETTINGS=-I../../src/kernel/detail -I../../include

v0: v0-1d v0c-1d v0-2d v0c-2d v0-3d v0r1-3d v0r1c-3d

v0-1d: test_l23_v0_1d.cu $(SHARED_SOURCE)
	$(NVCC) test_l23_v0_1d.cu -o t-v0-1d $(COMPILE_SETTINGS)
v0c-1d: test_l23_v0c_1d.cu $(SHARED_SOURCE)
	$(NVCC) test_l23_v0c_1d.cu -o t-v0c-1d $(COMPILE_SETTINGS) 

v0-2d: test_l23_v0_2d.cu $(SHARED_SOURCE)
	$(NVCC) test_l23_v0_2d.cu -o t-v0-2d $(COMPILE_SETTINGS) 
v0c-2d: test_l23_v0c_2d.cu $(SHARED_SOURCE)
	$(NVCC) test_l23_v0c_2d.cu -o t-v0c-2d $(COMPILE_SETTINGS) 

v0-3d: test_l23_v0_3d.cu $(SHARED_SOURCE)
	$(NVCC) test_l23_v0_3d.cu -o t-v0-3d $(COMPILE_SETTINGS) 
v0r1-3d: test_l23_v0r1_3d.cu $(SHARED_SOURCE)
	$(NVCC) test_l23_v0r1_3d.cu -o t-v0r1-3d $(COMPILE_SETTINGS) 
v0r1c-3d: test_l23_v0r1c_3d.cu $(SHARED_SOURCE)
	$(NVCC) test_l23_v0r1c_3d.cu -o t-v0r1c-3d $(COMPILE_SETTINGS) 

clean:
	rm -f t-v0-1d t-v0c-1d t-v0-2d t-v0c-2d t-v0-3d t-v0r1-3d t-v0r1c-3d
