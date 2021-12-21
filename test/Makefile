all: splinedemo

interp_spline3.o: ../src/wrapper/interp_spline3.cu ../src/kernel/spline3.cuh
	@echo :: resolving dependency of Sp::DefaultCompressor, spline3...
	nvcc ../src/wrapper/interp_spline3.cu -c --extended-lambda

extrap_lorenzo.o: ../src/wrapper/extrap_lorenzo.cu ../src/kernel/lorenzo.cuh
	@echo :: resolving dependency of Sp::FallbackCompressor, lorenzo...
	nvcc ../src/wrapper/extrap_lorenzo.cu -c --extended-lambda

csr10.o: ../src/wrapper/csr10.cu
	@echo :: resolving spline3 dependency-2...
	nvcc ../src/wrapper/csr10.cu -c --extended-lambda

csr11.o: ../src/wrapper/csr11.cu
	@echo :: resolving spline3 dependency-3...
	nvcc ../src/wrapper/csr11.cu -c --extended-lambda

splinedemo: ex_spline3_demo.cu ../src/sp_path.cuh ex_common.cuh interp_spline3.o csr11.o
	@echo :: creating spline3 demo-1...
	nvcc \
		interp_spline3.o \
		ex_spline3_demo.cu \
		csr11.o \
		-std=c++14 --extended-lambda \
		-lcusparse \
		-o splinedemo

splinestack: ex_spline3_stack.cu ../src/sp_path.cuh ex_common.cuh interp_spline3.o csr11.o
	@echo :: creating spline3 stack imaging...
	nvcc \
		interp_spline3.o \
		ex_spline3_stack.cu \
		csr11.o \
		-std=c++14 --extended-lambda \
		-lcusparse \
		-o splinestack

dummylorenzo:
	nvcc dummy_cxlorenzo.cu -DnoCUSPARSE -o run.dummy_cxlorenzo


query: test_query.cc ../src/query.hh
	nvcc  -x c++ -I../src/ -DMAIN test_query.cc -o query


newgs: test_thrustgs.cu
	nvcc -std=c++14 test_thrustgs.cu cuda_mem.o --extended-lambda 


rle: test_rle.cu
	nvcc -std=c++14 test_rle.cu -o test_rle

huff: test_huff_class.cu
	nvcc -std=c++14 -G \
 		-arch=sm_75 -gencode=arch=compute_75,code=sm_75 \
 		-gencode=arch=compute_75,code=compute_75 \
		test_huff_class.cu \
		../src/par_merge.o \
		../src/huff_codec.o \
		../src/par_huffman_sortbyfreq.o \
		../src/par_huffman.o \
		-o test_huff_class

testtypebinding: ../src/binding.hh test_type_binding.cu
	clang++ test_type_binding.cu -std=c++14 --cuda-gpu-arch=sm_75 -L/usr/local/cuda/lib64 -lcudart_static -ldl -lrt -pthread
	nvcc test_type_binding.cu


proflorenzo: dummy_xlorenzo3d.cu dummy_xlorenzo2d.cu
	nvcc -lineinfo dummy_xlorenzo2d.cu -o dummy_xlorenzo2d
	nvcc -lineinfo dummy_xlorenzo3d.cu -o dummy_xlorenzo3d


clean:
	rm -f *.o spline3demo

purge:
	rm -f *.o spline3demo
