spline: ex_spline.cu ../src/utils/verify.hh ../src/kernel/spline.h ../src/utils/io.hh
	nvcc ex_spline.cu -std=c++14 --extended-lambda -o spline

splinedbg: ex_spline.cu ../src/utils/verify.hh ../src/kernel/spline.h ../src/utils/io.hh
	nvcc ex_spline.cu -std=c++14 --extended-lambda -G -o spline
	nvcc ex_spline.cu -lineinfo -std=c++14 --extended-lambda -o splineprof

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


proflorenzo: dummy_xlorenzo3d.cu dummy_xlorenzo2d.cu
	nvcc -lineinfo dummy_xlorenzo2d.cu -o dummy_xlorenzo2d
	nvcc -lineinfo dummy_xlorenzo3d.cu -o dummy_xlorenzo3d
