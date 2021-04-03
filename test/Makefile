
query: test_query.cc ../src/query.hh
	nvcc  -x c++ -I../src/ -DMAIN test_query.cc -o query


newgs: test_thrustgs.cu
	nvcc -std=c++14 test_thrustgs.cu cuda_mem.o --extended-lambda 


rle: test_rle.cu
	nvcc -std=c++14 test_rle.cu -o test_rle
