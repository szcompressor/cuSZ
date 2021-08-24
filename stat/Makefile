smalleb-macos:
	clang++ -Xpreprocessor -fopenmp -lomp exp_small_eb.cc -std=c++17 -I"$(brew --prefix libomp)/include" -L"$(brew --prefix libomp)/lib" -O3

smalleb:
	g++ -fopenmp exp_small_eb.cc -std=c++17 -O3
