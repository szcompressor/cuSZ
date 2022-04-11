lorenzo:
	nvcc -lineinfo -std=c++14 \
		-DDPCPP_SHOWCASE \
		src/component/extrap_lorenzo.cu \
		src/experimental/dpcpp_demo_lorenzo.cu \
		-o dpcpp_demo_lorenzo
