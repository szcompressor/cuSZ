lorenzo:
	dpcpp -std=c++14 \
		-DDPCPP_SHOWCASE \
		src/wrapper/extrap_lorenzo.dp.cpp \
		src/experimental/dpcpp_demo_lorenzo.dp.cpp \
		-o dpcpp_demo_lorenzo
