all: tscan v01d v02d

tscan: test_scan.cu test_utils.hh
	nvcc test_scan.cu -o tscan

_rand: ../../../test/src/rand.hh ../../../test/src/rand_g.cc 
	nvcc ../../../test/src/rand_g.cc -c 

v01d: _rand test_l23v0.cu test_utils.hh lorenzo23.inl subroutine.inl subsub.inl
	nvcc test_l23v0.cu rand_g.o -lcurand -o tv0

v02d: _rand test_l23v02d.cu test_utils.hh lorenzo23.inl subroutine.inl subsub.inl
	nvcc test_l23v02d.cu rand_g.o -lcurand -o tv02d

clean:
	rm -f tscan tv0 tv02d
