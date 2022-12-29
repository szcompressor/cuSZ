all: tscan v01d v02d v03d

tscan: test_scan.cu test_utils.hh
	nvcc test_scan.cu -o tscan

v01d: test_l23v01d.cu test_utils.hh lorenzo23.inl subroutine.inl subsub.inl
	nvcc test_l23v01d.cu -o tv01d

v02d: test_l23v02d.cu test_utils.hh lorenzo23.inl subroutine.inl subsub.inl
	nvcc test_l23v02d.cu -o tv02d

v03d: test_l23v03d.cu test_utils.hh lorenzo23.inl subroutine.inl subsub.inl
	nvcc test_l23v03d.cu -o tv03d

clean:
	rm -f tscan tv01d tv02d tv03d
