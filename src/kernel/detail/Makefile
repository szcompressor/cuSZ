all: l23
l23: scan v0

SHARED_SOURCE = t-l23-utils.hh lorenzo23.inl subroutine.inl subsub.inl

scan: t-l23-scan.cu $(SHARED_SOURCE)
	nvcc t-l23-scan.cu -o t-scan

v0: v0-1d v0c-1d v0-2d v0c-2d v0-3d v0r1-3d v0r1c-3d

v0-1d: t-l23-v0-1d.cu $(SHARED_SOURCE)
	nvcc t-l23-v0-1d.cu -o t-v0-1d
v0c-1d: t-l23-v0c-1d.cu $(SHARED_SOURCE)
	nvcc t-l23-v0c-1d.cu -o t-v0c-1d

v0-2d: t-l23-v0-2d.cu $(SHARED_SOURCE)
	nvcc t-l23-v0-2d.cu -o t-v0-2d
v0c-2d: t-l23-v0c-2d.cu $(SHARED_SOURCE)
	nvcc t-l23-v0c-2d.cu -o t-v0c-2d

v0-3d: t-l23-v0-3d.cu $(SHARED_SOURCE)
	nvcc t-l23-v0-3d.cu -o t-v0-3d
v0r1-3d: t-l23-v0r1-3d.cu $(SHARED_SOURCE)
	nvcc t-l23-v0r1-3d.cu -o t-v0r1-3d
v0r1c-3d: t-l23-v0r1c-3d.cu $(SHARED_SOURCE)
	nvcc t-l23-v0r1c-3d.cu -o t-v0r1c-3d

clean:
	rm -f t-scan t-v0-1d t-v0c-1d t-v0-2d t-v0c-2d t-v0-3d t-v0r1-3d t-v0r1c-3d
