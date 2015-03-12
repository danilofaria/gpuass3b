build: 
	/usr/local/cuda/bin/nvcc im1.cu im1.cc -I. -I/usr/include/OpenEXR -lIlmImf -lImath -lHalf -o prog_out -arch=sm_30

test: build 
	./prog_out input.exr

.PHONY: exec 
exec: build
