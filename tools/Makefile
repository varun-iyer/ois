CC=nvcc

NVFLAGS = \
-arch=sm_61 \
-Xcompiler '-fPIC'

NVLFLAGS = \
	-shared \
	-L .

main : test.c libnvmas.so
	g++ -L. -o $@ $< -lnvmas

libnvmas.so : nvmas.o libdevice.a
	$(CC) $(NVFLAGS) $(NVLFLAGS) -ldevice $< -o $@

libdevice.a : device.o
	$(CC) $(NVFLAGS) -lib $^ -o $@

%.o : %.cu
	$(CC) $(NVFLAGS) -dc $^

clean:
	rm *.o *.so *.a main
