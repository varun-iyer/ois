CC=nvcc
SRCS=mat.cu
CFLAGS= \
--shared \
--compiler-options '-fPIC'

mat.so : $(SRCS)
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm -rf *.so
