CC=nvcc

NVFLAGS = \
-arch=sm_61 \
-Xcompiler '-fPIC'

NVLFLAGS = \
	-shared \
	-L .

main : main.cpp libtest.so
	g++ -L. -o $@ $< -ltest

libtest.so : libutil.a test.o
	$(CC) -arch=sm_61 -shared -Xcompiler -fPIC -L. -lutil test.o -o $@

libutil.a : util.o
	$(CC) $(NVFLAGS) -lib $^ -o $@

%.o : %.cu
	$(CC) $(NVFLAGS) -dc $^

clean:
	rm *.o *.so *.a main
