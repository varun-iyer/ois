CUDA_CC=nvcc
CUDA_CFLAGS = \
-arch=sm_61 \
-Xcompiler '-fPIC' \
-g
CUDA_SRCS = cuda/oistools.cu
CUDA_LFLAGS = -lcudart
CUDA_OFLAGS = -dc

CC=gcc
CFLAGS = -I./cpu/ '-fPIC' -g
LFLAGS = -lm
OFLAGS = -c
SRCS = cpu/oistools.c

STATIC = liboistools.a
SHARED = liboistools.so

default : cuda_shared

cpu_shared : cpu_oistools.o
	$(CC) $(CFLAGS) -shared $^ -o $(SHARED) $(LFLAGS)

cuda_shared : cuda_oistools.o
	$(CUDA_CC) $(CUDA_CFLAGS) -shared $^ -o $(SHARED) $(CUDA_LFLAGS)

cpu_static : cpu_oistools.o
	ar rcs $(STATIC) $^

cuda_static : cuda_oistools.o
	$(CUDA_CC) -arch=sm_61 -lib $^ -o $(STATIC) $(CUDA_LFLAGS)

cuda_oistools.o : $(CUDA_SRCS)
	$(CUDA_CC) -o $@ $(CUDA_CFLAGS) $(CUDA_OFLAGS) $^ $(CUDA_LFLAGS)

cpu_oistools.o : $(SRCS)
	$(CC) -o $@ $(CFLAGS) $(OFLAGS) $^ $(LFLAGS)

clean:
	-@rm *.a *.so *.o
