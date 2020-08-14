ifeq ($(CUDA), True)
CC=nvcc
CFLAGS = \
-arch=sm_61 \
-Xcompiler '-fPIC'
SRCS = cuda/oistools.cu
LFLAGS =
OFLAGS = -dc
else
CC=gcc
CFLAGS = -I./cpu/ -c
LFLAGS = -lm
OFLAGS =
SRCS = cpu/oistools.c
endif

liboistools.a : oistools.o
ifeq ($(CUDA), True)
	$(CC) $(CFLAGS) -lib $^ -o $@ $(LFLAGS)
else
	ar rcs $@ $^
endif

oistools.o : $(SRCS)
	$(CC) $(CFLAGS) $(OFLAGS) $^ $(LFLAGS)

clean:
	rm liboistools.a oistools.o
