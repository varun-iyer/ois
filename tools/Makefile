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
CFLAGS = -I./cpu/ '-fPIC'
LFLAGS = -lm
OFLAGS = -c
SRCS = cpu/oistools.c
endif

liboistools.so : oistools.o
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LFLAGS)
# ifeq ($(CUDA), True)
	# $(CC) -shared $^ -o $@ $(LFLAGS)
# else
	# ar rcs $@ $^
	# $(CC) -shared $* -o
# endif

oistools.o : $(SRCS)
	$(CC) $(CFLAGS) $(OFLAGS) $^ $(LFLAGS)

clean:
	rm liboistools.so oistools.o
