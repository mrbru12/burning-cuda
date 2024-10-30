CC = nvcc
CFLAGS = -I/home/bru/dev/raylib-5.0/include -I/home/bru/dev/campary_01.06.17/CAMPARY/Doubles/src_gpu
LDFLAGS = -L/home/bru/dev/raylib-5.0/lib -lraylib

all: bin src/main.cu
	$(CC) src/main.cu $(CFLAGS) $(LDFLAGS) -o bin/main
	$(CC) src/precision-main.cu $(CFLAGS) $(LDFLAGS) -o bin/precision-main

bin:
	mkdir bin

clean:
	rm -r bin

.PHONY: all 
