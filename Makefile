CC = nvcc
CFLAGS = -I/home/bru/raylib-5.0/include
LDFLAGS = -L/home/bru/raylib-5.0/lib -lraylib

all: bin src/main.cu
	# $(CC) $(CFLAGS) src/main.cu $(LDFLAGS) -o bin/main
	$(CC) src/main.cu $(CFLAGS) $(LDFLAGS) -o bin/main

bin:
	mkdir bin

clean:
	rm -r bin

.PHONY: all 
