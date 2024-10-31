CC = nvcc
LDFLAGS = -lraylib

all: bin src/main.cu
	$(CC) src/main.cu $(LDFLAGS) -o bin/burning-ship

bin:
	mkdir bin

clean:
	rm -r bin

.PHONY: all 
