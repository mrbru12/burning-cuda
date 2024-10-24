CC = nvcc
# CFLAGS = -Wall -Wextra -Wpedantic
# LDFLAGS = -lm

all: bin src/main.cu
	# $(CC) $(CFLAGS) src/main.cu $(LDFLAGS) -o bin/main
	$(CC) src/main.cu -o bin/main

bin:
	mkdir bin

clean:
	rm -r bin

.PHONY: all 
