# burning-cuda

A Burning Ship fractal implementation in CUDA.

## Screenshots

## Build from Source

### Any Debian Linux

> Steps 1. and 2. can be skipped if you already have raylib installed!

#### Install Dependencies

1. Get the latest release of [raylib]("https://github.com/raysan5/raylib/releases") (mine was v5.0):
```bash
$ wget https://github.com/raysan5/raylib/releases/download/5.0/raylib-5.0_linux_amd64.tar.gz
$ tar -xvf raylib-5.0_linux_amd64.tar.gz
```

2. Install raylib locally:
```bash
$ cd raylib-5.0_linux_amd64
$ sudo cp include/* /usr/local/include
$ sudo cp lib/* /usr/local/lib
$ sudo ldconfig
```

#### Clone and build the project:
```bash
$ git clone https://github.com/mrbru12/burning-cuda.git
$ cd burning-cuda
$ make
```

