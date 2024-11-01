# burning-cuda

A Burning Ship fractal implementation in CUDA.

## Screenshots

## Build from Source

### Any Debian Linux

> This project obviously depends on CUDA Toolkit. If you are having trouble installing it on Linux, I suggest installing it on Windows and using it trough WSL.
> This project depends on Raylib. If you already have it installed, feel free to skip the next section. 

#### Install Dependencies

1. Get the latest release of [Raylib](https://github.com/raysan5/raylib/releases) (mine was v5.0):
```bash
$ wget https://github.com/raysan5/raylib/releases/download/5.0/raylib-5.0_linux_amd64.tar.gz
$ tar -xvf raylib-5.0_linux_amd64.tar.gz
```

2. Install Raylib locally:
```bash
$ cd raylib-5.0_linux_amd64
$ sudo cp include/* /usr/local/include
$ sudo cp lib/* /usr/local/lib
$ sudo ldconfig
```

#### Clone and Build the Project
```bash
$ git clone https://github.com/mrbru12/burning-cuda.git
$ cd burning-cuda
$ make
```

