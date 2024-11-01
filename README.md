# burning-cuda

A [Burning Ship](https://en.wikipedia.org/wiki/Burning_Ship_fractal) fractal implementation in CUDA.

## Screenshots

<div align="center" style="display: flex; flex-direction: row; flex-wrap: wrap">
	<img width="49%" src="screenshots/screenshot-1730433841.png" alt="First armada ship."/>
	<img width="49%" src="screenshots/screenshot-1730434208.png" alt="Ship from the far left of the armada."/>
	<img width="49%" src="screenshots/screenshot-1730433816.png" alt="Spiral pattern near the first armada ship bulbous bow."/>
	<img width="49%" src="screenshots/screenshot-1730435790.png" alt="Pattern near the edge on the bottom of the main ship."/>
</div>

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

