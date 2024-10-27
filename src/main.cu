#include <cuda_runtime.h>
#include <SDL2/SDL.h>

#include <stdio.h>
#include <stdint.h>

#define THREADS_PER_BLOCK 256

typedef uint32_t pixel_t;

__managed__ pixel_t *pixels;

__device__ pixel_t pixel_from_rgb(uint8_t r, uint8_t g, uint8_t b) {
	pixel_t pixel;
	pixel  = b << 0;
	pixel |= g << 8;
	pixel |= r << 16;
	// pixel |= a << 24;

	return pixel;
}

__global__ void burning_ship_kernel(pixel_t *pixels, int width, int height, int max_iterations) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	double zx = 0.0, zy = 0.0;

	// Here is where the magic happens :)
	double scale = 4.0;
	double cx = ((double)x - (double)width / 2.0) * scale / (double)width;
	double cy = ((double)y - (double)height / 2.0) * scale / (double)height;

	int iteration = 0;
	while (zx * zx + zy * zy < 4.0 && iteration < max_iterations) {
		double temp = zx * zx - zy * zy + cx;

		zy = abs(2.0 * zx * zy) + cy;
		zx = temp;

		iteration++;
	}

	pixel_t color;
	if (iteration == max_iterations) {
		color = pixel_from_rgb(0, 0, 0);
	} else {
		color = pixel_from_rgb(0, (max_iterations / iteration) * 255, 0);
	}

	pixels[y * width + x] = color;
}

// TODO: Maybe use grid_size insdead of grid_width and grid_height
void draw_grid(SDL_Surface *surface, int grid_width, int grid_height, int width, int height) {
	SDL_Rect grid_cell;
	grid_cell.w = width / grid_width;
	grid_cell.h = height / grid_height;

	for (int y = 0; y < grid_height; y++) {
		grid_cell.y = y * grid_cell.h;

		for (int x = 0; x < grid_width; x++) {
			grid_cell.x = x * grid_cell.w;

			// SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
			// SDL_RenderDrawRect(renderer, &grid_cell);
		}
	}
}

// Create squared dims
void calculate_dims(dim3 *block_dim, dim3 *thread_dim, int width, int height) {
	int block_count = (width * height) / THREADS_PER_BLOCK;

	/*
	if (block_count % 2 != 0) {
		printf("WARN: Cannot perfectly fit the grid on the screen!\n");
	}
	*/

	int block_row = sqrt(block_count);
	int thread_row = sqrt(THREADS_PER_BLOCK);

	*block_dim = dim3(block_row, block_row);
	*thread_dim = dim3(thread_row, thread_row);
}

int main(int argc, char *argv[]) {
	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		printf("ERROR: Failed to initialize SDL!\n");
	}

	int width = 2000, height = 2000;
	// int grid_width = 4, grid_height = 4;

	cudaMallocManaged(&pixels, width * height * sizeof(pixel_t));

	dim3 block_dim, thread_dim;
	calculate_dims(&block_dim, &thread_dim, width, height);

	printf("Generating fractal...\n");

	burning_ship_kernel<<<block_dim, thread_dim>>>(pixels, width, height, 1000);

	cudaDeviceSynchronize();

	printf("Done!\n");

	SDL_Surface *surface = SDL_CreateRGBSurfaceFrom(
		pixels, width, height, 32, width * sizeof(pixel_t),
		0x00FF0000, 0x0000FF00, 0x000000FF, 0
	);

	SDL_SaveBMP(surface, "/mnt/c/Users/Bruno/Desktop/out.bmp");

	SDL_FreeSurface(surface);

	/*
	int grid_width = 4, grid_height = 4;
	SDL_Texture* grid_texture = SDL_CreateTexture(
		renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_TARGET, width, height
	);
	*/

	// SDL_DestroyTexture(grid_texture);
	// SDL_DestroyRenderer(renderer);

	cudaFree(pixels);

	return 0;
}

