#include <cuda_runtime.h>
#include <raylib.h>

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 640

#define THREADS_PER_BLOCK 256

#define FRACTAL_MAX_ITERATIONS 1000
#define FRACTAL_CENTERX -1.749816864467520
#define FRACTAL_CENTERY 0.000008025484745
#define FRACTAL_SCALE 0.000000043630546

#define FRACTAL_ZOOM_FACTOR 0.95
#define FRACTAL_PAN_FACTOR 0.05

__managed__ Color *pixels;

__device__ Color fractalColorGradient(int iteration, int maxIterations) {
	if (iteration == maxIterations) {
		return BLACK;
	} 

	int r = (int)(128.0 + 127.0 * sin(0.16 * iteration + 4));
	int g = (int)(128.0 + 127.0 * sin(0.16 * iteration + 2));
	int b = (int)(128.0 + 127.0 * sin(0.16 * iteration + 0));

	return (Color){ r, g, b, 255 };
}

//
// https://en.wikipedia.org/wiki/Burning_Ship_fractal#Implementation
// https://paulbourke.net/fractals/burnship/
//
__global__ void generateFractalKernel(Color *pixels, int width, int height, double centerX, double centerY, double scale, int maxIterations) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

	// Map x and y to the complex plane
	double cx = centerX + 2.0 * scale * ((double)x / width - 0.5);
	double cy = centerY + 2.0 * scale * ((double)y / height - 0.5);

	double zx = 0.0, zy = 0.0;

	int iteration = 0;
	while (zx * zx + zy * zy < 4.0 && iteration < maxIterations) {
		double temp = zx * zx - zy * zy + cx;
		zy = fabs(2.0 * zx * zy) + cy;
		zx = temp;

		iteration++;
	}

	pixels[y * width + x] = fractalColorGradient(iteration, maxIterations);
}

Texture2D createFractalTexture(int width, int height) {
	Image image = GenImageColor(width, height, BLANK);
	Texture2D texture = LoadTextureFromImage(image);
	UnloadImage(image);

	return texture;
}

static dim3 blockCount, threadsPerBlock;

void updateFractalTexture(Texture2D texture, double centerX, double centerY, double scale) {
	generateFractalKernel<<<blockCount, threadsPerBlock>>>(
		pixels, texture.width, texture.height, 
		centerX, centerY, scale, FRACTAL_MAX_ITERATIONS
	);
	cudaDeviceSynchronize();

	UpdateTexture(texture, pixels);
}

void calculateGridSize(dim3 *blockCount, dim3 *threadsPerBlock, int width, int height) {
    int threadsPerRow = (int)sqrt(THREADS_PER_BLOCK);

    *blockCount = dim3(width / threadsPerRow, height / threadsPerRow);
    *threadsPerBlock = dim3(threadsPerRow, threadsPerRow);
}

void showFractalInfo(double centerX, double centerY, double scale) {
	char info[512];
	snprintf(
		info, sizeof(info), 
		"CenterX: %.15lf\nCenterY: %.15lf\nScale: %.15lf", 
		centerX, centerY, scale
	);

	DrawText(info, 10, 10, 14, WHITE);
}

void showHelp(const char *binary) {
	printf("Usage: %s <centerX> <centerY> <scale>\n", binary); 
	printf("Example: %s -1.75 -0.035 0.05\n", binary);
}

int main(int argc, char *argv[]) {
    double centerX = FRACTAL_CENTERX;
    double centerY = FRACTAL_CENTERY;
    double scale = FRACTAL_SCALE;

	if (argc > 1) {
		if (argc != 4) {
			showHelp(argv[0]);
			exit(1);
		}

		sscanf(argv[1], "%lf", &centerX);
		sscanf(argv[2], "%lf", &centerY);
		sscanf(argv[3], "%lf", &scale);
	}

    const int width = SCREEN_WIDTH;
    const int height = SCREEN_HEIGHT;

	SetTraceLogLevel(LOG_NONE);
    InitWindow(width, height, "Burning Ship Fractal");
    SetTargetFPS(60);

    cudaMallocManaged(&pixels, width * height * sizeof(Color));
    calculateGridSize(&blockCount, &threadsPerBlock, width, height);

	Texture2D texture = createFractalTexture(width, height);

    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_UP)) scale *= FRACTAL_ZOOM_FACTOR;
        if (IsKeyDown(KEY_DOWN)) scale /= FRACTAL_ZOOM_FACTOR;

        double panFactor = FRACTAL_PAN_FACTOR * scale;
        if (IsKeyDown(KEY_W)) centerY -= panFactor;
        if (IsKeyDown(KEY_S)) centerY += panFactor;
        if (IsKeyDown(KEY_D)) centerX += panFactor;
        if (IsKeyDown(KEY_A)) centerX -= panFactor;

        updateFractalTexture(texture, centerX, centerY, scale);

        BeginDrawing();
		{
			ClearBackground(RAYWHITE);

			DrawTexture(texture, 0, 0, WHITE);
			showFractalInfo(centerX, centerY, scale);
		}
        EndDrawing();
    }

	UnloadTexture(texture);

    cudaFree(pixels);
    CloseWindow();

    return 0;
}
