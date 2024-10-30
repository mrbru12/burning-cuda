#include <cuda_runtime.h>
#include <raylib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 640

#define THREADS_PER_BLOCK 256

#define FRACTAL_MAX_ITERATIONS 1000
#define FRACTAL_CENTERX -1.749816864467520 // -1.75
#define FRACTAL_CENTERY 0.000008025484745 // -0.035
#define FRACTAL_SCALE 0.000000043630546 // 0.05

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

	// Mapeamento de x e y para o plano complexo de -2 a +2
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

int main(int argc, char *argv[]) {
	SetTraceLogLevel(LOG_NONE);

    const int width = SCREEN_WIDTH;
    const int height = SCREEN_HEIGHT;

    cudaMallocManaged(&pixels, width * height * sizeof(Color));

    calculateGridSize(&blockCount, &threadsPerBlock, width, height);

    double centerX = FRACTAL_CENTERX;
    double centerY = FRACTAL_CENTERY;
    double scale = FRACTAL_SCALE;

    InitWindow(width, height, "Burning Ship Fractal");
    SetTargetFPS(60);

	Texture2D texture = createFractalTexture(width, height);

    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_UP)) scale *= FRACTAL_ZOOM_FACTOR;
        if (IsKeyDown(KEY_DOWN)) scale /= FRACTAL_ZOOM_FACTOR;

		// scale *= 0.99;

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

			char info[512];
			snprintf(
				info, sizeof(info), 
				"CenterX: %.15lf\nCenterY: %.15lf\nScale: %.15lf", 
				centerX, centerY, scale
			);
			DrawText(info, 10, 10, 14, WHITE);

			/*
			char zoomText[256];
			snprintf(zoomText, sizeof(zoomText), "Scale: %lf%%", 1.0 - scale);
			DrawText(zoomText, 10, 10, 12, WHITE);
			*/
		}
        EndDrawing();
    }

	UnloadTexture(texture); // Descarrega a textura para regenerar na próxima iteração

    cudaFree(pixels); // Libera a memória CUDA
    CloseWindow(); // Fecha a janela e o contexto OpenGL

    return 0;
}
