#include <cuda_runtime.h>
#include <raylib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define THREADS_PER_BLOCK 256

__managed__ Color *pixels;

__device__ Color fractalColorGradient(int iteration, int max_iterations) {
	if (iteration == max_iterations) {
		return BLACK; // Preta para pontos fora do fractal
	} 

	// Gradiente de cores suave com função de seno
	int r = (int)(128.0 + 127.0 * sin(0.16 * iteration + 4));
	int g = (int)(128.0 + 127.0 * sin(0.16 * iteration + 2));
	int b = (int)(128.0 + 127.0 * sin(0.16 * iteration + 0));

	return (Color){ r, g, b, 255 };
}

__global__ void burning_ship_kernel(Color *pixels, int width, int height, double centerX, double centerY, double scale, int max_iterations) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        double zx = 0.0, zy = 0.0;

        double cx = centerX + ((double)x - (double)width / 2.0) * scale;
        double cy = centerY + ((double)y - (double)height / 2.0) * scale;

        int iteration = 0;
        while (zx * zx + zy * zy < 4.0 && iteration < max_iterations) {
            double temp = zx * zx - zy * zy + cx;
            zy = fabs(2.0 * zx * zy) + cy;
            zx = temp;
            iteration++;
        }

        pixels[y * width + x] = fractalColorGradient(iteration, max_iterations);
    }
}

void calculate_dims(dim3 *block_dim, dim3 *thread_dim, int width, int height) {
    int thread_row = (int)sqrt(THREADS_PER_BLOCK);
    *block_dim = dim3((width + thread_row - 1) / thread_row, (height + thread_row - 1) / thread_row);
    *thread_dim = dim3(thread_row, thread_row);
}

int main(int argc, char *argv[]) {
    const int width = 800;
    const int height = 600;

    // Inicializa a Raylib
    InitWindow(width, height, "Burning Ship Fractal");
    SetTargetFPS(60);

	SetTraceLogLevel(LOG_NONE);

    cudaMallocManaged(&pixels, width * height * sizeof(Color));

    dim3 block_dim, thread_dim;
    calculate_dims(&block_dim, &thread_dim, width, height);

    // Defina as coordenadas e escala para o fractal
    double centerX = -1.761485;
    double centerY = -0.03; // -0.000040;
    double scale = 0.0002;
    int max_iterations = 2000; // 12000;
    const double zoomFactor = 0.9;
    const double panFactorBase = 20.0; // 0.1;

    while (!WindowShouldClose()) {
        // Controles de zoom
        if (IsKeyDown(KEY_UP)) scale *= zoomFactor;
        if (IsKeyDown(KEY_DOWN)) scale /= zoomFactor;

        // Controles de movimentação
        double panFactor = panFactorBase * scale;
        if (IsKeyDown(KEY_W)) centerY -= panFactor;
        if (IsKeyDown(KEY_S)) centerY += panFactor;
        if (IsKeyDown(KEY_D)) centerX += panFactor;
        if (IsKeyDown(KEY_A)) centerX -= panFactor;

        // Gera o fractal com os novos valores de centro e escala
        burning_ship_kernel<<<block_dim, thread_dim>>>(pixels, width, height, centerX, centerY, scale, max_iterations);
        cudaDeviceSynchronize();

        // Cria uma imagem e textura a partir do array de pixels
        Image image = GenImageColor(width, height, BLACK);
        memcpy(image.data, pixels, width * height * sizeof(Color));
        Texture2D texture = LoadTextureFromImage(image);
        UnloadImage(image); // Descarrega a imagem já que agora temos a textura

        BeginDrawing();
		{
			ClearBackground(RAYWHITE);
			DrawTexture(texture, 0, 0, WHITE);

			/*
			char zoomText[256];
			snprintf(zoomText, sizeof(zoomText), "Scale: %lf%%", 1.0 - scale);
			DrawText(zoomText, 10, 10, 12, WHITE);
			*/
		}
        EndDrawing();

        UnloadTexture(texture); // Descarrega a textura para regenerar na próxima iteração
    }

    cudaFree(pixels); // Libera a memória CUDA
    CloseWindow(); // Fecha a janela e o contexto OpenGL

    return 0;
}
