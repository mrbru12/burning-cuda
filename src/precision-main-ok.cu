#include <cuda_runtime.h>
#include <raylib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 640

#define THREADS_PER_BLOCK 256

#define FRACTAL_MAX_ITERATIONS 10000
#define FRACTAL_CENTERX -1.749816864467520 // -1.75
#define FRACTAL_CENTERY 0.000008025484745 // -0.035
#define FRACTAL_SCALE 0.050000043630546 // 0.05

#define FRACTAL_ZOOM_FACTOR 0.85
#define FRACTAL_PAN_FACTOR 0.05

__managed__ Color *pixels;

// Estrutura para double-double arithmetic
struct DoubleDouble {
    double hi;
    double lo;
};

// Funções de operações de precisão dupla-dupla

__device__ __host__ DoubleDouble dd_add(DoubleDouble a, DoubleDouble b) {
    DoubleDouble result;
    double t1 = a.hi + b.hi;
    double e = t1 - a.hi;
    double t2 = ((b.hi - e) + (a.hi - (t1 - e))) + a.lo + b.lo;
    result.hi = t1 + t2;
    result.lo = t2 - (result.hi - t1);
    return result;
}

__device__ __host__ DoubleDouble dd_sub(DoubleDouble a, DoubleDouble b) {
    DoubleDouble result;
    double t1 = a.hi - b.hi;
    double e = t1 - a.hi;
    double t2 = ((-b.hi - e) + (a.hi - (t1 - e))) + a.lo - b.lo;
    result.hi = t1 + t2;
    result.lo = t2 - (result.hi - t1);
    return result;
}

__device__ __host__ DoubleDouble dd_mul(DoubleDouble a, DoubleDouble b) {
    DoubleDouble result;
    double c11 = a.hi * b.hi;
    double c21 = a.lo * b.hi + a.hi * b.lo;
    result.hi = c11 + c21;
    result.lo = c21 - (result.hi - c11);
    return result;
}

__device__ __host__ DoubleDouble dd_set(double val) {
    DoubleDouble result;
    result.hi = val;
    result.lo = 0.0;
    return result;
}

__device__ Color fractalColorGradient(int iteration, int maxIterations) {
    if (iteration == maxIterations) {
        return BLACK;
    } 

    int r = (int)(128.0 + 127.0 * sin(0.16 * iteration + 4));
    int g = (int)(128.0 + 127.0 * sin(0.16 * iteration + 2));
    int b = (int)(128.0 + 127.0 * sin(0.16 * iteration + 0));

    return (Color){ r, g, b, 255 };
}

__global__ void generateFractalKernel(Color *pixels, int width, int height, DoubleDouble centerX, DoubleDouble centerY, DoubleDouble scale, int maxIterations) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height) return;

    DoubleDouble two = dd_set(2.0);
    DoubleDouble cx = dd_add(centerX, dd_mul(dd_set(2.0 * ((double)x / width - 0.5)), scale));
    DoubleDouble cy = dd_add(centerY, dd_mul(dd_set(2.0 * ((double)y / height - 0.5)), scale));

    DoubleDouble zx = dd_set(0.0), zy = dd_set(0.0);
    int iteration = 0;
    while ((zx.hi * zx.hi + zy.hi * zy.hi < 4.0) && (iteration < maxIterations)) {
        DoubleDouble zx2 = dd_mul(zx, zx);
        DoubleDouble zy2 = dd_mul(zy, zy);

        DoubleDouble temp = dd_add(dd_sub(zx2, zy2), cx);
        zy = dd_add(dd_mul(two, dd_set(fabs(zx.hi) * zy.hi)), cy);
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

void updateFractalTexture(Texture2D texture, DoubleDouble centerX, DoubleDouble centerY, DoubleDouble scale) {
    generateFractalKernel<<<blockCount, threadsPerBlock>>>(
        pixels, texture.width, texture.height, 
        centerX, centerY, scale, FRACTAL_MAX_ITERATIONS
    );
    cudaDeviceSynchronize();

    UpdateTexture(texture, pixels);
}

void calculateGridSize(dim3 *blockCount, dim3 *threadsPerBlock, int width, int height) {
    int threadsPerRow = (int)sqrt(THREADS_PER_BLOCK);

    *blockCount = dim3((width + threadsPerRow - 1) / threadsPerRow, (height + threadsPerRow - 1) / threadsPerRow);
    *threadsPerBlock = dim3(threadsPerRow, threadsPerRow);
}

int main(int argc, char *argv[]) {
    SetTraceLogLevel(LOG_NONE);

    const int width = SCREEN_WIDTH;
    const int height = SCREEN_HEIGHT;

    cudaMallocManaged(&pixels, width * height * sizeof(Color));

    calculateGridSize(&blockCount, &threadsPerBlock, width, height);

    DoubleDouble centerX = dd_set(FRACTAL_CENTERX);
    DoubleDouble centerY = dd_set(FRACTAL_CENTERY);
    DoubleDouble scale = dd_set(FRACTAL_SCALE);

    InitWindow(width, height, "Burning Ship Fractal");
    SetTargetFPS(60);

    Texture2D texture = createFractalTexture(width, height);

    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_UP)) {
            scale = dd_mul(scale, dd_set(FRACTAL_ZOOM_FACTOR));
        }
        if (IsKeyDown(KEY_DOWN)) {
            scale = dd_mul(scale, dd_set(1.0 / FRACTAL_ZOOM_FACTOR));
        }

        DoubleDouble panFactor = dd_mul(scale, dd_set(FRACTAL_PAN_FACTOR));
        if (IsKeyDown(KEY_W)) centerY = dd_sub(centerY, panFactor);
        if (IsKeyDown(KEY_S)) centerY = dd_add(centerY, panFactor);
        if (IsKeyDown(KEY_D)) centerX = dd_add(centerX, panFactor);
        if (IsKeyDown(KEY_A)) centerX = dd_sub(centerX, panFactor);

        updateFractalTexture(texture, centerX, centerY, scale);

        BeginDrawing();
        {
            ClearBackground(RAYWHITE);
            DrawTexture(texture, 0, 0, WHITE);

            char info[512];
            snprintf(
                info, sizeof(info), 
                "CenterX: %.15lf\nCenterY: %.15lf\nScale: %.15lf", 
                centerX.hi, centerY.hi, scale.hi
            );
            DrawText(info, 10, 10, 14, WHITE);
        }
        EndDrawing();
    }

    UnloadTexture(texture);

    cudaFree(pixels);

    CloseWindow();

    return 0;
}
