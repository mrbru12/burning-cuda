#include <cuda_runtime.h>

#include <stdio.h>

int main(int argc, char *argv[]) {
	int device_count;
	cudaGetDeviceCount(&device_count);

	for (int i = 0; i < device_count; i++) {
		cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

		printf("pageableMemoryAccess: %d\n", prop.pageableMemoryAccess);
		printf("concurrentManagedAccess: %d\n", prop.concurrentManagedAccess);
		printf("managedMemory: %d\n", prop.managedMemory);
	}

	return 0;
}
