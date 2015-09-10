
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>

cudaError_t addWithCuda(int *c, int *a, unsigned int size, int gridx, int gridy, int dimBlock);
int * createArray(int amountToAdd, int arraySize, int * a);

__global__ void addKernel(int *c, int *a)
{
	//declares space in shared memory
	extern __shared__ int sdata[];

	//gets unique thread id
	int tid = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x;

	//each thread copies information from global into shared memory
	sdata[threadIdx.x] = a[tid];
	
	//threads are synced as all data must be copied over before any addition can be done
	__syncthreads();


	//TODO: improve efficiency by changing from interleaved addressing to sequential addressing
	//This for loop adds two adjacent numbers together, then sums the combination until only one number remains per block
	for (int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * threadIdx.x;


		if (index < blockDim.x) {
			sdata[index] += sdata[index + s];
		}
		//threads are synced to ensure all addition has completed before moving on
		__syncthreads();
	}
	//each block stores its final value in the array c at the position of its unique block id
	if (threadIdx.x == 0) c[blockIdx.x + blockIdx.y * gridDim.x] = sdata[0];

}

int main()
{
	//number of values we want to add (memory restrictions make it impossible to add 1 billion at once)
	int amountToAdd = 125000000;

	//TODO: Optimize by removing the need to have an array of a power of two
	//array is set to a power of two to make the addition easier
	const int arraySize = 134217728;
	int* a = new int[arraySize];
	int* b = new int[arraySize];
	int* c = new int[arraySize];

	//create the array up to 8 times and reduce to a manageable amount of numbers (set i to 1 for 125,000,000 numbers and 8 for 1,000,000,000)
	for (int i = 0; i < 8; i++){

		createArray(amountToAdd, arraySize, a);
		// Add vectors in parallel. reduces 125,000,000 numbers (8 times for 1 billion) stores in an array for further reduction
		cudaError_t cudaStatus = addWithCuda(c, a, arraySize, 32768, 4, 1024);
		for (int j = 0; j < 32768 * 4; j++){
			b[i*32768*4+j] = a[j];
			//printf("{%d}", i*32768*4+j);
		}
	}
	a = b;
	//reduces the values further until only a single value remains
	cudaError_t cudaStatus = addWithCuda(c, a, arraySize, 1024, 1, 1024);
	addWithCuda(c, a, arraySize, 1, 1, 1024);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	//Should print 1,000,000,000
	printf("{%d}\n", a[0]);



    //cudaDeviceReset must be called before exiting in order for profiling and
    //tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;

	
}

int * createArray(int amountToAdd, int arraySize, int * a){

	//populate the array with data
	for (int i = 0; i<amountToAdd; i++) {
		a[i] = 1;
	}

	//fill the rest of the array with 0's so as not to affect the final result
	for (int i = amountToAdd; i < arraySize; i++){
		a[i] = 0;
	}

	return a;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, int *a, unsigned int size, int gridx, int gridy, int dimBlock)
{
    int *dev_a = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "c cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "a cudaMalloc failed!");
        goto Error;
    }


    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<dim3(gridx,gridy,1), dimBlock, dimBlock *sizeof(int)>>>(dev_c, dev_a);


    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(a, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    
    return cudaStatus;
}
