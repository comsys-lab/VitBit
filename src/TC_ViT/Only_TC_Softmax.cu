#include <stdio.h>
#include <stdlib.h>

__global__ void Only_TC_Softmax(int8_t* input, int8_t* output, int N, int C) {
    const int warpsPerBlock = blockDim.x / warpSize;
    int tid = threadIdx.x;

    if (tid >= C) {
        return;
    }

    int warpId = tid / warpSize;
    int laneId = tid % warpSize;
    // one warp one row
    int row = blockIdx.x * warpsPerBlock + warpId;

    if (row >= N) {
        return;
    }

    const int8_t* x = input + row * C;
    int8_t* const y = output + row * C;

    float maxval = -INFINITY, sumval = 0, bigger;
    for (int i = laneId; i < C; i += warpSize) {
        bigger = fmaxf(maxval, x[i]);
        sumval = sumval * expf(maxval - bigger) + expf(x[i] - bigger);
        maxval = bigger;
    }

    float offsetMaxval, offsetSumval;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        __syncwarp();
        offsetMaxval = __shfl_down_sync(0xFFFFFFFF, maxval, offset);
        offsetSumval = __shfl_down_sync(0xFFFFFFFF, sumval, offset);
        if (offsetMaxval > maxval) {
            sumval *= expf(maxval - offsetMaxval);
            maxval = offsetMaxval;
        } else {
            offsetSumval *= expf(offsetMaxval - maxval);
        }
        sumval += offsetSumval;
    }

    maxval = __shfl_sync(0xFFFFFFFF, maxval, 0);
    sumval = __shfl_sync(0xFFFFFFFF, sumval, 0);

    for (int i = laneId; i < C; i += warpSize) {
        y[i] = expf(x[i] - maxval) / sumval;
    }
}

// int main() {
// 	// var declaration
// 	int N = 2304;
//     int C = 784;
// 	int8_t h_in[N*C];
// 	int8_t h_out[N*C];
// 	int8_t *d_in, *d_out;


// 	// memory allocation
// 	cudaMalloc((void**)&d_in, N*C * sizeof(int8_t));
// 	cudaMalloc((void**)&d_out, N*C * sizeof(int8_t));

// 	// data initialization
// 	for (int i = 0; i < N*C; ++i) {
// 		h_in[i] = (int8_t)(rand() % 5 + 1);
// 	}
	
// 	cudaMemcpy(d_in, h_in, N*C * sizeof(int8_t), cudaMemcpyHostToDevice);
	
// 	// launch softmax kernel
// 	dim3 threadsPerBlock(32,24);
// 	dim3 blocksPerGrid(2304/32, 784/24);

//     cudaEvent_t start1, stop1;
//     cudaEventCreate(&start1);
//     cudaEventCreate(&stop1);

//     cudaEventRecord(start1);

// 	Only_TC_Softmax<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N, C);

//     cudaEventRecord(stop1);
//     cudaEventSynchronize(stop1);

//     float milliseconds1 = 0;
//     cudaEventElapsedTime(&milliseconds1, start1, stop1);

//     cudaEventDestroy(start1);
//     cudaEventDestroy(stop1);

//     printf("%f\n", milliseconds1);

// 	// copy result to host
// 	cudaMemcpy(h_out, d_out, N * C * sizeof(int8_t), cudaMemcpyDeviceToHost);

// 	// clean device memory
// 	cudaFree(d_in);
// 	cudaFree(d_out);

// 	return 0;
// }
