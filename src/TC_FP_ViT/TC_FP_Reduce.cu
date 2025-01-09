__global__ void TC_FP_Reduce(int8_t *matrix, int8_t *result, int rows, int cols) {
    __shared__ int8_t partialSums[32];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x;
    int index = row * cols + blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    partialSums[tid] = 0;

    for (int i = index; i < row * cols + cols; i += stride) {
        if(i < rows * cols) {
            partialSums[tid] += matrix[i];
        }
    }
    
    if (tid == 0) {
        int8_t sum = 0;
        for (int i = 0; i < blockDim.x; ++i) {
            sum += partialSums[i];
        }
        result[blockIdx.y * gridDim.x + blockIdx.x] = sum;
    }
}