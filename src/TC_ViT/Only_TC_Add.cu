__global__ void Only_TC_Add(const int8_t* input, const int8_t* weight, int8_t* output, int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for(int i = 0; i < width; i++)
        output[idx + i] = input[idx + i] + weight[idx];
}
