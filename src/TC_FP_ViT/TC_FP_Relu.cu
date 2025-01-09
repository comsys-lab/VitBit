__global__ void TC_FP_Relu(const int8_t* input, int8_t* output, int width) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * width + col;
        
    output[idx] = max(0, input[idx]);
}
