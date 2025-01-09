__global__ void Only_TC_Normalization(const int8_t* input, int8_t* output, const int8_t* gamma, const int8_t* beta, int channels, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * width + col;
    int channel_idx = idx / width;
    int channel_start = channel_idx * width;

    if (row < channels && col < width) {
        // Calculate mean and variance for the current channel using integer arithmetic
        int8_t sum = 0;
        int8_t sum_squared = 0;
        for (int i = 0; i < width; ++i) {
            int input_idx = channel_start + i;
            int8_t val = input[input_idx];
            sum += val;
            sum_squared += val * val;
        }
        int8_t mean = sum / width;
        int8_t variance = sum_squared / width - mean * mean;

        // Normalize and scale the data
        int8_t normalized_val = ((input[idx] - mean) * 1000000) / (sqrt(variance + 1e-5f) * 1000) + 500000;
        int8_t scaled_val = (gamma[channel_idx] * normalized_val + beta[channel_idx]) / 1000000;

        // Convert back to integer format
        output[idx] = scaled_val;
    }
}