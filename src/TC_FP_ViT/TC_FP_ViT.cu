#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <mma.h>

#include "TC_FP_Functions.cuh"

#define warp_size 32
#define block_size1 32
#define block_size2 24

#define TOTAL_WIDTH 784
#define TC_WIDTH 720
#define CC_WIDTH 64

void initializeRandom_int8(int8_t* array, int size) {
    for (int i = 0; i < size; ++i) {
        array[i] = rand() % 11;
    }
}

void initializeRandom_int(int* array, int size) {
    for (int i = 0; i < size; ++i) {
        array[i] = rand() % 11;
    }
}

void initializeRandom_float(float* array, int size) {
    for (int i = 0; i < size; ++i) {
        array[i] = static_cast<float>(rand()) / RAND_MAX; // 0부터 1까지의 랜덤값
    }
}

void rearrange_int8(int8_t *input, int8_t *output, int output_rows, int output_cols) {
    for (int row = 0; row < output_rows; ++row) {
        for (int col = 0; col < output_cols; ++col) {
            int idx = row * output_cols + col;
            if (row < output_rows && col < output_cols) {
                int out_row = idx / output_cols;
                int out_col = idx % output_cols;
                int in_idx = (out_row * 14 + out_col / 48) * 224 + out_col % 48;

                output[idx] = input[in_idx];
            }
        }
    }
}

void rearrange_int(int *input, int *output, int output_rows, int output_cols) {
    for (int row = 0; row < output_rows; ++row) {
        for (int col = 0; col < output_cols; ++col) {
            int idx = row * output_cols + col;
            if (row < output_rows && col < output_cols) {
                int out_row = idx / output_cols;
                int out_col = idx % output_cols;
                int in_idx = (out_row * 14 + out_col / 48) * 224 + out_col % 48;

                output[idx] = input[in_idx];
            }
        }
    }
}

int main(){

    ///// Initalizing Input Data, Weight, Bias, Gamma, Beta /////
    /* Rearange 1 */
    int8_t *Rearrange_TC_input = new int8_t[224 * 224 * 3 * TC_WIDTH / (TC_WIDTH + CC_WIDTH)];
    initializeRandom_int8(Rearrange_TC_input, 224 * 224 * 3 * TC_WIDTH / (TC_WIDTH + CC_WIDTH));
    float *Rearrange_CC_input = new float[224 * 224 * 3 * CC_WIDTH / (TC_WIDTH + CC_WIDTH)];
    initializeRandom_float(Rearrange_CC_input, 224 * 224 * 3 * CC_WIDTH / (TC_WIDTH + CC_WIDTH));

    /* Measuring Preprocessng Time*/
    // clock_t Pre_Processing_Start, Pre_Processing_End;
    // double Pre_Processing_Time;

    // Pre_Processing_Start = clock();

    int8_t *Rearrange1_TC_output_GPU;
    cudaMalloc(&Rearrange1_TC_output_GPU, (TC_WIDTH*192) * sizeof(int8_t));
    cudaMemcpy(Rearrange1_TC_output_GPU, Rearrange_TC_input, (TC_WIDTH*192) * sizeof(int8_t), cudaMemcpyHostToDevice);
    float *Rearrange1_CC_output_GPU;
    cudaMalloc(&Rearrange1_CC_output_GPU, (CC_WIDTH*192) * sizeof(float));
    cudaMemcpy(Rearrange1_CC_output_GPU, Rearrange_CC_input, (CC_WIDTH*192) * sizeof(float), cudaMemcpyHostToDevice);

    /* Layer Normalization 2 */
    int8_t *Norm2_TC_gamma_CPU = new int8_t[192];
    initializeRandom_int8(Norm2_TC_gamma_CPU, 192);
    float *Norm2_CC_gamma_CPU = new float[192];
    initializeRandom_float(Norm2_CC_gamma_CPU, 192);

    int8_t *Norm2_TC_gamma_GPU;
    cudaMalloc(&Norm2_TC_gamma_GPU, 192 * sizeof(int8_t));
    cudaMemcpy(Norm2_TC_gamma_GPU, Norm2_TC_gamma_CPU, 192 * sizeof(int8_t), cudaMemcpyHostToDevice);
    float *Norm2_CC_gamma_GPU;
    cudaMalloc(&Norm2_CC_gamma_GPU, 192 * sizeof(float));
    cudaMemcpy(Norm2_CC_gamma_GPU, Norm2_CC_gamma_CPU, 192 * sizeof(float), cudaMemcpyHostToDevice);

    int8_t *Norm2_TC_beta_CPU = new int8_t[192];
    initializeRandom_int8(Norm2_TC_beta_CPU, 192);
    int8_t *Norm2_TC_beta_GPU;
    cudaMalloc(&Norm2_TC_beta_GPU, 192 * sizeof(int8_t));
    cudaMemcpy(Norm2_TC_beta_GPU, Norm2_TC_beta_CPU, 192 * sizeof(int8_t), cudaMemcpyHostToDevice);
    float *Norm2_CC_beta_CPU = new float[192];
    initializeRandom_float(Norm2_CC_beta_CPU, 192);
    float *Norm2_CC_beta_GPU;
    cudaMalloc(&Norm2_CC_beta_GPU, 192 * sizeof(float));
    cudaMemcpy(Norm2_CC_beta_GPU, Norm2_CC_beta_CPU, 192 * sizeof(float), cudaMemcpyHostToDevice);

    int8_t *Norm2_TC_output;
    cudaMalloc(&Norm2_TC_output, (TC_WIDTH*192) * sizeof(int8_t));
    float *Norm2_CC_output;
    cudaMalloc(&Norm2_CC_output, (CC_WIDTH*192) * sizeof(float));

    /* Linear 3 */
    // TC Parameters
    int8_t *Linear3_TC_weight_CPU = new int8_t[192 * 768];
    initializeRandom_int8(Linear3_TC_weight_CPU, 192 * 768);
    int8_t *Linear3_TC_weight_GPU;
    cudaMalloc(&Linear3_TC_weight_GPU, (192 * 768) * sizeof(int8_t));
    cudaMemcpy(Linear3_TC_weight_GPU, Linear3_TC_weight_CPU, (192 * 768) * sizeof(int8_t), cudaMemcpyHostToDevice);

    int8_t *Linear3_TC_bias_CPU = new int8_t[768];
    initializeRandom_int8(Linear3_TC_bias_CPU, 768);
    int8_t *Linear3_TC_bias_GPU;
    cudaMalloc(&Linear3_TC_bias_GPU, 768 * sizeof(int8_t));
    cudaMemcpy(Linear3_TC_bias_GPU, Linear3_TC_bias_CPU, 768 * sizeof(int8_t), cudaMemcpyHostToDevice);

    int8_t *Linear3_TC_output;
    cudaMalloc(&Linear3_TC_output, (TC_WIDTH*768) * sizeof(int8_t));
    int8_t *Linear3_1_TC_output;
    cudaMalloc(&Linear3_1_TC_output, (TC_WIDTH*768) * sizeof(int8_t));

    // CC Parameters
    float *Linear3_CC_weight_CPU = new float[192 * 768];
    initializeRandom_float(Linear3_CC_weight_CPU, 192 * 768);
    float *Linear3_CC_weight_GPU;
    cudaMalloc(&Linear3_CC_weight_GPU, (192 * 768) * sizeof(float));
    cudaMemcpy(Linear3_CC_weight_GPU, Linear3_CC_weight_CPU, (192 * 768) * sizeof(float), cudaMemcpyHostToDevice);

    float *Linear3_CC_bias_CPU = new float[768];
    initializeRandom_float(Linear3_CC_bias_CPU, 768);
    float *Linear3_CC_bias_GPU;
    cudaMalloc(&Linear3_CC_bias_GPU, 768 * sizeof(float));
    cudaMemcpy(Linear3_CC_bias_GPU, Linear3_CC_bias_CPU, 768 * sizeof(float), cudaMemcpyHostToDevice);

    float *Linear3_CC_output;
    cudaMalloc(&Linear3_CC_output, (CC_WIDTH*768) * sizeof(float));
    float *Linear3_1_CC_output;
    cudaMalloc(&Linear3_1_CC_output, (CC_WIDTH*768) * sizeof(float));

    /* Layer Normalization 4 */
    // TC Parameters
    int8_t *Norm4_TC_gamma_CPU = new int8_t[768];
    initializeRandom_int8(Norm4_TC_gamma_CPU, 768);
    int8_t *Norm4_TC_gamma_GPU;
    cudaMalloc(&Norm4_TC_gamma_GPU, 768 * sizeof(int8_t));
    cudaMemcpy(Norm4_TC_gamma_GPU, Norm4_TC_gamma_CPU, 768 * sizeof(int8_t), cudaMemcpyHostToDevice);

    int8_t *Norm4_TC_beta_CPU = new int8_t[768];
    initializeRandom_int8(Norm4_TC_beta_CPU, 768);
    int8_t *Norm4_TC_beta_GPU;
    cudaMalloc(&Norm4_TC_beta_GPU, 768 * sizeof(int8_t));
    cudaMemcpy(Norm4_TC_beta_GPU, Norm4_TC_beta_CPU, 768 * sizeof(int8_t), cudaMemcpyHostToDevice);

    int8_t *Norm4_TC_output;
    cudaMalloc(&Norm4_TC_output, (TC_WIDTH*768) * sizeof(int8_t));

    // CC Parameters
    float *Norm4_CC_gamma_CPU = new float[768];
    initializeRandom_float(Norm4_CC_gamma_CPU, 768);
    float *Norm4_CC_gamma_GPU;
    cudaMalloc(&Norm4_CC_gamma_GPU, 768 * sizeof(float));
    cudaMemcpy(Norm4_CC_gamma_GPU, Norm4_CC_gamma_CPU, 768 * sizeof(float), cudaMemcpyHostToDevice);

    float *Norm4_CC_beta_CPU = new float[768];
    initializeRandom_float(Norm4_CC_beta_CPU, 768);
    float *Norm4_CC_beta_GPU;
    cudaMalloc(&Norm4_CC_beta_GPU, 768 * sizeof(float));
    cudaMemcpy(Norm4_CC_beta_GPU, Norm4_CC_beta_CPU, 768 * sizeof(float), cudaMemcpyHostToDevice);

    float *Norm4_CC_output;
    cudaMalloc(&Norm4_CC_output, (CC_WIDTH*768) * sizeof(float));

    /* Dropout 5 */
    // TC Parameters
    int8_t *Drop5_TC_output;
    cudaMalloc(&Drop5_TC_output, (TC_WIDTH*768) * sizeof(int8_t));

    // CC Parameters
    float *Drop5_CC_output;
    cudaMalloc(&Drop5_CC_output, (CC_WIDTH*768) * sizeof(float));

    const float dropout_prob = 0.5f;

    /* Layer Normalization 6 */
    // TC Parameters
    int8_t *Norm6_TC_gamma_CPU = new int8_t[768];
    initializeRandom_int8(Norm6_TC_gamma_CPU, 768);
    int8_t *Norm6_TC_gamma_GPU;
    cudaMalloc(&Norm6_TC_gamma_GPU, 768 * sizeof(int8_t));
    cudaMemcpy(Norm6_TC_gamma_GPU, Norm6_TC_gamma_CPU, 768 * sizeof(int8_t), cudaMemcpyHostToDevice);

    int8_t *Norm6_TC_beta_CPU = new int8_t[768];
    initializeRandom_int8(Norm6_TC_beta_CPU, 768);
    int8_t *Norm6_TC_beta_GPU;
    cudaMalloc(&Norm6_TC_beta_GPU, 768 * sizeof(int8_t));
    cudaMemcpy(Norm6_TC_beta_GPU, Norm6_TC_beta_CPU, 768 * sizeof(int8_t), cudaMemcpyHostToDevice);

    int8_t *Norm6_TC_output;
    cudaMalloc(&Norm6_TC_output, (TC_WIDTH*768) * sizeof(int8_t));

    // CC Parameters
    float *Norm6_CC_gamma_CPU = new float[768];
    initializeRandom_float(Norm6_CC_gamma_CPU, 768);
    float *Norm6_CC_gamma_GPU;
    cudaMalloc(&Norm6_CC_gamma_GPU, 768 * sizeof(float));
    cudaMemcpy(Norm6_CC_gamma_GPU, Norm6_CC_gamma_CPU, 768 * sizeof(float), cudaMemcpyHostToDevice);

    float *Norm6_CC_beta_CPU = new float[768];
    initializeRandom_float(Norm6_CC_beta_CPU, 768);
    float *Norm6_CC_beta_GPU;
    cudaMalloc(&Norm6_CC_beta_GPU, 768 * sizeof(float));
    cudaMemcpy(Norm6_CC_beta_GPU, Norm6_CC_beta_CPU, 768 * sizeof(float), cudaMemcpyHostToDevice);

    float *Norm6_CC_output;
    cudaMalloc(&Norm6_CC_output, (CC_WIDTH*768) * sizeof(float));

    //// Iteration Start
    int num_iteration = 12;

    /* Linear 7 */
    // TC Parameters
    int8_t *Linear7_TC_weight_CPU[num_iteration];
    int8_t *Linear7_TC_weight_GPU[num_iteration];
    int8_t *Linear7_TC_bias_CPU[num_iteration];
    int8_t *Linear7_TC_bias_GPU[num_iteration];
    int8_t *Linear7_TC_output[num_iteration];
    int8_t *Linear7_1_TC_output[num_iteration];

    // CC Parameters
    float *Linear7_CC_weight_CPU[num_iteration];
    float *Linear7_CC_weight_GPU[num_iteration];
    float *Linear7_CC_bias_CPU[num_iteration];
    float *Linear7_CC_bias_GPU[num_iteration];
    float *Linear7_CC_output[num_iteration];
    float *Linear7_1_CC_output[num_iteration];

    /* Softmax 8 */
    // TC Parameters
    int8_t *Soft8_TC_output[num_iteration];

    // CC Parameters
    float *Soft8_CC_output[num_iteration];

    /* Dropout 9 */
    // TC Parametsers
    int8_t *Drop9_TC_output[num_iteration];

    // CC Parametsers
    float *Drop9_CC_output[num_iteration];

    /* Linear 10 */
    // TC Parameters
    int8_t *Linear10_TC_weight_CPU[num_iteration];
    int8_t *Linear10_TC_weight_GPU[num_iteration];
    int8_t *Linear10_TC_bias_CPU[num_iteration];
    int8_t *Linear10_TC_bias_GPU[num_iteration];
    int8_t *Linear10_TC_output[num_iteration];
    int8_t *Linear10_1_TC_output[num_iteration];

    // CC Parameters
    float *Linear10_CC_weight_CPU[num_iteration];
    float *Linear10_CC_weight_GPU[num_iteration];
    float *Linear10_CC_bias_CPU[num_iteration];
    float *Linear10_CC_bias_GPU[num_iteration];
    float *Linear10_CC_output[num_iteration];
    float *Linear10_1_CC_output[num_iteration];

    /* Dropout 11 */
    // TC Parameters
    int8_t *Drop11_TC_output[num_iteration];

    // CC Parameters
    float *Drop11_CC_output[num_iteration];

    /* Layer Normalization 13 */
    // TC Parameters
    int8_t *Norm13_TC_gamma_CPU[num_iteration];
    int8_t *Norm13_TC_gamma_GPU[num_iteration];
    int8_t *Norm13_TC_beta_CPU[num_iteration];
    int8_t *Norm13_TC_beta_GPU[num_iteration];
    int8_t *Norm13_TC_output[num_iteration];

    // CC Parameters
    float *Norm13_CC_gamma_CPU[num_iteration];
    float *Norm13_CC_gamma_GPU[num_iteration];
    float *Norm13_CC_beta_CPU[num_iteration];
    float *Norm13_CC_beta_GPU[num_iteration];
    float *Norm13_CC_output[num_iteration];

    /* Linear 14 */
    // TC Parameters
    int8_t *Linear14_TC_weight_CPU[num_iteration];
    int8_t *Linear14_TC_weight_GPU[num_iteration];
    int8_t *Linear14_TC_bias_CPU[num_iteration];
    int8_t *Linear14_TC_bias_GPU[num_iteration];
    int8_t *Linear14_TC_output[num_iteration];
    int8_t *Linear14_1_TC_output[num_iteration];

    // CC Parameters
    float *Linear14_CC_weight_CPU[num_iteration];
    float *Linear14_CC_weight_GPU[num_iteration];
    float *Linear14_CC_bias_CPU[num_iteration];
    float *Linear14_CC_bias_GPU[num_iteration];
    float *Linear14_CC_output[num_iteration];
    float *Linear14_1_CC_output[num_iteration];

    /* Gelu 15 */
    // TC Parameters
    int8_t *Gelu15_TC_output[num_iteration];

    // CC Parameters
    float *Gelu15_CC_output[num_iteration];

    /* Dropout 16 */
    // TC Parameters
    int8_t *Drop16_TC_output[num_iteration];

    // CC Parameters
    float *Drop16_CC_output[num_iteration];

    /* Linear 17 */
    // TC Parameters
    int8_t *Linear17_TC_weight_CPU[num_iteration];
    int8_t *Linear17_TC_weight_GPU[num_iteration];
    int8_t *Linear17_TC_bias_CPU[num_iteration];
    int8_t *Linear17_TC_bias_GPU[num_iteration];
    int8_t *Linear17_TC_output[num_iteration];
    int8_t *Linear17_1_TC_output[num_iteration];

    // CC Parameters
    float *Linear17_CC_weight_CPU[num_iteration];
    float *Linear17_CC_weight_GPU[num_iteration];
    float *Linear17_CC_bias_CPU[num_iteration];
    float *Linear17_CC_bias_GPU[num_iteration];
    float *Linear17_CC_output[num_iteration];
    float *Linear17_1_CC_output[num_iteration];

    /* Dropout 18 */
    // TC Parameters
    int8_t *Drop18_TC_output[num_iteration];

    // CC Parameters
    float *Drop18_CC_output[num_iteration];

    /* Layer Normalization 20 */
    // TC Parameters
    int8_t *Norm20_TC_gamma_CPU[num_iteration];
    int8_t *Norm20_TC_gamma_GPU[num_iteration];
    int8_t *Norm20_TC_beta_CPU[num_iteration];
    int8_t *Norm20_TC_beta_GPU[num_iteration];
    int8_t *Norm20_TC_output[num_iteration];

    // CC Parameters
    float *Norm20_CC_gamma_CPU[num_iteration];
    float *Norm20_CC_gamma_GPU[num_iteration];
    float *Norm20_CC_beta_CPU[num_iteration];
    float *Norm20_CC_beta_GPU[num_iteration];
    float *Norm20_CC_output[num_iteration];
    
    for(int i = 0; i < num_iteration; ++i){
        /* Linear 7 */
        // TC Parameters
        Linear7_TC_weight_CPU[i] = new int8_t[768 * 2304];
        initializeRandom_int8(Linear7_TC_weight_CPU[i], 768 * 2304);
        cudaMalloc(&Linear7_TC_weight_GPU[i], (768 * 2304) * sizeof(int8_t));
        cudaMemcpy(Linear7_TC_weight_GPU[i], Linear7_TC_weight_CPU[i], (768 * 2304) * sizeof(int8_t), cudaMemcpyHostToDevice);
        Linear7_TC_bias_CPU[i] = new int8_t[2304];
        initializeRandom_int8(Linear7_TC_bias_CPU[i], 2304);   
        cudaMalloc(&Linear7_TC_bias_GPU[i], 2304 * sizeof(int8_t));
        cudaMemcpy(Linear7_TC_bias_GPU[i], Linear7_TC_bias_CPU[i], 2304 * sizeof(int8_t), cudaMemcpyHostToDevice);
        cudaMalloc(&Linear7_TC_output[i], (TC_WIDTH*2304) * sizeof(int8_t));
        cudaMalloc(&Linear7_1_TC_output[i], (TC_WIDTH*2304) * sizeof(int8_t));

        // CC Parameters
        Linear7_CC_weight_CPU[i] = new float[768 * 2304];
        initializeRandom_float(Linear7_CC_weight_CPU[i], 768 * 2304);
        cudaMalloc(&Linear7_CC_weight_GPU[i], (768 * 2304) * sizeof(float));
        cudaMemcpy(Linear7_CC_weight_GPU[i], Linear7_CC_weight_CPU[i], (768 * 2304) * sizeof(float), cudaMemcpyHostToDevice);
        Linear7_CC_bias_CPU[i] = new float[2304];
        initializeRandom_float(Linear7_CC_bias_CPU[i], 2304);   
        cudaMalloc(&Linear7_CC_bias_GPU[i], 2304 * sizeof(float));
        cudaMemcpy(Linear7_CC_bias_GPU[i], Linear7_CC_bias_CPU[i], 2304 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&Linear7_CC_output[i], (CC_WIDTH*2304) * sizeof(float));
        cudaMalloc(&Linear7_1_CC_output[i], (CC_WIDTH*2304) * sizeof(float));

        /* Softmax 8 */
        // TC Parameters
        cudaMalloc(&Soft8_TC_output[i], (TC_WIDTH*2304) * sizeof(int8_t));

        // CC Parameters
        cudaMalloc(&Soft8_CC_output[i], (CC_WIDTH*2304) * sizeof(float));

        /* Dropout 9 */
        // TC Parameters
        cudaMalloc(&Drop9_TC_output[i], (TC_WIDTH*768) * sizeof(int8_t));

        // CC Parameters
        cudaMalloc(&Drop9_CC_output[i], (CC_WIDTH*768) * sizeof(float));

        /* Linear 10 */
        // TC Parameters
        Linear10_TC_weight_CPU[i] = new int8_t[768 * 768];
        initializeRandom_int8(Linear10_TC_weight_CPU[i], 768 * 768);
        cudaMalloc(&Linear10_TC_weight_GPU[i], (768 * 768) * sizeof(int8_t));
        cudaMemcpy(Linear10_TC_weight_GPU[i], Linear10_TC_weight_CPU[i], (768 * 768) * sizeof(int8_t), cudaMemcpyHostToDevice);
        Linear10_TC_bias_CPU[i] = new int8_t[768];
        initializeRandom_int8(Linear10_TC_bias_CPU[i], 768);
        cudaMalloc(&Linear10_TC_bias_GPU[i], 768 * sizeof(int8_t));
        cudaMemcpy(Linear10_TC_bias_GPU[i], Linear10_TC_bias_CPU[i], 768 * sizeof(int8_t), cudaMemcpyHostToDevice);
        cudaMalloc(&Linear10_TC_output[i], (TC_WIDTH*768) * sizeof(int8_t));
        cudaMalloc(&Linear10_1_TC_output[i], (TC_WIDTH*768) * sizeof(int8_t));

        // CC Parameters
        Linear10_CC_weight_CPU[i] = new float[768 * 768];
        initializeRandom_float(Linear10_CC_weight_CPU[i], 768 * 768);
        cudaMalloc(&Linear10_CC_weight_GPU[i], (768 * 768) * sizeof(float));
        cudaMemcpy(Linear10_CC_weight_GPU[i], Linear10_CC_weight_CPU[i], (768 * 768) * sizeof(float), cudaMemcpyHostToDevice);
        Linear10_CC_bias_CPU[i] = new float[768];
        initializeRandom_float(Linear10_CC_bias_CPU[i], 768);
        cudaMalloc(&Linear10_CC_bias_GPU[i], 768 * sizeof(float));
        cudaMemcpy(Linear10_CC_bias_GPU[i], Linear10_CC_bias_CPU[i], 768 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&Linear10_CC_output[i], (CC_WIDTH*768) * sizeof(float));
        cudaMalloc(&Linear10_1_CC_output[i], (CC_WIDTH*768) * sizeof(float));

        /* Dropout 11 */
        // TC Parameters
        cudaMalloc(&Drop11_TC_output[i], (TC_WIDTH*768) * sizeof(int8_t));

        // CC Parameters
        cudaMalloc(&Drop11_CC_output[i], (CC_WIDTH*768) * sizeof(float));

        /* Layer Normalization 13 */
        // TC Parameters
        Norm13_TC_gamma_CPU[i] = new int8_t[768];
        initializeRandom_int8(Norm13_TC_gamma_CPU[i], 768);
        cudaMalloc(&Norm13_TC_gamma_GPU[i], 768 * sizeof(int8_t));
        cudaMemcpy(Norm13_TC_gamma_GPU[i], Norm13_TC_gamma_CPU[i], 768 * sizeof(int8_t), cudaMemcpyHostToDevice);
        Norm13_TC_beta_CPU[i] = new int8_t[768];
        initializeRandom_int8(Norm13_TC_beta_CPU[i], 768);
        cudaMalloc(&Norm13_TC_beta_GPU[i], 768 * sizeof(int8_t));
        cudaMemcpy(Norm13_TC_beta_GPU[i], Norm13_TC_beta_CPU[i], 768 * sizeof(int8_t), cudaMemcpyHostToDevice);
        cudaMalloc(&Norm13_TC_output[i], (TC_WIDTH*768) * sizeof(int8_t));

        // CC Parameters
        Norm13_CC_gamma_CPU[i] = new float[768];
        initializeRandom_float(Norm13_CC_gamma_CPU[i], 768);
        cudaMalloc(&Norm13_CC_gamma_GPU[i], 768 * sizeof(float));
        cudaMemcpy(Norm13_CC_gamma_GPU[i], Norm13_CC_gamma_CPU[i], 768 * sizeof(float), cudaMemcpyHostToDevice);
        Norm13_CC_beta_CPU[i] = new float[768];
        initializeRandom_float(Norm13_CC_beta_CPU[i], 768);
        cudaMalloc(&Norm13_CC_beta_GPU[i], 768 * sizeof(float));
        cudaMemcpy(Norm13_CC_beta_GPU[i], Norm13_CC_beta_CPU[i], 768 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&Norm13_CC_output[i], (CC_WIDTH*768) * sizeof(float));

        /* Linear 14 */
        // TC Parameters
        Linear14_TC_weight_CPU[i] = new int8_t[768 * 3072];
        initializeRandom_int8(Linear14_TC_weight_CPU[i], 768 * 3072);
        cudaMalloc(&Linear14_TC_weight_GPU[i], (768 * 3072) * sizeof(int8_t));
        cudaMemcpy(Linear14_TC_weight_GPU[i], Linear14_TC_weight_CPU[i], (768 * 3072) * sizeof(int8_t), cudaMemcpyHostToDevice);
        Linear14_TC_bias_CPU[i] = new int8_t[3072];
        initializeRandom_int8(Linear14_TC_bias_CPU[i], 3072);
        cudaMalloc(&Linear14_TC_bias_GPU[i], 3072 * sizeof(int8_t));
        cudaMemcpy(Linear14_TC_bias_GPU[i], Linear14_TC_bias_CPU[i], 3072 * sizeof(int8_t), cudaMemcpyHostToDevice);
        cudaMalloc(&Linear14_TC_output[i], (TC_WIDTH*3072) * sizeof(int8_t));
        cudaMalloc(&Linear14_1_TC_output[i], (TC_WIDTH*3072) * sizeof(int8_t));

        // CC Parameters
        Linear14_CC_weight_CPU[i] = new float[768 * 3072];
        initializeRandom_float(Linear14_CC_weight_CPU[i], 768 * 3072);
        cudaMalloc(&Linear14_CC_weight_GPU[i], (768 * 3072) * sizeof(float));
        cudaMemcpy(Linear14_CC_weight_GPU[i], Linear14_CC_weight_CPU[i], (768 * 3072) * sizeof(float), cudaMemcpyHostToDevice);
        Linear14_CC_bias_CPU[i] = new float[3072];
        initializeRandom_float(Linear14_CC_bias_CPU[i], 3072);
        cudaMalloc(&Linear14_CC_bias_GPU[i], 3072 * sizeof(float));
        cudaMemcpy(Linear14_CC_bias_GPU[i], Linear14_CC_bias_CPU[i], 3072 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&Linear14_CC_output[i], (CC_WIDTH*3072) * sizeof(float));
        cudaMalloc(&Linear14_1_CC_output[i], (CC_WIDTH*3072) * sizeof(float));

        /* Gelu 15 */
        // TC Parameters
        cudaMalloc(&Gelu15_TC_output[i], (TC_WIDTH*3072) * sizeof(int8_t));

        // CC Parameters
        cudaMalloc(&Gelu15_CC_output[i], (CC_WIDTH*3072) * sizeof(float));

        /* Dropout 16 */
        // TC Parameters
        cudaMalloc(&Drop16_TC_output[i], (TC_WIDTH*3072) * sizeof(int8_t));

        // CC Parameters
        cudaMalloc(&Drop16_CC_output[i], (CC_WIDTH*3072) * sizeof(float));

        /* Linear 17 */
        // TC Parameters
        Linear17_TC_weight_CPU[i] = new int8_t[768 * 3072];
        initializeRandom_int8(Linear17_TC_weight_CPU[i], 768 * 3072);
        cudaMalloc(&Linear17_TC_weight_GPU[i], (768 * 3072) * sizeof(int8_t));
        cudaMemcpy(Linear17_TC_weight_GPU[i], Linear17_TC_weight_CPU[i], (768 * 3072) * sizeof(int8_t), cudaMemcpyHostToDevice);
        Linear17_TC_bias_CPU[i] = new int8_t[768];
        initializeRandom_int8(Linear17_TC_bias_CPU[i], 768);
        cudaMalloc(&Linear17_TC_bias_GPU[i], 3072 * sizeof(int8_t));
        cudaMemcpy(Linear17_TC_bias_GPU[i], Linear17_TC_bias_CPU[i], 768 * sizeof(int8_t), cudaMemcpyHostToDevice);
        cudaMalloc(&Linear17_TC_output[i], (TC_WIDTH*768) * sizeof(int8_t));
        cudaMalloc(&Linear17_1_TC_output[i], (TC_WIDTH*768) * sizeof(int8_t));

        // CC Parameters
        Linear17_CC_weight_CPU[i] = new float[768 * 3072];
        initializeRandom_float(Linear17_CC_weight_CPU[i], 768 * 3072);
        cudaMalloc(&Linear17_CC_weight_GPU[i], (768 * 3072) * sizeof(float));
        cudaMemcpy(Linear17_CC_weight_GPU[i], Linear17_CC_weight_CPU[i], (768 * 3072) * sizeof(float), cudaMemcpyHostToDevice);
        Linear17_CC_bias_CPU[i] = new float[768];
        initializeRandom_float(Linear17_CC_bias_CPU[i], 768);
        cudaMalloc(&Linear17_CC_bias_GPU[i], 3072 * sizeof(float));
        cudaMemcpy(Linear17_CC_bias_GPU[i], Linear17_CC_bias_CPU[i], 768 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&Linear17_CC_output[i], (CC_WIDTH*768) * sizeof(float));
        cudaMalloc(&Linear17_1_CC_output[i], (CC_WIDTH*768) * sizeof(float));

        /* Dropout 18 */
        // TC Parameters
        cudaMalloc(&Drop18_TC_output[i], (TC_WIDTH*768) * sizeof(int8_t));

        // CC Parameters
        cudaMalloc(&Drop18_CC_output[i], (CC_WIDTH*768) * sizeof(float));

        /* Layer Normalization 20 */
        // TC Parameters
        Norm20_TC_gamma_CPU[i] = new int8_t[768];
        initializeRandom_int8(Norm20_TC_gamma_CPU[i], 768);
        cudaMalloc(&Norm20_TC_gamma_GPU[i], 768 * sizeof(int8_t));
        cudaMemcpy(Norm20_TC_gamma_GPU[i], Norm20_TC_gamma_CPU[i], 768 * sizeof(int8_t), cudaMemcpyHostToDevice);
        Norm20_TC_beta_CPU[i] = new int8_t[768];
        initializeRandom_int8(Norm20_TC_beta_CPU[i], 768);
        cudaMalloc(&Norm20_TC_beta_GPU[i], 768 * sizeof(int8_t));
        cudaMemcpy(Norm20_TC_beta_GPU[i], Norm20_TC_beta_CPU[i], 768 * sizeof(int8_t), cudaMemcpyHostToDevice);
        cudaMalloc(&Norm20_TC_output[i], (TC_WIDTH*768) * sizeof(int8_t));

        // CC Parameters
        Norm20_CC_gamma_CPU[i] = new float[768];
        initializeRandom_float(Norm20_CC_gamma_CPU[i], 768);
        cudaMalloc(&Norm20_CC_gamma_GPU[i], 768 * sizeof(float));
        cudaMemcpy(Norm20_CC_gamma_GPU[i], Norm20_CC_gamma_CPU[i], 768 * sizeof(float), cudaMemcpyHostToDevice);
        Norm20_CC_beta_CPU[i] = new float[768];
        initializeRandom_float(Norm20_CC_beta_CPU[i], 768);
        cudaMalloc(&Norm20_CC_beta_GPU[i], 768 * sizeof(float));
        cudaMemcpy(Norm20_CC_beta_GPU[i], Norm20_CC_beta_CPU[i], 768 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&Norm20_CC_output[i], (CC_WIDTH*768) * sizeof(float));
    }

    // Pre_Processing_End = clock();

    // Pre_Processing_Time = ((double) (Pre_Processing_End - Pre_Processing_Start)) / CLOCKS_PER_SEC;

    // printf("Pre_Processing_Time: %f s\n", Pre_Processing_Time);

    ///// Starting Computation /////

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    dim3 dimBlock(block_size1, block_size2);

    /* Layer Normalization 2 */
    dim3 dimGrid2(ceil(192/block_size1), ceil(TOTAL_WIDTH/block_size2));
    TC_FP_Normalization<<<dimGrid2,dimBlock>>>(Rearrange1_TC_output_GPU, Norm2_TC_output, Norm2_TC_gamma_GPU, Norm2_TC_beta_GPU,
                                                Rearrange1_CC_output_GPU, Norm2_CC_output, Norm2_CC_gamma_GPU, Norm2_CC_beta_GPU,
                                                768, TC_WIDTH, CC_WIDTH);

    /* Linear 3 */
    dim3 dimGrid3(ceil(TC_WIDTH/warp_size), ceil(768/warp_size));
    TC_FP_Linear<<<dimGrid3,dimBlock>>>(Norm2_TC_output, Linear3_TC_weight_GPU, Linear3_TC_output, 
                                        Norm2_CC_output, Linear3_CC_weight_GPU, Linear3_CC_output, 
                                        768, TC_WIDTH, 768,
                                        768, CC_WIDTH, 768);

    /* Bias Add 3_1 */
    dim3 dimGrid3_1(ceil(768/block_size1), 1);
    dim3 dimblock3_1(block_size1, 1);
    TC_FP_Add<<<dimGrid3_1,dimblock3_1>>>(Linear3_TC_output, Linear3_TC_bias_GPU, Linear3_1_TC_output,
                                            Linear3_CC_output, Linear3_CC_bias_GPU, Linear3_1_CC_output,
                                            TC_WIDTH, CC_WIDTH);

    /* Layer Normalization 4 */
    dim3 dimGrid4(ceil(768/block_size1), ceil(TOTAL_WIDTH/block_size2));
    TC_FP_Normalization<<<dimGrid4,dimBlock>>>(Linear3_1_TC_output, Norm4_TC_output, Norm4_TC_gamma_GPU, Norm4_TC_beta_GPU, 
                                                Linear3_1_CC_output, Norm4_CC_output, Norm4_CC_gamma_GPU, Norm4_CC_beta_GPU,
                                                768, TC_WIDTH, CC_WIDTH);

    /* Dropout 5 */
    dim3 dimGrid5(ceil(768/block_size1), ceil(TOTAL_WIDTH/block_size2));
    TC_FP_Dropout<<<dimGrid5, dimBlock>>>(Norm4_TC_output, Drop5_TC_output, 
                                            Norm4_CC_output, Drop5_CC_output, 
                                            dropout_prob, TC_WIDTH, CC_WIDTH, 768, 768);
 
    /* Layer Normalization 6 */
    dim3 dimGrid6(ceil(768/block_size1), ceil(TOTAL_WIDTH/block_size2));
    TC_FP_Normalization<<<dimGrid6,dimBlock>>>(Drop5_TC_output, Norm6_TC_output, Norm6_TC_gamma_GPU, Norm6_TC_beta_GPU, 
                                                Drop5_CC_output, Norm6_CC_output, Norm6_CC_gamma_GPU, Norm6_CC_beta_GPU, 
                                                768, TC_WIDTH, CC_WIDTH);


    //////////////////////////////// First Layer ////////////////////////////////

    for(int i = 0; i < num_iteration; ++i){
        if(i == 0){
            /* Linear 7 */
            dim3 dimGrid7(ceil(TC_WIDTH/warp_size), ceil(2304/warp_size));
            TC_FP_Linear<<<dimGrid7,dimBlock>>>(Norm6_TC_output, Linear7_TC_weight_GPU[i], Linear7_TC_output[i], 
                                                Norm6_CC_output, Linear7_CC_weight_GPU[i], Linear7_CC_output[i], 
                                                2304, TC_WIDTH, 768,
                                                2304, CC_WIDTH, 768);
        }
        else{
            /* Linear 7 */
            dim3 dimGrid7(ceil(TC_WIDTH/warp_size), ceil(2304/warp_size));
            TC_FP_Linear<<<dimGrid7,dimBlock>>>(Norm20_TC_output[i-1], Linear7_TC_weight_GPU[i], Linear7_TC_output[i], 
                                                Norm20_CC_output[i-1], Linear7_CC_weight_GPU[i], Linear7_CC_output[i], 
                                                2304, TC_WIDTH, 768,
                                                2304, CC_WIDTH, 768);
        }

        /* Bias Add 7_1 */
        dim3 dimGrid7_1(ceil(2304/block_size1), 1);
        dim3 dimblock7_1(block_size1, 1);
        TC_FP_Add<<<dimGrid7_1,dimblock7_1>>>(Linear7_TC_output[i], Linear7_TC_bias_GPU[i], Linear7_1_TC_output[i],
                                            Linear7_CC_output[i], Linear7_CC_bias_GPU[i], Linear7_1_CC_output[i],
                                            TC_WIDTH, CC_WIDTH);

        /* Softmax 8 */
        dim3 dimGrid8(ceil(2304/block_size1), ceil(TOTAL_WIDTH/block_size2));
        TC_FP_Softmax<<<dimGrid8, dimBlock>>>(Linear7_1_TC_output[i], Soft8_TC_output[i],
                                                Linear7_1_CC_output[i], Soft8_CC_output[i], 
                                                2304, TC_WIDTH, CC_WIDTH);

        /* Dropout 9 */
        dim3 dimGrid9(ceil(768/block_size1), ceil(TOTAL_WIDTH/block_size2));
        TC_FP_Dropout<<<dimGrid9, dimBlock>>>(Soft8_TC_output[i], Drop9_TC_output[i], 
                                                Soft8_CC_output[i], Drop9_CC_output[i], 
                                                dropout_prob, TC_WIDTH, CC_WIDTH, 2304, 768);

        /* Linear 10 */
        dim3 dimGrid10(ceil(TC_WIDTH/warp_size), ceil(768/warp_size));
        TC_FP_Linear<<<dimGrid10,dimBlock>>>(Drop9_TC_output[i], Linear10_TC_weight_GPU[i], Linear10_TC_output[i],
                                            Drop9_CC_output[i], Linear10_CC_weight_GPU[i], Linear10_CC_output[i],
                                            768, TC_WIDTH, 768,
                                            768, CC_WIDTH, 768);

        /* Bias Add 10_1 */
        dim3 dimGrid10_1(ceil(768/block_size1), 1);
        dim3 dimblock10_1(block_size1, 1);
        TC_FP_Add<<<dimGrid10_1,dimblock10_1>>>(Linear10_TC_output[i], Linear10_TC_bias_GPU[i], Linear10_1_TC_output[i],
                                                Linear10_CC_output[i], Linear10_CC_bias_GPU[i], Linear10_1_CC_output[i],
                                                TC_WIDTH, CC_WIDTH);

        /* Dropout 11 */
        dim3 dimGrid11(ceil(768/block_size1), ceil(TOTAL_WIDTH/block_size2));
        TC_FP_Dropout<<<dimGrid11, dimBlock>>>(Linear10_1_TC_output[i], Drop11_TC_output[i], 
                                                Linear10_1_CC_output[i], Drop11_CC_output[i], 
                                                dropout_prob, TC_WIDTH, CC_WIDTH, 768, 768);

        /* Attention */

        /* Layer Normalization 13 */
        dim3 dimGrid13(ceil(768/block_size1), ceil(TOTAL_WIDTH/block_size2));
        TC_FP_Normalization<<<dimGrid13,dimBlock>>>(Drop11_TC_output[i], Norm13_TC_output[i], Norm13_TC_gamma_GPU[i], Norm13_TC_beta_GPU[i], 
                                                    Drop11_CC_output[i], Norm13_CC_output[i], Norm13_CC_gamma_GPU[i], Norm13_CC_beta_GPU[i], 
                                                    768, TC_WIDTH, CC_WIDTH);
        
        /* Linear 14 */
        dim3 dimGrid14(ceil(TC_WIDTH/warp_size), ceil(3072/warp_size));
        TC_FP_Linear<<<dimGrid14,dimBlock>>>(Norm13_TC_output[i], Linear14_TC_weight_GPU[i], Linear14_TC_output[i], 
                                            Norm13_CC_output[i], Linear14_CC_weight_GPU[i], Linear14_CC_output[i], 
                                            3072, TC_WIDTH, 768,
                                            3072, CC_WIDTH, 768);

        /* Bias Add 14_1 */
        dim3 dimGrid14_1(ceil(3072/block_size1), 1);
        dim3 dimblock14_1(block_size1, 1);
        TC_FP_Add<<<dimGrid14_1,dimblock14_1>>>(Linear14_TC_output[i], Linear14_TC_bias_GPU[i], Linear14_1_TC_output[i],
                                                Linear14_CC_output[i], Linear14_CC_bias_GPU[i], Linear14_1_CC_output[i],
                                                TC_WIDTH, CC_WIDTH);

        /* Gelu 15 */
        dim3 dimGrid15(ceil(3072/block_size1), ceil(TOTAL_WIDTH/block_size2));
        TC_FP_Gelu<<<dimGrid15,dimBlock>>>(Linear14_1_TC_output[i], Gelu15_TC_output[i], 
                                            Linear14_1_CC_output[i], Gelu15_CC_output[i],
                                            TC_WIDTH, CC_WIDTH, 3072);

        /* Dropout 16 */
        dim3 dimGrid16(ceil(3072/block_size1), ceil(TOTAL_WIDTH/block_size2));
        TC_FP_Dropout<<<dimGrid16, dimBlock>>>(Gelu15_TC_output[i], Drop16_TC_output[i], 
                                                Gelu15_CC_output[i], Drop16_CC_output[i], 
                                                dropout_prob, TC_WIDTH, CC_WIDTH, 3072, 3072);

        /* Linear 17 */
        dim3 dimGrid17(ceil(TC_WIDTH/warp_size), ceil(768/warp_size));
        TC_FP_Linear<<<dimGrid17,dimBlock>>>(Drop16_TC_output[i], Linear17_TC_weight_GPU[i], Linear17_TC_output[i], 
                                            Drop16_CC_output[i], Linear17_CC_weight_GPU[i], Linear17_CC_output[i],
                                            768, TC_WIDTH, 3072,
                                            768, CC_WIDTH, 3072);
        
        /* Bias Add 17_1 */
        dim3 dimGrid17_1(ceil(768/block_size1), 1);
        dim3 dimblock17_1(block_size1, 1);
        TC_FP_Add<<<dimGrid17_1,dimblock17_1>>>(Linear17_TC_output[i], Linear17_TC_bias_GPU[i], Linear17_1_TC_output[i],
                                                Linear17_CC_output[i], Linear17_CC_bias_GPU[i], Linear17_1_CC_output[i],
                                                TC_WIDTH, CC_WIDTH);

        /* Dropout 18 */
        dim3 dimGrid18(ceil(768/block_size1), ceil(TOTAL_WIDTH/block_size2));
        TC_FP_Dropout<<<dimGrid18, dimBlock>>>(Linear17_1_TC_output[i], Drop18_TC_output[i], 
                                                Linear17_1_CC_output[i], Drop18_CC_output[i], 
                                                dropout_prob, TC_WIDTH, CC_WIDTH, 768, 768);

        /* Feedforward */

        /* Layer Normalization 20 */
        dim3 dimGrid20(ceil(768/block_size1), ceil(TOTAL_WIDTH/block_size2));
        TC_FP_Normalization<<<dimGrid20,dimBlock>>>(Drop18_TC_output[i], Norm20_TC_output[i], Norm20_TC_gamma_GPU[i], Norm20_TC_beta_GPU[i], 
                                                    Drop18_CC_output[i], Norm20_CC_output[i], Norm20_CC_gamma_GPU[i], Norm20_CC_beta_GPU[i], 
                                                    768, TC_WIDTH, CC_WIDTH);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Inference Time: %fms\n", milliseconds);

    // Freeing CPU Memory
    delete[] Rearrange_TC_input;
    delete[] Rearrange_CC_input;

    delete[] Norm2_TC_gamma_CPU;
    delete[] Norm2_CC_gamma_CPU;
    delete[] Norm2_TC_beta_CPU;
    delete[] Norm2_CC_beta_CPU;

    delete[] Linear3_TC_weight_CPU;
    delete[] Linear3_TC_bias_CPU;
    delete[] Linear3_CC_weight_CPU;
    delete[] Linear3_CC_bias_CPU;

    delete[] Norm4_TC_gamma_CPU;
    delete[] Norm4_TC_beta_CPU;
    delete[] Norm4_CC_gamma_CPU;
    delete[] Norm4_CC_beta_CPU;

    delete[] Norm6_TC_gamma_CPU;
    delete[] Norm6_TC_beta_CPU;
    delete[] Norm6_CC_gamma_CPU;
    delete[] Norm6_CC_beta_CPU;

    for(int i = 0; i < num_iteration; ++i) {
        delete[] Linear7_TC_weight_CPU[i];
        delete[] Linear7_TC_bias_CPU[i];
        delete[] Linear7_CC_weight_CPU[i];
        delete[] Linear7_CC_bias_CPU[i];

        delete[] Linear10_TC_weight_CPU[i];
        delete[] Linear10_TC_bias_CPU[i];
        delete[] Linear10_CC_weight_CPU[i];
        delete[] Linear10_CC_bias_CPU[i];

        delete[] Norm13_TC_gamma_CPU[i];
        delete[] Norm13_TC_beta_CPU[i];
        delete[] Norm13_CC_gamma_CPU[i];
        delete[] Norm13_CC_beta_CPU[i];

        delete[] Linear14_TC_weight_CPU[i];
        delete[] Linear14_TC_bias_CPU[i];
        delete[] Linear14_CC_weight_CPU[i];
        delete[] Linear14_CC_bias_CPU[i];

        delete[] Linear17_TC_weight_CPU[i];
        delete[] Linear17_TC_bias_CPU[i];
        delete[] Linear17_CC_weight_CPU[i];
        delete[] Linear17_CC_bias_CPU[i];

        delete[] Norm20_TC_gamma_CPU[i];
        delete[] Norm20_TC_beta_CPU[i];
        delete[] Norm20_CC_gamma_CPU[i];
        delete[] Norm20_CC_beta_CPU[i];
    }

    // Freeing GPU Memory
    cudaFree(Rearrange1_TC_output_GPU);
    cudaFree(Rearrange1_CC_output_GPU);

    cudaFree(Norm2_TC_gamma_GPU);
    cudaFree(Norm2_CC_gamma_GPU);
    cudaFree(Norm2_TC_beta_GPU);
    cudaFree(Norm2_CC_beta_GPU);
    cudaFree(Norm2_TC_output);
    cudaFree(Norm2_CC_output);

    cudaFree(Linear3_TC_weight_GPU);
    cudaFree(Linear3_TC_bias_GPU);
    cudaFree(Linear3_TC_output);
    cudaFree(Linear3_1_TC_output);
    cudaFree(Linear3_CC_weight_GPU);
    cudaFree(Linear3_CC_bias_GPU);
    cudaFree(Linear3_CC_output);
    cudaFree(Linear3_1_CC_output);

    cudaFree(Norm4_TC_gamma_GPU);
    cudaFree(Norm4_TC_beta_GPU);
    cudaFree(Norm4_TC_output);
    cudaFree(Norm4_CC_gamma_GPU);
    cudaFree(Norm4_CC_beta_GPU);
    cudaFree(Norm4_CC_output);

    cudaFree(Drop5_TC_output);
    cudaFree(Drop5_CC_output);

    cudaFree(Norm6_TC_gamma_GPU);
    cudaFree(Norm6_TC_beta_GPU);
    cudaFree(Norm6_TC_output);
    cudaFree(Norm6_CC_gamma_GPU);
    cudaFree(Norm6_CC_beta_GPU);
    cudaFree(Norm6_CC_output);

    for(int i = 0; i < num_iteration; ++i) {
        cudaFree(Linear7_TC_weight_GPU[i]);
        cudaFree(Linear7_TC_bias_GPU[i]);
        cudaFree(Linear7_TC_output[i]);
        cudaFree(Linear7_1_TC_output[i]);
        cudaFree(Linear7_CC_weight_GPU[i]);
        cudaFree(Linear7_CC_bias_GPU[i]);
        cudaFree(Linear7_CC_output[i]);
        cudaFree(Linear7_1_CC_output[i]);

        cudaFree(Soft8_TC_output[i]);
        cudaFree(Soft8_CC_output[i]);

        cudaFree(Drop9_TC_output[i]);
        cudaFree(Drop9_CC_output[i]);

        cudaFree(Linear10_TC_weight_GPU[i]);
        cudaFree(Linear10_TC_bias_GPU[i]);
        cudaFree(Linear10_TC_output[i]);
        cudaFree(Linear10_1_TC_output[i]);
        cudaFree(Linear10_CC_weight_GPU[i]);
        cudaFree(Linear10_CC_bias_GPU[i]);
        cudaFree(Linear10_CC_output[i]);
        cudaFree(Linear10_1_CC_output[i]);

        cudaFree(Drop11_TC_output[i]);
        cudaFree(Drop11_CC_output[i]);

        cudaFree(Norm13_TC_gamma_GPU[i]);
        cudaFree(Norm13_TC_beta_GPU[i]);
        cudaFree(Norm13_TC_output[i]);
        cudaFree(Norm13_CC_gamma_GPU[i]);
        cudaFree(Norm13_CC_beta_GPU[i]);
        cudaFree(Norm13_CC_output[i]);

        cudaFree(Linear14_TC_weight_GPU[i]);
        cudaFree(Linear14_TC_bias_GPU[i]);
        cudaFree(Linear14_TC_output[i]);
        cudaFree(Linear14_1_TC_output[i]);
        cudaFree(Linear14_CC_weight_GPU[i]);
        cudaFree(Linear14_CC_bias_GPU[i]);
        cudaFree(Linear14_CC_output[i]);
        cudaFree(Linear14_1_CC_output[i]);

        cudaFree(Gelu15_TC_output[i]);
        cudaFree(Gelu15_CC_output[i]);

        cudaFree(Drop16_TC_output[i]);
        cudaFree(Drop16_CC_output[i]);

        cudaFree(Linear17_TC_weight_GPU[i]);
        cudaFree(Linear17_TC_bias_GPU[i]);
        cudaFree(Linear17_TC_output[i]);
        cudaFree(Linear17_1_TC_output[i]);
        cudaFree(Linear17_CC_weight_GPU[i]);
        cudaFree(Linear17_CC_bias_GPU[i]);
        cudaFree(Linear17_CC_output[i]);
        cudaFree(Linear17_1_CC_output[i]);

        cudaFree(Drop18_TC_output[i]);
        cudaFree(Drop18_CC_output[i]);

        cudaFree(Norm20_TC_gamma_GPU[i]);
        cudaFree(Norm20_TC_beta_GPU[i]);
        cudaFree(Norm20_TC_output[i]);
        cudaFree(Norm20_CC_gamma_GPU[i]);
        cudaFree(Norm20_CC_beta_GPU[i]);
        cudaFree(Norm20_CC_output[i]);
    }

    return 0;
}
