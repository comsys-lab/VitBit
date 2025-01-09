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

#include "Only_TC_Functions.cuh"

#define warp_size 32
#define block_size1 32
#define block_size2 24

#define WIDTH 784

void initializeRandom_int8(int8_t* array, int size) {
    for (int i = 0; i < size; ++i) {
        array[i] = rand() % 11;
    }
}

void initializeRandom_float(float* array, int size) {
    for (int i = 0; i < size; ++i) {
        array[i] = static_cast<float>(rand()) / RAND_MAX; // 0부터 1까지의 랜덤값
    }
}

void rearrange(int8_t *input, int8_t *output, int output_rows, int output_cols) {
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
    int8_t *Rearrange_input = new int8_t[224*224*3];
    initializeRandom_int8(Rearrange_input, 224*224*3);

    /* Measuring Preprocessng Time*/
    // clock_t Pre_Processing_Start, Pre_Processing_End;
    // double Pre_Processing_Time;

    // Pre_Processing_Start = clock();

    int8_t *Rearrange1_output_GPU;
    cudaMalloc(&Rearrange1_output_GPU, (WIDTH*192) * sizeof(int8_t));
    cudaMemcpy(Rearrange1_output_GPU, Rearrange_input, (WIDTH*192) * sizeof(int8_t), cudaMemcpyHostToDevice);

    /* Layer Normalization 2 */
    int8_t *Norm2_gamma_CPU = new int8_t[192];
    initializeRandom_int8(Norm2_gamma_CPU, 192);
    int8_t *Norm2_gamma_GPU;
    cudaMalloc(&Norm2_gamma_GPU, 192 * sizeof(int8_t));
    cudaMemcpy(Norm2_gamma_GPU, Norm2_gamma_CPU, 192 * sizeof(int8_t), cudaMemcpyHostToDevice);

    int8_t *Norm2_beta_CPU = new int8_t[192];
    initializeRandom_int8(Norm2_beta_CPU, 192);
    int8_t *Norm2_beta_GPU;
    cudaMalloc(&Norm2_beta_GPU, 192 * sizeof(int8_t));
    cudaMemcpy(Norm2_beta_GPU, Norm2_beta_CPU, 192 * sizeof(int8_t), cudaMemcpyHostToDevice);

    int8_t *Norm2_output;
    cudaMalloc(&Norm2_output, (WIDTH*192) * sizeof(int8_t));

    /* Linear 3 */
    int8_t *Linear3_weight_CPU = new int8_t[192 * 768];
    initializeRandom_int8(Linear3_weight_CPU, 192 * 768);
    int8_t *Linear3_weight_GPU;
    cudaMalloc(&Linear3_weight_GPU, (192 * 768) * sizeof(int8_t));
    cudaMemcpy(Linear3_weight_GPU, Linear3_weight_CPU, (192 * 768) * sizeof(int8_t), cudaMemcpyHostToDevice);

    int8_t *Linear3_bias_CPU = new int8_t[768];
    initializeRandom_int8(Linear3_bias_CPU, 768);
    int8_t *Linear3_bias_GPU;
    cudaMalloc(&Linear3_bias_GPU, 768 * sizeof(int8_t));
    cudaMemcpy(Linear3_bias_GPU, Linear3_bias_CPU, 768 * sizeof(int8_t), cudaMemcpyHostToDevice);

    int8_t *Linear3_output;
    cudaMalloc(&Linear3_output, (WIDTH*768) * sizeof(int8_t));
    int8_t *Linear3_1_output;
    cudaMalloc(&Linear3_1_output, (WIDTH*768) * sizeof(int8_t));

    /* Layer Normalization 4 */
    int8_t *Norm4_gamma_CPU = new int8_t[768];
    initializeRandom_int8(Norm4_gamma_CPU, 768);
    int8_t *Norm4_gamma_GPU;
    cudaMalloc(&Norm4_gamma_GPU, 768 * sizeof(int8_t));
    cudaMemcpy(Norm4_gamma_GPU, Norm4_gamma_CPU, 768 * sizeof(int8_t), cudaMemcpyHostToDevice);

    int8_t *Norm4_beta_CPU = new int8_t[768];
    initializeRandom_int8(Norm4_beta_CPU, 768);
    int8_t *Norm4_beta_GPU;
    cudaMalloc(&Norm4_beta_GPU, 768 * sizeof(int8_t));
    cudaMemcpy(Norm4_beta_GPU, Norm4_beta_CPU, 768 * sizeof(int8_t), cudaMemcpyHostToDevice);

    int8_t *Norm4_output;
    cudaMalloc(&Norm4_output, (WIDTH*768) * sizeof(int8_t));

    /* Dropout 5 */
    int8_t *Drop5_output;
    cudaMalloc(&Drop5_output, (WIDTH*768) * sizeof(int8_t));

    const float dropout_prob = 0.5f;

    /* Layer Normalization 6 */
    int8_t *Norm6_gamma_CPU = new int8_t[768];
    initializeRandom_int8(Norm6_gamma_CPU, 768);
    int8_t *Norm6_gamma_GPU;
    cudaMalloc(&Norm6_gamma_GPU, 768 * sizeof(int8_t));
    cudaMemcpy(Norm6_gamma_GPU, Norm6_gamma_CPU, 768 * sizeof(int8_t), cudaMemcpyHostToDevice);

    int8_t *Norm6_beta_CPU = new int8_t[768];
    initializeRandom_int8(Norm6_beta_CPU, 768);
    int8_t *Norm6_beta_GPU;
    cudaMalloc(&Norm6_beta_GPU, 768 * sizeof(int8_t));
    cudaMemcpy(Norm6_beta_GPU, Norm6_beta_CPU, 768 * sizeof(int8_t), cudaMemcpyHostToDevice);

    int8_t *Norm6_output;
    cudaMalloc(&Norm6_output, (WIDTH*768) * sizeof(int8_t));

    //// Iteration Start
    int num_iteration = 12;

    /* Linear 7 */
    int8_t *Linear7_weight_CPU[num_iteration];
    int8_t *Linear7_weight_GPU[num_iteration];
    int8_t *Linear7_bias_CPU[num_iteration];
    int8_t *Linear7_bias_GPU[num_iteration];
    int8_t *Linear7_output[num_iteration];
    int8_t *Linear7_1_output[num_iteration];

    /* Softmax 8 */
    int8_t *Soft8_output[num_iteration];

    /* Dropout 9 */
    int8_t *Drop9_output[num_iteration];

    /* Linear 10 */
    int8_t *Linear10_weight_CPU[num_iteration];
    int8_t *Linear10_weight_GPU[num_iteration];
    int8_t *Linear10_bias_CPU[num_iteration];
    int8_t *Linear10_bias_GPU[num_iteration];
    int8_t *Linear10_output[num_iteration];
    int8_t *Linear10_1_output[num_iteration];

    /* Dropout 11 */
    int8_t *Drop11_output[num_iteration];

    /* Layer Normalization 13 */
    int8_t *Norm13_gamma_CPU[num_iteration];
    int8_t *Norm13_gamma_GPU[num_iteration];
    int8_t *Norm13_beta_CPU[num_iteration];
    int8_t *Norm13_beta_GPU[num_iteration];
    int8_t *Norm13_output[num_iteration];

    /* Linear 14 */
    int8_t *Linear14_weight_CPU[num_iteration];
    int8_t *Linear14_weight_GPU[num_iteration];
    int8_t *Linear14_bias_CPU[num_iteration];
    int8_t *Linear14_bias_GPU[num_iteration];
    int8_t *Linear14_output[num_iteration];
    int8_t *Linear14_1_output[num_iteration];

    /* Gelu 15 */
    int8_t *Gelu15_output[num_iteration];

    /* Dropout 16 */
    int8_t *Drop16_output[num_iteration];

    /* Linear 17 */
    int8_t *Linear17_weight_CPU[num_iteration];
    int8_t *Linear17_weight_GPU[num_iteration];
    int8_t *Linear17_bias_CPU[num_iteration];
    int8_t *Linear17_bias_GPU[num_iteration];
    int8_t *Linear17_output[num_iteration];
    int8_t *Linear17_1_output[num_iteration];

    /* Dropout 18 */
    int8_t *Drop18_output[num_iteration];

    /* Layer Normalization 20 */
    int8_t *Norm20_gamma_CPU[num_iteration];
    int8_t *Norm20_gamma_GPU[num_iteration];
    int8_t *Norm20_beta_CPU[num_iteration];
    int8_t *Norm20_beta_GPU[num_iteration];
    int8_t *Norm20_output[num_iteration];
    
    for(int i = 0; i < num_iteration; ++i){
        /* Linear 7 */
        Linear7_weight_CPU[i] = new int8_t[768 * 2304];
        initializeRandom_int8(Linear7_weight_CPU[i], 768 * 2304);
        
        cudaMalloc(&Linear7_weight_GPU[i], (768 * 2304) * sizeof(int8_t));
        cudaMemcpy(Linear7_weight_GPU[i], Linear7_weight_CPU[i], (768 * 2304) * sizeof(int8_t), cudaMemcpyHostToDevice);

        Linear7_bias_CPU[i] = new int8_t[2304];
        initializeRandom_int8(Linear7_bias_CPU[i], 2304);
        
        cudaMalloc(&Linear7_bias_GPU[i], 2304 * sizeof(int8_t));
        cudaMemcpy(Linear7_bias_GPU[i], Linear7_bias_CPU[i], 2304 * sizeof(int8_t), cudaMemcpyHostToDevice);

        
        cudaMalloc(&Linear7_output[i], (WIDTH*2304) * sizeof(int8_t));
        cudaMalloc(&Linear7_1_output[i], (WIDTH*2304) * sizeof(int8_t));

        /* Softmax 8 */
        cudaMalloc(&Soft8_output[i], (WIDTH*2304) * sizeof(int8_t));

        /* Dropout 9 */
        cudaMalloc(&Drop9_output[i], (WIDTH*768) * sizeof(int8_t));

        /* Linear 10 */
        Linear10_weight_CPU[i] = new int8_t[768 * 768];
        initializeRandom_int8(Linear10_weight_CPU[i], 768 * 768);
       
        cudaMalloc(&Linear10_weight_GPU[i], (768 * 768) * sizeof(int8_t));
        cudaMemcpy(Linear10_weight_GPU[i], Linear10_weight_CPU[i], (768 * 768) * sizeof(int8_t), cudaMemcpyHostToDevice);

        Linear10_bias_CPU[i] = new int8_t[768];
        initializeRandom_int8(Linear10_bias_CPU[i], 768);
        
        cudaMalloc(&Linear10_bias_GPU[i], 768 * sizeof(int8_t));
        cudaMemcpy(Linear10_bias_GPU[i], Linear10_bias_CPU[i], 768 * sizeof(int8_t), cudaMemcpyHostToDevice);

        cudaMalloc(&Linear10_output[i], (WIDTH*768) * sizeof(int8_t));
        cudaMalloc(&Linear10_1_output[i], (WIDTH*768) * sizeof(int8_t));

        /* Dropout 11 */
        cudaMalloc(&Drop11_output[i], (WIDTH*768) * sizeof(int8_t));

        /* Layer Normalization 13 */
        Norm13_gamma_CPU[i] = new int8_t[768];
        initializeRandom_int8(Norm13_gamma_CPU[i], 768);

        cudaMalloc(&Norm13_gamma_GPU[i], 768 * sizeof(int8_t));
        cudaMemcpy(Norm13_gamma_GPU[i], Norm13_gamma_CPU[i], 768 * sizeof(int8_t), cudaMemcpyHostToDevice);

        Norm13_beta_CPU[i] = new int8_t[768];
        initializeRandom_int8(Norm13_beta_CPU[i], 768);
        
        cudaMalloc(&Norm13_beta_GPU[i], 768 * sizeof(int8_t));
        cudaMemcpy(Norm13_beta_GPU[i], Norm13_beta_CPU[i], 768 * sizeof(int8_t), cudaMemcpyHostToDevice);

        cudaMalloc(&Norm13_output[i], (WIDTH*768) * sizeof(int8_t));

        /* Linear 14 */
        Linear14_weight_CPU[i] = new int8_t[768 * 3072];
        initializeRandom_int8(Linear14_weight_CPU[i], 768 * 3072);

        cudaMalloc(&Linear14_weight_GPU[i], (768 * 3072) * sizeof(int8_t));
        cudaMemcpy(Linear14_weight_GPU[i], Linear14_weight_CPU[i], (768 * 3072) * sizeof(int8_t), cudaMemcpyHostToDevice);

        Linear14_bias_CPU[i] = new int8_t[3072];
        initializeRandom_int8(Linear14_bias_CPU[i], 3072);

        cudaMalloc(&Linear14_bias_GPU[i], 3072 * sizeof(int8_t));
        cudaMemcpy(Linear14_bias_GPU[i], Linear14_bias_CPU[i], 3072 * sizeof(int8_t), cudaMemcpyHostToDevice);

        cudaMalloc(&Linear14_output[i], (WIDTH*3072) * sizeof(int8_t));
        cudaMalloc(&Linear14_1_output[i], (WIDTH*3072) * sizeof(int8_t));

        /* Gelu 15 */
        cudaMalloc(&Gelu15_output[i], (WIDTH*3072) * sizeof(int8_t));

        /* Dropout 16 */;
        cudaMalloc(&Drop16_output[i], (WIDTH*3072) * sizeof(int8_t));

        /* Linear 17 */
        Linear17_weight_CPU[i] = new int8_t[768 * 3072];
        initializeRandom_int8(Linear17_weight_CPU[i], 768 * 3072);

        cudaMalloc(&Linear17_weight_GPU[i], (768 * 3072) * sizeof(int8_t));
        cudaMemcpy(Linear17_weight_GPU[i], Linear17_weight_CPU[i], (768 * 3072) * sizeof(int8_t), cudaMemcpyHostToDevice);

        Linear17_bias_CPU[i] = new int8_t[768];
        initializeRandom_int8(Linear17_bias_CPU[i], 768);

        cudaMalloc(&Linear17_bias_GPU[i], 3072 * sizeof(int8_t));
        cudaMemcpy(Linear17_bias_GPU[i], Linear17_bias_CPU[i], 768 * sizeof(int8_t), cudaMemcpyHostToDevice);

        cudaMalloc(&Linear17_output[i], (WIDTH*768) * sizeof(int8_t));
        cudaMalloc(&Linear17_1_output[i], (WIDTH*768) * sizeof(int8_t));

        /* Dropout 18 */
        cudaMalloc(&Drop18_output[i], (WIDTH*768) * sizeof(int8_t));

        /* Layer Normalization 20 */
        Norm20_gamma_CPU[i] = new int8_t[768];
        initializeRandom_int8(Norm20_gamma_CPU[i], 768);

        cudaMalloc(&Norm20_gamma_GPU[i], 768 * sizeof(int8_t));
        cudaMemcpy(Norm20_gamma_GPU[i], Norm20_gamma_CPU[i], 768 * sizeof(int8_t), cudaMemcpyHostToDevice);

        Norm20_beta_CPU[i] = new int8_t[768];
        initializeRandom_int8(Norm20_beta_CPU[i], 768);

        cudaMalloc(&Norm20_beta_GPU[i], 768 * sizeof(int8_t));
        cudaMemcpy(Norm20_beta_GPU[i], Norm20_beta_CPU[i], 768 * sizeof(int8_t), cudaMemcpyHostToDevice);

        cudaMalloc(&Norm20_output[i], (WIDTH*768) * sizeof(int8_t));
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
    dim3 dimGrid2(ceil(192/block_size1), ceil(WIDTH/block_size2));
    Only_TC_Normalization<<<dimGrid2,dimBlock>>>(Rearrange1_output_GPU, Norm2_output, Norm2_gamma_GPU, Norm2_beta_GPU, 768, WIDTH);

    /* Linear 3 */
    dim3 dimGrid3(ceil(WIDTH/warp_size), ceil(768/warp_size));
    Only_TC_Linear<<<dimGrid3,dimBlock>>>(Norm2_output, Linear3_weight_GPU, Linear3_output, 768, WIDTH, 768);

    /* Bias Add 3_1 */
    dim3 dimGrid3_1(ceil(768/block_size1), 1);
    dim3 dimblock3_1(block_size1, 1);
    Only_TC_Add<<<dimGrid3_1,dimblock3_1>>>(Linear3_output, Linear3_bias_GPU, Linear3_1_output, WIDTH);

    /* Layer Normalization 4 */
    dim3 dimGrid4(ceil(768/block_size1), ceil(WIDTH/block_size2));
    Only_TC_Normalization<<<dimGrid4,dimBlock>>>(Linear3_1_output, Norm4_output, Norm4_gamma_GPU, Norm4_beta_GPU, 768, WIDTH);

    /* Dropout 5 */
    dim3 dimGrid5(ceil(768/block_size1), ceil(WIDTH/block_size2));
    Only_TC_Dropout<<<dimGrid5, dimBlock>>>(Norm4_output, Drop5_output, dropout_prob, WIDTH, 768, 768);
 
    /* Layer Normalization 6 */
    dim3 dimGrid6(ceil(768/block_size1), ceil(WIDTH/block_size2));
    Only_TC_Normalization<<<dimGrid6,dimBlock>>>(Drop5_output, Norm6_output, Norm6_gamma_GPU, Norm6_beta_GPU, 768, WIDTH);


    //////////////////////////////// First Layer ////////////////////////////////

    for(int i = 0; i < num_iteration; ++i){
        if(i == 0){
            /* Linear 7 */
            dim3 dimGrid7(ceil(WIDTH/warp_size), ceil(2304/warp_size));
            Only_TC_Linear<<<dimGrid7,dimBlock>>>(Norm6_output, Linear7_weight_GPU[i], Linear7_output[i], 2304, WIDTH, 768);
        }
        else{
            /* Linear 7 */
            dim3 dimGrid7(ceil(WIDTH/warp_size), ceil(2304/warp_size));
            Only_TC_Linear<<<dimGrid7,dimBlock>>>(Norm20_output[i-1], Linear7_weight_GPU[i], Linear7_output[i], 2304, WIDTH, 768);
        }

        /* Bias Add 7_1 */
        dim3 dimGrid7_1(ceil(2304/block_size1), 1);
        dim3 dimblock7_1(block_size1, 1);
        Only_TC_Add<<<dimGrid7_1,dimblock7_1>>>(Linear7_output[i], Linear7_bias_GPU[i], Linear7_1_output[i],WIDTH);

        /* Softmax 8 */
        dim3 dimGrid8(ceil(2304/block_size1), ceil(WIDTH/block_size2));
        Only_TC_Softmax<<<dimGrid8, dimBlock>>>(Linear7_1_output[i], Soft8_output[i], WIDTH, 2304);

        /* Dropout 9 */
        dim3 dimGrid9(ceil(768/block_size1), ceil(WIDTH/block_size2));
        Only_TC_Dropout<<<dimGrid9, dimBlock>>>(Soft8_output[i], Drop9_output[i], dropout_prob, WIDTH, 2304, 768);

        /* Linear 10 */
        dim3 dimGrid10(ceil(WIDTH/warp_size), ceil(768/warp_size));
        Only_TC_Linear<<<dimGrid10,dimBlock>>>(Drop9_output[i], Linear10_weight_GPU[i], Linear10_output[i], 768, WIDTH, 768);

        /* Bias Add 10_1 */
        dim3 dimGrid10_1(ceil(768/block_size1), 1);
        dim3 dimblock10_1(block_size1, 1);
        Only_TC_Add<<<dimGrid10_1,dimblock10_1>>>(Linear10_output[i], Linear10_bias_GPU[i], Linear10_1_output[i], WIDTH);

        /* Dropout 11 */
        dim3 dimGrid11(ceil(768/block_size1), ceil(WIDTH/block_size2));
        Only_TC_Dropout<<<dimGrid11, dimBlock>>>(Linear10_1_output[i], Drop11_output[i], dropout_prob, WIDTH, 768, 768);

        /* Attention */

        /* Layer Normalization 13 */
        dim3 dimGrid13(ceil(768/block_size1), ceil(WIDTH/block_size2));
        Only_TC_Normalization<<<dimGrid13,dimBlock>>>(Drop11_output[i], Norm13_output[i], Norm13_gamma_GPU[i], Norm13_beta_GPU[i], 768, WIDTH);


        /* Linear 14 */
        dim3 dimGrid14(ceil(WIDTH/warp_size), (3072/warp_size));
        Only_TC_Linear<<<dimGrid14,dimBlock>>>(Norm13_output[i], Linear14_weight_GPU[i], Linear14_output[i], 3072, WIDTH, 768);

        /* Bias Add 14_1 */
        dim3 dimGrid14_1(ceil(3072/block_size1), 1);
        dim3 dimblock14_1(block_size1, 1);
        Only_TC_Add<<<dimGrid14_1,dimblock14_1>>>(Linear14_output[i], Linear14_bias_GPU[i], Linear14_1_output[i], WIDTH);

        /* Gelu 15 */
        dim3 dimGrid15(ceil(3072/block_size1), ceil(WIDTH/block_size2));
        Only_TC_Gelu<<<dimGrid15,dimBlock>>>(Linear14_1_output[i], Gelu15_output[i], WIDTH);

        /* Dropout 16 */
        dim3 dimGrid16(ceil(3072/block_size1), ceil(WIDTH/block_size2));
        Only_TC_Dropout<<<dimGrid16, dimBlock>>>(Gelu15_output[i], Drop16_output[i], dropout_prob, WIDTH, 3072, 3072);

        /* Linear 17 */
        dim3 dimGrid17(ceil(WIDTH/warp_size), ceil(768/warp_size));
        Only_TC_Linear<<<dimGrid17,dimBlock>>>(Drop16_output[i], Linear17_weight_GPU[i], Linear17_output[i], 768, WIDTH, 3072);

        /* Bias Add 17_1 */
        dim3 dimGrid17_1(ceil(768/block_size1), 1);
        dim3 dimblock17_1(block_size1, 1);
        Only_TC_Add<<<dimGrid17_1,dimblock17_1>>>(Linear17_output[i], Linear17_bias_GPU[i], Linear17_1_output[i], WIDTH);

        /* Dropout 18 */
        dim3 dimGrid18(ceil(768/block_size1), ceil(WIDTH/block_size2));
        Only_TC_Dropout<<<dimGrid18, dimBlock>>>(Linear17_1_output[i], Drop18_output[i], dropout_prob, WIDTH, 768, 768);

        /* Feedforward */

        /* Layer Normalization 20 */
        dim3 dimGrid20(ceil(768/block_size1), ceil(WIDTH/block_size2));
        Only_TC_Normalization<<<dimGrid20,dimBlock>>>(Drop18_output[i], Norm20_output[i], Norm20_gamma_GPU[i], Norm20_beta_GPU[i], 768, WIDTH);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Inference Time: %fms\n", milliseconds);

    // Freeing CPU Memory
    delete[] Rearrange_input;

    delete[] Norm2_gamma_CPU;
    delete[] Norm2_beta_CPU;
    delete[] Linear3_weight_CPU;
    delete[] Linear3_bias_CPU;
    delete[] Norm4_gamma_CPU;
    delete[] Norm4_beta_CPU;
    delete[] Norm6_gamma_CPU;
    delete[] Norm6_beta_CPU;

    for(int i = 0; i < num_iteration; ++i) {
        delete[] Linear7_weight_CPU[i];
        delete[] Linear7_bias_CPU[i];
        delete[] Linear10_weight_CPU[i];
        delete[] Linear10_bias_CPU[i];
        delete[] Norm13_gamma_CPU[i];
        delete[] Norm13_beta_CPU[i];
        delete[] Linear14_weight_CPU[i];
        delete[] Linear14_bias_CPU[i];
        delete[] Linear17_weight_CPU[i];
        delete[] Linear17_bias_CPU[i];
        delete[] Norm20_gamma_CPU[i];
        delete[] Norm20_beta_CPU[i];
    }

    // Freeing GPU Memory
    cudaFree(Rearrange1_output_GPU);

    cudaFree(Norm2_gamma_GPU);
    cudaFree(Norm2_beta_GPU);
    cudaFree(Norm2_output);

    cudaFree(Linear3_weight_GPU);
    cudaFree(Linear3_bias_GPU);
    cudaFree(Linear3_output);
    cudaFree(Linear3_1_output);

    cudaFree(Norm4_gamma_GPU);
    cudaFree(Norm4_beta_GPU);
    cudaFree(Norm4_output);

    cudaFree(Drop5_output);

    cudaFree(Norm6_gamma_GPU);
    cudaFree(Norm6_beta_GPU);
    cudaFree(Norm6_output);

    for(int i = 0; i < num_iteration; ++i) {
        cudaFree(Linear7_weight_GPU[i]);
        cudaFree(Linear7_bias_GPU[i]);
        cudaFree(Linear7_output[i]);
        cudaFree(Linear7_1_output[i]);

        cudaFree(Soft8_output[i]);

        cudaFree(Drop9_output[i]);

        cudaFree(Linear10_weight_GPU[i]);
        cudaFree(Linear10_bias_GPU[i]);
        cudaFree(Linear10_output[i]);
        cudaFree(Linear10_1_output[i]);

        cudaFree(Drop11_output[i]);

        cudaFree(Norm13_gamma_GPU[i]);
        cudaFree(Norm13_beta_GPU[i]);
        cudaFree(Norm13_output[i]);

        cudaFree(Linear14_weight_GPU[i]);
        cudaFree(Linear14_bias_GPU[i]);
        cudaFree(Linear14_output[i]);
        cudaFree(Linear14_1_output[i]);

        cudaFree(Gelu15_output[i]);

        cudaFree(Drop16_output[i]);

        cudaFree(Linear17_weight_GPU[i]);
        cudaFree(Linear17_bias_GPU[i]);
        cudaFree(Linear17_output[i]);
        cudaFree(Linear17_1_output[i]);

        cudaFree(Drop18_output[i]);

        cudaFree(Norm20_gamma_GPU[i]);
        cudaFree(Norm20_beta_GPU[i]);
        cudaFree(Norm20_output[i]);
    }

    return 0;
}
