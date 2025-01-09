#pragma once
#include <cuda.h>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <mma.h>
#include <limits>   

using namespace nvcuda;
namespace cg = cooperative_groups;

#define STAGE 2

#define warp_size 32
#define TILE_WIDTH1 32
#define TILE_WIDTH2 24

constexpr bool GEMM_OP_T = true;
constexpr bool GEMM_OP_N = false;

__device__ void TC_gemm_kernel(const int8_t* A, const int8_t* B, int8_t* C, 
                            int M, int N, int K) {
    constexpr int TC_SIZE = 16;
    constexpr int CP_SIZE_BYTES = 16;

    int warp_id = threadIdx.x/warp_size;
    int lane_id = threadIdx.x%warp_size;

    __shared__ int8_t SLB[STAGE * (warp_size*warp_size + warp_size*warp_size)];

    int8_t* smem_a[2];
    int8_t* smem_b[2];

    smem_a[0] = SLB;
    smem_a[1] = SLB + warp_size*warp_size;
    smem_b[0] = SLB + STAGE*warp_size*warp_size;
    smem_b[1] = SLB + STAGE*warp_size*warp_size + warp_size*warp_size;

    const int BCM = warp_size * blockIdx.y;
    const int BCN = warp_size * blockIdx.x;

    const int LDA = GEMM_OP_T ? K : M;
    const int LDB = GEMM_OP_N ? N : K;
    const int LDC = GEMM_OP_T ? N : M;

    const int WCM = warp_id;
    const int WCN = warp_id;

    const int BLOCK_K_LOOP = K / warp_size;

    const int8_t* BA = A + BCM * LDA;
    const int8_t* BB = B + BCN * LDB;
    int8_t* BC = C + BCM * LDC + BCN;
    int8_t* BWC = BC + WCM * warp_size * LDC + WCN * warp_size;

    constexpr int WARP_M_LOOP = warp_size / TC_SIZE;
    constexpr int WARP_N_LOOP = warp_size / TC_SIZE;
    constexpr int WARP_K_LOOP = warp_size / TC_SIZE;

    wmma::fragment<wmma::matrix_a, TC_SIZE, TC_SIZE, TC_SIZE, int8_t, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, TC_SIZE, TC_SIZE, TC_SIZE, int8_t, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, TC_SIZE, TC_SIZE, TC_SIZE, int> frag_c[WARP_M_LOOP][WARP_N_LOOP];

    #pragma unroll
    for (int i = 0; i < WARP_M_LOOP; i++) {
        #pragma unroll
        for (int j = 0; j < WARP_N_LOOP; j++) {
            wmma::fill_fragment(frag_c[i][j], 0);
        }
    }  

    constexpr int WARP_SIZE_X = 2;
    int lane_id_x = lane_id % (WARP_SIZE_X); 
    int lane_id_y = lane_id / (WARP_SIZE_X); 

    const int8_t* load_gmem_addr_a, *load_gmem_addr_b;
    int store_smem_addr_a, store_smem_addr_b;
    int k;

    k = 0;

    #pragma unroll
    for(int j = 0; j < warp_size/(CP_SIZE_BYTES*WARP_SIZE_X); j++){
        #pragma unroll
        for(int i=warp_id; i<(warp_size/TC_SIZE); i+=1)
        {
            load_gmem_addr_a = BA + (i*TC_SIZE + lane_id_y) * LDA + k*warp_size + j*(CP_SIZE_BYTES*WARP_SIZE_X) + lane_id_x*CP_SIZE_BYTES;
            store_smem_addr_a = __cvta_generic_to_shared(smem_a[k%2] + (i*TC_SIZE + lane_id_y)*warp_size + j*(CP_SIZE_BYTES*WARP_SIZE_X) + lane_id_x*CP_SIZE_BYTES);
            asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" :: "r"(store_smem_addr_a), "l"(load_gmem_addr_a), "n"(CP_SIZE_BYTES));
        }
        
        #pragma unroll
        for(int i=warp_id; i<(warp_size/TC_SIZE); i+=1)
        {
            load_gmem_addr_b = BB + (i*TC_SIZE + lane_id_y) * LDB + k*warp_size + j*(CP_SIZE_BYTES*WARP_SIZE_X) + lane_id_x*CP_SIZE_BYTES;
            store_smem_addr_b = __cvta_generic_to_shared(smem_b[k%2] + (i*TC_SIZE + lane_id_y)*warp_size + j*(CP_SIZE_BYTES*WARP_SIZE_X) + lane_id_x*CP_SIZE_BYTES);
            asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" :: "r"(store_smem_addr_b), "l"(load_gmem_addr_b), "n"(CP_SIZE_BYTES));
        }
    }

    #pragma unroll
    for(k=1; k<BLOCK_K_LOOP; k++){
      asm ("cp.async.commit_group;\n" ::);
      asm ("cp.async.wait_group 0;\n" ::);
      __syncthreads();

      #pragma unroll
      for(int j = 0; j < warp_size/(CP_SIZE_BYTES*WARP_SIZE_X); j++){
        #pragma unroll
        for(int i=warp_id; i<(warp_size/TC_SIZE); i+=1)
        {
          load_gmem_addr_a = BA + (i*TC_SIZE + lane_id_y) * LDA + k*warp_size + j*(CP_SIZE_BYTES*WARP_SIZE_X) + lane_id_x*CP_SIZE_BYTES;
          store_smem_addr_a = __cvta_generic_to_shared(smem_a[k%2] + (i*TC_SIZE + lane_id_y)*warp_size + j*(CP_SIZE_BYTES*WARP_SIZE_X) + lane_id_x*CP_SIZE_BYTES);
          asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" :: "r"(store_smem_addr_a), "l"(load_gmem_addr_a), "n"(CP_SIZE_BYTES));
        }
        
        #pragma unroll
        for(int i=warp_id; i<(warp_size/TC_SIZE); i+=1)
        {
          load_gmem_addr_b = BB + (i*TC_SIZE + lane_id_y) * LDB + k*warp_size + j*(CP_SIZE_BYTES*WARP_SIZE_X) + lane_id_x*CP_SIZE_BYTES;
          store_smem_addr_b = __cvta_generic_to_shared(smem_b[k%2] + (i*TC_SIZE + lane_id_y)*warp_size + j*(CP_SIZE_BYTES*WARP_SIZE_X) + lane_id_x*CP_SIZE_BYTES);
          asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" :: "r"(store_smem_addr_b), "l"(load_gmem_addr_b), "n"(CP_SIZE_BYTES));
        }
      }

      #pragma unroll
      for(int ki=0; ki<WARP_K_LOOP; ki++)
        #pragma unroll
        for(int yi=0; yi<WARP_M_LOOP; yi++){
            wmma::load_matrix_sync(frag_a, &smem_a[(k-1)%2][(WCM*warp_size+yi*TC_SIZE)*warp_size+ki*TC_SIZE], warp_size);
            #pragma unroll
            for(int xi=0; xi<WARP_N_LOOP; xi++){
                wmma::load_matrix_sync(frag_b, &smem_b[(k-1)%2][(WCN*warp_size+xi*TC_SIZE)*warp_size+ki*TC_SIZE], warp_size);
                wmma::mma_sync(frag_c[yi][xi], frag_a, frag_b, frag_c[yi][xi]);
            }
        }
    }

    asm ("cp.async.commit_group;\n" ::);
    asm ("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    k = BLOCK_K_LOOP -1;
    #pragma unroll
    for(int ki=0; ki<WARP_K_LOOP; ki++)
        #pragma unroll
        for(int yi=0; yi<WARP_M_LOOP; yi++){
            wmma::load_matrix_sync(frag_a, &smem_a[(k)%2][(WCM*warp_size+yi*TC_SIZE)*warp_size+ki*TC_SIZE], warp_size);
            #pragma unroll
            for(int xi=0; xi<WARP_N_LOOP; xi++){
                wmma::load_matrix_sync(frag_b, &smem_b[(k)%2][(WCN*warp_size+xi*TC_SIZE)*warp_size+ki*TC_SIZE], warp_size);
                wmma::mma_sync(frag_c[yi][xi], frag_a, frag_b, frag_c[yi][xi]);
            }
        }
      

    int gmem_lane_id_x = lane_id % 4; // [0,4]
    int gmem_lane_id_y = lane_id / 4; // [0 8]
    #pragma unroll
    for(int yi=0; yi<WARP_M_LOOP; yi++)
        #pragma unroll
        for(int xi=0; xi<WARP_N_LOOP; xi++)
        {
            for(int tc_yi=0; tc_yi<2; tc_yi++){
                for(int tc_xi=0; tc_xi<2; tc_xi++){
                    auto* store_gmem_addr = reinterpret_cast<char2*>(BWC + (yi*TC_SIZE + tc_yi*TC_SIZE/2 + gmem_lane_id_y) * LDC + xi*TC_SIZE + tc_xi*TC_SIZE/2 + gmem_lane_id_x*2);
                    char2 tmp_char2;
                    tmp_char2.x = static_cast<int8_t>(frag_c[yi][xi].x[tc_xi*4+tc_yi*2+0]); 
                    tmp_char2.y = static_cast<int8_t>(frag_c[yi][xi].x[tc_xi*4+tc_yi*2+1]);
                    *store_gmem_addr = tmp_char2; 
                }
            }
        }
}


__device__ void CC_INT_gemm_kernel(const int* CC_input_int, const int* CC_weight_int, int* CC_output_int, 
                               int CC_M, int CC_N, int CC_K) {

    __shared__ int input_shared[TILE_WIDTH2][TILE_WIDTH1];
    __shared__ int weight_shared[TILE_WIDTH2][TILE_WIDTH1];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int X_index = bx * blockDim.x + tx;
    int Y_index = by * blockDim.y + ty;

    for (int tile = 0; tile < CC_N / TILE_WIDTH1;  ++tile) {
        input_shared[ty][tx] = CC_input_int[(tile * TILE_WIDTH2 + ty) * CC_N + tx];
        weight_shared[ty][tx] = CC_weight_int[X_index * CC_N + (tile * TILE_WIDTH1 + tx)];
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH2; ++i) {
            CC_output_int[X_index * CC_N + Y_index] += weight_shared[ty][i] * input_shared[i][tx];
        }
        __syncthreads();
    }
}

__device__ void CC_FP_gemm_kernel(const float* CC_input_fp, const float* CC_weight_fp, float* CC_output_fp, 
                               int CC_M, int CC_N, int CC_K) {

    __shared__ float input_shared[TILE_WIDTH2][TILE_WIDTH1];
    __shared__ float weight_shared[TILE_WIDTH2][TILE_WIDTH1];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int X_index = bx * blockDim.x + tx;
    int Y_index = by * blockDim.y + ty;

    for (int tile = 0; tile < CC_N / TILE_WIDTH1;  ++tile) {
        input_shared[ty][tx] = CC_input_fp[(tile * TILE_WIDTH2 + ty) * CC_N + tx];
        weight_shared[ty][tx] = CC_weight_fp[X_index * CC_N + (tile * TILE_WIDTH1 + tx)];
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH2; ++i) {
            CC_output_fp[X_index * CC_N + Y_index] += weight_shared[ty][i] * input_shared[i][tx];
        }
        __syncthreads();
    }
}

__device__ void CC_INT_FP_gemm_kernel(const int* CC_input_int, const int* CC_weight_int, int* CC_output_int,
                                    const float* CC_input_fp, const float* CC_weight_fp, float* CC_output_fp, 
                                    int CC_M, int CC_N_half, int CC_K) {

    __shared__ int input_int_shared[TILE_WIDTH2][TILE_WIDTH1];
    __shared__ int weight_int_shared[TILE_WIDTH2][TILE_WIDTH1];
    __shared__ float input_fp_shared[TILE_WIDTH2][TILE_WIDTH1];
    __shared__ float weight_fp_shared[TILE_WIDTH2][TILE_WIDTH1];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int tid = (by * blockDim.y + ty) * TILE_WIDTH2 * TILE_WIDTH1 + bx * blockDim.x + tx;
    int Row = by * blockDim.y + ty;
    int Col = (bx < CC_N_half/TILE_WIDTH1) ? (bx * blockDim.x + tx) : ((bx - CC_N_half/TILE_WIDTH1) * blockDim.x + tx);
   
    if ((tid / warpSize) % 2 == 0) {
        for (int i = 0; i < CC_N_half / TILE_WIDTH1; ++i) {
            input_int_shared[threadIdx.y][threadIdx.x] = CC_input_int[(i * TILE_WIDTH2 + ty) * CC_N_half + Col];
            weight_int_shared[threadIdx.y][threadIdx.x] = CC_weight_int[Row * CC_N_half + (i * TILE_WIDTH1 + tx)];
            __syncthreads();

            for (int j = 0; j < TILE_WIDTH2; ++j) {
                CC_output_int[Row * CC_N_half + Col] += weight_int_shared[ty][j] * input_int_shared[j][tx];
            }
            __syncthreads();
        }
    }
    else {
        for (int i = 0; i < CC_N_half / TILE_WIDTH1; ++i) {
            input_fp_shared[threadIdx.y][threadIdx.x] = CC_input_fp[(i * TILE_WIDTH2 + ty) * CC_N_half + Col];
            weight_fp_shared[threadIdx.y][threadIdx.x] = CC_weight_fp[Row * CC_N_half + (i * TILE_WIDTH1 + tx)];
            __syncthreads();

            for (int j = 0; j < TILE_WIDTH2; ++j) {
                CC_output_fp[Row * CC_N_half + Col] += weight_fp_shared[ty][j] * input_fp_shared[j][tx];
            }
            __syncthreads();
        }
    }
}

__device__ void CC_VitBit_gemm_kernel(const int* CC_input_int, const int* CC_weight_int, int* CC_output_int,
                                    const float* CC_input_fp, const float* CC_weight_fp, float* CC_output_fp, 
                                    int CC_M, int CC_N_VB, int CC_K) {

    __shared__ int input_int_shared[TILE_WIDTH2][TILE_WIDTH1];
    __shared__ int weight_int_shared[TILE_WIDTH2][TILE_WIDTH1];
    __shared__ float input_fp_shared[TILE_WIDTH2][TILE_WIDTH1];
    __shared__ float weight_fp_shared[TILE_WIDTH2][TILE_WIDTH1];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int tid = (by * blockDim.y + ty) * TILE_WIDTH1 * TILE_WIDTH2 + bx * blockDim.x + tx;
    int Row = by * blockDim.y + ty;
    int Col = (bx < CC_N_VB/TILE_WIDTH1) ? (bx * blockDim.x + tx) : ((bx - CC_N_VB/TILE_WIDTH1) * blockDim.x + tx);
   
    if ((tid / warpSize) % 2 == 0) {
        for (int i = 0; i < CC_N_VB / TILE_WIDTH1; ++i) {
            input_int_shared[threadIdx.y][threadIdx.x] = CC_input_int[(i * TILE_WIDTH2 + ty) * CC_N_VB + Col];
            weight_int_shared[threadIdx.y][threadIdx.x] = CC_weight_int[Row * CC_N_VB + (i * TILE_WIDTH1 + tx)];
            __syncthreads();

            for (int j = 0; j < TILE_WIDTH2; ++j) {
                CC_output_int[Row * CC_N_VB + Col] += weight_int_shared[ty][j] * input_int_shared[j][tx];
            }
            __syncthreads();
        }
    }
    else {
        for (int i = 0; i < CC_N_VB / TILE_WIDTH1; ++i) {
            input_fp_shared[threadIdx.y][threadIdx.x] = CC_input_fp[(i * TILE_WIDTH2 + ty) * CC_N_VB + Col];
            weight_fp_shared[threadIdx.y][threadIdx.x] = CC_weight_fp[Row * CC_N_VB + (i * TILE_WIDTH1 + tx)];
            __syncthreads();

            for (int j = 0; j < TILE_WIDTH2; ++j) {
                CC_output_fp[Row * CC_N_VB + Col] += weight_fp_shared[ty][j] * input_fp_shared[j][tx];
            }
            __syncthreads();
        }
    }
}
