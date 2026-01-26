#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#include <math.h>
#include <string.h>
#include <stddef.h>

#ifndef LN_EPS
#define LN_EPS 1e-5f
#endif





#ifndef LN_EPS
#define LN_EPS 1e-5f
#endif



int min(int a, int b){
    return a < b ? a : b;
}

void transpose_matrix(const float *src_matrix, float *dest_matrix, size_t src_num_rows, size_t src_num_cols){
    for (size_t idx_row = 0; idx_row < src_num_rows; idx_row++){
        for (size_t idx_col = 0; idx_col < src_num_cols; idx_col++){
            dest_matrix[idx_col * src_num_rows + idx_row] = src_matrix[idx_row * src_num_cols + idx_col];
        }
    }
}


void naive_matmul(const float* A, const float *B, float * C, size_t m, size_t n, size_t k){
    // transpose B to col-major
    float* B_transposed = (float*)malloc(k * n * sizeof(float));
    transpose_matrix(B, B_transposed, k, n);
    for (size_t idx_m = 0; idx_m < m; idx_m++ ){
        for (size_t idx_n = 0; idx_n < n; idx_n++){
            for (size_t idx_k = 0; idx_k < k; idx_k++){            
                // C[idx_m][idx_n] = A[idx_m][idx_k] * B[idx_k][idx_n]
                C[idx_m * n + idx_n] += A[idx_m * k + idx_k] * B_transposed[idx_k + idx_n * k];
            }
        }
    }
}







/*

m = 3 n= 4 k = 2

1 10    1 1 1 1
2 20    2 2 2 2
3 30    

21 21 21 21
42 42 42 42
63 63 63 63

sum of outer products

first column by first row:
1  (x)   1 1 1 1
2
3

=

1 1 1 1
2 2 2 2
3 3 3 3



second column by second row:

10  (x)   2 2 2 2
20
30

20 20 20 20
40 40 40 40
60 60 60 60
 


sum of outer products:

1 1 1 1   20 20 20 20
2 2 2 2 + 40 40 40 40
3 3 3 3   60 60 60 60
=
21 21 21 21
42 42 42 42
63 63 63 63


*/

void outer_product_matmul(const float * a, const float * b, float * c, size_t m, size_t n, size_t k ){
    for (size_t idx_k = 0; idx_k < k; idx_k++){
        for (size_t idx_m = 0; idx_m < m; idx_m++){
            for (size_t idx_n = 0; idx_n < n; idx_n++){
                // size_t offset_a = idx_k * m  + idx_m; // a[m][k] col major 
                // size_t offset_b = idx_k * n + idx_n;  // b[k][n]  row major
                // size_t offset_c = idx_m * n + idx_n; // row major
                // c[offset_c] += a[offset_a] * b[offset_b];
                c[idx_m * n +idx_n] += a[idx_k * m +idx_m] * b[idx_k * n + idx_n];
            }
        }
    }
}

void tiled_matmul(const float * A, const float * B, float * C, size_t m, size_t n, size_t k, size_t size_tile){
    for (size_t tile_start_m = 0; tile_start_m < m; tile_start_m += size_tile){
        for (size_t tile_start_n = 0; tile_start_n < n; tile_start_n += size_tile){
            for (size_t tile_start_k = 0; tile_start_k < k; tile_start_k += size_tile){


                for (size_t idx_m = tile_start_m; idx_m < tile_start_m + size_tile && idx_m < m; idx_m++){
                    for (size_t idx_n = tile_start_n; idx_n < tile_start_n + size_tile && idx_n < n; idx_n++){
                        float sum = 0.0;
                        // each k (col of A and row of B)
                        size_t offset_a = idx_m * k;
                        size_t offset_b = idx_n * n;
                        for (size_t idx_k = tile_start_k; idx_k < tile_start_k + size_tile && idx_k < k; idx_k++){
                            sum += A[offset_a + idx_k] * B[idx_k + offset_b];
                            }
                        C[idx_m * n + idx_n] += sum;
                    }
                }


            }
        }
    }
}

/*
Naïve: 
Each element of the result matrix is equal to the dot product of the row vector of A and column vector of B. 
	- C[0][0] = A[0][] * B[][0]
	- C[0][1] = A[0][] * B[][1]
	- C[0][n] = A[0][] * B[][n]

    - C[1][0] = A[1][] * B[][0]
    - C[1][1] = A[1][] * B[][1]
    - C[1][n] = A[1][] * B[][n]

    - C[m][0] = A[m][] * B[][0]
    - C[m][1] = A[m][] * B[][1]
    - C[m][n] = A[m][] * B[][n]


Memory Layout with Large Matrices:
If A, B and C are all row major then when obtaining the dot product of A's row by B's column the A row vector will be contiguous in memory but B elements will be spaced M elements apart and M will be larger than the cache line leading to more misses and fetches from slower memories. So A should be row major while B should be column major.

Tiling:
In the naïve approach each element of C is calculated from the dot product of its row vector in A and its column vector in B. 
Elements are calculated one by one. Let us say we have matrices all of 1024 x1024 floats.  
With the naïve approach, by the time C[0][1] is being calculated there is a strong chance that the intial elements of A[0][]
have been evicted during the fetching of elements of B.

*/


void l1_tiled_matmul(const float * A, const float * B, float * C, size_t m, size_t n, size_t k, size_t size_outer_tile, size_t size_inner_tile){
    for (size_t idx_m = 0; idx_m < m; idx_m += size_outer_tile){
        for (size_t idx_n = 0; idx_n < n; idx_n += size_outer_tile){
            for (size_t idx_k = 0; idx_k < k; idx_k += size_outer_tile){
                
                for (size_t idx_mm = idx_m; idx_mm < idx_m + size_outer_tile && idx_mm < m; idx_mm += size_inner_tile){
                    for (size_t idx_nn = idx_n; idx_nn < idx_n + size_outer_tile && idx_nn < n; idx_nn += size_inner_tile){
                        for (size_t idx_kk = idx_k; idx_kk < idx_k + size_outer_tile && idx_kk < k; idx_kk += size_inner_tile){

                            for (size_t idx_mmm = idx_mm; idx_mmm < idx_mm + size_inner_tile && idx_mmm < m; idx_mmm++){
                                for (size_t idx_nnn = idx_nn; idx_nnn < idx_nn + size_inner_tile && idx_nnn < n; idx_nnn++){
                                    float sum = 0;
                                    size_t offset_a = idx_mmm * k; // A[idx_mmm][0] row major
                                    size_t offset_b = idx_nnn * k; // B[0][idx_nnn] col major
                                    size_t idx_kkk = idx_kk;
                                    for (;idx_kkk < idx_kk + size_inner_tile && idx_kkk < k; idx_kkk+= 8){
                                        sum += A[offset_a + idx_kkk] * B[offset_b + idx_kkk];
                                        sum += A[offset_a + idx_kkk + 1] * B[offset_b + idx_kkk + 1];
                                        sum += A[offset_a + idx_kkk + 2] * B[offset_b + idx_kkk + 2];
                                        sum += A[offset_a + idx_kkk + 3] * B[offset_b + idx_kkk + 3];
                                        sum += A[offset_a + idx_kkk + 4] * B[offset_b + idx_kkk + 4];
                                        sum += A[offset_a + idx_kkk + 5] * B[offset_b + idx_kkk + 5];
                                        sum += A[offset_a + idx_kkk + 6] * B[offset_b + idx_kkk + 6];
                                        sum += A[offset_a + idx_kkk + 7] * B[offset_b + idx_kkk + 7];
                                    }
                                    for (; idx_kkk < idx_kk + size_inner_tile && idx_kkk < k; idx_kkk++){
                                        sum += A[offset_a + idx_kkk] * B[offset_b +idx_kkk];
                                    }
                                    C[idx_mmm * n + idx_nnn] += sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}




/*

M = 4 N = 4 K = 4

tile_M = 2 tile_N = 2
c c
c c

a a a a   b b
a a a a   b b
          b b
          b b

a        b b
a

ab ab  + ab ab + ab ab  + ab ab
ab ab    ab ab   ab ab    ab ab

= 4ab 4ab
  4ab 4ab

*/



void simd_kernel_unrolled(const float * tile_A, const float * tile_B, float * C, size_t M, size_t N, size_t K, size_t tile_m, size_t tile_n, size_t offset_tile_C){
    __m256 reg_array_tile_C[8] = {}; // 8 256 bit regs for 8x8 floats of C
    __m256 reg_col_tile_strip_A; // 256 bit reg for 8 float row slice of 8 x K row strip of A
    __m256 reg_row_tile_strip_B_element; // 256 bit reg for broadcast of an element from K x 8 col strip of B


    for (size_t idx_k = 0; idx_k < K; idx_k++){
        // Load the 8 floats from the idx_kth column of the 8 row strip of A
        reg_col_tile_strip_A = _mm256_loadu_ps(&tile_A[idx_k * M]);
        
        reg_row_tile_strip_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N]);

        reg_array_tile_C[0] = _mm256_fmadd_ps(reg_col_tile_strip_A, reg_row_tile_strip_B_element, reg_array_tile_C[0]);

        reg_row_tile_strip_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 1]);

        reg_array_tile_C[1] = _mm256_fmadd_ps(reg_col_tile_strip_A, reg_row_tile_strip_B_element, reg_array_tile_C[1]);


        reg_row_tile_strip_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 2]);

        reg_array_tile_C[2] = _mm256_fmadd_ps(reg_col_tile_strip_A, reg_row_tile_strip_B_element, reg_array_tile_C[2]);


        reg_row_tile_strip_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 3]);

        reg_array_tile_C[3] = _mm256_fmadd_ps(reg_col_tile_strip_A, reg_row_tile_strip_B_element, reg_array_tile_C[3]);


        reg_row_tile_strip_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 4]);

        reg_array_tile_C[4] = _mm256_fmadd_ps(reg_col_tile_strip_A, reg_row_tile_strip_B_element, reg_array_tile_C[4]);


        reg_row_tile_strip_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 5]);
        reg_array_tile_C[5] = _mm256_fmadd_ps(reg_col_tile_strip_A, reg_row_tile_strip_B_element, reg_array_tile_C[5]);

        reg_row_tile_strip_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 6]);
        reg_array_tile_C[6] = _mm256_fmadd_ps(reg_col_tile_strip_A, reg_row_tile_strip_B_element, reg_array_tile_C[6]);
        reg_row_tile_strip_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 7]);
        reg_array_tile_C[7] = _mm256_fmadd_ps(reg_col_tile_strip_A, reg_row_tile_strip_B_element, reg_array_tile_C[7]);
    }   

    for (size_t idx_col = 0; idx_col * M + offset_tile_C < M * N && idx_col < 8; idx_col++){
        _mm256_storeu_ps(&C[idx_col * M + offset_tile_C], reg_array_tile_C[idx_col]);
    }
}

void simd_kernel_rolled(const float * tile_A, const float * tile_B, float * C, 
    size_t M, size_t N, size_t K, 
    size_t tile_m, size_t tile_n, 
    size_t offset_tile_C, size_t offset_tile_A, size_t offset_tile_B){

    __m256 reg_array_tile_C[8] = {}; // 8 256 bit regs for 8x8 floats of C
    __m256 reg_col_tile_strip_A; // 256 bit reg for 8 float row slice of 8 x K row strip of A
    __m256 reg_row_tile_strip_B_element; // 256 bit reg for broadcast of an element from K x 8 col strip of B

        // Create a mask for the first tile_m elements
    int mask_arr[8] = {0,0,0,0,0,0,0,0};
    for (size_t i = 0; i < tile_m; ++i) {
        mask_arr[i] = -1; // -1 means load, 0 means zero
    }
    __m256i mask = _mm256_loadu_si256((__m256i*)mask_arr);


    for (size_t idx_k = 0; idx_k < K; idx_k++){
        // Load the 8 floats from the idx_kth column of the 8 row strip of A
        reg_col_tile_strip_A = _mm256_maskload_ps(&tile_A[idx_k * M], mask);

        for (size_t idx_tile_B = 0; idx_tile_B < tile_n; idx_tile_B++){
            reg_row_tile_strip_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + idx_tile_B]);
            reg_array_tile_C[idx_tile_B] = _mm256_fmadd_ps(reg_col_tile_strip_A, reg_row_tile_strip_B_element, reg_array_tile_C[idx_tile_B]);
        }
        
        
    }   

    for (size_t idx_col = 0; idx_col * M + offset_tile_C < M * N && idx_col < 8; idx_col++){
        _mm256_maskstore_ps(&C[idx_col * M + offset_tile_C], mask, reg_array_tile_C[idx_col]);
    }
}

void simd_kernel(const float * tile_A, const float * tile_B, float * C, 
    size_t M, size_t N, size_t K, 
    size_t tile_m, size_t tile_n, 
    size_t offset_tile_C, size_t offset_tile_A, size_t offset_tile_B){

    __m256 reg_array_tile_C[8] = {}; // 8 256 bit regs for 8x8 floats of C
    __m256 reg_col_tile_strip_A; // 256 bit reg for 8 float row slice of 8 x K row strip of A
    __m256 reg_row_tile_strip_B_element; // 256 bit reg for broadcast of an element from K x 8 col strip of B
    
    
        // Create a mask for the first tile_m elements
    int mask_arr[8] = {0,0,0,0,0,0,0,0};
    for (size_t i = 0; i < tile_m; ++i) {
        mask_arr[i] = -1; // -1 means load, 0 means zero
    }
     __m256i mask = _mm256_loadu_si256((__m256i*)mask_arr);


    for (size_t idx_k = 0; idx_k < K; idx_k++){
        // Load the 8 floats from the idx_kth column of the row strip of A
        reg_col_tile_strip_A = _mm256_maskload_ps(&tile_A[idx_k * M], mask);

        for (size_t idx_tile_B = 0; idx_tile_B < tile_n; idx_tile_B++){
            reg_row_tile_strip_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + idx_tile_B]);
            reg_array_tile_C[idx_tile_B] = _mm256_fmadd_ps(reg_col_tile_strip_A, reg_row_tile_strip_B_element, reg_array_tile_C[idx_tile_B]);
        }
        
        
    }   

    for (size_t idx_col = 0; idx_col * M + offset_tile_C < M * N && idx_col < 8; idx_col++){
        _mm256_storeu_ps(&C[idx_col * M + offset_tile_C], reg_array_tile_C[idx_col]);
    }
}

void simd_matmul(const float *A, const float *B, float *C, size_t M, size_t N, size_t K)
{
    float * A_col_major = (float*)malloc(M * K * sizeof(float));
    transpose_matrix(A, A_col_major, M, K);
    float * C_col_major = (float*)malloc(M * N * sizeof(float));
    memset(C_col_major, 0, M * N * sizeof(float));
    const size_t tile_m = 8;
    const size_t tile_n = 8;
    const size_t remainder_m = M % tile_m;
    const size_t remainder_n = N % tile_n;
    size_t offset_C = 0;

    if (remainder_m == 0 && remainder_n == 0)
    {

        for (size_t idx_m = 0; idx_m < M; idx_m += tile_m)
        {
            for (size_t idx_n = 0; idx_n < N; idx_n += tile_n)
            {
                offset_C = idx_m + idx_n * M;
                simd_kernel_unrolled(&A_col_major[idx_m], &B[idx_n], C_col_major, M, N, K, tile_m, tile_n, offset_C);
            }
        }
    }
    else
    {
        printf("remainder_m = %zu, remainder_n = %zu\n", remainder_m, remainder_n);
        size_t idx_m = 0;
        for (; idx_m < M - remainder_m; idx_m += tile_m)
        {
            size_t idx_n = 0;
            for ( ; idx_n < N - remainder_n; idx_n += tile_n)
            {
                offset_C = idx_m + idx_n * M;
                simd_kernel_unrolled(&A_col_major[idx_m], &B[idx_n], C_col_major, M, N, K, tile_m, tile_n, offset_C);
            }
            offset_C = idx_m + idx_n * M;
            simd_kernel_rolled(&A_col_major[idx_m], &B[idx_n], C_col_major, M, N, K, tile_m, remainder_n, offset_C, idx_m, remainder_n);
        }
        for (size_t idx_n = 0; idx_n < N; idx_n += tile_n)
        {
            offset_C = idx_m + idx_n * M;
            simd_kernel_rolled(&A_col_major[idx_m], &B[idx_n], C_col_major, M, N, K, remainder_m, tile_n, offset_C, idx_m, tile_n);
        }
    }

    transpose_matrix(C_col_major, C, N, M); // since col to row major dimensions are swapped
    free(A_col_major);
    free(C_col_major);
}

     /*
     C = A * B
     grads_C is dL/dC
     grads_B is dL/dB
     grads_A is dL/dA 
     
     A is M x K 
     B is K x N 
     C is M x N 
     
     
     K: input dimension
     M: batch size
     N: output dimension


     */


void matmul_backwards(const float * grads_C, const float * B, const float * A, float * grads_B, float * grads_A, size_t M, size_t N, size_t K){
    
    float * A_transpose = (float *)malloc(M * K * sizeof(float));
    transpose_matrix(A, A_transpose, M, K);

    naive_matmul(A_transpose, grads_C, grads_B, K, N, M); // grads_B = A-transpose * grads_C


    float * B_transpose = (float *)malloc(K * N * sizeof(float));
    transpose_matrix(B, B_transpose, K, N);

    naive_matmul(grads_C, B_transpose, grads_A, M, N, K); // grads_A = grads_C * B-transpose

    free(A_transpose);
    free(B_transpose);
}



void simd_matmul_backwards(const float * grads_C, const float * B, const float * A, float * grads_B, float * grads_A, size_t M, size_t N, size_t K)
{
    // grads_B = A-transpose * grads_C
    float * A_transpose = (float *)malloc(M * K * sizeof(float));
    transpose_matrix(A, A_transpose, M, K);
    // k x n = k x m * m x n
    simd_matmul(A_transpose, grads_C, grads_B, M, N, K);

    if (grads_A == NULL){ // must be first layer where input grads are not needed
        free(A_transpose);
        return;
    }
    // grads_A = grads_C * B-transpose
    // m x k = m x n * n x k
    float * B_transpose = (float *)malloc(M * N * sizeof(float));
    transpose_matrix(B, B_transpose, K, N);
    simd_matmul(grads_C, B_transpose, grads_A, M, N, K);
    free(A_transpose);
    free(B_transpose);
}

float cross_entropy_forward(const float *logits, const long *targets, float *log_probs, size_t batch_size, size_t num_classes)
{
    float loss = 0.0f;
    for (size_t idx_sample = 0; idx_sample < batch_size; idx_sample++)
    {
        float logit_sum = 0.0f;
        // subtract logit max for numerical stability
        float max_logit = logits[idx_sample * num_classes];
        for (size_t idx_class = 1; idx_class < num_classes; idx_class++)
        {
            if (logits[idx_sample * num_classes + idx_class] > max_logit)
            {
                max_logit = logits[idx_sample * num_classes + idx_class];
            }
        }
        
        for (size_t idx_class = 0; idx_class < num_classes; idx_class++)
        {
            logit_sum += expf(logits[idx_sample * num_classes + idx_class] - max_logit);
        }
        for (size_t idx_class = 0; idx_class < num_classes; idx_class++)
        {
            float log_prob = logits[idx_sample * num_classes + idx_class] - max_logit - logf(logit_sum);
            log_probs[idx_sample * num_classes + idx_class] = log_prob;
            if (idx_class == targets[idx_sample])
            {
                loss -= log_prob; // accumulate loss only for the target class
            }
        }
    }

    return loss/batch_size;
}

void loss_backward(const float *logits, const long *targets, float *grad_logits, size_t size_batch, size_t num_classes)
{
    for (size_t idx_sample = 0; idx_sample < size_batch; idx_sample++)
    {
        float max_logit = logits[idx_sample * num_classes];
        for (size_t idx_class = 0; idx_class < num_classes; idx_class++)
        {
            if (logits[idx_sample * num_classes + idx_class] > max_logit)
            {
                max_logit = logits[idx_sample * num_classes + idx_class];
            }
        }
        float logit_sum = 0.0f;
        for (size_t idx_class = 0; idx_class < num_classes; idx_class++)
        {
            logit_sum += expf(logits[idx_sample * num_classes + idx_class] - max_logit);
        }

        for (size_t idx_class = 0; idx_class < num_classes; idx_class++)
        {
            float prob = expf(logits[idx_sample * num_classes + idx_class] - max_logit) / logit_sum;
            float target = (idx_class == targets[idx_sample]) ? 1.0f : 0.0f;
            grad_logits[idx_sample * num_classes + idx_class] = (prob - target) / size_batch;
        }
    }
}


void layer_normalization_forward(const float *input, float *output, size_t batch_size, size_t num_features,
float * gamma, float * beta)
{
    for (size_t idx_sample = 0; idx_sample < batch_size; idx_sample++)
    {
        float mean = 0.0f;
        float variance = 0.0f;

        // Calculate mean
        for (size_t idx_feature = 0; idx_feature < num_features; idx_feature++)
        {
            mean += input[idx_sample * num_features + idx_feature];
        }
        mean /= num_features;

        // Calculate variance
        for (size_t idx_feature = 0; idx_feature < num_features; idx_feature++)
        {
            float diff = input[idx_sample * num_features + idx_feature] - mean;
            variance += diff * diff;
        }
        variance /= num_features;

        // Normalize and apply scale and shift
        for (size_t idx_feature = 0; idx_feature < num_features; idx_feature++)
        {
            output[idx_sample * num_features + idx_feature] = gamma[idx_feature] * 
                (input[idx_sample * num_features + idx_feature] - mean) / sqrtf(variance + 1e-5f) + beta[idx_feature];
        }
    }
}

/*
    Layer Normalization (per sample)
    K: number of features
    B: batch size
    x_k: input feature k
    y_k: output feature k
    gamma_k: scale parameter for feature k
    beta_k: shift parameter for feature k

    mean = 1/K * sum_j x_j
    x_centered_k = x_k - mean
    variance = 1/K * sum_j (x_centered_j)^2
    x_hat_k = x_centered_k / sqrt(variance + eps)
    y_k = gamma_k * x_hat_k + beta_k

    Gradients:

    dL/dgamma_k = 1 / B * sum_over_samples( dL/dy_k * x_hat_k )
    dL/dbeta_k  = 1 / B * sum_over_samples( dL/dy_k )

    dL/dx_k = 1 / sqrt(variance + eps) *
        ( dL/dy_k
          - 1/K * sum_j(dL/dy_j * gamma_j)
          - x_hat_k * 1/K * sum_j(gamma_j * dL/dy_j * x_hat_j) )
*/


void layer_normalization_backward(const float *inputs,
                                  const float *grad_outputs,
                                  float *grad_inputs,
                                  size_t size_batch,
                                  size_t num_features,
                                  const float *gammas,
                                  float *grad_gammas,
                                  float *grad_betas)
{
    // Initialize accumulators for grad_gamma and grad_beta
    for (size_t idx_feature = 0; idx_feature < num_features; ++idx_feature) {
        grad_gammas[idx_feature] = 0.0f;
        grad_betas[idx_feature]  = 0.0f;
    }

    for (size_t idx_sample = 0; idx_sample < size_batch; ++idx_sample)
    {
        const size_t offset_sample = idx_sample * num_features;

        // Mean
        float x_mean = 0.0f;
        for (size_t idx_feature = 0; idx_feature < num_features; ++idx_feature) {
            x_mean += inputs[offset_sample + idx_feature];
        }
        x_mean /= (float)num_features;
        
        // print mean
        printf("Mean of sample %zu: %f\n", idx_sample, x_mean);

        // Variance
        float variance = 0.0f;
        for (size_t idx_feature = 0; idx_feature < num_features; ++idx_feature) {
            float diff = inputs[offset_sample + idx_feature] - x_mean;
            variance += diff * diff;
        }
        variance /= (float)num_features;

        // print variance
        printf("Variance of sample %zu: %f\n", idx_sample, variance);

        // Inverse standard deviation plus epsilon for numerical stability
        const float inv_stddev = 1.0f / sqrtf(variance + LN_EPS);

        float mean_dl_dy_gamma = 0.0f;
        float mean_dl_dy_gamma_xhat = 0.0f;

        // Precompute sums for input gradient calculation
        for (size_t idx_feature = 0; idx_feature < num_features; ++idx_feature)
        {
            const float x = inputs[offset_sample + idx_feature];
            const float grad_output = grad_outputs[offset_sample + idx_feature];
            const float x_hat = (x - x_mean) * inv_stddev;
            mean_dl_dy_gamma += grad_output * gammas[idx_feature];
            mean_dl_dy_gamma_xhat += grad_output * gammas[idx_feature] * x_hat;
        }
        mean_dl_dy_gamma /= (float)num_features;
        mean_dl_dy_gamma_xhat /= (float)num_features;

        // Input gradients and accumulate parameter gradients
        for (size_t idx_feature = 0; idx_feature < num_features; ++idx_feature)
        {
            const float x = inputs[offset_sample + idx_feature];
            const float grad_output = grad_outputs[offset_sample + idx_feature];
            const float x_hat = (x - x_mean) * inv_stddev;
            
            grad_inputs[offset_sample + idx_feature] = inv_stddev * 
                (grad_output * gammas[idx_feature]
                 - mean_dl_dy_gamma
                 - x_hat * mean_dl_dy_gamma_xhat);

            grad_gammas[idx_feature] += grad_output * x_hat;  
            grad_betas[idx_feature]  += grad_output;

        }
    }

}

void relu_forward(float *activations, size_t num_features, size_t size_batch)
{
    for (size_t idx_sample = 0; idx_sample < size_batch; idx_sample++) {
        for (size_t idx_feature = 0; idx_feature < num_features; idx_feature++) {
            activations[idx_sample * num_features + idx_feature] = fmaxf(0.0f, activations[idx_sample * num_features + idx_feature]);
        }
    }
}

void relu_backward(const float *input, float *gradients, const unsigned char *labels, size_t num_features, size_t size_batch)
{
    for (size_t idx_sample = 0; idx_sample < size_batch; idx_sample++) {
        for (size_t idx_feature = 0; idx_feature < num_features; idx_feature++) {
            gradients[idx_sample * num_features + idx_feature] = 
            (input[idx_sample * num_features + idx_feature] > 0.0f) ? gradients[idx_sample * num_features + idx_feature] : 0.0f;
        }
    }
}

void softmax_forward(float *activations, size_t num_classes, size_t size_batch)
{
    for (size_t idx_sample = 0; idx_sample < size_batch; idx_sample++) {
        float max_logit = activations[idx_sample * num_classes];
        for (size_t idx_neuron = 1; idx_neuron < num_classes; idx_neuron++) {
            if (activations[idx_sample * num_classes + idx_neuron] > max_logit) {
                max_logit = activations[idx_sample * num_classes + idx_neuron];
            }
        }

        float sum_exp = 0.0f;
        for (size_t idx_neuron = 0; idx_neuron < num_classes; idx_neuron++) {
            activations[idx_sample * num_classes + idx_neuron] = expf(activations[idx_sample * num_classes + idx_neuron] - max_logit);
            sum_exp += activations[idx_sample * num_classes + idx_neuron];
        }

        for (size_t idx_neuron = 0; idx_neuron < num_classes; idx_neuron++) {
            activations[idx_sample * num_classes + idx_neuron] /= sum_exp;
        }
    }
}
