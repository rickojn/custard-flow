#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#include <math.h>



int min(int a, int b){
    return a < b ? a : b;
}

void naive_matmul(const float* A, const float *B, float * C, size_t m, size_t n, size_t k){
    // A is m x k row major
    // B is k x n col major
    // C is m x n row major
    for (size_t idx_m = 0; idx_m < m; idx_m++ ){
        for (size_t idx_n = 0; idx_n < n; idx_n++){
            for (size_t idx_k = 0; idx_k < k; idx_k++){            
                // C[idx_m][idx_n] = A[idx_m][idx_k] * B[idx_k][idx_n]
                C[idx_m * n + idx_n] += A[idx_m * k + idx_k] * B[idx_k + idx_n * k];
                float db_1 = A[idx_m * k + idx_k];
                float db_2 = B[idx_k + idx_n * k];
                float db_3 = C[idx_m * n + idx_n];
                int db_rando = idx_k;
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



void simd_kernel(const float * tile_A, const float * tile_B, float * C, size_t M, size_t N, size_t K, size_t tile_m, size_t tile_n, size_t offset_tile_C){
    __m256 reg_array_C[8][1] = {};
    __m256 reg_col_tile_A_1;
    __m256 reg_tile_B_element;

    for (size_t idx_k = 0; idx_k < K; idx_k++){
        reg_col_tile_A_1 = _mm256_loadu_ps(&tile_A[idx_k * M]);

        
        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N]);

        reg_array_C[0][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[0][0]);


        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 1]);

        reg_array_C[1][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[1][0]);


        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 2]);

        reg_array_C[2][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[2][0]);


        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 3]);

        reg_array_C[3][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[3][0]);


        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 4]);

        reg_array_C[4][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[4][0]);


        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 5]);
        reg_array_C[5][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[5][0]);

        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 6]);
        reg_array_C[6][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[6][0]);
        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 7]);
        reg_array_C[7][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[7][0]);
    }   

    for (size_t idx_col = 0; idx_col * M + offset_tile_C < M * N && idx_col < 8; idx_col++){
        _mm256_storeu_ps(&C[idx_col * M + offset_tile_C], reg_array_C[idx_col][0]);
    }
}

void simd_matmul(const float *A, const float *B, float *C, size_t M, size_t N, size_t K)
{
    const size_t tile_m = 8;
    const size_t tile_n = 8;
    size_t offset_C = 0;
    for (size_t idx_m = 0; idx_m < M; idx_m += tile_m)
    {
        for (size_t idx_n = 0; idx_n < N; idx_n += tile_n)
        {
            offset_C= idx_m + idx_n * M;
            simd_kernel(&A[idx_m], &B[idx_n], C, M, N, K, tile_m, tile_n, offset_C);
        }
    }
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
    for (size_t idx_m = 0; idx_m < M; idx_m++){
        for (size_t idx_n = 0; idx_n < N; idx_n++){
            for (size_t idx_k = 0; idx_k < K; idx_k++){
                // grads_B = A-transpose * grads_C
                // grads_B[idx_k][idx_n] += A[idx_m][idx_k] * grads_C[idx_m][idx_n];
                grads_B[idx_k + idx_n * K] += A[idx_m * K + idx_k] * grads_C[idx_m * N + idx_n];
            }
        }
    }


    for (size_t idx_m = 0; idx_m < M; idx_m++){
        for (size_t idx_k = 0; idx_k < K; idx_k++){
            for (size_t idx_n = 0; idx_n < N; idx_n++){
                // grads_A = grads_C * B-transpose
                // grads_A[idx_m][idx_k] += grads_C[idx_m][idx_n] * B[idx_k][idx_n];
                grads_A[idx_m * K + idx_k] += grads_C[idx_m * N + idx_n] * B[idx_k + idx_n * K];
            }
        }
    }
}


void transpose_matrix(const float *src_matrix, float *dest_matrix, size_t src_num_rows, size_t src_num_cols){
    for (size_t idx_row = 0; idx_row < src_num_rows; idx_row++){
        for (size_t idx_col = 0; idx_col < src_num_cols; idx_col++){
            dest_matrix[idx_col * src_num_rows + idx_row] = src_matrix[idx_row * src_num_cols + idx_col];
        }
    }
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