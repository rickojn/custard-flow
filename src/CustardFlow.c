#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>




void naive_matmul(const float* A, const float *B, float * C, size_t m, size_t n, size_t k, size_t lead_dim_a, size_t lead_dim_b, size_t lead_dim_c){
    for (size_t idx_m = 0; idx_m < m; idx_m++ ){
        for (size_t idx_n = 0; idx_n < n; idx_n++){
            for (size_t idx_k = 0; idx_k < k; idx_k++){            
                // C[m][n] = A[m][k] * B[k][n]
                C[idx_m * lead_dim_c + idx_n] += A[idx_m * lead_dim_a + idx_k] * B[idx_n * lead_dim_b + idx_k]; 
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



void simd_kernel(const float * tile_A, const float * tile_B, float * tile_C, size_t M, size_t N, size_t K, size_t tile_m, size_t tile_n){
    __m256 reg_array_C[6][2] = {};
    __m256 reg_col_tile_A_1;
    __m256 reg_col_tile_A_2;
    __m256 reg_tile_B_element;

    for (size_t idx_k = 0; idx_k < K; idx_k++){
        int db_1 = idx_k * M;
        float db_2 = tile_A[db_1];
        reg_col_tile_A_1 = _mm256_loadu_ps(&tile_A[idx_k * M]);
        reg_col_tile_A_2 = _mm256_loadu_ps(&tile_A[idx_k * M  + 8]);
        
        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N]);

        reg_array_C[0][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[0][0]);
        reg_array_C[0][1] = _mm256_fmadd_ps(reg_col_tile_A_2, reg_tile_B_element, reg_array_C[0][1]);

        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 1]);

        reg_array_C[1][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[1][0]);
        reg_array_C[1][1] = _mm256_fmadd_ps(reg_col_tile_A_2, reg_tile_B_element, reg_array_C[1][1]);

        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 2]);

        reg_array_C[2][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[2][0]);
        reg_array_C[2][1] = _mm256_fmadd_ps(reg_col_tile_A_2, reg_tile_B_element, reg_array_C[2][1]);

        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 3]);

        reg_array_C[3][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[3][0]);
        reg_array_C[3][1] = _mm256_fmadd_ps(reg_col_tile_A_2, reg_tile_B_element, reg_array_C[3][1]);

        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 4]);

        reg_array_C[4][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[4][0]);
        reg_array_C[4][1] = _mm256_fmadd_ps(reg_col_tile_A_2, reg_tile_B_element, reg_array_C[4][1]);

        reg_tile_B_element = _mm256_broadcast_ss(&tile_B[idx_k * N + 5]);
        reg_array_C[5][0] = _mm256_fmadd_ps(reg_col_tile_A_1, reg_tile_B_element, reg_array_C[5][0]);
        reg_array_C[5][1] = _mm256_fmadd_ps(reg_col_tile_A_2, reg_tile_B_element, reg_array_C[5][1]);
    }   

    for (size_t idx = 0; idx < 6; idx++){
        _mm256_storeu_ps(&tile_C[idx * tile_m], reg_array_C[idx][0]);
        _mm256_storeu_ps(&tile_C[idx * tile_m + 8], reg_array_C[idx][1]);
    }
}

void simd_matmul(const float *A, const float *B, float *C, size_t M, size_t N, size_t K)
{
    const size_t tile_m = 16;
    const size_t tile_n = 6;
    size_t offset_C = 0;
    for (size_t idx_m = 0; idx_m < M; idx_m += tile_m)
    {
        for (size_t idx_n = 0; idx_n < N; idx_n += tile_n)
        {
            int db_c_offset = offset_C;
            simd_kernel(&A[idx_m], &B[idx_n], &C[offset_C], M, N, K, tile_m, tile_n);
            offset_C += tile_m * tile_n;
        }
    }
}
