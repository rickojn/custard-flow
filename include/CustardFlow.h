#ifndef CUSTARD_FLOW_H
#define CUSTARD_FLOW_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>


// Function declarations
size_t min(size_t a, size_t b);

void naive_matmul(const float *A, const float *B, float *C, size_t m, size_t n, size_t k, size_t lead_dim_a, size_t lead_dim_b, size_t lead_dim_c);
void outer_product_matmul(const float *A, const float *B, float *C, size_t rows, size_t cols, size_t inner_dim);
void tiled_matmul(const float *A, const float *B, float *C, size_t rows, size_t cols, size_t inner_dim, size_t tile_size);
void l1_tiled_matmul(const float *A, const float *B, float *C, size_t rows, size_t cols, size_t inner_dim, size_t tile_size, size_t inner_tile_size);
void initialise_large_matrices(float *A, float *B, float *C);
void check_result(const float *ref_C, const float *C, size_t rows, size_t cols);
void simd_matmul(const float *A, const float *B, float *C, size_t M, size_t N, size_t K);
void matmul_backwards(const float *grads_C, const float *B, const float *A, float *grads_B, float *grads_A, size_t M, size_t N, size_t K);
void transpose_matrix(const float *src_matrix, float *dest_matrix, size_t src_num_rows, size_t src_num_cols);

#endif // CUSTARD_FLOW_H