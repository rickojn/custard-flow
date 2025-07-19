#ifndef CUSTARD_FLOW_H
#define CUSTARD_FLOW_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>





// Function declarations
int min(int a, int b);

void naive_matmul(const float *A, const float *B, float *C, size_t m, size_t n, size_t k);
void outer_product_matmul(const float *A, const float *B, float *C, size_t rows, size_t cols, size_t inner_dim);
void tiled_matmul(const float *A, const float *B, float *C, size_t rows, size_t cols, size_t inner_dim, size_t tile_size);
void l1_tiled_matmul(const float *A, const float *B, float *C, size_t rows, size_t cols, size_t inner_dim, size_t tile_size, size_t inner_tile_size);
void initialise_large_matrices(float *A, float *B, float *C);
void check_result(const float *ref_C, const float *C, size_t rows, size_t cols);
void simd_matmul(const float *A, const float *B, float *C, size_t M, size_t N, size_t K);
void matmul_backwards(const float *grads_C, const float *B, const float *A, float *grads_B, float *grads_A, size_t M, size_t N, size_t K);
void transpose_matrix(const float *src_matrix, float *dest_matrix, size_t src_num_rows, size_t src_num_cols);
void simd_matmul_backward(const float *grads_C, const float *B, const float *A, float *grads_B, float *grads_A, size_t M, size_t N, size_t K);
float cross_entropy_forward(const float *logits, const long *targets, float * log_probs, size_t batch_size, size_t num_classes);

#ifdef __cplusplus
}
#endif

#endif // CUSTARD_FLOW_H