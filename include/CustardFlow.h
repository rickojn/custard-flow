#ifndef CUSTARD_FLOW_H
#define CUSTARD_FLOW_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>





// Function declarations
int cf_min(int a, int b);

void cf_naive_matmul(const float *A, const float *B, float *C, size_t m, size_t n, size_t k);
void cf_outer_product_matmul(const float *A, const float *B, float *C, size_t rows, size_t cols, size_t inner_dim);
void cf_tiled_matmul(const float *A, const float *B, float *C, size_t rows, size_t cols, size_t inner_dim, size_t tile_size);
void cf_l1_tiled_matmul(const float *A, const float *B, float *C, size_t rows, size_t cols, size_t inner_dim, size_t tile_size, size_t inner_tile_size);
void cf_initialise_large_matrices(float *A, float *B, float *C);
void cf_check_result(const float *ref_C, const float *C, size_t rows, size_t cols);
void cf_simd_matmul(const float *A, const float *B, float *C, size_t M, size_t N, size_t K);
void cf_matmul_backwards(const float *grads_C, const float *B, const float *A, float *grads_B, float *grads_A, size_t M, size_t N, size_t K);
void cf_transpose_matrix(const float *src_matrix, float *dest_matrix, size_t src_num_rows, size_t src_num_cols);
void cf_simd_matmul_backwards(const float *grads_C, const float *B, const float *A, float *grads_B, float *grads_A, size_t M, size_t N, size_t K);
float cf_cross_entropy_forward(const float *logits, const long *targets, float * log_probs, size_t batch_size, size_t num_classes);
void cf_loss_backward(const float *logits, const long *targets, float *grad_logits, size_t batch_size, size_t num_classes);
void cf_layer_normalization_forward(const float *input, float *output, size_t batch_size, size_t num_features,
float * gamma, float * beta);
void cf_layer_normalization_backward(const float *inputs, const float *grad_outputs, float *grad_inputs, size_t size_batch, size_t num_features,
float * gammas, float * grad_gammas, float * grad_betas);



#ifdef __cplusplus
}
#endif

#endif // CUSTARD_FLOW_H