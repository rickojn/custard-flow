#include <gtest/gtest.h>
#include <torch/torch.h>
#include "CustardFlow.h"
#include <stdlib.h>
// This is a simple test case for the CustardFlow library using Google Test.

TEST(MinTest, BasicFunctionality) {
  // Test the min function.
  EXPECT_EQ(min(3, 5), 3);
  EXPECT_EQ(min(10, -2), -2);
  EXPECT_EQ(min(0, 0), 0);
}


TEST(MatmulNaiveTest, BasicFunctionality) {
    // Test the naive_matmul function.

    /*
    A [1,2,3
       4,5,6]
    B [1,4
       2,5
       3,6]
   C  [14, 32
       32, 77]
    */
    float A[] = {1, 2, 3, 4, 5, 6};
    float B[] = {1, 2, 3, 4, 5, 6};
    float C[4] = {0};

    naive_matmul(A, B, C, 2, 2, 3);

    // Expected result for C is {22, 28, 34}
    EXPECT_FLOAT_EQ(C[0], 14);
    EXPECT_FLOAT_EQ(C[1], 32);
    EXPECT_FLOAT_EQ(C[2], 32);
    EXPECT_FLOAT_EQ(C[3], 77);
}

TEST(MatmulSIMDTest, BasicFunctionality) {


    /*
    A [1,
       2,
       3,
       4,
       5,
       6,
       7,
       8]
    B [1,2,3,4,5,6,7,8]
   C  [1, 2, 3, 4, 5, 6, 7, 8,
       2, 4, 6, 8, 10, 12, 14, 16,
       3, 6, 9, 12, 15, 18, 21, 24,
       4, 8, 12, 16, 20, 24, 28, 32,
       5, 10, 15, 20, 25, 30, 35, 40,
       6, 12, 18, 24, 30, 36, 42, 48,
       7, 14, 21, 28, 35, 42, 49, 56,
       8, 16, 24, 32, 40, 48, 56, 64]
   
    */
    float A[] = {1, 2, 3, 4, 5, 6, 7, 8}; 
    float B[] = {1, 2, 3, 4, 5, 6, 7, 8};
    // float B[] = {1, 2, 3, 4, 5, 6};
    float C[64] = {0};

    simd_matmul(A, B, C, 8, 8, 1);

    EXPECT_FLOAT_EQ(C[0], 1);
    EXPECT_FLOAT_EQ(C[1], 2);
    EXPECT_FLOAT_EQ(C[2], 3);
    EXPECT_FLOAT_EQ(C[3], 4);
}

/* Test matmul_backwards function
    A [1,2,3      
       4,5,6]           2 x 3
    B [1,4
       2,5              3 x 2
       3,6]
grdsC [10,20,
       30,40]       2 x 2


   C  [14, 32
       32, 77]

   Bt [1,2,3,
       4,5,6]

c11 = a11*b11 + a12*b21 + a13*b31
c12 = a11*b12 + a12*b22 + a13*b32
c21 = a21*b11 + a22*b21 + a23*b31
c22 = a21*b12 + a22*b22 + a23*b32


grads_A [90,120,150,
         190,260,330]

grads_B [65, 90,
         85, 120
         105, 150]
*/

TEST(MatmulBackwardsTest, BasicFunctionality) {
    float A[] = {1, 2, 3, 4, 5, 6}; // 2x3
    float B[] = {1, 2, 3, 4, 5, 6}; // 3x2
    float grads_C[] = {10, 20, 30, 40}; // 2x2
    float grads_A[6] = {0};
    float grads_B[6] = {0};

    matmul_backwards(grads_C, B, A, grads_B, grads_A, 2, 2, 3);

    EXPECT_FLOAT_EQ(grads_A[0], 90);
    EXPECT_FLOAT_EQ(grads_A[1], 120);
    EXPECT_FLOAT_EQ(grads_A[2], 150);
    EXPECT_FLOAT_EQ(grads_A[3], 190);
    EXPECT_FLOAT_EQ(grads_A[4], 260);
    EXPECT_FLOAT_EQ(grads_A[5], 330);

    EXPECT_FLOAT_EQ(grads_B[0], 130);
    EXPECT_FLOAT_EQ(grads_B[1], 170);
    EXPECT_FLOAT_EQ(grads_B[2], 210);
    EXPECT_FLOAT_EQ(grads_B[3], 180);
    EXPECT_FLOAT_EQ(grads_B[4], 240);
    EXPECT_FLOAT_EQ(grads_B[5], 300);
}

TEST(MatrixMultiplicationTest, CompareWithLibTorch) {
    // ARRANGE
    torch::manual_seed(42);
    torch::Tensor A = torch::rand({3, 3});
    torch::Tensor B = torch::rand({3, 3});
    torch::Tensor expected = torch::mm(A, B);

    float* A_ptr = A.data_ptr<float>();
    float* A_ptr_transposed = A.t().data_ptr<float>();
    float* B_ptr = B.data_ptr<float>();
    float* B_ptr_transposed = (float*)malloc(B.size(0) * B.size(1) * sizeof(float));
    transpose_matrix(B_ptr, B_ptr_transposed, B.size(0), B.size(1));
    float* expected_ptr = expected.data_ptr<float>();

    int m = A.size(0);
    int k = A.size(1);
    int n = B.size(1);

    float* my_result_ptr = new float[m * n];
    std::fill(my_result_ptr, my_result_ptr + m * n, 0.0f); 

    // ACT
    naive_matmul(A_ptr, B_ptr_transposed, my_result_ptr, m, k, n);


    // ASSERT
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_FLOAT_EQ(my_result_ptr[i * n + j], expected_ptr[i * n + j])
                << "Mismatch at (" << i << ", " << j << ")";
        }
    }

    // Clean up
    delete[] my_result_ptr;
    free(B_ptr_transposed);
}


 
 void mat_mul_backwards_test(
     std::function<void(const float *, const float *, const float *, float *, float *, size_t, size_t, size_t)> matmul_backwards_func,
     std::string func_name)
 {
     // ARRANGE
     torch::manual_seed(42);
     torch::Tensor input = torch::rand({3, 3}, torch::requires_grad());
     torch::Tensor weights = torch::rand({3, 3}, torch::requires_grad());
     printf("inputs:\n");
     std::cout << input << std::endl;
     printf("weights:\n");
     std::cout << weights << std::endl;
     torch::Tensor output = torch::mm(input, weights);
     torch::Tensor grad_output = torch::rand_like(output);
     printf("grad output:\n");
     std::cout << grad_output << std::endl;
     output.backward(grad_output);
     
     float *input_grad = input.grad().data_ptr<float>();
     float *weights_grad = weights.grad().data_ptr<float>();
     float *input_ptr = input.data_ptr<float>();
     float *weights_ptr = weights.data_ptr<float>();
     float *transposed_weights_ptr = (float *)malloc(weights.size(0) * weights.size(1) * sizeof(float));
     transpose_matrix(weights_ptr, transposed_weights_ptr, weights.size(0), weights.size(1));
     float *grad_output_ptr = grad_output.data_ptr<float>();
     
     float *input_grad_computed = new float[3 * 3];
     float *weights_grad_computed = new float[3 * 3];
     std::fill(input_grad_computed, input_grad_computed + 3 * 3, 0.0f);
     std::fill(weights_grad_computed, weights_grad_computed + 3 * 3, 0.0f);

     // ACT
     matmul_backwards_func(grad_output_ptr, transposed_weights_ptr, input_ptr, weights_grad_computed, input_grad_computed, 3, 3, 3);

     // ASSERT
     for (int i = 0; i < 3; ++i)
     {
         for (int j = 0; j < 3; ++j)
         {
             EXPECT_NEAR(input_grad_computed[i * 3 + j], input_grad[i * 3 + j], 1e-3)
                 << "Mismatch in input gradient at (" << i << ", " << j << ")";
             EXPECT_NEAR(weights_grad_computed[i * 3 + j], weights_grad[j * 3 + i], 1e-3)
                 << "Mismatch in weights gradient at (" << i << ", " << j << ")";
                 // break out of the loop if weights mismatch is found
        //         if (std::abs(input_grad_computed[i * 3 + j] - input_grad[i * 3 + j]) > 1e-3 ||
        //             std::abs(weights_grad_computed[i * 3 + j] - weights_grad[j * 3 + i]) > 1e-3) {
        //             i = 3; // break out of the outer loop if mismatch is found
        //             break; // break out of the loop if mismatch is found
        //  }
        }
     }

     

     delete[] input_grad_computed;
     delete[] weights_grad_computed;
     free(transposed_weights_ptr);
 }


TEST(MatrixMultiplicationBackwardsTest, MatmulBackwards) {
    mat_mul_backwards_test(matmul_backwards, "matmul_backwards");
 }

 TEST(MatrixMultiplicationBackwardsTest, SimdMatmulBackwards) {
    mat_mul_backwards_test(simd_matmul_backwards, "simd_matmul_backwards");
 }


 TEST(CrossEntropyLossTest, BasicFunctionality) {
    // ARRANGE
    torch::manual_seed(42);
    int batch_size = 4;
    int num_classes = 5;

    // Random logits and targets
    torch::Tensor logits = torch::randn({batch_size, num_classes}, torch::requires_grad());
    torch::Tensor targets = torch::randint(0, num_classes, {batch_size}, torch::kLong);

    auto loss = torch::nn::functional::cross_entropy(logits, targets, torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kMean));


    float *logits_ptr = logits.data_ptr<float>();
    float * transposed_logits_ptr = (float *)malloc(logits.size(0) * logits.size(1) * sizeof(float));
    transpose_matrix(logits_ptr, transposed_logits_ptr, logits.size(0), logits.size(1));
    long *targets_ptr = targets.data_ptr<long>();

    float *log_probs = new float[batch_size * num_classes];

    // ACT
    float loss_value = cross_entropy_forward(logits_ptr, targets_ptr, log_probs, batch_size, num_classes);

    // ASSERT
    EXPECT_FLOAT_EQ(loss.item<float>(), loss_value);

    delete[] log_probs;

 }

 TEST(LossBackwardTest, BasicFunctionality) {
    // ARRANGE
    torch::manual_seed(42);
    int batch_size = 4;
    int num_classes = 5;

    // Random logits and targets
    torch::Tensor logits = torch::randn({batch_size, num_classes}, torch::requires_grad());
    torch::Tensor targets = torch::randint(0, num_classes, {batch_size}, torch::kLong);

    auto loss = torch::nn::functional::cross_entropy(logits, targets, torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kMean));
    loss.backward();

    float *logits_ptr = logits.data_ptr<float>();
    long *targets_ptr = targets.data_ptr<long>();
    float *grad_logits = logits.grad().data_ptr<float>();

    // ACT
    loss_backward(logits_ptr, targets_ptr, grad_logits, batch_size, num_classes);

    // ASSERT
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            EXPECT_FLOAT_EQ(grad_logits[i * num_classes + j], logits.grad()[i][j].item<float>())
                << "Mismatch in gradient at (" << i << ", " << j << ")";
        }
    }
 }

 // 1) Define a struct to hold your dimensions:
struct MatMulDims {
    int m, k, n;
};

// 2) Create a fixture that derives from TestWithParam, templated on MatMulDims:
class SIMDMatrixMultiplicationTest
  : public ::testing::TestWithParam<MatMulDims> {
protected:
    void run_compare_with_libtorch() {
        auto dims = GetParam();
        int m = dims.m, k = dims.k, n = dims.n;

        // seed and random tensors
        torch::manual_seed(42);
        auto A = torch::rand({m, k});
        auto B = torch::rand({k, n});
        auto expected = torch::mm(A, B);

        // get raw pointers
        float* A_ptr = A.data_ptr<float>();
        float* A_ptr_t = (float*)malloc(m * k * sizeof(float));
        transpose_matrix(A_ptr, A_ptr_t, m, k);

        float* B_ptr = B.data_ptr<float>();
        float* B_ptr_t = (float*)malloc(k * n * sizeof(float));
        transpose_matrix(B_ptr, B_ptr_t, k, n);

        float* actual = new float[m * n];
        std::fill(actual, actual + m * n, 0.0f);

        // call your SIMD matmul (note argument order may vary)
        simd_matmul(A_ptr_t, B_ptr, actual, m, n, k);

        // transpose expected to col-major for comparison
        float* exp_t = (float*)malloc(m * n * sizeof(float));
        transpose_matrix(expected.data_ptr<float>(), exp_t, m, n);

        // compare
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                // actual is column-major: [j*m + i]
                EXPECT_NEAR(actual[j * m + i], exp_t[j * m + i], 1e-3)
                    << "Failed for dims m="<<m<<", k="<<k<<", n="<<n
                    << " at ("<<i<<","<<j<<") offset: " << j * m + i;
                    if (std::abs(actual[j * m + i] - exp_t[j * m + i]) > 1e-3) {
                            i = m; // break outer loop
                            break; // break inner loop
                        }
            }
        }

        // cleanup
        delete[] actual;
        free(A_ptr_t);
        free(B_ptr_t);
        free(exp_t);
    }
};

// 3) Define the parameterized test using TEST_P:
TEST_P(SIMDMatrixMultiplicationTest, CompareWithLibTorch) {
    run_compare_with_libtorch();
}

// 4) Instantiate the suite with the size tuples you want to cover:
INSTANTIATE_TEST_SUITE_P(
    VariousSizes,
    SIMDMatrixMultiplicationTest,
    ::testing::Values(
        MatMulDims{1, 1, 1},
        MatMulDims{3, 3, 3},
        MatMulDims{8, 8, 1},
        MatMulDims{257, 512, 1024},
        MatMulDims{257, 512, 1023}
    ),
    [](auto const& info) {
        auto dims = info.param;
        return "m" + std::to_string(dims.m)
             + "_k" + std::to_string(dims.k)
             + "_n" + std::to_string(dims.n);
    }
);

 