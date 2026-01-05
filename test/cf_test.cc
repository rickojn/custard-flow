#include <gtest/gtest.h>
#include <torch/torch.h>
#include "../include/CustardFlow.h"
#include <stdlib.h>

TEST(MatrixMultiplicationTest, CompareWithLibTorch) {
    // ARRANGE
    torch::manual_seed(42);
    torch::Tensor A = torch::rand({3, 3});
    torch::Tensor B = torch::rand({3, 3});
    torch::Tensor expected = torch::mm(A, B);

    float* A_ptr = A.data_ptr<float>();
    float* B_ptr = B.data_ptr<float>();
    float* expected_ptr = expected.data_ptr<float>();

    int m = A.size(0);
    int k = A.size(1);
    int n = B.size(1);

    float* actual_ptr = new float[m * n];
    std::fill(actual_ptr, actual_ptr + m * n, 0.0f); 

    // ACT
    naive_matmul(A_ptr, B_ptr, actual_ptr, m, k, n);


    // ASSERT
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_FLOAT_EQ(actual_ptr[i * n + j], expected_ptr[i * n + j])
                << "Mismatch at (" << i << ", " << j << ")";
        }
    }

    // Clean up
    delete[] actual_ptr;
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
     
     float *expected_input_grad = input.grad().data_ptr<float>();
     float *expected_weights_grad = weights.grad().data_ptr<float>();
     float *input_ptr = input.data_ptr<float>();
     float *weights_ptr = weights.data_ptr<float>();
     float *grad_output_ptr = grad_output.data_ptr<float>();
     
     float *actual_input_grad = new float[3 * 3];
     float *actual_weights_grad = new float[3 * 3];
     std::fill(actual_input_grad, actual_input_grad + 3 * 3, 0.0f);
     std::fill(actual_weights_grad, actual_weights_grad + 3 * 3, 0.0f);

     // ACT
     matmul_backwards_func(grad_output_ptr, weights_ptr, input_ptr, actual_weights_grad, actual_input_grad, 3, 3, 3);

     // ASSERT
     for (int i = 0; i < 3; ++i)
     {
         for (int j = 0; j < 3; ++j)
         {
             EXPECT_NEAR(actual_input_grad[i * 3 + j], expected_input_grad[i * 3 + j], 1e-3)
                 << "Mismatch in input gradient at (" << i << ", " << j << ")";
             EXPECT_NEAR(actual_weights_grad[i * 3 + j], expected_weights_grad[i * 3 + j], 1e-3)
                 << "Mismatch in weights gradient at (" << i << ", " << j << ")";
                 // break out of the loop if weights mismatch is found
                if (std::abs(actual_input_grad[i * 3 + j] - expected_input_grad[i * 3 + j]) > 1e-3 ||
                    std::abs(actual_weights_grad[i * 3 + j] - expected_weights_grad[i * 3 + j]) > 1e-3) {
                    i = 3; // break out of the outer loop if mismatch is found
                    break; // break out of the loop if mismatch is found
         }
        }
     }

     

     delete[] actual_input_grad;
     delete[] actual_weights_grad;
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
        // ARRANGE
        auto dims = GetParam();
        int m = dims.m, k = dims.k, n = dims.n;

        torch::manual_seed(42);
        auto A = torch::rand({m, k});
        auto B = torch::rand({k, n});
        auto expected = torch::mm(A, B);


        float* A_ptr = A.data_ptr<float>();

        float* B_ptr = B.data_ptr<float>();

        float* expected_ptr = expected.data_ptr<float>();

        float* actual = new float[m * n];
        std::fill(actual, actual + m * n, 0.0f);

        // ACT
        simd_matmul(A_ptr, B_ptr, actual, m, n, k);


        // compare
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                // actual is column-major: [j*m + i]
                EXPECT_NEAR(actual[i * n + j], expected_ptr[i * n + j], 1e-3)
                    << "Failed for dims m="<<m<<", k="<<k<<", n="<<n
                    << " at ("<<i<<","<<j<<") offset: " << i * n + j;
                    if (std::abs(actual[i * n + j] - expected_ptr[i * n + j]) > 1e-3) {
                            i = m; // break outer loop
                            break; // break inner loop
                        }
            }
        }

        // cleanup
        delete[] actual;
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
        MatMulDims{8,8,8},
        MatMulDims{16, 16, 16},
        MatMulDims{8, 16,8},
        MatMulDims{8, 15, 1},
        MatMulDims{256, 512, 1024},
        MatMulDims{8, 16, 32},
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

TEST(layer_norm_test, basic_functionality) {
    // ARRANGE
    torch::manual_seed(42);
    int batch_size = 4;
    int num_features = 5;

    torch::Tensor input = torch::randn({batch_size, num_features}, torch::requires_grad());
    torch::Tensor gamma = torch::randn({num_features}, torch::requires_grad());
    torch::Tensor beta = torch::randn({num_features}, torch::requires_grad());
    // torch layer norm
    auto layer_norm = torch::nn::LayerNorm(torch::nn::LayerNormOptions({num_features}).elementwise_affine(true));

    auto output = layer_norm->forward(input);

    // ACT
    float *input_ptr = input.data_ptr<float>();
    float *output_ptr = output.data_ptr<float>();
    float *gamma_ptr = layer_norm->weight.data_ptr<float>();
    float *beta_ptr = layer_norm->bias.data_ptr<float>();
    float *expected_output = new float[batch_size * num_features];
    layer_normalization_forward(input_ptr, expected_output, batch_size, num_features, gamma_ptr, beta_ptr);
    // ASSERT
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_features; ++j) {
            EXPECT_NEAR(output_ptr[i * num_features + j], expected_output[i * num_features + j], 1e-3)
                << "Mismatch at (" << i << ", " << j << ")";}
        }
}

// Layer normalization backward test
TEST(layer_norm_backward_test, basic_functionality) {
    // ARRANGE
    torch::manual_seed(42);
    int batch_size = 4;
    int num_features = 5; 
    torch::Tensor input = torch::randn({batch_size, num_features}, torch::requires_grad());
    auto mean = input.mean(1, true);
    //print torch mean
    for (size_t i = 0; i < batch_size; i++)
    {
        std::cout << "Mean of torch sample " << i << ": " << mean[i].item<float>() << std::endl;
    }
    
    auto variance = input.var(1, false, true);
    //print torch variance
    for (size_t i = 0; i < batch_size; i++)
    {
        std::cout << "Variance of torch sample " << i << ": " << variance[i].item<float>() << std::endl;
    }
    torch::Tensor gamma = torch::randn({num_features}, torch::requires_grad());
    torch::Tensor beta = torch::randn({num_features}, torch::requires_grad());
    auto layer_norm = torch::nn::LayerNorm(torch::nn::LayerNormOptions({num_features}).elementwise_affine(true));
    auto output = layer_norm->forward(input);
    torch::Tensor grad_output = torch::randn_like(output);
    output.backward(grad_output);
    float *input_ptr = input.data_ptr<float>();
    float *gamma_ptr = layer_norm->weight.data_ptr<float>();
    float *beta_ptr = layer_norm->bias.data_ptr<float>();
    float *grad_output_ptr = grad_output.data_ptr<float>();
    float *expected_input_grad = input.grad().data_ptr<float>();
    float *expected_gamma_grad = layer_norm->weight.grad().data_ptr<float>();
    float *expected_beta_grad = layer_norm->bias.grad().data_ptr<float>();  
    float *actual_input_grad = new float[batch_size * num_features];
    float *actual_gamma_grad = new float[num_features];
    float *actual_beta_grad = new float[num_features];
    std::fill(actual_input_grad, actual_input_grad + batch_size * num_features, 0.0f);
    std::fill(actual_gamma_grad, actual_gamma_grad + num_features, 0.0f);
    std::fill(actual_beta_grad, actual_beta_grad + num_features, 0.0f);
    // ACT
    layer_normalization_backward(input_ptr, grad_output_ptr, actual_input_grad, batch_size, num_features,
        gamma_ptr, actual_gamma_grad, actual_beta_grad);
    // ASSERT
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_features; ++j) {
            EXPECT_NEAR(actual_input_grad[i * num_features + j], expected_input_grad[i * num_features + j], 1e-3)
                << "Mismatch in input gradient at (" << i << ", " << j << ")";
        }
    }
    for (int j = 0; j < num_features; ++j) {
        EXPECT_NEAR(actual_gamma_grad[j], expected_gamma_grad[j], 1e-3)
            << "Mismatch in gamma gradient at (" << j << ")";
        EXPECT_NEAR(actual_beta_grad[j], expected_beta_grad[j], 1e-3)
            << "Mismatch in beta gradient at (" << j << ")";
    }   
}

 