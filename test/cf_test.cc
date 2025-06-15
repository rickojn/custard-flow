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

    EXPECT_FLOAT_EQ(grads_B[0], 65);
    EXPECT_FLOAT_EQ(grads_B[1], 85);
    EXPECT_FLOAT_EQ(grads_B[2], 105);
    EXPECT_FLOAT_EQ(grads_B[3], 90);
    EXPECT_FLOAT_EQ(grads_B[4], 120);
    EXPECT_FLOAT_EQ(grads_B[5], 150);
}

TEST(MatrixMultiplicationTest, CompareWithLibTorch) {
    // Create random matrices using LibTorch
    torch::manual_seed(42); // For reproducibility
    torch::Tensor A = torch::rand({3, 3});
    torch::Tensor B = torch::rand({3, 3});
    torch::Tensor expected = torch::mm(A, B);

    // Extract raw pointers
    float* A_ptr = A.data_ptr<float>();
    float* A_ptr_transposed = A.t().data_ptr<float>();
    float* B_ptr = B.data_ptr<float>();
    float* B_ptr_transposed = (float*)malloc(B.size(0) * B.size(1) * sizeof(float));
    transpose_matrix(B_ptr, B_ptr_transposed, B.size(0), B.size(1));
    float* expected_ptr = expected.data_ptr<float>();
    // Get dimensions
    int m = A.size(0);
    int k = A.size(1);
    int n = B.size(1);

    // Allocate memory for the result matrix
    float* my_result_ptr = new float[m * n];
    std::fill(my_result_ptr, my_result_ptr + m * n, 0.0f); // Initialize to zero

    // Call your custom function
    naive_matmul(A_ptr, B_ptr_transposed, my_result_ptr, m, k, n);

    // Print the matrices for debugging
    std::cout << "Matrix A:\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            std::cout << A_ptr[i * k + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "Matrix B:\n";
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << B_ptr[i * n + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "Expected Result:\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << expected_ptr[i * n + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "My Result:\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << my_result_ptr[i * n + j] << " ";
        }
        std::cout << "\n";
    }
    // Print the dimensions
    std::cout << "Dimensions: A(" << m << ", " << k << "), B(" << k << ", " << n << "), Result(" << m << ", " << n << ")\n";

    // Compare results
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_FLOAT_EQ(my_result_ptr[i * n + j], expected_ptr[i * n + j])
                << "Mismatch at (" << i << ", " << j << ")";
        }
    }
    // Free allocated memory
    delete[] my_result_ptr;
    free(B_ptr_transposed);
}

TEST(MatrixMultiplicaitonBackwards, CompareWithTorch){
    // Create random matrices using LibTorch
    torch::manual_seed(42); // For reproducibility
    torch::Tensor A = torch::rand({3, 3});
    torch::Tensor B = torch::rand({3, 3});
    torch::Tensor grads_C = torch::rand({3, 3});

    // Extract raw pointers
    float* A_ptr = A.data_ptr<float>();
    float* B_ptr = B.data_ptr<float>();
    float* grads_C_ptr = grads_C.data_ptr<float>();

    // Get dimensions
    int m = A.size(0);
    int k = A.size(1);
    int n = B.size(1);

    // Allocate memory for the result matrices
    float* grads_A_ptr = new float[m * k];
    float* grads_B_ptr = new float[k * n];
    std::fill(grads_A_ptr, grads_A_ptr + m * k, 0.0f); // Initialize to zero
    std::fill(grads_B_ptr, grads_B_ptr + k * n, 0.0f); // Initialize to zero

    // Call your custom function
    matmul_backwards(grads_C_ptr, B_ptr, A_ptr, grads_B_ptr, grads_A_ptr, m, n, k);

    // Print the matrices for debugging
    std::cout << "Matrix A:\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            std::cout << A_ptr[i * k + j] << " ";
        }
        std::cout << "\n";
    }
    
    std::cout << "Matrix B:\n";
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << B_ptr[i * n + j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "Grads C:\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << grads_C_ptr[i * n + j] << " ";
        }
        std::cout << "\n";
    }
}