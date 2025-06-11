#include <gtest/gtest.h>
#include "CustardFlow.h"
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
   C  [14, 32
       32, 77]

   Bt [1,2,3,
       4,5,6]

c11 = a11*b11 + a12*b21 + 

grads_C [10,20,
         30,40]       2 x 2

grads_A [90,120,150,
         190,260,330]
	 
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
}


