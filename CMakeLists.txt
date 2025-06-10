cmake_minimum_required(VERSION 3.14)
project(my_project)

# GoogleTest requires at least C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Build the CustardFlow library
add_library(custard-flow
    src/CustardFlow.c
)
target_include_directories(custard-flow PUBLIC ${CMAKE_SOURCE_DIR}/include)

enable_testing()

add_executable(
  hello_test
  test/hello_test.cc
)
target_link_libraries(
  hello_test
  GTest::gtest_main
  custard-flow
)

# Enable intrinsics (SSE/AVX) if needed
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(custard-flow PRIVATE -march=native)
elseif(MSVC)
    target_compile_options(custard-flow PRIVATE /arch:AVX)
endif()

include(GoogleTest)
gtest_discover_tests(hello_test)