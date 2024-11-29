cmake_minimum_required(VERSION 3.14)
project(ALL_MINIMAX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set runtime library settings
if(MSVC)
    # Use static runtime
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    # Add compiler warnings
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra)
endif()

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
)

# Configure Google Test to use static runtime
set(gtest_force_shared_crt OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# Add the library
add_library(opt opt.cpp)
target_include_directories(opt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Set runtime library for opt library
if(MSVC)
    target_compile_options(opt PRIVATE /MT$<$<CONFIG:Debug>:d>)
endif()

enable_testing()

# Add the test executable
add_executable(test_opt test_opt.cpp)
target_link_libraries(test_opt PRIVATE opt gtest_main)

# Set runtime library for test executable
if(MSVC)
    target_compile_options(test_opt PRIVATE /MT$<$<CONFIG:Debug>:d>)
endif()

include(GoogleTest)
gtest_discover_tests(test_opt)
