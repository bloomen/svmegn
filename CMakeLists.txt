cmake_minimum_required(VERSION 3.0)
project(svmegn)

option(svmegn_build_tests "Build svmegn tests." OFF)
option(svmegn_enable_asan "Build svmegn with address sanitizer." OFF)
option(svmegn_enable_tsan "Build svmegn with thread sanitizer." OFF)

if(svmegn_enable_asan AND svmegn_enable_tsan)
    message(FATAL_ERROR "svmegn_enable_asan and svmegn_enable_tsan cannot both be ON")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

if(EXISTS "${PROJECT_SOURCE_DIR}/cmake/${CMAKE_BUILD_TYPE}.cmake")
    include("${PROJECT_SOURCE_DIR}/cmake/${CMAKE_BUILD_TYPE}.cmake")
endif()

function(svmegn_add_flags target warnings)
    set_property(TARGET ${target} PROPERTY C_STANDARD 11)
    set_property(TARGET ${target} PROPERTY C_EXTENSIONS OFF)
    set_property(TARGET ${target} PROPERTY CXX_STANDARD 17)
    set_property(TARGET ${target} PROPERTY CXX_EXTENSIONS OFF)
    target_include_directories(${target} PRIVATE include)
    if(MSVC)
        target_compile_options(${target} PRIVATE /bigobj /EHsc /wd4996)
        if(warnings)
            target_compile_options(${target} PRIVATE /W4 /wd4503 /wd4702 /wd4100 /wd4706 /wd4068 /wd4127)
        endif()
        if(${MSVC_VERSION} GREATER_EQUAL 1929)
            if(svmegn_enable_asan)
                target_compile_options(${target} PRIVATE /fsanitize=address)
            endif()
        endif()
        add_definitions(-D_CRT_SECURE_NO_WARNINGS)
        add_definitions(-D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
    else()
        if(warnings)
            target_compile_options(${target} PRIVATE -Wall -Wconversion -Wextra -Wpedantic)
        endif()
        target_link_libraries(${target} PRIVATE ${CMAKE_THREAD_LIBS_INIT})
        if(CMAKE_COMPILER_IS_GNUCC)
            target_compile_options(${target} PRIVATE -pthread)
        endif()
        if(svmegn_enable_asan)
            if(APPLE)
                target_compile_options(${target} PRIVATE -fsanitize=address,undefined)
                set_target_properties(${target} PROPERTIES LINK_FLAGS "-fsanitize=address,undefined")
            else()
                target_compile_options(${target} PRIVATE -fsanitize=address,leak,undefined)
                set_target_properties(${target} PROPERTIES LINK_FLAGS "-fsanitize=address,leak,undefined")
            endif()
        endif()
        if(svmegn_enable_tsan)
            target_compile_options(${target} PRIVATE -fsanitize=thread)
            set_target_properties(${target} PROPERTIES LINK_FLAGS "-fsanitize=thread")
        endif()
    endif()
endfunction()

find_package(Threads)

add_library(svmegn_libsvm
    src/libsvm/svm.h
    src/libsvm/svm.cpp
)
svmegn_add_flags(svmegn_libsvm OFF)

add_library(svmegn_liblinear
    src/liblinear/blas/blas.h
    src/liblinear/blas/blasp.h
    src/liblinear/blas/daxpy.c
    src/liblinear/blas/ddot.c
    src/liblinear/blas/dnrm2.c
    src/liblinear/blas/dscal.c
    src/liblinear/linear.h
    src/liblinear/linear.cpp
    src/liblinear/newton.h
    src/liblinear/newton.cpp
)
svmegn_add_flags(svmegn_liblinear OFF)

add_library(svmegn
    include/svmegn.h
    src/svmegn.cpp
)
svmegn_add_flags(svmegn ON)
target_link_libraries(svmegn PRIVATE svmegn_libsvm svmegn_liblinear)

set_target_properties(svmegn PROPERTIES PUBLIC_HEADER include/svmegn.h)
install(TARGETS svmegn ARCHIVE DESTINATION lib LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION include)

if(svmegn_build_tests)
    enable_testing()
    add_executable(svmegn_test include/svmegn.h test/test.cpp)
    svmegn_add_flags(svmegn_test ON)
    target_link_libraries(svmegn_test PRIVATE svmegn gtest$<$<CONFIG:Debug>:d>)
    add_test(svmegn_test svmegn_test)
endif()

set(svmegn_source_files
    ${PROJECT_SOURCE_DIR}/include/svmegn.h
    ${PROJECT_SOURCE_DIR}/src/svmegn.cpp
    ${PROJECT_SOURCE_DIR}/test/test.cpp)

add_custom_target(
    svmegn_format
    COMMAND clang-format
    -style=file
    -i
    ${svmegn_source_files}
)
