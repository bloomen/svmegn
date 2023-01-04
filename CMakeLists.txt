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

function(svmegn_add_flags target)
    set_property(TARGET ${target} PROPERTY CXX_STANDARD 17)
    set_property(TARGET ${target} PROPERTY CXX_EXTENSIONS OFF)
    target_include_directories(${target} PRIVATE include)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /bigobj /EHsc /wd4503 /wd4996 /wd4702 /wd4100 /wd4706 /wd4068)
        if(${MSVC_VERSION} GREATER_EQUAL 1929)
            if(svmegn_enable_asan)
                target_compile_options(${target} PRIVATE /fsanitize=address)
            endif()
        endif()
        add_definitions(-D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wconversion -Wextra -Wpedantic)
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

add_library(svmegn include/svmegn.h src/svmegn.cpp src/svm.h src/svm.cpp)
svmegn_add_flags(svmegn)

set_target_properties(svmegn PROPERTIES PUBLIC_HEADER include/svmegn.h)
install(TARGETS svmegn ARCHIVE DESTINATION lib LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION include)

if(svmegn_build_tests)
    enable_testing()
    add_executable(svmegn_test include/svmegn.h test/test.cpp)
    svmegn_add_flags(svmegn_test)
    target_link_libraries(svmegn_test svmegn)
    add_test(svmegn_test svmegn_test)
endif()

add_custom_target(
    svmegn_format
    COMMAND clang-format
    -style=file
    -i
    include/svmegn.h
    src/svmegn.cpp
    test/test.cpp
)
