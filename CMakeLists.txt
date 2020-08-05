cmake_minimum_required(VERSION 3.12)
cmake_policy(SET CMP0048 NEW)

project(linSolver VERSION 0.0)
set(LIB_NAME "linSolver")

option(WITH_CUDA "select whether CUDA is used or not" OFF)
option(BUILD_TESTS "select whether to build tests or not" ON)
option(BUILD_DOCS "select whether to build docs or not" OFF)
option(WITH_OPENMP "select whether OpenMP is used or not" ON)
option(WITH_COVERAGE "select whether code coverage is on or off (only works with DEBUG build)" OFF)

set(BUILD "DEBUG" CACHE STRING "select BUILD type: DEBUG or RELEASE")

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(LINK_LIBS "")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(GNU_WARNINGS "-Wall -Wextra -Wnon-virtual-dtor -Wpedantic -Wshadow")

if (${BUILD} MATCHES "DEBUG")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
	if (WITH_COVERAGE)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
	endif()
elseif (${BUILD} MATCHES "RELEASE")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GNU_WARNINGS}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

if (${WITH_CUDA})
	set(CUDA_ENABLED ON)
    SET(CUDA_SEPARABLE_COMPILATION ON)
	find_package(CUDA 10.1 REQUIRED)
	message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
    cuda_select_nvcc_arch_flags(ARCH_FLAGS)
    LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS} -Wno-deprecated-gpu-targets --expt-relaxed-constexpr -rdc=true)
    set(CUDA_PROPAGATE_HOST_FLAG OFF)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

if (${WITH_OPENMP})
	find_package(OpenMP)
endif()

add_subdirectory(src)

if (${BUILD_TESTS})
	include (CTest)
	add_subdirectory(tests)
endif()

if (${BUILD_DOCS})
	add_subdirectory(docs)
endif()