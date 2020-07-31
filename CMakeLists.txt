cmake_minimum_required(VERSION 3.12)
cmake_policy(SET CMP0048 NEW)

project(LAS VERSION 0.1)
set(LIB_NAME "LAS")

option(WITH_CUDA "select whether CUDA is used or not" OFF)
option(BUILD_TESTS "select whether to build tests or not" OFF)
option(BUILD_DOCS "select whether to build docs or not" OFF)
option(WITH_VALGRIND "select if valgrind is ON or OFF (for tests)" OFF)

set(BUILD "DEBUG" CACHE STRING "select BUILD type: DEBUG or RELEASE")

set(LINK_LIBS "")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(GNU_WARNINGS "-Wall -Wextra -Wnon-virtual-dtor -Wpedantic -Wshadow")

if (${BUILD} MATCHES "DEBUG")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g --coverage")
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

add_subdirectory(src)

if (${BUILD_TESTS})
	enable_testing()
	add_subdirectory(tests)
endif()

if (${BUILD_DOCS})
	add_subdirectory(docs)
endif()