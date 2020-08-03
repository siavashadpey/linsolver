find_package(Threads REQUIRED)
find_program(MEMORYCHECK_COMMAND valgrind)

# taken from APS
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
                 ${CMAKE_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

function(add_tests test_set)
  # note that we include/link against Armadillo and MPI header/libraries
  include_directories(${project_include})
  foreach(filename ${test_set})
    add_executable(${filename}_test ${filename}_test.cpp)
    target_link_libraries(${filename}_test ${LIB_NAME} gtest_main)
    add_test(NAME ${filename}_test COMMAND ./${filename}_test --gtest_output=xml:./${filename}_test_result.xml)
  endforeach()
endfunction(add_tests)

# do not add suffix _test.cpp here
set(SERIAL_TEST_SET
	host_vector
	host_matrix
	gmres
)

add_tests("${SERIAL_TEST_SET}")

