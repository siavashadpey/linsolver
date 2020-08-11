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
  include_directories(${project_include})
  foreach(filename ${test_set})
    add_executable(${filename}_test ${filename}_test.cpp)
    target_link_libraries(${filename}_test ${LIB_NAME} gtest_main)
    add_test(NAME ${filename}_test COMMAND ./${filename}_test --gtest_output=xml:./${filename}_test_result.xml)
  endforeach()
endfunction(add_tests)

function(cuda_add_tests test_set)
  include_directories(${project_include})
  foreach(filename ${test_set})
    set_source_files_properties(${filename}_test.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    cuda_add_executable(${filename}_test ${filename}_test.cpp)
    target_link_libraries(${filename}_test ${LIB_NAME} gtest_main)
    add_test(NAME ${filename}_test COMMAND ./${filename}_test --gtest_output=xml:./${filename}_test_result.xml)
  endforeach()
endfunction(cuda_add_tests)

function(copy_data_files data_files)
  foreach(filename ${data_files})
    configure_file(${filename} ${CMAKE_CURRENT_BINARY_DIR}/${filename} COPYONLY)
  endforeach()
endfunction(copy_data_files)

# do not add suffix _test.cpp here
set(HOST_TEST_SET
	host_vector
	host_matrix
  host_matrix_coo
	host_gmres
  host_cg
  host_cgs
  host_jacobi
)

set(DATA_FILES 
  mm_test.mtx
  gre__115.mtx
)

add_tests("${HOST_TEST_SET}")
copy_data_files("${DATA_FILES}")

if (CUDA_ENABLED) 

  set(DEVICE_TEST_SET
    device_vector
    device_matrix
    device_gmres
  )

  cuda_add_tests("${DEVICE_TEST_SET}")

endif()

