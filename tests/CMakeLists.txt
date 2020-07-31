add_executable(host_vector_test host_vector_test.cpp)
target_link_libraries(host_vector_test ${LIB_NAME})

add_executable(host_matrix_test host_matrix_test.cpp)
target_link_libraries(host_matrix_test ${LIB_NAME})


add_executable(gmres_test gmres_test.cpp)
target_link_libraries(gmres_test ${LIB_NAME})