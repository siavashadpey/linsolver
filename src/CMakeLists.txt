add_subdirectory(solvers)
add_subdirectory(backends)
add_subdirectory(base)

if (CUDA_ENABLED)
    set(HOST_SRC 
	     ${HOST_SRC}
       ${BASE_BACKEND_SRC}
    )
    set (DEVICE_SRC
        ${DEVICE_SRC}
        ${SOLVERS_SRC}
        ${BASE_SRC}
        )
else()
    set(HOST_SRC 
       ${BASE_BACKEND_SRC}
       ${HOST_SRC}
       ${SOLVERS_SRC}
    )
endif()

add_library(${LIB_NAME} SHARED ${HOST_SRC})
target_link_libraries(${LIB_NAME} ${LINK_LIBS})
IF (OPENMP_FOUND)
	target_link_libraries(${LIB_NAME} OpenMP::OpenMP_CXX)
endif()

if (CUDA_ENABLED)
	foreach(i ${DEVICE_SRC})
        set_source_files_properties(${i} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
  	endforeach()

  	cuda_add_library(${LIB_NAME}_CUDA SHARED ${DEVICE_SRC})
    target_link_libraries(${LIB_NAME}_CUDA ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusparse_LIBRARY})
  	target_link_libraries(${LIB_NAME} ${LIB_NAME}_CUDA)
endif()