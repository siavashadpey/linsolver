add_subdirectory(solvers)
add_subdirectory(backends)

set(SRC 
	${SOLVERS_SRC}
	${BASE_SRC}
	${HOST_SRC}
)

add_library(${LIB_NAME} SHARED ${SRC})
target_link_libraries(${LIB_NAME} ${LINK_LIBS})


if (CUDA_ENABLED)
	foreach(i ${CUDA_SRC})
    	set_source_files_properties(${i} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
  	endforeach()

  	cuda_add_library(${LIB_NAME}_CUDA ${CUDA_SRC})
  	target_link_libraries(${LIB_NAME} ${LIB_NAME}_CUDA ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY})
endif()