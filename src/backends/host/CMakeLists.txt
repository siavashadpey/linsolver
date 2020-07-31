include_directories(${project_include})

set(HOST_SRC
	${CMAKE_CURRENT_SOURCE_DIR}/host_vector.cpp 
	${CMAKE_CURRENT_SOURCE_DIR}/host_matrix.cpp
	PARENT_SCOPE
)