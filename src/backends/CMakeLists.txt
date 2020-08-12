include_directories(${project_include})

set(BASE_BACKEND_SRC
	${CMAKE_CURRENT_SOURCE_DIR}/base_vector.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/base_matrix.cpp 
	PARENT_SCOPE
)

add_subdirectory(host)
add_subdirectory(device)

set (HOST_SRC 
	${HOST_SRC}
	PARENT_SCOPE
)

set (DEVICE_SRC 
	${DEVICE_SRC}
	PARENT_SCOPE
)