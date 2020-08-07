include_directories(${project_include})

set(DEVICE_SRC
	${CMAKE_CURRENT_SOURCE_DIR}/device_vector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device_matrix.cpp
	PARENT_SCOPE
)