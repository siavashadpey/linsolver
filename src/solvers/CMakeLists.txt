include_directories(${project_include})

set(SOLVERS_SRC
	${CMAKE_CURRENT_SOURCE_DIR}/base_solver.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/base_direct_solver.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/base_iterative_solver.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/gmres.cpp
	PARENT_SCOPE
)