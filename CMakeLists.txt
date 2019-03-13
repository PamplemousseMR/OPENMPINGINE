#========================================
#	project
#========================================
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT("OPENMPINGINE" CXX)

#================================================================================
#		init
#================================================================================

#========================================
#	variables
#========================================
SET(openmpingine_project_name "openmpingine")
SET(openmpingine_src_dir "src")
SET(openmpingine_header_dir "include")

#========================================
#	compiler specifications
#========================================
SET(CMAKE_CXX_STANDARD 17)

#========================================
#	find files
#========================================
FILE(GLOB_RECURSE openmpingine_src_files ${openmpingine_src_dir}/*)
FILE(GLOB_RECURSE openmpingine_header_files ${openmpingine_header_dir}/*)

#================================================================================
#		create
#================================================================================

#========================================
#	create project
#========================================
ADD_EXECUTABLE(${openmpingine_project_name} ${openmpingine_src_files} ${openmpingine_header_files})
TARGET_INCLUDE_DIRECTORIES(${openmpingine_project_name} PRIVATE ${openmpingine_header_dir})

#========================================
#	find OpenMPI
#========================================
FIND_PACKAGE("MPI")

IF(NOT ${MPI_CXX_FOUND})
   MESSAGE(FATAL_ERROR "Can't find OpenMPI")
ENDIF()

TARGET_INCLUDE_DIRECTORIES(${openmpingine_project_name} PRIVATE ${MPI_CXX_INCLUDE_PATH})
TARGET_LINK_LIBRARIES(${openmpingine_project_name} ${MPI_CXX_LIBRARIES})
