#=======================================
#                PROJECT
#=======================================

cmake_minimum_required(VERSION 3.0)
project("OPENMPINGINE")

#========================================
#               LIBRARIES
#========================================

find_package("MPI" REQUIRED)

#========================================
#               EXECUTABLE
#========================================

set(simdngine_src
    src/openmpingine.cpp)

add_executable(OPENMPINGINE ${simdngine_src})
set_target_properties(OPENMPINGINE PROPERTIES CXX_STANDARD 14)

target_link_libraries(OPENMPINGINE PRIVATE MPI::MPI_CXX)