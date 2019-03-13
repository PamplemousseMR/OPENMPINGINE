#include <mpi.h>
#include <iostream>

int main(int argc, char** argv)
{
    int size, rank;
    MPI_Init(&argc,&argv);

    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    std::cout << "Hello World from process " << rank << "/" << size << std::endl;
    std::flush(std::cout);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
