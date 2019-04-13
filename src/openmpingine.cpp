#include <mpi.h>
#include <iostream>
#include <thread>
#include <chrono>

static const unsigned s_MATRIX_SIZE = 10;
static const unsigned s_MASTER = 0;

int main(int argc, char** argv)
{
    // Initialize MPI
    int procNumber;
    int procRank;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&procNumber);
    MPI_Comm_rank(MPI_COMM_WORLD,&procRank);

    if(s_MATRIX_SIZE%procNumber != 0)
    {
        if(procRank == s_MASTER)
        {
            std::cout << "The number of process must be a divider of the matrix size" << std::endl;
        }
        return EXIT_FAILURE;
    }

    const float localSize = s_MATRIX_SIZE/static_cast<float>(procNumber);
    const unsigned begin = static_cast<unsigned>(procRank * localSize);
    const unsigned end = static_cast<unsigned>(procRank * localSize + localSize);
    const unsigned range = end-begin;

    // Create a slice of A
    unsigned** const a = new unsigned*[range];
    for(unsigned i=0; i<range ; ++i)
    {
        a[i] = new unsigned[s_MATRIX_SIZE];
    }

    for(unsigned i=0 ; i<range ; ++i)
    {
        for(unsigned j=0 ; j<s_MATRIX_SIZE ; ++j)
        {
            a[i][j] = (i + procRank*range)*s_MATRIX_SIZE + j;
        }
    }

    // Create a slice of B
    unsigned** const b = new unsigned*[s_MATRIX_SIZE];
    for(unsigned i=0; i<s_MATRIX_SIZE ; ++i)
    {
        b[i] = new unsigned[range];
    }

    for(unsigned i=0 ; i<s_MATRIX_SIZE ; ++i)
    {
        for(unsigned j=0 ; j<range ; ++j)
        {
            b[i][j] = i*s_MATRIX_SIZE + (j + procRank*range);
        }
    }

    // Create a slice of C
    unsigned** const c = new unsigned*[s_MATRIX_SIZE];
    for(int i=0; i<s_MATRIX_SIZE ; ++i)
    {
        c[i] = new unsigned[range];
        std::memset(c[i], 0, sizeof (unsigned)*range);
    }

    // Compute multiplication
    const int next = (procRank+1)%procNumber;
    const int previous = (procRank+procNumber-1)%procNumber;
    for (int count=0, currentRank=procRank; count<procNumber ; ++count, currentRank=(currentRank+1)%procNumber)
    {
        for(unsigned i=0 ; i<range ; ++i)
        {
            for(unsigned j=0 ; j<range ; ++j)
            {
                for(unsigned k=0 ; k<s_MATRIX_SIZE ; ++k)
                {
                    int line = i + (currentRank * range);
                    c[line][j] += a[i][k] * b[k][j];
                }
            }
        }
        for(unsigned i=0 ; i<range ; ++i)
        {
            MPI_Sendrecv_replace(static_cast<void*>(&a[i][0]), s_MATRIX_SIZE, MPI_UNSIGNED, previous, 42, next, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // If the process is the master, get all data to display them
    if(procRank == s_MASTER)
    {
        // Create a slice of R
        unsigned** const r = new unsigned*[s_MATRIX_SIZE];
        for(unsigned i=0; i<s_MATRIX_SIZE ; ++i)
        {
            r[i] = new unsigned[s_MATRIX_SIZE];
            std::memset(r[i], 0, sizeof (unsigned)*s_MATRIX_SIZE);
        }


        // Colpy slice C into R
        for(unsigned i=0 ; i<s_MATRIX_SIZE ; ++i)
        {
            for(unsigned j=0 ; j<range ; ++j)
            {
                r[i][j] = c[i][j];
            }
        }

        for(unsigned proc=1 ; proc<procNumber ; ++proc)
        {
            for(unsigned i=0 ; i<s_MATRIX_SIZE ; ++i)
            {
                MPI_Recv(&r[i][proc*range], range, MPI_UNSIGNED, proc, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        // Display the result
        for(unsigned i=0 ; i<s_MATRIX_SIZE ; ++i)
        {
            for(unsigned j=0 ; j<s_MATRIX_SIZE ; ++j)
            {
                std::cout << r[i][j] << ", ";
            }
            std::cout << std::endl;
        }

        // Delet R
        for(unsigned i=0; i<s_MATRIX_SIZE ; ++i)
            delete[] r[i];
        delete[] r;
    }
    else
    {
        // Send local data to master
        for(unsigned i=0 ; i<s_MATRIX_SIZE ; ++i)
        {
            MPI_Send(c[i], range, MPI_UNSIGNED, s_MASTER, 42, MPI_COMM_WORLD);
        }
    }

    // Finalize MPI
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    // Delete A
    for(unsigned i=0; i<range ; ++i)
        delete[] a[i];
    delete[] a;

    // Delete B
    for(unsigned i=0; i<s_MATRIX_SIZE ; ++i)
        delete[] b[i];
    delete[] b;

    // Delete C
    for(unsigned i=0; i<s_MATRIX_SIZE ; ++i)
        delete[] c[i];
    delete[] c;

    return EXIT_SUCCESS;
}
