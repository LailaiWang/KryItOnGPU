#include "mpi.h"

extern int driver1();

int main(int argc, char **args) {
 
#ifdef _MPI
    MPI_Init(&argc, &args);
    int size;
    int pid;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
#endif

    driver1();
#ifdef _MPI    
    MPI_Finalize();
#endif
    return 0;
}
