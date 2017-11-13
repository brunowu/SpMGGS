#include "ebmg.h"

static char help[] = "Matrix Generator by selected eigenvalues";

int main(int argc, char ** argv){
	 int             world_size;

	PetscInitialize(&argc,&argv,(char *)0,help);

	MPI_Comm_size(PETSC_COMM_WORLD, &world_size);
    	PetscPrintf(PETSC_COMM_WORLD, "--------------------------\n");
	PetscPrintf(PETSC_COMM_WORLD, "Using number of %d precessors for the generation...\n", world_size);
/*
        #if defined(PETSC_USE_COMPLEX)
                PetscPrintf(PETSC_COMM_WORLD,"Generating the Complex eigenvalues...\n\n");
        #elif defined(PETSC_USE_REAL)
                PetscPrintf(PETSC_COMM_WORLD,"Generating the REAL eigenvalues...\n\n");
        #endif
*/
	EBMG();

	PetscFinalize();

	return 0;
}
