#ifndef _LIBS_H
#define _LIBS_H

#define PI 3.1415926
#define epsilon 1

#define max(a,b) (a>=b?a:b)
#define min(a,b) (a<=b?a:b)

#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h> 
#include <petscvec.h>
#include <petscksp.h>
#include "petsc.h"

typedef struct _MatrixInfo{
	int n;
	int m;
	int nnz;
} MatrixInfo;

PetscErrorCode getFileSize(const char * name, PetscInt * size);

PetscErrorCode readBinaryScalarArray(const char * name, PetscInt * nb, PetscScalar * array);

void random_selection(PetscScalar *ret, PetscInt nombre);

void selection(PetscScalar *ret, PetscInt nombre, PetscInt min, PetscInt max);

void change(PetscScalar *array, PetscInt n, PetscReal ratio);

void shuffer(PetscScalar *array, PetscInt n);

PetscInt *indexShuffer(PetscInt n);

PetscInt factorial(PetscInt start, PetscInt end);

PetscScalar Random (PetscInt _iMin, PetscInt _iMax);

PetscInt IRandom (PetscInt _iMin, PetscInt _iMax);

PetscReal rnd(PetscReal min, PetscReal max);

void printarray(PetscInt n, PetscScalar *a);

#endif
