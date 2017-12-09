#ifndef MAT_CUDA_H
#define MAT_CUDA_H

#include <cusparse_v2.h>
#include <stdio.h>
#include <petscksp.h>
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petsctime.h>

#define CUSPARSE_CHECK(x) {cusparseStatus_t _c=x; if (_c != CUSPARSE_STATUS_SUCCESS) {printf("cusparse fail: %d, line: %d\n", (int)_c, __LINE__); exit(-1);}}

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

typedef struct MatCUDA{
	int row, col, nnz;
	int *csrRowPtr;
	int *csrColInd;
	double *csrVal;
}MatCUDA;

PetscErrorCode MatSeqCopy2GPU(Mat A,MatCUDA *B);

PetscErrorCode MatSeqCopy2HOST(MatCUDA B, Mat A);

PetscErrorCode MatMatMult_SeqGPU(MatCUDA A, MatCUDA B, MatCUDA *C, cusparseHandle_t hndl);

PetscErrorCode MatMatMult_MPIAIJGPU(Mat A,Mat P,Mat *C, cusparseHandle_t hndl);


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


#include "petscmat.h"
#include <math.h>

PetscErrorCode EBMG();


#endif



