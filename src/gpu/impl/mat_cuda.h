#ifndef MAT_CUDA_H
#define MAT_CUDA_H

#include <cusparse_v2.h>
#include <stdio.h>
#include <petscksp.h>
#include <../src/mat/impls/aij/seq/aij.h>

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

#endif



