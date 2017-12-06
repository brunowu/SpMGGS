static char help[] = "Mat Mat mult on GPU\n";

#include <cusparse_v2.h>
#include <stdio.h>
#include <petscksp.h>
#include <../src/mat/impls/aij/seq/aij.h>
#include "mat_cuda.h"
#define N 10

// matrix generation and validation depends on these relationships:
#define SCL 2
#define K N
#define M (SCL*N)
// A: MxK  B: KxN  C: MxN

// error check macros
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

// perform sparse-matrix multiplication C=AxB
PetscErrorCode MatSeqCopy2GPU(Mat A,MatCUDA *B){

  Mat_SeqAIJ         *a =(Mat_SeqAIJ*)A->data;
  PetscInt           *ai=a->i, *aj=a->j;
  PetscScalar        *aa=a->a;
  PetscInt	     m, n;
  PetscInt	     nnz;
  int i;
  MatGetSize(A,&m,&n);
  B->row = m;
  B->col = n;
  nnz = ai[m] - ai[0];

  B->nnz = nnz;
   
  int *h_csrRowPtrA, *h_csrColIndA;
  double *h_csrValA;

  h_csrRowPtrA = (int *)malloc((m+1)*sizeof(int));
  h_csrColIndA = (int *)malloc(nnz*sizeof(int));  
  h_csrValA  = (double *)malloc(nnz*sizeof(double));

  for(i=0;i<m+1;i++){
    h_csrRowPtrA[i] = ai[i];
  }

  for(i=0;i<nnz;i++){
    h_csrColIndA[i] = aj[i];
    h_csrValA[i] = PetscRealPart(aa[i]);
  }

  cudaMalloc(&B->csrRowPtr, (m+1)*sizeof(int));
  cudaMalloc(&B->csrColInd, nnz*sizeof(int));
  cudaMalloc(&B->csrVal, nnz*sizeof(double));
  
  cudaMemcpy(B->csrRowPtr, h_csrRowPtrA, (m+1)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(B->csrColInd, h_csrColIndA, nnz*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(B->csrVal, h_csrValA, nnz*sizeof(double), cudaMemcpyHostToDevice);

 
  return 0;

}

PetscErrorCode MatMatMult_SeqGPU(MatCUDA A, MatCUDA B, MatCUDA *C, cusparseHandle_t hndl){

  int baseC;
  int *nnzTotalDevHostPtr = &C->nnz;
  cusparseMatDescr_t descrA, descrB, descrC;
  cusparseStatus_t stat;
  CUSPARSE_CHECK(cusparseCreate(&hndl));
  stat = cusparseCreateMatDescr(&descrA);
  CUSPARSE_CHECK(stat);
  stat = cusparseCreateMatDescr(&descrB);
  CUSPARSE_CHECK(stat);
  stat = cusparseCreateMatDescr(&descrC);
  CUSPARSE_CHECK(stat);
  stat = cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
  CUSPARSE_CHECK(stat);
  stat = cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL);
  CUSPARSE_CHECK(stat);
  stat = cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);
  CUSPARSE_CHECK(stat);
  stat = cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
  CUSPARSE_CHECK(stat);
  stat = cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO);
  CUSPARSE_CHECK(stat);
  stat = cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);
  CUSPARSE_CHECK(stat);

  cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t transB = CUSPARSE_OPERATION_NON_TRANSPOSE;

  stat = cusparseSetPointerMode(hndl, CUSPARSE_POINTER_MODE_HOST);

  CUSPARSE_CHECK(stat);

  cudaMalloc((void**)&C->csrRowPtr, sizeof(int)*(A.row+1));

  cudaCheckErrors("cudaMalloc fail");

  C->row = A.row;
  C->col = B.col;
  stat = cusparseXcsrgemmNnz(hndl, transA, transB, A.row, B.col, A.col,
        descrA, A.nnz, A.csrRowPtr, A.csrColInd,
        descrB, B.nnz, B.csrRowPtr, B.csrColInd,
        descrC, C->csrRowPtr, nnzTotalDevHostPtr );
  CUSPARSE_CHECK(stat);
  if (NULL != nnzTotalDevHostPtr){
    C->nnz = *nnzTotalDevHostPtr;}
  else{
    cudaMemcpy(&C->nnz, C->csrRowPtr+A.row, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&baseC, C->csrRowPtr, sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy fail");
    C->nnz -= baseC;}

  cudaMalloc((void**)&C->csrColInd, sizeof(int)*C->nnz);
  cudaMalloc((void**)&C->csrVal, sizeof(double)*C->nnz);
  cudaCheckErrors("cudaMalloc fail");
// perform multiplication C = A*B
  stat = cusparseDcsrgemm(hndl, transA, transB, A.row, B.col, A.col,
        descrA, A.nnz,
        A.csrVal, A.csrRowPtr, A.csrColInd,
        descrB, B.nnz,
        B.csrVal, B.csrRowPtr, B.csrColInd,
        descrC,
        C->csrVal, C->csrRowPtr, C->csrColInd);
  CUSPARSE_CHECK(stat);

  return 0;
}

PetscErrorCode MatSeqCopy2HOST(MatCUDA B, Mat *A){
  
  int *h_csrRowPtr, *h_csrColInd;
  double *h_csrVal;
  PetscScalar *val;
  int i;

  PetscMalloc1(B.nnz, &val);

  h_csrRowPtr = (int *)malloc((B.row+1)*sizeof(int));
  h_csrColInd = (int *)malloc(B.nnz *sizeof(int));
  h_csrVal  = (double *)malloc(B.nnz *sizeof(double));

  cudaMemcpy(h_csrRowPtr, B.csrRowPtr, (B.row+1)*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_csrColInd, B.csrColInd,  B.nnz*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_csrVal, B.csrVal, B.nnz*sizeof(double), cudaMemcpyDeviceToHost);
  cudaCheckErrors("cudaMemcpy fail");
 
  for(i=0;i<B.nnz;i++){
    val[i] = h_csrVal[i];
  }  

  MatCreateSeqAIJWithArrays(MPI_COMM_SELF,B.row,B.col,h_csrRowPtr,h_csrColInd,val,A);
  
  return 0;
}

int main(int argc,char **argv){

  PetscErrorCode     ierr;
  Mat                A, B;
  PetscScalar	     v;
  cusparseHandle_t hndl;

  PetscInitialize(&argc,&argv,0,help);
  MatCreateSeqAIJ(PETSC_COMM_WORLD,M,N,1,NULL,&A);
  MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
  v = 1.0;
  for(int i = 0; i < N; i++){
	int j = 2*i;
        ierr = MatSetValues(A,1,&j,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
	int m = j+1;
	ierr = MatSetValues(A,1,&m,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  MatView(A,PETSC_VIEWER_STDOUT_SELF);

  MatCreateSeqAIJ(PETSC_COMM_WORLD,N,N,1,NULL,&B);

  MatSetOption(B, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
  
  v = 2.0;
  for (int i=0; i<N; i++) {
    ierr = MatSetValues(B,1,&i,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  MatView(B,PETSC_VIEWER_STDOUT_SELF);

  MatCUDA C;
  MatSeqCopy2GPU(A, &C);
  printf("------\n");
  printf("C row = %d \n", C.row);
  printf("------\n");
 
  MatCUDA D;
  MatSeqCopy2GPU(B, &D);
  printf("------\n");
  printf("D row = %d \n", D.row);
  printf("------\n"); 

  
/*
   |1.0 0.0 0.0 ...|
   |1.0 0.0 0.0 ...|
   |0.0 1.0 0.0 ...|
   |0.0 1.0 0.0 ...|
   |0.0 0.0 1.0 ...|
   |0.0 0.0 1.0 ...|
   ...

   B:
   |2.0 0.0 0.0 ...|
   |0.0 2.0 0.0 ...|
   |0.0 0.0 2.0 ...|
   ...                */

// set cusparse matrix types
  
  CUSPARSE_CHECK(cusparseCreate(&hndl));

  MatCUDA E;
  
  MatMatMult_SeqGPU(C, D, &E, hndl);

  printf("E->row = %d\n", E.row);
  printf("-------");
 
  Mat F;
  MatSeqCopy2HOST(E,&F);
  MatView(F,PETSC_VIEWER_STDOUT_SELF);
  printf("-------");
  printf("Success!\n");
  
  MatDestroy(&A);
  PetscFinalize();

  return ierr;
}

