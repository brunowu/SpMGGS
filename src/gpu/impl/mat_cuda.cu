static char help[] = "Mat Mat mult on GPU\n";

#include <cusparse_v2.h>
#include <stdio.h>
#include <petscksp.h>
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/utils/freespace.h>
#include "mat_cuda.h"
#include <petsctime.h>
#define N 10

// matrix generation and validation depends on these relationships:
#define SCL 1
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

PetscErrorCode MatMatMult_MPIAIJGPU(Mat A,Mat P,Mat *C, cusparseHandle_t hndl){
  PetscErrorCode ierr;
  MPI_Comm           comm;
  PetscMPIInt        size;
  Mat_PtAPMPI        *ptap;
  Mat_MPIAIJ         *a        =(Mat_MPIAIJ*)A->data;
  MatCUDA	     ploc, poth;
  MatCUDA	     Adia, Aoff;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  ierr = PetscNew(&ptap);CHKERRQ(ierr);
  ierr = MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_INITIAL_MATRIX,&ptap->startsj_s,&ptap->startsj_r,&ptap->bufa,&ptap->P_oth);CHKERRQ(ierr);
  ierr = MatMPIAIJGetLocalMat(P,MAT_INITIAL_MATRIX,&ptap->P_loc);CHKERRQ(ierr);


 
  MatSeqCopy2GPU(ptap->P_loc, &ploc);
  MatSeqCopy2GPU(ptap->P_oth, &poth);

  MatSeqCopy2GPU(a->A,&Adia);
  MatSeqCopy2GPU(a->B,&Aoff);

  MatCUDA Cdia, Coff;

  MatMatMult_SeqGPU(Adia, ploc, &Cdia,hndl);
  MatMatMult_SeqGPU(Aoff, poth, &Coff,hndl);

  Mat	Cdiag, Cofff;

  MatSeqCopy2HOST(Cdia,&Cdiag);
  MatSeqCopy2HOST(Coff,&Cofff);


  MatAXPY(Cdiag,1.0,Cofff, DIFFERENT_NONZERO_PATTERN);
  PetscPrintf(PETSC_COMM_WORLD,"TEST CDIAg + COFFdia\n");
  Mat_SeqAIJ *cd = (Mat_SeqAIJ*) Cdiag->data;
  PetscInt	*cdi = cd->i, *cdj = cd->j;
  PetscScalar *cda = cd->a;
  MatCreateMPIAIJWithArrays(MPI_COMM_WORLD,2,PETSC_DECIDE,4,4,cdi,cdj,cda,C);
 // MatView(*C,PETSC_VIEWER_STDOUT_WORLD);
  PetscFunctionReturn(0);

}
int main(int argc,char **argv){

  PetscErrorCode     ierr;
  Mat                A, B;
  PetscScalar	     v;
  cusparseHandle_t hndl;

  PetscInt        r, start, end;
 PetscInt m=2,n=2;
  PetscInitialize(&argc,&argv,0,help);
//  MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,M,N,1,NULL,0,NULL,&A);

 // MatView(A,PETSC_VIEWER_STDOUT_WORLD);

//  MatCreateSeqAIJ(PETSC_COMM_WORLD,N,N,1,NULL,&B);


PetscInt       i,j,Ii,J,Istart,Iend;
//  MatView(B,PETSC_VIEWER_STDOUT_WORLD);

 

 ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A,5,NULL);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0+Ii; i = Ii/n; j = Ii - i*n;
    if (i>0)   {J = Ii - n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (i<m-1) {J = Ii + n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (j>0)   {J = Ii - 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (j<n-1) {J = Ii + 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    v = 4.0+Ii; ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,ADD_VALUES);CHKERRQ(ierr);
  }


    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
   MatView(A,PETSC_VIEWER_STDOUT_WORLD); 



 ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(B,5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,5,NULL);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(B,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -100.0+Ii; i = Ii/n; j = Ii - i*n;
    if (i>0)   {J = Ii - n; ierr = MatSetValues(B,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (i<m-1) {J = Ii + n; ierr = MatSetValues(B,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (j>0)   {J = Ii - 1; ierr = MatSetValues(B,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (j<n-1) {J = Ii + 1; ierr = MatSetValues(B,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    v = 10.0+Ii; ierr = MatSetValues(B,1,&Ii,1,&Ii,&v,ADD_VALUES);CHKERRQ(ierr);
  }


    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
   MatView(B,PETSC_VIEWER_STDOUT_WORLD);


   Mat C;

   Mat D;
   PetscPrintf(PETSC_COMM_WORLD,"\n\nVERIFICATION!!!!---\n");
   MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&D);
   MatView(D,PETSC_VIEWER_STDOUT_WORLD);

  CUSPARSE_CHECK(cusparseCreate(&hndl));
   MatMatMult_MPIAIJGPU(A,B,&C, hndl);
//   MatView(C,PETSC_VIEWER_STDOUT_WORLD);  


  MatDestroy(&A);
  PetscFinalize();

  return ierr;
}

