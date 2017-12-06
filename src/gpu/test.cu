static char help[] = "Mat Mat mult on GPU\n";

#include <cusparse_v2.h>
#include <stdio.h>
#include <petscksp.h>
#include <../src/mat/impls/aij/seq/aij.h>

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
int main(int argc,char **argv){

  PetscErrorCode     ierr;
  Mat                A, B;
  PetscScalar	     v;
  cusparseStatus_t stat;
  cusparseHandle_t hndl;
  cusparseMatDescr_t descrA, descrB, descrC;
  int *csrRowPtrA, *csrRowPtrB, *csrRowPtrC, *csrColIndA, *csrColIndB, *csrColIndC;
  int *h_csrRowPtrA, *h_csrRowPtrB, *h_csrRowPtrC, *h_csrColIndA, *h_csrColIndB, *h_csrColIndC;
  float *csrValA, *csrValB, *csrValC, *h_csrValA, *h_csrValB, *h_csrValC;
  int nnzA, nnzB, nnzC;
  int m,n,k;
  m = M;
  n = N;
  k = K;

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


  Mat_SeqAIJ         *a =(Mat_SeqAIJ*)A->data, *b =(Mat_SeqAIJ*)B->data;
  PetscInt           *ai=a->i, *aj=a->j, *bi=b->i,*bj=b->j;
  PetscScalar        *aa=a->a, *bb=b->a;


  for(int i = 0; i < M+1; i++){
    printf("ai[%d] = %d \n",i,ai[i]);
  }


/* A:
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

  nnzA = m;
  nnzB = n;
  h_csrRowPtrA = (int *)malloc((m+1)*sizeof(int));
  h_csrRowPtrB = (int *)malloc((n+1)*sizeof(int));
  h_csrColIndA = (int *)malloc(m*sizeof(int));
  h_csrColIndB = (int *)malloc(n*sizeof(int));
  h_csrValA  = (float *)malloc(m*sizeof(float));
  h_csrValB  = (float *)malloc(n*sizeof(float));
  if ((h_csrRowPtrA == NULL) || (h_csrRowPtrB == NULL) || (h_csrColIndA == NULL) || (h_csrColIndB == NULL) || (h_csrValA == NULL) || (h_csrValB == NULL))
    {printf("malloc fail\n"); return -1;}
  for (int i = 0; i < m; i++){
    h_csrValA[i] = PetscRealPart(aa[i]);
    h_csrRowPtrA[i] = ai[i];
    h_csrColIndA[i] = aj[i];
    if (i < n){
      h_csrValB[i] = PetscRealPart(bb[i]);
      h_csrRowPtrB[i] = bi[i];
      h_csrColIndB[i] = bj[i];}
    }
  h_csrRowPtrA[m] = ai[m];
  h_csrRowPtrB[n] = ai[n];

// transfer data to device

  cudaMalloc(&csrRowPtrA, (m+1)*sizeof(int));
  cudaMalloc(&csrRowPtrB, (n+1)*sizeof(int));
  cudaMalloc(&csrColIndA, m*sizeof(int));
  cudaMalloc(&csrColIndB, n*sizeof(int));
  cudaMalloc(&csrValA, m*sizeof(float));
  cudaMalloc(&csrValB, n*sizeof(float));
  cudaCheckErrors("cudaMalloc fail");
  cudaMemcpy(csrRowPtrA, h_csrRowPtrA, (m+1)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(csrRowPtrB, h_csrRowPtrB, (n+1)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(csrColIndA, h_csrColIndA, m*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(csrColIndB, h_csrColIndB, n*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(csrValA, h_csrValA, m*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(csrValB, h_csrValB, n*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy fail");

// set cusparse matrix types
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

// figure out size of C
  int baseC;
// nnzTotalDevHostPtr points to host memory
  int *nnzTotalDevHostPtr = &nnzC;
  stat = cusparseSetPointerMode(hndl, CUSPARSE_POINTER_MODE_HOST);
  CUSPARSE_CHECK(stat);
  cudaMalloc((void**)&csrRowPtrC, sizeof(int)*(m+1));
  cudaCheckErrors("cudaMalloc fail");
  stat = cusparseXcsrgemmNnz(hndl, transA, transB, m, n, k,
        descrA, nnzA, csrRowPtrA, csrColIndA,
        descrB, nnzB, csrRowPtrB, csrColIndB,
        descrC, csrRowPtrC, nnzTotalDevHostPtr );
  CUSPARSE_CHECK(stat);
  if (NULL != nnzTotalDevHostPtr){
    nnzC = *nnzTotalDevHostPtr;}
  else{
    cudaMemcpy(&nnzC, csrRowPtrC+m, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&baseC, csrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy fail");
    nnzC -= baseC;}
  cudaMalloc((void**)&csrColIndC, sizeof(int)*nnzC);
  cudaMalloc((void**)&csrValC, sizeof(float)*nnzC);
  cudaCheckErrors("cudaMalloc fail");
// perform multiplication C = A*B
  stat = cusparseScsrgemm(hndl, transA, transB, m, n, k,
        descrA, nnzA,
        csrValA, csrRowPtrA, csrColIndA,
        descrB, nnzB,
        csrValB, csrRowPtrB, csrColIndB,
        descrC,
        csrValC, csrRowPtrC, csrColIndC);
  CUSPARSE_CHECK(stat);

// copy result (C) back to host
  h_csrRowPtrC = (int *)malloc((m+1)*sizeof(int));
  h_csrColIndC = (int *)malloc(nnzC *sizeof(int));
  h_csrValC  = (float *)malloc(nnzC *sizeof(float));
  if ((h_csrRowPtrC == NULL) || (h_csrColIndC == NULL) || (h_csrValC == NULL))
    {printf("malloc fail\n"); return -1;}
  cudaMemcpy(h_csrRowPtrC, csrRowPtrC, (m+1)*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_csrColIndC, csrColIndC,  nnzC*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_csrValC, csrValC, nnzC*sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckErrors("cudaMemcpy fail");

  for(int i = 0; i < m; i++){
    printf("ccolind[%d] = %d \n",i,h_csrColIndC[i]);
  }
  printf("-----\n");
  for(int i = 0; i < m; i++){
    printf("cVal[%d] = %f \n",i,h_csrValC[i]);
  }
// check result, C = 2A
  if (nnzC != m) {printf("invalid matrix size C: %d, should be: %d\n", nnzC, m); return -1;}
  for (int i = 0; i < m; i++){
    if (h_csrRowPtrA[i] != h_csrRowPtrC[i]) {printf("A/C row ptr mismatch at %d, A: %d, C: %d\n", i, h_csrRowPtrA[i], h_csrRowPtrC[i]); return -1;}
    if (h_csrColIndA[i] != h_csrColIndC[i]) {printf("A/C col ind mismatch at %d, A: %d, C: %d\n", i, h_csrColIndA[i], h_csrColIndC[i]); return -1;}
    if ((h_csrValA[i]*2.0f) != h_csrValC[i]) {printf("A/C value mismatch at %d, A: %f, C: %f\n", i, h_csrValA[i]*2.0f, h_csrValC[i]); return -1;}
    }
  printf("Success!\n");
  
  MatDestroy(&A);
  PetscFinalize();

  return ierr;
}

