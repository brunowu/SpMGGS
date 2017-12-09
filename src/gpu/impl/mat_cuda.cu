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

//  cudaCheckErrors("cudaMalloc fail");
  cudaMalloc((void**)&C->csrColInd, sizeof(int)*C->nnz);
  cudaCheckErrors("cudaMalloc fail");
  cudaMalloc((void**)&C->csrVal, sizeof(double)*C->nnz);
  cudaCheckErrors("cudaMalloc fail");
  PetscPrintf(PETSC_COMM_WORLD,"debugging----\n");
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

//  printf("Aoff size = %d\n",Aoff.col);
//  printf("Hey man!\n");

  PetscInt Arow, Pcol;
  MatGetSize(A, &Arow,NULL);
  MatGetSize(P, NULL,&Pcol);
//  PetscPrintf(PETSC_COMM_WORLD, "MPI COMM World Size = %d \n",size);

  MatCUDA Cdia, Coff;
  Mat   Cdiag, Cofff;
  MatMatMult_SeqGPU(Adia, ploc, &Cdia,hndl);
   MatSeqCopy2HOST(Cdia,&Cdiag);
  if(Aoff.col != 0){
  	MatMatMult_SeqGPU(Aoff, poth, &Coff,hndl);
//  	printf("Aoff col !=0!\n");
        MatSeqCopy2HOST(Coff,&Cofff);
	MatAXPY(Cdiag,1.0,Cofff, DIFFERENT_NONZERO_PATTERN);
 }
  
  PetscPrintf(PETSC_COMM_WORLD,"TEST CDIAg + COFFdia\n");
  Mat_SeqAIJ *cd = (Mat_SeqAIJ*) Cdiag->data;
  PetscInt	*cdi = cd->i, *cdj = cd->j;
  PetscScalar *cda = cd->a;
  MatCreateMPIAIJWithArrays(MPI_COMM_WORLD,(PetscInt)(Arow/size),PETSC_DECIDE,Arow,Pcol,cdi,cdj,cda,C);
 // MatView(*C,PETSC_VIEWER_STDOUT_WORLD);
  PetscFunctionReturn(0);

}

PetscErrorCode getFileSize(const char * name, PetscInt * size){
  FILE * fptr;
  *size = 0L;

#ifdef LINUX
  struct stat fs;

  if(stat(name,&fs)!=0){
    perror("Cannot state file\n");
  }
  *size=fs.st_size;

#else
fptr=fopen(name,"rb");
  if(fptr!=NULL){
    fseek(fptr,0L,SEEK_END);
    *size = ftell(fptr);
    fclose(fptr);
  }
#endif

  return 0;
}

PetscErrorCode readBinaryScalarArray(const char * name, PetscInt * nb, PetscScalar * array){
  int file_descriptor;
  PetscErrorCode ierr;
  PetscInt size;

  getFileSize(name,&size);

  if(*nb<=0) *nb=(PetscInt)size/((PetscInt)sizeof(PetscScalar));
  if(size/sizeof(PetscScalar)!=*nb) {
    return 1;
  }


  ierr=PetscBinaryOpen(name,FILE_MODE_READ,&file_descriptor);CHKERRQ(ierr);
  ierr=PetscBinarySynchronizedRead(PETSC_COMM_WORLD,file_descriptor,array,*nb,PETSC_SCALAR);CHKERRQ(ierr);
  ierr=PetscBinaryClose(file_descriptor);CHKERRQ(ierr);

  return ierr;
}

void random_selection(PetscScalar *ret, PetscInt nombre)
{

  PetscInt      i;

  int                   my_seed,my_rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&my_rank);
  my_seed=time(NULL)+my_rank;

  srand(my_seed);

  for(i = 0; i < nombre; i++){


        PetscReal real, imag;
        srand(i);
        real = 10+10*i;
        imag = 10+10*i;
        #ifdef PETSC_USE_COMPLEX
                ret[i] = real+ PETSC_i * imag;
//              ret[nombre-1-i] =  real -  PETSC_i * imag;
        #else
                ret[i] = real;
//                ret[nombre-1-i] = real;
        #endif

    }




}

void selection(PetscScalar *ret, PetscInt nombre, PetscInt min, PetscInt max)
{

  PetscScalar   *tab;
  PetscInt    i, indice, maxi = max - min;

  if(min >= max || nombre > maxi + 1 || nombre < 1)
    PetscPrintf(PETSC_COMM_WORLD,"Input values of tirage() are wrong.\n");

  PetscMalloc((maxi + 1)*sizeof(tab[0]),&tab);

  for(i = 0; i < maxi + 1; i++)
    tab[i] = i + min;

  for(i = 0; i < nombre; i++){
    indice = rand() % (maxi + 1);
    ret[i] = tab[indice];
    tab[indice] = tab[maxi];
    maxi--;
  }
  PetscFree(tab);
}

void change(PetscScalar *array, PetscInt n, PetscReal ratio)
{
        PetscInt i;

        for(i = n-1; i>0;i--){
                if(i < ratio * n){
                        array[i] = -array[i];
                }
                else{
                        array[i] = array[i];
                }
        }
}

void shuffer(PetscScalar *array, PetscInt n)
{
        PetscInt index, i;
        PetscScalar tmp;
        srand(time(NULL));

        for(i = n-1; i>0;i--){
                index = rand() % i;
                tmp = array[i];
                array[i]=array[index];
                array[index]=tmp;
        }
}

PetscInt *indexShuffer(PetscInt n)
{
        PetscInt index, i;
        PetscInt *a;
        a = (PetscInt *) malloc(n*sizeof(PetscInt));
        PetscInt tmp;
        srand(time(NULL));
        for (i = 0; i < n; i++)
                a[i] = i;

        for(i = n-1; i>0;i--){
                index = rand() % i;
                tmp = a[i];
                a[i]=a[index];
                a[index]=tmp;
        }
        return a;
}

PetscInt factorial(PetscInt start, PetscInt end) {
  PetscInt i;
  PetscInt valeur;
  if(start>end){
    valeur = 1;
  }else{
    valeur = start;
    for(i= start+1; i <= end; i++){
        valeur *= i;
    }
  }
  return valeur;
}

PetscReal rnd(PetscReal min, PetscReal max){
       return (min + (max - min)*(rand() / (PetscReal)(RAND_MAX)));
}

PetscScalar Random (PetscInt _iMin, PetscInt _iMax)
{
  return (_iMin + (rand () % (_iMax-_iMin+1)));
}

PetscInt IRandom (PetscInt _iMin, PetscInt _iMax)
{
  return (_iMin + (rand () % (_iMax-_iMin+1)));
}

void printarray(PetscInt n, PetscScalar *a) {

        PetscScalarView(n, a, PETSC_VIEWER_STDOUT_WORLD);
}

PetscErrorCode EBMG(){

        char fileb[PETSC_MAX_PATH_LEN];

        Vec            eigenvalues;
        Mat            Mt, AM, MA, matAop;
        Mat            A;
        PetscErrorCode ierr;
        PetscInt       n,i,j,k,degree;
        PetscScalar    rRandom1;
        PetscInt       iRandom, d1;
        PetscInt        size;
        PetscBool      flagb,n_flg,degree_flg,d1_flg;
        MatInfo         Ainfo;
        double          gnnz;
        char            matrixOutputFile[PETSC_MAX_PATH_LEN];
        PetscViewer     output_viewer;
        clock_t start, finish;
        double  duration;
        PetscLogDouble st,ed;

        PetscInt Istart, Iend;

        ierr=PetscOptionsGetInt(NULL, PETSC_NULL,"-dim",&n, &n_flg);CHKERRQ(ierr); // flag -n read the dimension of matrix to generate
        ierr=PetscOptionsGetInt(NULL, PETSC_NULL,"-degree",&degree, &degree_flg);CHKERRQ(ierr); // flag -degree read the degree of nipotent matrix
        ierr=PetscOptionsGetInt(NULL, PETSC_NULL,"-d1",&d1, &d1_flg);CHKERRQ(ierr); //flag -d1 real the width of initial low-band matrix

        if (!n_flg){
                ierr = PetscPrintf(PETSC_COMM_WORLD,"!!!Please set the dimension of matrix to be generated\n");CHKERRQ(ierr);
       		return 0;
        }

        if (!degree_flg){
                degree = 5;
                ierr = PetscPrintf(PETSC_COMM_WORLD,"Unset degree, using default degree = %d\n", degree);CHKERRQ(ierr);
        }

        if (!d1_flg){
                d1 = 5;
                ierr = PetscPrintf(PETSC_COMM_WORLD,"Unset d1, using default degree = %d\n", d1);CHKERRQ(ierr);
        }

        #ifdef PETSC_USE_COMPLEX
                PetscPrintf(PETSC_COMM_WORLD,"Generating the Complex eigenvalues...\n\n");
        #else
                PetscPrintf(PETSC_COMM_WORLD,"Generating the REAL eigenvalues...\n\n");
        #endif

        PetscPrintf(PETSC_COMM_WORLD, "To generate matrix with dim = %d, degree = %d, d1 = %d ... \n", n, degree,d1);

        ierr = VecCreate(PETSC_COMM_WORLD,&eigenvalues);CHKERRQ(ierr);
        ierr = VecSetSizes(eigenvalues,PETSC_DECIDE,n);CHKERRQ(ierr);
        ierr = VecSetFromOptions(eigenvalues);CHKERRQ(ierr);

        PetscInt istart, iend;
        ierr = VecGetOwnershipRange(eigenvalues, &istart, &iend);

        PetscScalar *Deigenvalues;
        PetscMalloc1(n,&Deigenvalues);

        PetscPrintf(PETSC_COMM_WORLD, "--------------------------\n");
        ierr=PetscOptionsGetString(NULL,PETSC_NULL,"-vfile",fileb,PETSC_MAX_PATH_LEN-1,&flagb);CHKERRQ(ierr); //maybe read the eigenvalues we need from outside file

        if (!flagb){ //no input eigenvalues file, generate with some strategies
                PetscPrintf(PETSC_COMM_WORLD, "Not providing the outside eigenvalues files, using the internal functions to generate them...\n");
                random_selection(Deigenvalues,n);
        //      shuffer(Deigenvalues,n);
        }
        else{ //real the required eigenvalues from outside files
        PetscPrintf(PETSC_COMM_WORLD, "Using the eigenvalues provides by outside files: %s ...\n", fileb);
                readBinaryScalarArray(fileb, &size, Deigenvalues);
                change(Deigenvalues, size, 0.5);
                shuffer(Deigenvalues,size);
                ierr = PetscPrintf(PETSC_COMM_WORLD,"read file size = %ld\n", size);CHKERRQ(ierr);
                if(size != n){
                        ierr = PetscPrintf(PETSC_COMM_WORLD,"!!!read file size and vec dimemson do not match and mat dim set to be equal to vec dim\n");CHKERRQ(ierr);
                        return 0;
                }
                n = size;
        }

        PetscScalar tmp;

        PetscTime(&st);
        for(i=istart;i<iend;i++){
                tmp = Deigenvalues[i];
                VecSetValue(eigenvalues, i, tmp, INSERT_VALUES);
        }

///     VecView(eigenvalues, PETSC_VIEWER_STDOUT_WORLD);
        start = clock();

        ierr = MatCreate(PETSC_COMM_WORLD,&Mt);CHKERRQ(ierr);
        ierr = MatSetSizes(Mt,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
        ierr = MatSetType(Mt,MATMPIAIJ);CHKERRQ(ierr);
        ierr = MatSetFromOptions(Mt);CHKERRQ(ierr);
        ierr = MatSetUp(Mt);CHKERRQ(ierr);
//        MatSetOption(Mt, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
        ierr = MatGetOwnershipRange(Mt,&Istart,&Iend); CHKERRQ(ierr);
        ierr = MatDiagonalSet(Mt,eigenvalues,INSERT_VALUES);CHKERRQ(ierr);

/*Set up the low-band part of initial matrix*/
        for (i=Istart; i<Iend; i++){
          for (j=i-d1; j<i; j++){
                if(j >= 0)
                {
                rRandom1 = 0.5*Random (0,10);
                ierr = MatSetValues(Mt,1,&i,1,&j,&rRandom1,INSERT_VALUES);CHKERRQ(ierr);
                }
           }
        }


        ierr = MatAssemblyBegin(Mt,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(Mt,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatDuplicate(Mt,MAT_DO_NOT_COPY_VALUES,&A);CHKERRQ(ierr);

        PetscInt iI, jJ;
        PetscScalar vV=1;
        i=0;

/*set up the part with the largest degree of nipotent matrix A */
        for (i=Istart; i<Iend; i++){
                if (i<degree) {
                        iI=i;
                        jJ=i+1;
                        ierr = MatSetValues(A,1,&iI,1,&jJ,&vV,INSERT_VALUES);CHKERRQ(ierr);
                        i++;
                }
        }

        i=degree+1;

/*set up the other part of nipotent matrix with degree random generated but small than the input one*/

        for (i=Istart; i<Iend; i++){
                if (i<n-1) {
                        srand (time (NULL));
                        iRandom = IRandom (1,degree);
                        for(j=0; j<min(iRandom,n-1-i); j++)
                        {
                        iI=i+j;
                        jJ=i+j+1;
                        ierr = MatSetValues(A,1,&iI,1,&jJ,&vV,INSERT_VALUES);CHKERRQ(ierr);
                        }

                        i=i+min(iRandom,n-1-i);
                        srand (time (NULL));
                        iRandom = IRandom (1,max(1,n-1-i));
                        i=i+iRandom;
                }
        }

        ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

        PetscInt my_factorielle_bornes=1;

        ierr = MatCreate(PETSC_COMM_WORLD,&matAop);CHKERRQ(ierr);
        ierr = MatSetSizes(matAop,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
        ierr = MatSetType(matAop,MATMPIAIJ);CHKERRQ(ierr);
        ierr = MatSetFromOptions(matAop);CHKERRQ(ierr);
//      MatSetOption(matAop, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
        ierr = MatSetUp(matAop);CHKERRQ(ierr);

        ierr = MatDuplicate(Mt,MAT_COPY_VALUES,&matAop);CHKERRQ(ierr);
//      MatCopy(Mt,matAop,DIFFERENT_NONZERO_PATTERN);
        my_factorielle_bornes =  factorial(1,(2*degree-2));
        ierr = MatScale(Mt,my_factorielle_bornes);CHKERRQ(ierr);

	PetscPrintf(PETSC_COMM_WORLD, "starting mmmult loops\n");
        cusparseHandle_t hndl;
        CUSPARSE_CHECK(cusparseCreate(&hndl));
        for (k=1; k<=(2*degree-2); k++) {
           MatMatMult_MPIAIJGPU(matAop,A, &MA, hndl);
           MatMatMult_MPIAIJGPU(A,matAop,&AM,hndl);
          ierr = MatAYPX(matAop,0,AM, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
          ierr = MatAXPY(matAop,-1,MA,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);//A*M-M*A
          my_factorielle_bornes =  factorial(k+1,2*degree-2); //2*(d-1)*(2d-3)****(k+1)
          ierr = MatAXPY(Mt,my_factorielle_bornes,matAop,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
          ierr = MatZeroEntries(AM);CHKERRQ(ierr);
          ierr = MatZeroEntries(MA);CHKERRQ(ierr);
        }

        my_factorielle_bornes =  factorial(1,(2*degree-2));
        PetscReal inv = (PetscReal) 1 / my_factorielle_bornes;
//        PetscPrintf(PETSC_COMM_WORLD,"\n@>inv = %f\n",inv);
        ierr = MatScale(Mt,inv);CHKERRQ(ierr);

        finish = clock();
        PetscTime(&ed);

        MatGetInfo(Mt,MAT_GLOBAL_SUM,&Ainfo);

        gnnz = Ainfo.nz_used;

/*
        sprintf(matrixOutputFile,"Rectangular_nb_%d_%dx%d_%g_nnz.gz",n, n,n,gnnz);

        PetscPrintf(PETSC_COMM_WORLD,"\n@>Dumping matrix to PETSc binary %s\n",matrixOutputFile);

        PetscViewerBinaryOpen(PETSC_COMM_WORLD,matrixOutputFile,FILE_MODE_WRITE,&output_viewer);
        PetscViewerPushFormat(output_viewer,PETSC_VIEWER_ASCII_INFO_DETAIL);
//      MatView(Mt,PETSC_VIEWER_STDOUT_WORLD);
        MatView(Mt,output_viewer);
        PetscViewerDestroy(&output_viewer);


        PetscPrintf(PETSC_COMM_WORLD,"\n@>Matrix %s Dumped\n\n",matrixOutputFile);
        PetscPrintf(PETSC_COMM_WORLD,"\n>>>>>>Please use the command 'gzip -d **' to unzip the file to binary file\n\n");
*/
        //MatView(Mt,PETSC_VIEWER_STDOUT_WORLD);

        duration = (double)(finish - start) / CLOCKS_PER_SEC;
        PetscLogDouble time = ed - st;
    	PetscPrintf(PETSC_COMM_WORLD, "--------------------------\n");
    	PetscPrintf(PETSC_COMM_WORLD,"\nElapsed time is %f seconds\n\n",duration);
    	PetscPrintf(PETSC_COMM_WORLD, "--------------------------\n");

        PetscLogDouble a,b, benchmark;


        PetscTime(&a);
        Mat Mult;
        MatMatMultSymbolic(Mt,A,PETSC_DEFAULT, &Mult);
        MatMatMultNumeric(Mt,A,Mult);
        PetscTime(&b);
        benchmark = b - a;
        PetscPrintf(PETSC_COMM_WORLD,"\nMatMult Bencnmark time is %f seconds\n\n",benchmark);
        PetscPrintf(PETSC_COMM_WORLD, "--------------------------\n");
        PetscPrintf(PETSC_COMM_WORLD, "---------Done!!!!!!-------\n");

        MatDestroy(&Mt);
        MatDestroy(&AM);
        MatDestroy(&MA);
        MatDestroy(&matAop);
        VecDestroy(&eigenvalues);

        return ierr;

}



int main(int argc, char ** argv){
         int             world_size;

        PetscInitialize(&argc,&argv,(char *)0,help);

        MPI_Comm_size(PETSC_COMM_WORLD, &world_size);
        PetscPrintf(PETSC_COMM_WORLD, "--------------------------\n");
        PetscPrintf(PETSC_COMM_WORLD, "Using number of %d precessors for the generation...\n", world_size);
        
        EBMG();

        PetscFinalize();

        return 0;
}







