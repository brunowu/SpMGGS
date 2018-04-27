#include "ebmg.h"

PetscErrorCode EBMG(){

	char fileb[PETSC_MAX_PATH_LEN];

	Vec            eigenvalues;
  	Mat            Mt, AM, MA, matAop;
  	Mat            A;
  	PetscErrorCode ierr;
  	PetscInt       n,i,j,k,degree;
  	PetscScalar    rRandom1;
  	PetscInt       iRandom, d1;
  	PetscInt    	size;
  	PetscBool      flagb,n_flg,degree_flg,d1_flg;
	MatInfo     	Ainfo;
	double        	gnnz;
	char           	matrixOutputFile[PETSC_MAX_PATH_LEN];
	PetscViewer    	output_viewer;
	clock_t start, finish;
	double  duration;
	PetscLogDouble st,ed;

	PetscInt Istart, Iend;

	PetscTime(&st);

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
	//	shuffer(Deigenvalues,n);
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

	for(i=istart;i<iend;i++){
		tmp = Deigenvalues[i];
		VecSetValue(eigenvalues, i, tmp, INSERT_VALUES);

	}
	
///	VecView(eigenvalues, PETSC_VIEWER_STDOUT_WORLD);
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
	PetscScalar vV=1, vvV=0;
	i=0;

    for(i = Istart; i < Iend; i++){
            if(i < n-1){
                    if((i+1) % degree == 0){
                            iI=i;
                            jJ=i+1;
                            ierr = MatSetValues(A,1,&iI,1,&jJ,&vvV,INSERT_VALUES);CHKERRQ(ierr);
                    }
                    else{
                            iI=i;
                            jJ=i+1;
                            ierr = MatSetValues(A,1,&iI,1,&jJ,&vV,INSERT_VALUES);CHKERRQ(ierr);
                    }
            }

    }

	ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	
	PetscInt my_factorielle_bornes=1;

        ierr = MatCreate(PETSC_COMM_WORLD,&matAop);CHKERRQ(ierr);
        ierr = MatSetSizes(matAop,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
        ierr = MatSetType(matAop,MATMPIAIJ);CHKERRQ(ierr);
        ierr = MatSetFromOptions(matAop);CHKERRQ(ierr);
        ierr = MatSetUp(matAop);CHKERRQ(ierr);

		MatCopy(Mt,matAop,DIFFERENT_NONZERO_PATTERN);
		my_factorielle_bornes =  factorial(1,(2*degree-2));
		ierr = MatScale(Mt,my_factorielle_bornes);CHKERRQ(ierr);

	for (k=1; k<=(2*degree-2); k++) {
          MatMatMult(matAop, A, MAT_INITIAL_MATRIX,  PETSC_DEFAULT , &MA);
          MatMatMult(A, matAop, MAT_INITIAL_MATRIX,  PETSC_DEFAULT , &AM);

		  ierr = MatAYPX(matAop,0,AM, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
		  ierr = MatAXPY(matAop,-1,MA,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);//A*M-M*A
		  MatView(MA, PETSC_VIEWER_STDOUT_WORLD);
		  my_factorielle_bornes =  factorial(k+1,2*degree-2); //2*(d-1)*(2d-3)****(k+1) 
		  ierr = MatAXPY(Mt,my_factorielle_bornes,matAop,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
		  ierr = MatZeroEntries(AM);CHKERRQ(ierr);
		  ierr = MatZeroEntries(MA);CHKERRQ(ierr);
	}

  	my_factorielle_bornes =  factorial(1,(2*degree-2));
	PetscReal inv = (PetscReal) 1 / my_factorielle_bornes;
        PetscPrintf(PETSC_COMM_WORLD,"\n@>inv = %f\n",inv);
	ierr = MatScale(Mt,inv);CHKERRQ(ierr);	

	finish = clock();	
	PetscTime(&ed);
	MatGetInfo(Mt,MAT_GLOBAL_SUM,&Ainfo);

	gnnz = Ainfo.nz_used;

	sprintf(matrixOutputFile,"Rectangular_nb_%d_%dx%d_%g_nnz.gz",n, n,n,gnnz);
		
	PetscPrintf(PETSC_COMM_WORLD,"\n@>Dumping matrix to PETSc binary %s\n",matrixOutputFile);
			
	PetscViewerBinaryOpen(PETSC_COMM_WORLD,matrixOutputFile,FILE_MODE_WRITE,&output_viewer);
	PetscViewerPushFormat(output_viewer,PETSC_VIEWER_ASCII_INFO_DETAIL);
//	MatView(Mt,PETSC_VIEWER_STDOUT_WORLD);
	MatView(Mt,output_viewer);
	PetscViewerDestroy(&output_viewer);
		
	PetscPrintf(PETSC_COMM_WORLD,"\n@>Matrix %s Dumped\n\n",matrixOutputFile);
	PetscPrintf(PETSC_COMM_WORLD,"\n>>>>>>Please use the command 'gzip -d **' to unzip the file to binary file\n\n");

	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	PetscLogDouble time = ed - st;
    PetscPrintf(PETSC_COMM_WORLD, "--------------------------\n");
    PetscPrintf(PETSC_COMM_WORLD,"\nElapsed time is %f seconds\n\n",time);
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
	return ierr;

}
