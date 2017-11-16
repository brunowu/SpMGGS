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

	PetscTime(&st);
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
//	MatSetOption(matAop, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
        ierr = MatSetUp(matAop);CHKERRQ(ierr);

//	ierr = MatDuplicate(Mt,MAT_COPY_VALUES,&matAop);CHKERRQ(ierr);
	MatCopy(Mt,matAop,DIFFERENT_NONZERO_PATTERN);
	my_factorielle_bornes =  factorial(1,(2*degree-2));
	ierr = MatScale(Mt,my_factorielle_bornes);CHKERRQ(ierr);

	for (k=1; k<=(2*degree-2); k++) {
	  ierr = MatMatMultSymbolic(matAop,A,PETSC_DEFAULT,&MA);CHKERRQ(ierr); 
//          ierr = MatMatMultSymbolic(matAop,A,1,&MA);CHKERRQ(ierr);
	  ierr = MatMatMultNumeric(matAop,A,MA);CHKERRQ(ierr);//M*A
//          ierr = MatMatMultSymbolic(A,matAop,1,&AM);CHKERRQ(ierr);
	  ierr = MatMatMultSymbolic(A,matAop,PETSC_DEFAULT,&AM);CHKERRQ(ierr);
	  ierr = MatMatMultNumeric(A,matAop,AM);CHKERRQ(ierr); //A*M

	  ierr = MatAYPX(matAop,0,AM, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
	  ierr = MatAXPY(matAop,-1,MA,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);//A*M-M*A
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

/*
	PetscInt nn;
	PetscScalar *va;

	MatGetRow(Mt,4, &nn,NULL,&va);
	for(i=0;i<nn;i++){
		PetscPrintf(PETSC_COMM_WORLD,"va[%d] = %.20f\n",i,va[i]);
	}
*/
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
