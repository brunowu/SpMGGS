#include "libs.h"

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

  PetscInt	i;

  int 			my_seed,my_rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&my_rank);
  my_seed=time(NULL)+my_rank;

  srand(my_seed); 

  for(i = 0; i < nombre/2; i++){

/*
    if(i<1*nombre/4) ret = 1*(cos(2*PI*i/nombre)+2) + PETSC_i*1*(sin(2*PI*i/nombre)+2);
    else if(i<2*nombre/4) ret =1* (cos(2*PI*i/nombre)+2) + PETSC_i*1*(sin(2*PI*i/nombre)-2);
    else if(i<3*nombre/4) ret = 1*(cos(2*PI*i/nombre)+4) + PETSC_i*1*(sin(2*PI*i/nombre)+2);
    else ret = 1*(cos(2*PI*i/nombre)+4) + PETSC_i*1*(sin(2*PI*i/nombre)-2);
*/

/*
    if(i < 9*nombre / 10)
    	ret[i] = -1*(cos(2*PI*i/nombre)+2+5*rand()/RAND_MAX) + PETSC_i*1*(sin(2*PI*i/nombre));
    else ret[i] =-1*(cos(2*PI*i/nombre)+2) + PETSC_i*1*(sin(2*PI*i/nombre) );
*/


/*
    if(i < 1*nombre / 10)
        ret[i] = 1*(cos(2*PI*i/nombre)+1.2500) + PETSC_i*2*(sin(2*PI*i/nombre));
    else ret[i] =1*(cos(2*PI*i/nombre)+1.250) + PETSC_i*2*(sin(2*PI*i/nombre));

*/
/*
    if(i < 2*nombre / 10)
        ret[i] = 5*i + PETSC_i*10;
    else ret[i] =5*(cos(PI*i/nombre))+100+0.01*PETSC_i;
*/

//    ret[i] =200+  (PetscScalar)PETSC_i*2*cos(i / nombre);

//    ret[i] = 2123*i+1+(-123.5423121*i+1000)*PETSC_i;

//  good case for UCGLE

/* 
    if(i < 3*nombre / 10)
        ret[i] = 1*(cos(2*PI*i/nombre)+2) + PETSC_i*1*sin(2*PI*i/nombre);
    else if(i < 6 * nombre / 10) ret[i] = 10*(cos(2*PI*i/nombre)+200) + PETSC_i*1*(sin(2*PI*i/nombre) + 277);
    else ret[i] = 1*(cos(2*PI*i/nombre)+3000) + PETSC_i*1*(sin(2*PI*i/nombre) - 2800); 

*/

/*
   if(i < 5 * nombre / 10){
	ret[i] = 1 + (0.8* (sin(4*PI*i/nombre)+1.5))  + PETSC_i* (0.8* (sin(4*PI*i/nombre)+1.5));
   }
   else{
	ret[i] = 1 + (0.8* (sin(4*PI*i/nombre)+1.5))  - PETSC_i* (0.8* (sin(4*PI*i/nombre)+1.5));
   }   
*/
/*
   if(i < 5 * nombre / 20){
        ret[i] = 5  + PETSC_i* (8* (sin(4*PI*i/nombre)+1.5));
   }
   else if(i < 5 * nombre / 10){
        ret[i] = 5  - PETSC_i* (8* (sin(4*PI*i/nombre)+1.5));
   }
   else if(i < 15 * nombre / 20){
        ret[i] = 10  + PETSC_i* (8* (sin(4*PI*i/nombre)+1.5));
   }
   else{
        ret[i] = 10  - PETSC_i* (8* (sin(4*PI*i/nombre)+1.5));
   }
*/

/*
   if(i < 5 * nombre / 20){
        ret[i] = 10+0.02* sin(4*PI*i/nombre) + PETSC_i* (8* (sin(4*PI*i/nombre)+1.5));
   }
   else if(i < 5 * nombre / 10){
        ret[i] = 10+0.02* sin(4*PI*i/nombre)  - PETSC_i* (8* (sin(4*PI*i/nombre)+1.5));
   }
   else if(i < 15 * nombre / 20){
        ret[i] = 10 + 3* sin(4*PI*i/nombre) + PETSC_i* (8* (sin(4*PI*i/nombre)+1.5));
   }
   else{
        ret[i] = 10 + 3* sin(4*PI*i/nombre) - PETSC_i* (8* (sin(4*PI*i/nombre)+1.5));
   }
*/
/*
 if(i < 5 * nombre / 20){
        ret[i] = 20+8* cos(4*PI*i/nombre) + PETSC_i* (8* (sin(4*PI*i/nombre)));
   }
   else if(i < 10 * nombre / 20) 
  {
        ret[i] = 20+8* cos(4*PI*i/nombre)  - PETSC_i* (8* (sin(4*PI*i/nombre)));
  }
   else if(i < 15 * nombre / 20)
   {
        ret[i] = 20+8* cos(4*PI*i/nombre)  + PETSC_i* (8* (sin(4*PI*i/nombre)));
   }
   else{
        ret[i] = 20+8* cos(4*PI*i/nombre)  - PETSC_i* (8* (sin(4*PI*i/nombre)));
	}

*/
//ret[0] = -10;

//good case for the generation
/*
 if(i < 10 * nombre / 20){
        ret[i] = 10.+8* cos(4*PI*i/nombre) + PETSC_i* (8* (sin(4*PI*i/nombre)));
   }
   else{
        ret[i] = 10.+8* cos(4*PI*i/nombre)  - PETSC_i* (8* (sin(4*PI*i/nombre)));
        }
*/
/*
  if(i < 0.5*nombre)
	ret[i] = 1.0+PETSC_i * 1.0;
  else ret[i] = 1.0-PETSC_i * 1.0;
*/
/*
 if(i < 10 * nombre / 20){
        ret[i] = 13+8* cos(4*PI*i/nombre) + PETSC_i* (8* (sin(4*PI*i/nombre)));
   }
   else{
        ret[i] = 13+8* cos(4*PI*i/nombre)  - PETSC_i* (8* (sin(4*PI*i/nombre)));
        }
*/
	PetscReal real, imag;
	srand(i);

//good case 1

	real = rnd(3,10);
	imag = rnd(3,10);


//good case 2

/*
	PetscReal rd= rnd(0,1);
	real = -10 + 12 * cos(rd*PI);
	imag = 12 * sin(rd*PI);
*/

//three leafed formula
/*
	PetscReal rd = rnd(0,1);
	real = 10 + 9*cos(3*rd*PI)*cos(rd*PI);
	imag = 9*cos(3*rd*PI)*sin(rd*PI);
*/

	#if defined(PETSC_USE_COMPLEX)
//		PetscPrintf(PETSC_COMM_WORLD,"Generating the Complex eigenvalues...\n\n");
        	ret[i] = real+ PETSC_i * imag;
        	ret[nombre-1-i] =  real -  PETSC_i * imag;
	#elif defined(PETSC_USE_REAL)
//		PetscPrintf(PETSC_COMM_WORLD,"Generating the REAL eigenvalues...\n\n");
		ret[i] = real;
                ret[nombre-1-i] = real;
	#endif

//arrow formula
/*
	if(i < nombre/10){
		real = rnd(2,10);
		imag = rnd(0,1);
		ret[i] = real+ PETSC_i * imag;
		ret[nombre-1-i] =  real -  PETSC_i * imag;
	}
	else{
		PetscReal rd = rnd(-0.5,0.5);
                real = 10 + 3 * cos(rd*PI);
                imag = 3 * sin(rd*PI);
                ret[i] = real+ PETSC_i * imag;
                ret[nombre-1-i] =  real -  PETSC_i * imag;
	}
*/
//	PetscPrintf(PETSC_COMM_WORLD,"ret[%d] = %f + %fi\n", i, real, imag);

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
