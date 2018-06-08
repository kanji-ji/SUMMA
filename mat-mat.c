#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <mpi.h>

#define  N      5440 
#define  DEBUG  0
#define  EPS    1.0e-18

int     myid, numprocs;
char filename[]="input.txt";

void init_matrix(double* A,int n);
void print_matrix(double* A,int n);
void read_matrix(double* A,char* filename,MPI_Comm comm_cart);
double* plus_matrix(double* A,double* B,int n);
void MyMatMat(double* C, double* A, double* B, int n); 
double* SUMMA(MPI_Comm comm_cart,double* C, double* A, double* B, int n);
void Strassen(double* C_sub,double* A_sub, double* B_sub,int n);

int main(int argc, char* argv[]) {

    double  t0, t1, t2, t_w;
    double  d_mflops;

    int     ierr;
    int     i, j;      
    int     iflag, iflag_t;

    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm comm_cart;

    /* making a new communicator in order to communicate between blocks*/
    int num_block=sqrt(numprocs);

    if(num_block*num_block!=numprocs){ //confirm numprocs is a square number
        printf("ERROR: number of processors is not a square number\n");
        MPI_Abort(MPI_COMM_WORLD,1);    
    }

    int ndims=2; //because it's matrix
    const int dims[2]={num_block,num_block}; //designate the shape of block matrix
    const int periods[2]={0,0};
    int reorder=0;
    MPI_Cart_create(MPI_COMM_WORLD,ndims,dims,periods,reorder,&comm_cart); //create new communicator

    int subn=N/num_block; //get the size of submatrix
    if(subn*num_block!=N){
        printf("ERROR: N have to be dividable by num_block\n");
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    
    /* getting the index of block matrix*/
    int coords[2];
    MPI_Cart_coords(comm_cart,myid,2,coords);
    int ib_start_row=num_block*coords[0];
    int ib_start_col=num_block*coords[1];

    /* matrix memory allocation -------------------*/
    double *A=NULL,*B=NULL,*C=NULL;
    A=(double *) calloc(subn*subn,sizeof(double));
    B=(double *) calloc(subn*subn,sizeof(double));
    C=(double *) calloc(subn*subn,sizeof(double));

    /* matrix generation --------------------------*/
    init_matrix(A,subn);
    init_matrix(B,subn);
    memset(C,0,subn*subn*sizeof(double));
    /* end of matrix generation --------------------------*/

    /* Start of mat-vec routine ----------------------------*/
    ierr = MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();

    C=SUMMA(comm_cart,C, A, B, subn);

    t2 = MPI_Wtime();
    t0 =  t2 - t1; 
    ierr = MPI_Reduce(&t0, &t_w, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    /* End of mat-vec routine --------------------------- */

    if (myid == 0) {

    printf("N  = %d \n",N);
    printf("Mat-Mat time  = %lf [sec.] \n",t_w);

    d_mflops = 2.0*(double)N*(double)N*(double)N/t_w;
    d_mflops = d_mflops * 1.0e-6;
    printf(" %lf [MFLOPS] \n", d_mflops);
    }


    if (DEBUG == 1) {
        /* Verification routine ----------------- */
        iflag = 0;
        for(i=0; i<subn; i++) { 
            
            for(j=0; j<subn; j++) { 
            
                if (fabs(C[i*subn+j] - (double)N) > EPS && myid==0) {
                    printf(" Error! in ( %d , %d ) th argument. Expected value:%d Calculated value:%d\n",i, j, N, C[i*subn+j]);
                    iflag = 1;
                    MPI_Abort(MPI_COMM_WORLD,1);
                } 
            }
        }
        /* ------------------------------------- */

        MPI_Reduce(&iflag, &iflag_t, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (myid == 0) {
            if (iflag_t == 0) printf(" OK! \n");
        }
    }

    ierr = MPI_Finalize();

    exit(0);
}


//if DEBUG=1 fill A,B with 1.0, else fill them with random numbers
void init_matrix(double* A,int n){
    double dc_inv;
    int i,j;
    if (DEBUG == 1) {
        for(i=0; i<n; i++) {
            for(j=0; j<n; j++) {
                A[i*n+j] = 1.0; //using 1D array as 2D array
            }
        }
    }
    else{
        srand(1);
        double dc_inv = 1.0/(double)RAND_MAX;
        for(i=0; i<n; i++) {
            for(j=0; j<n; j++) {
               A[i*n+j] = rand()*dc_inv;
            }
        }
    }
}

//unrecommended. file format is complicated
void read_matrix(double* A,char* filename,MPI_Comm comm_cart){
    int coords[2];
    MPI_Cart_coords(comm_cart,myid,2,coords);
    FILE *fp;
    int n;
    fp=fopen(filename,"r");
    if(fp==NULL){
        printf("ERROR: can't open a file, file may not exist or filepath may be wrong\n");
        return;    
    }
    fscanf(fp,"%d %d",&n,&n);

    int num_block=N/sqrt(numprocs);
    int subn=N/num_block;
    int ib_start_row=coords[0]*num_block;
    int ib_start_col=coords[1]*num_block;
    int ib_end_row=ib_start_row+subn;
    int ib_end_col=ib_start_col+subn;
    int n_col; double buf;
    int i;
    for(i=0;i<n;++i){
        while(fscanf(fp,"%d",&n_col)!=EOF && n_col!=-1){
            
            fscanf(fp,"%lf",&buf);
            if(i>=ib_start_row && i<ib_end_row && n_col>=ib_start_col && n_col<ib_end_col){
                A[(i-ib_start_row)*subn+(n_col-ib_start_col)]=buf;
            }
        }
    }
        fclose(fp);
}

//normal matrix multiplication
void MyMatMat(double* C, double* A, double* B, int n) 
{
     int  i, j, k;
#pragma omp parallel for private(j,k)
     for(i=0; i<n; i++) {
       for(j=0; j<n; j++) {
           C[i*n+j] =0.0;
         for(k=0; k<n; k++) {
           C[i*n+j] += A[i*n+k] * B[k*n+j]; 
         }
       }
     }
   
}

double* plus_matrix(double* A,double* B,int n){
    int i,j;
    double* C=(double *) calloc(n*n,sizeof(double));
    
    for(i=0;i<n;++i){
        for(j=0;j<n;++j){
            C[i*n+j]=A[i*n+j]+B[i*n+j];
        }
    }
    return C;
}

double* SUMMA(MPI_Comm comm_cart,double* C, double* A, double* B, int n){
    
    /*getting the index of block matrix*/
    int coords[2];
    MPI_Cart_coords(comm_cart,myid,2,coords);
    int my_row=coords[0],my_col=coords[1];
    
    int num_block=sqrt(numprocs);
    
    /*creating new communicators to broadcast A_ij/B_ij along rows/columns*/
    MPI_Comm row_comm,col_comm;
    
    int remain_dims[2];
    
    remain_dims[0]=1; remain_dims[1]=0;
    MPI_Cart_sub(comm_cart,remain_dims,&row_comm);
    
    remain_dims[0]=0; remain_dims[1]=1;
    MPI_Cart_sub(comm_cart,remain_dims,&col_comm);

    /*save A_ij,B_ij to avoid being overwritten by broadcasting*/
    double *A_loc_save=(double *) calloc(n*n,sizeof(double));
    double *B_loc_save=(double *) calloc(n*n,sizeof(double));
    double *C_loc_tmp =(double *) calloc(n*n,sizeof(double));
    
    memcpy(A_loc_save,A,n*n*sizeof(double));
    memcpy(B_loc_save,B,n*n*sizeof(double));
    memset(C_loc_tmp,0,n*n*sizeof(double));

    
    /*SUMMA main part*/
    int bcast_root,i,j;
    for(bcast_root=0;bcast_root<num_block;++bcast_root){
        
        if(my_col==bcast_root){
            memcpy(A,A_loc_save,n*n*sizeof(double));
        }

        MPI_Bcast(A,n*n,MPI_DOUBLE,bcast_root,row_comm);
        
        if(my_row==bcast_root){
            memcpy(B,B_loc_save,n*n*sizeof(double));
        }
        
        MPI_Bcast(B,n*n,MPI_DOUBLE,bcast_root,col_comm);
        
        MyMatMat(C_loc_tmp,A,B,n);
        
        C=plus_matrix(C,C_loc_tmp,n);
    }

    free(A_loc_save);
    free(B_loc_save);
    free(C_loc_tmp);
    return C;
}

//for debugging
void print_matrix(double* A,int n){
    int i;
    for(i=0;i<n*n;++i) printf("%lf ",A[i]);
    printf("\n");
}

//under construction
void Strassen(double* C_sub,double* A_sub, double* B_sub,int n){
        
}
