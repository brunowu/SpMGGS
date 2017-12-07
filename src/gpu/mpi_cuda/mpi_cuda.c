#include <mpi.h>
#include <stdio.h>
#include <cusparse_v2.h>

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Print off a hello world message
    printf("Hello world from processor %s, rank %d"
           " out of %d processors\n",
           processor_name, world_rank, world_size);

    int *data;
    int i;
    data = (int *)malloc((2)*sizeof(int));
    data[0] = world_rank;
    data[1] = world_rank+5;
    printf("I am rank %d, data[0] = %d, data[1] = %d \n", world_rank,data[0],data[1] );
    int *d_data;
    cudaMalloc(&d_data, 2*sizeof(int));
    cudaMemcpy(d_data, data, 2*sizeof(int), cudaMemcpyHostToDevice);
    
    for(i=0;i<2;i++){d_data[i] = d_data[i]+2;}

    int *h_data;
    h_data = (int *)malloc((2)*sizeof(int));

    cudaMemcpy(h_data, d_data, 2*sizeof(int), cudaMemcpyDeviceToHost);
    printf("I am rank %d, h_data[0] = %d, h_data[1] = %d \n", world_rank,h_data[0],h_data[1] );
    // Finalize the MPI environment.
    MPI_Finalize();
}
