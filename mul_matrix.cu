#include <stdio.h>
#include <cuda.h>

int *a, *b;  // host data
int *c, *c2;  // results

//Cuda error checking - non mandatory
void cudaCheckError() {
 cudaError_t e=cudaGetLastError();
 if(e!=cudaSuccess) {
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));
   exit(0);
 }
}

//GPU kernel
__global__
void matMul(int *A,int *B,int *C, int N){
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    int col=blockIdx.y*blockDim.y+threadIdx.y;
    if(row < N && col < N) {
     float product_val = 0;
     for(int k=0;k<N;k++) {
        product_val += A[row*N+k]*B[k*N+col];
      }
    C[row*N+col] = product_val;
    }

}

int main(int argc,char **argv)
{
    printf("Begin \n");
    //Iterations
    int n=4;
    //Number of blocks
    int nBytes = n*n*sizeof(int);

    //Block size and number
    int block_size, block_no;

    //memory allocation
    a = (int *) malloc(nBytes);
    b = (int *) malloc(nBytes);
    c = (int *) malloc(nBytes);
    c2 = (int *) malloc(nBytes);

    int *a_d,*b_d,*c_d;
    block_size = 8; //threads per block
    block_no = n*n/block_size;

    //Work definition
    dim3 dimBlock(block_size, 1, 1);
    dim3 dimGrid(block_no, 1, 1);

    // Data filling
    for(int i=0;i<n*n;i++)
    a[i]=i,b[i]=i;


    printf("Allocating device memory on host..\n");
   //GPU memory allocation
    cudaMalloc((void **) &a_d, n*n*sizeof(int));
    cudaMalloc((void **) &b_d, n*n*sizeof(int));
    cudaMalloc((void **) &c_d, n*n*sizeof(int));

    printf("Copying to device..\n");
    cudaMemcpy(a_d, a, n*n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, n*n*sizeof(int), cudaMemcpyHostToDevice);

    clock_t start_d=clock();
    printf("Doing GPU Vector multiplication\n");
    matMul<<<block_no,block_size>>>(a_d, b_d, c_d, n);
    cudaCheckError();

    //Wait for kernel call to finish
    cudaThreadSynchronize();

    clock_t end_d = clock();

    //Time computing
    double time_d = (double)(end_d-start_d)/CLOCKS_PER_SEC;

    //Copying data back to host, this is a blocking call and will not start until all kernels are finished
    cudaMemcpy(c, c_d, n*sizeof(int), cudaMemcpyDeviceToHost);
    printf("n = %d \t GPU time = %fs \n", n, time_d);

    //Free GPU memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    return 0;
}
