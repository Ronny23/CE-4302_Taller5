#include <stdio.h>
#include <cuda.h>

int *ha, *hb, *hc;  // host data
int *hd;  // results

__global__
void add(int *a, int *b, int *c, int *d, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N*N) {
        d[i] = 5 * a[i] + 3/4 * b[i] + 2/3 * c[i];
    }
}

//CPU function
void addCPU(int *a,int *b, int *c, int *d, int N){
    for(int i = 0; i < N; i++) {
      for(int j = 0; j < N; j++) {
          d[i*N+j] = 5 * a[i*N+j] + 3/4 * b[i*N+j] + 2/3 * c[i*N+j];
      }
    }
}


int main() {
    int N = 50000;

    int nBytes = N*N*sizeof(int);
    //Block size and number
    int block_size, block_no;

    //memory allocation
    ha = (int *) malloc(nBytes);
    hb = (int *) malloc(nBytes);
    hc = (int *) malloc(nBytes);
    hd = (int *) malloc(nBytes);

    block_size = 8; //threads per block
    block_no = N*N/block_size;

    //Work definition
    dim3 dimBlock(block_size, 1, 1);
    dim3 dimGrid(block_no, 1, 1);


    for (int i = 0; i < N*N; ++i) {
        ha[i] = i;
        hb[i] = i*i;
        hc[i] = i*i-2*i;
    }

    int *da, *db, *dc, *dd;
    cudaMalloc((void **)&da, N*N*sizeof(int));
    cudaMalloc((void **)&db, N*N*sizeof(int));
    cudaMalloc((void **)&dc, N*N*sizeof(int));
    cudaMalloc((void **)&dd, N*N*sizeof(int));


    cudaMemcpy(da, ha, N*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, N*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dc, hc, N*N*sizeof(int), cudaMemcpyHostToDevice);

    clock_t start_d=clock();
    add<<<block_no,block_size>>>(da, db, dc, dd, N);

    cudaMemcpy(hd, dd, N*sizeof(int), cudaMemcpyDeviceToHost);
    clock_t end_d = clock();

    clock_t start_h = clock();
    addCPU(ha, hb, hc, hd, N);
    clock_t end_h = clock();

    //Time computing
    double time_d = (double)(end_d-start_d)/CLOCKS_PER_SEC;
    double time_h = (double)(end_h-start_h)/CLOCKS_PER_SEC;

    printf("n = %d \t block size = %d \t GPU time = %fs \t CPU time = %fs\n", N, block_size, time_d, time_h);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    cudaFree(dd);
    return 0;
}
