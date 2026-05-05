#include<iostream>
#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>

#define N 1024

__global__ void vec_add(float *a, float *b, float *c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    float *h_a, *h_b, *h_c;  // host pointers
    float *d_a, *d_b, *d_c;  // device pointers

    // Allocate host memory
    h_a = (float*)malloc(N * sizeof(float));
    h_b = (float*)malloc(N * sizeof(float));
    h_c = (float*)malloc(N * sizeof(float));

    // Initialize host data
    for(int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
        h_c[i] = 0.0f;
    }

    // Allocate device memory
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    vec_add<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print first few results
    for(int i = 0; i < 5; i++) {
        printf("h_c[%d] = %f\n", i, h_c[i]);
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
