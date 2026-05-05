#include<iostream>
#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>


#define N 1024;


__global__ void ele_mul(float *a, float *b, float *c, int n){
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if(id < n){
        c[id] = a[id] * b[id];
    }
}