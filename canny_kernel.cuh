#include <iostream>
#include <math.h>
#define BLOCK_SIZE 256
using namespace std;



__global__ 
void add(int n, float *x, float *y);

void calculation();

void canny(int *imageLine, int rows, int cols);

