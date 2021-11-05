#include <iostream>
#include <math.h>
using namespace std;



__global__ 
void add(int n, float *x, float *y);

void calculation();

void canny(int *imageLine, int rows, int cols);

