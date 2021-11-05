#include <iostream>
#include "canny_kernel.cuh"
#include "io.h"
#include <sys/time.h>
using namespace std;
int main(void)
{
  string projectAddr = "/users/zgao34/cuda-practice/canny-gpu/";
  string inputAddr = projectAddr + "input/sunflower.jpg";
  string outputAddr = projectAddr + "output/output.jpg";
  ImageIO img(inputAddr);
  int rows = img.getRows();
  int cols = img.getCols();
  int *image;
  img.getImageLine(&image);
  
  struct timeval t1, t2;

  gettimeofday(&t1, 0);

  canny(image, rows, cols);

  gettimeofday(&t2, 0);

  double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

  printf("Time to generate:  %3.1f ms \n", time);


  ImageIO out(image, rows, cols);
  out.imageWrite(outputAddr);
  return 0;
}