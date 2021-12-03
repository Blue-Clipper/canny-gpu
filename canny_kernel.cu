#include "canny_kernel.cuh"

__global__ 
void gaussian(int *img, int *origin, int rows, int cols) {
    int gaussianMask[5][5] = {
                                {1, 4, 7, 4, 1},
                                {4, 16, 26, 16, 4},
                                {7, 26, 41, 26, 7},
                                {4, 16, 26, 16, 4},
                                {1, 4, 7, 4, 1}
                               };
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < rows * cols; i += stride) {
        int curRow = i / cols;
        int curCol = i % cols;
        int newPixel = 0;
        for(int rowOffset = -2; rowOffset <= 2; rowOffset ++) {
          for(int colOffset = -2; colOffset <= 2; colOffset ++) {
              int neighbourRow = curRow + rowOffset;
              int neighbourCol = curCol + colOffset;
              if(neighbourRow < 0 || neighbourRow >= rows || neighbourCol < 0 || neighbourCol >= cols) {
                continue;
              }
              int neighbourIndex = neighbourRow * cols + neighbourCol;
              newPixel += origin[neighbourIndex] * gaussianMask[2 + rowOffset][2 + colOffset];
          }
        }
        img[i] = newPixel / 273;
    }
    
}

__global__
void gradient(int *strength, int *direction, int *origin, int rows, int cols) {
    int gxMask[3][3] = {
                         {-1, 0, 1}, 
                         {-2, 0, 2}, 
                         {-1, 0, 1}
                        };
    int gyMask[3][3] = {
                         {-1, -2, -1}, 
                         {0, 0, 0}, 
                         {1, 2, 1}
                        };
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < rows * cols; i += stride) {
      int curRow = i / cols;
      int curCol = i % cols;
      int gx = 0;
      int gy = 0;
      for(int rowOffset = -1; rowOffset <= 1; rowOffset ++) {
        for(int colOffset = -1; colOffset <= 1; colOffset ++) {
          int neighbourRow = curRow + rowOffset;
          int neighbourCol = curCol + colOffset;
          if(neighbourRow < 0 || neighbourRow >= rows || neighbourCol < 0 || neighbourCol >= cols) {
            continue;
          }
          gx += origin[neighbourRow * cols + neighbourCol] * gxMask[rowOffset + 1][colOffset + 1];
          gy += origin[neighbourRow * cols + neighbourCol] * gyMask[rowOffset + 1][colOffset + 1]; 
        }
      }
      strength[i] = sqrtf(gx * gx + gy * gy);
      double angle = (atan2(float(gx), float(gy)) / M_PI) * 180.0;
      if ( ( (angle < 22.5) && (angle > -22.5) ) || (angle > 157.5) || (angle < -157.5))
				direction[i] = 0;
			else if ( ( (angle > 22.5) && (angle < 67.5) ) || ( (angle < -112.5) && (angle > -157.5) ) )
				direction[i] = 45;
			else if ( ( (angle > 67.5) && (angle < 112.5) ) || ( (angle < -67.5) && (angle > -112.5) ) )
				direction[i] = 90;
			else if ( ( (angle > 112.5) && (angle < 157.5) ) || ( (angle < -22.5) && (angle > -67.5) ) )
				direction[i] = 135;
    }
}

__device__
void findEdge(int *strength, int *direction, int *edge, int rows, int cols, 
              int rowShift, int colShift, int i, int dir, int lowerThreshold, int* visited) {
	bool edgeEnd = false;
  int newRow = (i / cols) + rowShift;
	int newCol = (i % cols) + colShift;
  if(newRow < 0 || newRow >= rows) {
      edgeEnd = true;
  }
  if(newCol < 0 || newCol >= cols) {
      edgeEnd = true;
  }
  int idx = newRow * cols + newCol;
  if(visited[idx] == 1) {
    return;
  } else {
    visited[idx] = 1;
  }
  while((direction[idx] == dir) && !edgeEnd &&
      (strength[idx] > lowerThreshold)) {
          edge[idx] = 255;
          newRow = newRow + rowShift;
          newCol = newCol + colShift;
          idx = newRow * cols + newCol;
          if(newRow < 0 || newRow >= rows) {
              break;
          }
          if(newCol < 0 || newRow >= cols) {
              break;
          }
          if(visited[idx] == 1) {
            return;
          } else {
            visited[idx] = 1;
          }
  }
}

__global__
void traceEdge(int *strength, int *direction, int *edge, 
                int rows, int cols, int upperThreshold, int lowerThreshold, int *visited) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;


    for(int i = index; i < rows * cols; i += stride) {
      if(strength[i] > upperThreshold) {
          switch(direction[i]) {
              case 0:
                  findEdge(strength, direction, edge, rows, cols, 0, 1, i, 0, lowerThreshold, visited);
                  break;
              case 45:
                  findEdge(strength, direction, edge, rows, cols, 1, 1, i, 45, lowerThreshold, visited);
                  break;
              case 90:
                  findEdge(strength, direction, edge, rows, cols, 1, 0, i, 90, lowerThreshold, visited);
                  break;
              case 135:
                  findEdge(strength, direction, edge, rows, cols, 1, -1, i, 135, lowerThreshold, visited);
                  break;
              default :
                  edge[i] = 0;     
          }
          edge[i] = edge[i] == 255 ? 255 : 0;
      } else {
          edge[i] = 0;     
      }
    }
}

void canny(int *imageLine, int rows, int cols){

  int *img = NULL, *origin = NULL;

  cudaMallocManaged(&img, rows*cols*sizeof(int));
  cudaMallocManaged(&origin, rows * cols * sizeof(int));
  cudaMemcpy(origin,imageLine,rows * cols * sizeof(int),cudaMemcpyHostToDevice);
 
  int numBlocks = (rows * cols + BLOCK_SIZE - 1) / BLOCK_SIZE;

  //gaussian_filter  
  gaussian<<<numBlocks, BLOCK_SIZE>>>(img, origin, rows, cols);
  cudaDeviceSynchronize();


  int *strength = NULL, *direction = NULL;
  cudaMallocManaged(&strength, rows*cols*sizeof(int));
  cudaMallocManaged(&direction, rows * cols * sizeof(int));
  if(strength == NULL || direction == NULL) {
    cout << "GPU Malloc Failed." << endl;
    return;
  }
  gradient<<<numBlocks, BLOCK_SIZE>>>(strength, direction, img, rows, cols);
  cudaDeviceSynchronize();
  int *visited;
  cudaMallocManaged(&visited, rows*cols*sizeof(int));
  traceEdge<<<numBlocks, BLOCK_SIZE>>>(strength, direction, img, rows, cols, 100, 35, visited);
  cudaDeviceSynchronize();

  cudaMemcpy(imageLine, img, rows * cols * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(img);
  cudaFree(origin);
  cudaFree(strength);
  cudaFree(direction);

}