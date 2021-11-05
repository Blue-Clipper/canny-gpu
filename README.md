# canny-gpu
GPU accelerated canny edge detection

## Run

Change the path in main.cu

### Load module
```
module load cmake/3.15.4 && \
module load gcc/10.2 && \
module load cuda/11.1.1 && \
module load opencv/3.4.1
```
### Get resource
```
interact -q gpu -t 0x:00:00 -g 1
```

### Compile
```
mkdir debug && cd debug
cmake ..
make
./canny
```
