# EECE7368 Lab 1 Report

### Darknet Profiling and Floating to Fixed Conversion

Robert D'Antonio and Baibhav Pathak

## 1. Introduction

In this first lab, we were tasked with running, profiling, and improving Darknet. Darknet is a light-weight, open-source deep learning framework that is suitable for many applications where strong compute may not be available. For our lab, we worked with Darknet's YoloV3 model which performs real-time object detection. We first compiled and tested the model to understand its general detection performance. Next, we profiled YoloV3 on Darknet to gain some insight into bottleneck instructions. Finally, we implemented software optimizations to address these bottlenecks, achieiving quicker runtimes with significantly fewer computation-intensive floating-point operations. 

## 2. Implementation

### 2.1. Compiling and Running YoloV3 on Darknet (Issue #2)

For compilation and running the algorithm, we simply followed provided instructions to clone, build, and run the model. With this initial running, this model achieved solid detection performance:

`
dog: 80%    
bicycle: 37%    
car: 71%    
truck: 42%    
truck: 61%    
car: 40%  
`

### 2.2. Profiling Darknet with the YoloV3 Algorithm

We want to better understand the performance of Darknet and the bottlenecks that exist in its existing codebase. To do this, we use Darknet's included profiling feature. The outputted performance summary has information on how many times each instruction is called, how much time it takes to execute, and what percentage of the program's runtime is dominated by it. For a non-optimized Darknet, we found the following results for the top 10 instructions:

|   % seconds   | cumulative seconds | self seconds  | calls   | self s/call | total s/call | name                                     |
|-------|------------|-------|---------|--------|-------|------------------------------------------|
| 95.56 | 7.10       | 7.10  | 3694    | 0.00   | 0.00  | gemm_nn                                  |
|  0.81 | 7.16       | 0.06  | 6       | 0.01   | 0.01  | forward_maxpool_layer_avx               |
|  0.67 | 7.21       | 0.05  | 9       | 0.01   | 0.01  | im2col_cpu_ext                          |
|  0.40 | 7.24       | 0.03  | 761     | 0.00   | 0.00  | load_image_stb                          |
|  0.27 | 7.26       | 0.02  | 1571292 | 0.00   | 0.00  | set_pixel                                |
|  0.27 | 7.28       | 0.02  | 667114  | 0.00   | 0.00  | stbiw__jpg_writeBits                    |
|  0.27 | 7.30       | 0.02  | 663552  | 0.00   | 0.00  | stbiw__jpg_DCT                          |
|  0.27 | 7.32       | 0.02  | 41472   | 0.00   | 0.00  | stbiw__jpg_processDU                    |
|  0.27 | 7.34       | 0.02  | 13      | 0.00   | 0.00  | activate_array_cpu_custom                |
|  0.27 | 7.36       | 0.02  | 1       | 0.02   | 0.02  | fuse_conv_batchnorm                      |

Evidently, `gemm_nn` is dominating, acounting for 95.56% of Darknet's runtime. We want to optimize this!

### 2.3. Optimizing `gemm_nn` by Converting to Fixed-Point Arithmetic

`gemm_nn` is Darknet's matrix multiplication function. In the existing codebase, it performs floating point multiplication in a triple-nested loop. Thus, for multiplication of two matrices with dimensions $M \times K$ and $K \times N$, `gemm_nn` will execute $M \times K \times N$ floating-point multiplications. Floating point operations are very slow, so this is an obvious area where we can improve. 

Our solution is to use fixed point matrix multiplication. 

