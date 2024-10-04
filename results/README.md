# EECE7368 Lab 1 Report

### Darknet Profiling and Floating to Fixed Conversion

Robert D'Antonio and Baibhav Pathak

## 1. Introduction

In this first lab, we were tasked with running, profiling, and improving Darknet. Darknet is a light-weight, open-source deep learning framework that is suitable for many applications where strong compute may not be available. For our lab, we worked with Darknet's YoloV3 model which performs real-time object detection. We first compiled and tested the model to understand its general detection performance. Next, we profiled YoloV3 on Darknet to gain some insight into bottleneck instructions. Finally, we implemented software optimizations to address these bottlenecks, achieiving quicker runtimes with significantly fewer computation-intensive floating-point operations. 

## 2. Implementation

### 2.1. Compiling and Running YoloV3 on Darknet (Issue #2)

For compilation and running the algorithm, we simply followed provided instructions to clone, build, and run the model. With this initial running, this model achieved solid detection performance:

```
dog: 80%    
bicycle: 37%    
car: 71%    
truck: 42%    
truck: 61%    
car: 40%  
```

### 2.2. Profiling Darknet with the YoloV3 Algorithm

We want to better understand the performance of Darknet and the bottlenecks that exist in its existing codebase. To do this, we use Darknet's included profiling feature. The outputted performance summary has information on how many times each instruction is called, how much time it takes to execute, and what percentage of the program's runtime is dominated by it. For a non-optimized Darknet, we found the following results for the top 10 instructions:

|   % seconds   | Cumulative Seconds | Self Seconds  | Calls   | Self s/call | Total s/call | Name                                     |
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

#### 2.3.1. Implementation

`gemm_nn` is Darknet's matrix multiplication function. In the existing codebase, it performs floating point multiplication in a triple-nested loop. Thus, for multiplication of two matrices with dimensions $M \times K$ and $K \times N$, `gemm_nn` will execute $M \times K \times N$ floating-point multiplications. Floating point operations are very slow, so this is an obvious area where we can improve. 

Our solution is to use fixed point matrix multiplication. We did this by writing two new helper functions, `mm_float_to_fixed` and `mm_fixed_to_float`, and modifying the existing `gemm` function to operate on 32-bit integers rather than floating point values. As the name suggests, `mm_float_to_fixed` converts a matrix of floating point values to a matrix of 32-bit integers. A key input this is the `scale` value, which determines how much to multiply the initial floating point values by prior to casting them to an integer. This value is tuned to maximize precision while avoiding overflow, and getting it right leads to better overall numerical accuracy. We multiply by $2^\text{scale}$. `mm_fixed_to_float` simply reverses this operation, and divides the fixed values by $2^\text{scale}$

Combining all of this, we have a new `gemm` function in Darknet that looks like this:

```c
gemm(A, B, C):  // A, B (float) are inputs, C (float) is output
  A_fixed = mm_float_to_fixed(A)
  B_fixed = mm_float_to_fixed(B)

  C_fixed = gemm_fixed_point(A_fixed, B_fixed)

  C = mm_fixed_to_float(C_fixed)

  return C
```

#### 2.3.2. Tuning the `scale` Parameter using SNR

`scale` is important, as it directly affects the numerical stability. It is also tricky to get right as it is application-dependent. In our initial test cases (provided to us), we optimized our scale by calculating a signal-to-noise ratio (SNR). This calculation compares the output matrix from our fixed-point GEMM implementation with the desired output values. We evaluated different values for `scale` in a loop and found that `scale=14` provided us the best SNR, 50.736 dB:

Note the SNR of 103.845 dB for the original floating point implementation. The -186.470 dB at `scale=16` corresponds to overflow. 
```
SNR for float gemm is 103.8454589844 dB 
SNR for fixed gemm is -5.6096243858 dB with scale=1
SNR for fixed gemm is 1.9803683758 dB with scale=2
SNR for fixed gemm is -0.5879516006 dB with scale=3
SNR for fixed gemm is 3.2709569931 dB with scale=4
SNR for fixed gemm is 2.7874507904 dB with scale=5
SNR for fixed gemm is 6.5597982407 dB with scale=6
SNR for fixed gemm is 9.7240753174 dB with scale=7
SNR for fixed gemm is 14.3678665161 dB with scale=8
SNR for fixed gemm is 20.3883152008 dB with scale=9
SNR for fixed gemm is 24.6961498260 dB with scale=10
SNR for fixed gemm is 31.2855548859 dB with scale=11
SNR for fixed gemm is 36.7923278809 dB with scale=12
SNR for fixed gemm is 42.3394432068 dB with scale=13
SNR for fixed gemm is 50.7365760803 dB with scale=14
SNR for fixed gemm is 8.3975419998 dB with scale=15
SNR for fixed gemm is -186.4700622559 dB with scale=16
```

#### 2.3.3. Tuning `scale` in Darknet

After implementing our fixed point GEMM function in Darknet, we evaluated its accuracy using by computing the MaP of our implemation against the original floating point GEMM. With `scale=14`, the results were very sad with MaP showing up as `nan`. This likely means we were getting overflow and that the data we're operating on in Darknet is different than what we used for the SNR testbench. 

So, we had to tune it again, this time optimizing for the MaP rather than the SNR. We found that `scale=10` led to a MaP of 1.000, meaning that our Darknet with the fixed-point GEMM performs as well as the original floating point version. 

## 3. Profiling Darknet with Fixed-Point GEMM

Profiling Darknet with the Fixed-Point GEMM presented the following results. We observe that our gemm_fixed_point takes longer overall than the original floating point implementation. There are different explanations for this, but it's worth noting that we are profiling this on a modern PC with strong floating-point hardware. This results would likely look much different on a lighter-weight system. 

| % Time | Cumulative Seconds | Self Seconds | Calls      | Self s/Call | Total s/Call | Function Name                |
|--------|--------------------|--------------|------------|-------------|--------------|------------------------------|
| 92.07  | 7.31               | 7.31         | 13         | 0.56        | 0.56         | gemm_fixed_point           |
| 2.77   | 7.53               | 0.22         | 29,568,944 | 0.00        | 0.00         | roundup                    |
| 0.63   | 7.58               | 0.05         | 26         | 0.00        | 0.01         | mm_float_to_fixed          |
| 0.63   | 7.63               | 0.05         | 6          | 0.01        | 0.01         | forward_maxpool_layer_avx   |
| 0.50   | 7.67               | 0.04         | 9          | 0.00        | 0.01         | im2col_cpu_ext             |
| 0.38   | 7.70               | 0.03         | 20,361,120 | 0.00        | 0.00         | is_a_ge_zero_and_a_lt_b    |
| 0.38   | 7.73               | 0.03         | 41,472     | 0.00        | 0.00         | stbiw__jpg_processDU       |
| 0.38   | 7.76               | 0.03         |            |             |              | _init                      |
| 0.25   | 7.78               | 0.02         | 663,552    | 0.00        | 0.00         | stbiw__jpg_DCT             |
| 0.25   | 7.80               | 0.02         | 761        | 0.00        | 0.00         | load_image_stb             |
| 0.25   | 7.82               | 0.02         | 13         | 0.00        | 0.00         | add_bias                   |

## 4. Further Optimizations

We tried to implement some different optimizations here, but ultimatly turned up unsuccessful with all of them. 

### 4.1. Idea 1: Making all Conv2D Layers Fixed-Point

A simple backtrace of our `gemm_fixed_point` function showed that it is only being used by the Convolution layers: 

```console
(gdb) backtrace
#0  gemm_fixed_point (TA=0, TB=0, M=16, N=173056, K=27, ALPHA=1, A=0xaaaaaabfb990, lda=27, 
    B=0xffffeae00010, ldb=173056, BETA=1, C=0xffffea200010, ldc=173056) at ./src/gemm.c:148
#1  0x0000aaaaaaaaa2b8 in gemm (TA=0, TB=0, M=16, N=173056, K=27, ALPHA=1, A=0xaaaaaabc0960, lda=27, 
    B=0xfffff1e00010, ldb=173056, BETA=1, C=0xfffff6e00010, ldc=173056) at ./src/gemm.c:177
#2  0x0000aaaaaaab55a0 in forward_convolutional_layer ()
#3  0x0000aaaaaab07ea4 in forward_network ()
#4  0x0000aaaaaab0922c in network_predict ()
#5  0x0000aaaaaab31618 in validate_detector_map ()
#6  0x0000aaaaaab33b70 in run_detector ()
#7  0x0000aaaaaaaa5140 in main ()
(gdb) 
```

We also observe that many convolution layers are sequential (max pooling layers generally won't be affected by a change to fixed point):

```
 layer   filters  size/strd(dil)      input                output
   0 conv     16       3 x 3/ 1    416 x 416 x   3 ->  416 x 416 x  16 0.150 BF
   1 max                2x 2/ 2    416 x 416 x  16 ->  208 x 208 x  16 0.003 BF
   2 conv     32       3 x 3/ 1    208 x 208 x  16 ->  208 x 208 x  32 0.399 BF
   3 max                2x 2/ 2    208 x 208 x  32 ->  104 x 104 x  32 0.001 BF
   4 conv     64       3 x 3/ 1    104 x 104 x  32 ->  104 x 104 x  64 0.399 BF
   5 max                2x 2/ 2    104 x 104 x  64 ->   52 x  52 x  64 0.001 BF
   6 conv    128       3 x 3/ 1     52 x  52 x  64 ->   52 x  52 x 128 0.399 BF
   7 max                2x 2/ 2     52 x  52 x 128 ->   26 x  26 x 128 0.000 BF
   8 conv    256       3 x 3/ 1     26 x  26 x 128 ->   26 x  26 x 256 0.399 BF
   9 max                2x 2/ 2     26 x  26 x 256 ->   13 x  13 x 256 0.000 BF
  10 conv    512       3 x 3/ 1     13 x  13 x 256 ->   13 x  13 x 512 0.399 BF
  11 max                2x 2/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.000 BF
  12 conv   1024       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x1024 1.595 BF
  13 conv    256       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 256 0.089 BF
  14 conv    512       3 x 3/ 1     13 x  13 x 256 ->   13 x  13 x 512 0.399 BF
  15 conv    255       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x 255 0.044 BF
  16 yolo
[yolo] params: iou loss: mse (2), iou_norm: 0.75, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.00
  17 route  13 		                           ->   13 x  13 x 256 
  18 conv    128       1 x 1/ 1     13 x  13 x 256 ->   13 x  13 x 128 0.011 BF
  19 upsample                 2x    13 x  13 x 128 ->   26 x  26 x 128
  20 route  19 8 	                           ->   26 x  26 x 384 
  21 conv    256       3 x 3/ 1     26 x  26 x 384 ->   26 x  26 x 256 1.196 BF
  22 conv    255       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 255 0.088 BF
  23 yolo
```

With this information we believe it should be possible to make one float-to-fixed conversion in `network_predict()` prior to the convolution layers, and one fixed-to-float conversion after the convolution layers (and before the yolo layer). Ultimately we were not able to implement this as there is a lot of complexity buried inside each layer. This optimization is certainly doable, but more time/effort is needed to implement it.

### 4.2. Performing float-to-fixed in `im2col`. 

This is an idea that got thrown around a lot in lecture. We didn't explore it much as we were more committed to the previous idea. We need to spend more time thinking about this one to decide if it is a better option than 4.1.

## 5. What's Next?

On modern processors, using fixed-point operations instead of floating-point operations may not be faster. Floating point units do their job well, especially on modern PCs like those we used to run all of these tests. Additionally, fixed-point hardware is consistantly busy with the countless other fixed-point tasks that a system demands (addressing, etc.). The motivation of this project is to eventually implement fixed-point GEMM on a FPGA, and what we have done here presents a solid starting point for that goal. 
