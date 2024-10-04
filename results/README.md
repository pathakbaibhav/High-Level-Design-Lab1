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

We want to better understand the performance of Darknet and the bottlenecks that exist in its existing codebase. To do this, we use Darknet's included profiling feature. The outputted performance summary has information on how many times each instruction is called, how much time it takes to execute, and what percentage of the program's runtime is dominated by it. For a non-optimized Darknet, we found the following results:


