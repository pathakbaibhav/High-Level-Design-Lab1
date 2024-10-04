# EECE7368 Lab 1 Report

### Darknet Profiling and Floating to Fixed Conversion

Robert D'Antonio and Baibhav Pathak

## 1. Introduction

In this first lab, we were tasked with running, profiling, and improving Darknet. Darknet is a light-weight, open-source deep learning framework that is suitable for many applications where strong compute may not be available. For our lab, we worked with Darknet's YoloV3 model which performs real-time object detection. We first compiled and tested the model to understand its general detection performance. Next, we profiled YoloV3 on Darknet to gain some insight into bottleneck instructions. Finally, we implemented software optimizations to address these bottlenecks, achieiving quicker runtimes with significantly fewer computation-intensive floating-point operations. 
