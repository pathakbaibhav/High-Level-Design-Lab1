# EECE 7368 Lab 1
# Darknet Profiling and Floating to Fixed Conversion

The lab largely based on work by Andreas Gerstlauer, UT Austin.

## Overview
The goals of this lab are to:

- Learn the structure of the Darknet source code and compile the code
- Identify and propose ways to remove the bottleneck of the code when running on CPU
 
The assignment of this lab includes the following:

- Set up the design environment
- Profile the code to identify the time consuming portions of the code
- Complete an exercise to remove a type of bottleneck
- Isolate modules of the Darknet and perform floating-to-fixed point conversion
- Perform additional software optimizations 

To simplify this lab, compilation, execution and profiling will occur on the host (e.g. x86). Execution on the simulator or actual board will follow once the environment is established. 

This lab's code stretches over two repositories. In addition to this repo, there is the darknet repository for which you have accepted a separate assignment (see Canvas assignment), which created a repository with the name `f<year>-darknet-<name>`. 

The instructions for ths lab are detailed in the following steps:

 1. (Reserved for feedback branch pull request. You will receive top level feedback there).
 2. [Compiling and Running Darknet](.github/STARTING_ISSUES/2.%20Compiling%20and%20Running%20Darknet.md)
 3. [Profiling Darknet](.github/STARTING_ISSUES/3.%20Profiling%20Darknet.md)
 4. [Floating- to Fixed-point Conversion](.github/STARTING_ISSUES/4.%20Floating-%20to%20Fixed-point%20Conversion.md)
 5. [Integrate FP GEMM into Darknet](.github/STARTING_ISSUES/5.%20Integrate%20FP%20GEMM%20into%20Darknet.md)
 6. [Additional SW Optimizations - Extra Credit](.github/STARTING_ISSUES/6.%20Additional%20SW%20Optimizations%20-%20Extra%20Credit.md )

After accepting this assignment in github classroom, each step is converted into a [github issue](https://docs.github.com/en/issues). Follow the issues in numerically increasing issue number (the first issue is typically on the bottom of the list). 

## Report

Deliver your report as part of your lab 1 assignment repository. Create s folder `results` and make the results/README.md your report. Any media (e.g. images) used in the report should be committed into the same directory. If Markdown is not your thing and rather want to use a different editor, you can commit a PDF version of your report into the results folder.

The report should list the bottlenecks identified during profiling and discuss/propose ways used to remove them. List the differences between the original floating point and fixed point versions of the Darknet code with respect to what you observed by profiling them. Finally, report on the results of floating-point to fixed-point conversion and any additional optimizations performed. 

## General Rules

Please commit your code frequently or at e very logical break. Each commit should have a meaningful commit message and [cross reference](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/autolinked-references-and-urls#issues-and-pull-requests) the issue the commit belongs to. Ideally, there would be no commits without referencing to a github issue. 

Please comment on each issue with the problems faced and your approach to solve them. Close an issue when done. 



