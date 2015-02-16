CUDAkNN
=============
An implementation of the k-nearest neighbors in CUDA

Introduction
-------------
This is an implementation of the k-nearest neighbors in CUDA using some naive implementations plus a static kd-tree optimization

Install
-------------
This software is Linux supported only, but it might works in Windows as well

*Dependencies*
- CUDA 6.5 
- GNU Make 4.1
- Git 2.2.2

**Linux**

*Compiling*

Open a terminal and type

    git clone https://github.com/tlgimenes/CUDAkNN.git

You shoud now be downloading the source code. After finished downloading, type

    cd CUDAkNN
  
    make

Congratulations ! You are done !

TODO
-------------
- [x] Implement a naive k-nearest neighbors in CUDA
- [ ] Implement a static kd-tree in CUDA
- [ ] Implement kNN using the kd-tree in CUDA
- [ ] Implement real time parallel kd-tree in CUDA
- [ ] Implement kNN using the dynamic real-time kd-tree in CUDA
