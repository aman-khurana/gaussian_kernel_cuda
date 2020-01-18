### Calculating Gaussian Kernel Density Estimate using  CUDA

NVIDIA CUDA Kernel for fast computing of Gaussian kernel density estimate. The goal is to estimate the probability density function of a random variable. Let vector $X=(x_1,x_2,…,x_n)$ be a sample drawn from some univariate distribution with an unknown density function f. We can estimate f as:

$f^h(x)=1/(n⋅h)∑_{i=1}^nK((x−x_i)/h)$

where h is bandwidth, and K is a kernel function. In the specific case of Gaussian kernel density estimate, K is defined as follows:

$K(x)=1/\sqrt2π* exp(−x^2/2)$

Given an input vector $X=(x_1,x_2,…,x_n)$, and some predefined bandwidth h, we can estimate density over X, as $Y=(y_1,y_2,…,y_n)$, where $y_i = \hat{f}_h(x_i)$.

This code was tested and benchmarked on UB's HPC center, [Center for Computational Research.](http://www.buffalo.edu/ccr.html)



