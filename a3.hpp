/*  YOUR_FIRST_NAME
 *  YOUR_LAST_NAME
 *  YOUR_UBIT_NAME
 */

#ifndef A3_HPP
#define A3_HPP
#define _USE_MATH_DEFINES

#include<iostream>
#include <cmath>
#include <math.h>


using namespace std;

__global__ void kde(int n, float h, float* dx, float* dy){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (i < n){
            
                float x_curr = dx[i];
		float K = 1/sqrt(x_curr * M_PI) * exp(-pow(x_curr, 2)/2);
                float sum_temp = 0;
		for(int j=0; j<n;j++){
			float x = dx[j];
			sum_temp += K * ((x - x_curr)/h) ;	
		
		} 	        
		
		float sum = sum_temp/(n * h) ;
                dy[i] = sum;
		
        }
}



__host__

void gaussian_kde(int n, float h, const std::vector<float>& x, std::vector<float>& y) {

float* dx ;
float* dy ;

cudaMalloc((void **)&dx, n* sizeof(float));
cudaMalloc((void **)&dy, n* sizeof(float));

cudaMemcpy(dx, &x[0], n*sizeof(float), cudaMemcpyHostToDevice);

int threads_per_block = 2048;
kde<<<(n + threads_per_block - 1)/threads_per_block, threads_per_block>>>(n,h, dx,dy);

cudaMemcpy(&y[0], dy, n* sizeof(float), cudaMemcpyDeviceToHost);
 

//for(int i =0; i< x.size();i++)
//	cout<<x[i]<<",";
//cout<<endl;

//for(int i=0; i< y.size(); i++)
//	cout<<y[i]<<" ";
//cout<<endl;

cudaFree(dx);
cudaFree(dy);

} // gaussian_kde



#endif // A3_HPP
