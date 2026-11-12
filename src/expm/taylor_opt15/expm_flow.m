function [E,m,s,nProd]=expm_flow(A,tol)
% Compute the matrix exponential: sum_{k=0}^{\infty} A^k / k!
% Generative Flows with Matrix Exponential. Proceedings of the 37th 
% International Conference on Machine Learning, Online, PMLR 119, 2020.
%
% Inputs:
% - A:     input matrix.
% - tol:   tolerance.
%
% Outputs: 
% - E:     exponential of matrix A.
% - m:     approximation polynomial order used.
% - s:     scaling parameter.
% - nProd: cost in terms of number of matrix products.

if nargin<2
    tol=1e-8;
end

% Scale the matrix A
s=ceil(log2(max(norm(A,1),0.5)))+1;
A=A/(2^s);

% Initialize variables
E=eye(size(A));  % Identity matrix of the same size as A
t=A;
m=0;
nProd=0;
k=2;

% Loop until the norm of t is sufficiently small
while norm(t,1)>tol
    E=E+t;
    t=A*t/k;
    k=k+1;
    m=m+1;
    nProd=nProd+1;
end
    
% Square the result E 's' times
for i=1:s
    E=E*E;
    nProd=nProd+1;
end
end