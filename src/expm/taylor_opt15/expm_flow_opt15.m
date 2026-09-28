function [E,m,s,nProd]=expm_flow_opt15(A,tol)
% Computes the matrix exponential of A using the scaling and squaring
% algorithm and the Taylor polynomial approximation. Polynomials are
% efficiently evaluated by means of Sastre formulas. 
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
%
% Group of High Performance Scientific Computation (HiPerSC)
% Universitat Politècnica de València (Spain)
% http://hipersc.blogs.upv.es

if nargin<2
    tol=1e-8;
end
[E,m,s,nProd]=expm_taylor_flow(A,tol,2);
