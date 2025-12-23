function [E,m,s,nProd]=expm_flow_sastre(A)
% Computes the matrix exponential of A using the scaling and squaring
% algorithm and the Taylor polynomial approximation. Polynomials are
% efficiently evaluated by means of Sastre formulas. 
%
% Inputs:
% - A:        input matrix.
%
% Outputs: 
% - E:        exponential of matrix A.
% - m:        approximation polynomial order used.
% - s:        scaling parameter.
% - nProd:    cost in terms of number of matrix products. 
%
% Group of High Performance Scientific Computation (HiPerSC)
% Universitat Politècnica de València (Spain)
% http://hipersc.blogs.upv.es
[E,m,s,nProd]=expm_taylor_flow(A,2);
