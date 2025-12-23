function [E,m,s,nProd]=expm_taylor_flow(A,poleval)
% Computes the matrix exponential of A using the scaling and squaring
% algorithm and the Taylor polynomial approximation. Polynomials are
% efficiently evaluated by means of Paterson-Stockmeyer algorithm or
% Sastre formulas. Accuracy is fixed to 1e-8.
%
% Inputs:
% - A:       input matrix.
% - poleval: formulas to apply in the polynomial evaluation: 
%            Paterson-Stockmeyer (1) or Sastre (2).
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

if nargin<2
    poleval=2;
end

% Select m and s values
[m,s,pA,nProd_ms]=select_m_s_expm(A,poleval);

% Scaling technique
pA=scaling(pA,s);

% Get polynomial coefficients
c=coefs_expm(m,poleval);

% Evaluate efficiently the polynomial
[E,nProd_eval]=polyvalm(c,pA,poleval);

% Squaring technique
[E,nProd_sq]=squaring(E,s);

% Total number of matrix products
nProd=nProd_ms+nProd_eval+nProd_sq;

end

function [m, s, pA, nProd] = select_m_s_expm(A,poleval)
% Determines the polynomial order and the scaling parameter to compute the
% exponential of matrix A by means of Taylor approximation of order m<=16.
%
% Inputs:
% - A:       input matrix.
% - poleval: formulas to apply in the polynomial evaluation: 
%            Paterson-Stockmeyer (1) or Sastre (2).
% 
% Outputs: 
% - m:       approximation polynomial order used.
% - s:       scaling parameter.
% - pA:      list with the powers of A nedeed, such that pA{i} contains
%            A^i, for i=1,2,3,...,q.
% - nProd:   number of matrix products required by the function.

if poleval==1
    [m,s,pA,nProd]=select_m_s_expm_paterson_stockmeyer(A,1e-8);
else
    [m,s,pA,nProd]=select_m_s_expm_sastre(A,1e-8);
end
end


function [m, s, pA, nProd] = select_m_s_expm_paterson_stockmeyer(A,tol)
% Determines the polynomial order and the scaling parameter to compute the
% exponential of matrix A by means of Taylor approximation of order 
% m <= 16. The polynomial will be evaluated by means of Paterson-Stockmeyer 
% algorithm. 
%
% Inputs:
% - A:     input matrix.
% - tol:   tolerance.
% 
% Outputs: 
% - m:     approximation polynomial order used.
% - s:     scaling parameter.
% - pA:    list with the powers of A nedeed, such that pA{i} contains
%          A^i, for i=1,2,3,...,q.
% - nProd: number of matrix products required by the function.

M=[1 2 4 6 9 12 16];
J=[1 2 2 3 3 4 4];
K=[1 1 2 2 3 3 4];

m=0;
s=0;
nProd=0;
pA{1}=A;
norm_pA(1)=norm(pA{1},1);
if norm_pA(1)==0
    return;
end

norm_pA(1)=log2(norm_pA(1));
tol=log2(tol);
L=[-1.000000000000000e+00 -2.584962500721156e+00 -2.584962500721156e+00 -4.584962500721156e+00 -6.906890595608519e+00 -9.491853096329674e+00 -1.229920801838728e+01 -1.529920801838728e+01 -2.179106111471695e+01 -2.525049273335425e+01 -3.253589495221650e+01 -3.634324987427410e+01 -4.833760331113296e+01 -5.250752831257527e+01 -6.546970134368499e+01 -6.992913296232229e+01 -8.838195332701626e+01 -9.313684082917973e+01 -1.126632636523603e+02 -1.176632636523603e+02 -1.433037819969853e+02 -1.485517095104289e+02 -1.753351740062775e+02 -1.807946056249148e+02 -2.142081380635902e+02 -2.198805634055617e+02 -2.544854157301764e+02 -2.603433967253040e+02 -3.020175117547525e+02 -3.080619058741109e+02];
fin=0;
i=1;
while i<=length(M) && fin==0
    m=M(i);
    j=J(i);
    k=K(i);
    if j>k
        pA{j}=pA{j-1}*A;
        norm_pA(j)=log2(norm(pA{j},1));
        nProd=nProd+1;        
    end
    p=2*i-1;    
    if m==1
        error(1)=L(p) + 2*norm_pA(1);
        error(2)=L(p+1) + 3*norm_pA(1);
    else        
        error(1)=L(p) + k*norm_pA(j) + norm_pA(1);
        error(2)=L(p+1) + k*norm_pA(j) + norm_pA(2); 
    end
    if error(1)<=tol && error(2)<=tol
        fin=1;
    else
        i=i+1;
    end
end
if fin==0
    for i=1:2
        aux=error(i)-tol;
        s1=ceil(aux/(m+i));
        if s1>s
            s=s1;
        end
    end
end
if s>20
    s=20;
end
end

function [m, s, pA, nProd] = select_m_s_expm_sastre(A,tol)
% Determines the polynomial order and the scaling parameter to compute the
% exponential of matrix A by means of Taylor approximation of order 
% m <= 15. The polynomial will be evaluated by means of Sastre formulas. 
%
% Inputs:
% - A:     input matrix.
% - tol:   tolerance.
% 
% Outputs: 
% - m:     approximation polynomial order used.
% - s:     scaling parameter.
% - pA:    list with the powers of A nedeed, such that pA{i} contains
%          A^i, for i=1,2,3,...,q.
% - nProd: number of matrix products required by the function.

M=[1 2 4 8 15];
J=[1 2 2 2 2];
K=[1 1 2 4 8];

m=0;
s=0;
nProd=0;
pA{1}=A;
norm_pA(1)=norm(pA{1},1);
if norm_pA(1)==0
    return;
end

norm_pA(1)=log2(norm_pA(1));
tol=log2(tol);
L=[-1 -2.584962500721156 -2.584962500721156 -4.584962500721156 -6.906890595608519 -9.491853096329674 -1.846913301982959e+01 -2.179106111471695e+01 -4.538856141351141e+01 -4.833760331113296e+01];
fin=0;
i=1;
while i<=length(M) && fin==0
    m=M(i);
    j=J(i);
    k=K(i);
    if m==2
        pA{j}=pA{j-1}*A;
        norm_pA(j)=log2(norm(pA{j},1));
        nProd=nProd+1;        
    end
    p=2*i-1;    
    if m==1
        error(1)=L(p) + 2*norm_pA(1);
        error(2)=L(p+1) + 3*norm_pA(1);
    else        
        error(1)=L(p) + k*norm_pA(j);
        error(2)=L(p+1) + k*norm_pA(j);
        if j*k==m 
            error(1)=error(1) + norm_pA(1);
            error(2)=error(2) + norm_pA(2);
        else
            error(2)=error(2) + norm_pA(1);
        end
    end
    if error(1)<=tol && error(2)<=tol
        fin=1;
    else
        i=i+1;
    end
end
if fin==0
    for i=1:2
        aux=error(i)-tol;
        s1=ceil(aux/(m+i));
        if s1>s
            s=s1;
        end
    end
end
if s>20
    s=20;
end
end

function pA = scaling(pA,s)
% Scaling of A powers.
%
% Inputs:
% - pA: cell array with the powers of A nedeed. pA{i} contains A^i, 
%       for i=1,2,3,...,q.
% - s:  scaling parameter.
%
% Outputs:
% - pA: cell array with the powers of A after scaling.

if s>0
    q=length(pA);
    for k=1:q
        pA{k}=pA{k}/(2^(k*s));
    end
end
end

function [A,nProd] = squaring(A,s)
% Squaring of A.
%
% Inputs:
% - A:    matrix resulting of the approximation polynomial evaluation.
% - s:    scaling parameter.
%
% Outputs:
% - A:    the exponential of matrix A after squaring technique.
% -nProd: number of matrix products carried out by the function.  

for i=1:s
    A=A*A;
end
nProd=s;
end

function p=coefs_expm(m,poleval)
% Provides the approximation Taylor polynomial coefficients for the 
% matrix exponential function to be evaluated by means of the Paterson-
% Stockmeyer algorithm or the Sastre formulas.
%
% Inputs:
% - m:       approximation Taylor polynomial order.
% - poleval: formulas to apply in the polynomial evaluation: 
%            Paterson-Stockmeyer (1) or Sastre (2).
% Outputs:
% - p:       vector of m+1 components with the coefficients of the 
%            polynomial ordered in ascending (Paterson-Stockmeyer) or 
%            descending powers (Sastre).

if poleval==1
   p=coefs_expm_paterson_stockmeyer(m);
elseif poleval==2
    p=coefs_expm_sastre(m);
end
end

function p=coefs_expm_paterson_stockmeyer(m)
% Provides the approximation Taylor polynomial coefficients for the 
% matrix exponential function to be evaluated by means of the Paterson-
% Stockmeyer algorithm.
%
% Inputs:
% - m: approximation Taylor polynomial order.
% Outputs:
% - p: vector of m+1 components with the coefficients of the polynomial 
%      ordered from degree zero to degree m, i.e. in ascending powers.

p=[1.0000000000000000e+00, 1.0000000000000000e+00, 5.0000000000000000e-01, 1.6666666666666666e-01, 4.1666666666666664e-02, 8.3333333333333332e-03, 1.3888888888888889e-03, 1.9841269841269841e-04, 2.4801587301587302e-05, 2.7557319223985893e-06, 2.7557319223985888e-07, 2.5052108385441720e-08, 2.0876756987868100e-09, 1.6059043836821613e-10, 1.1470745597729725e-11, 7.6471637318198164e-13, 4.7794773323873853e-14, 2.8114572543455206e-15, 1.5619206968586225e-16, 8.2206352466243295e-18, 4.1103176233121648e-19, 1.9572941063391263e-20, 8.8967913924505741e-22, 3.8681701706306835e-23, 1.6117375710961184e-24, 6.4469502843844736e-26, 2.4795962632247972e-27, 9.1836898637955460e-29, 3.2798892370698385e-30, 1.1309962886447718e-31, 3.7699876288159061e-33, 1.2161250415535181e-34, 3.8003907548547441e-36, 1.1516335620771951e-37, 3.3871575355211618e-39, 9.6775929586318907e-41, 2.6882202662866367e-42, 7.2654601791530724e-44, 1.9119632050402823e-45, 4.9024697565135435e-47, 1.2256174391283860e-48, 2.9893108271424051e-50, 7.1174067312914405e-52, 1.6552108677421951e-53, 3.7618428812322623e-55, 8.3596508471828045e-57, 1.8173154015614793e-58, 3.8666285139605940e-60, 8.0554760707512382e-62, 1.6439747083165791e-63, 3.2879494166331584e-65, 6.4469596404571737e-67, 1.2397999308571486e-68, 2.3392451525606576e-70, 4.3319354677049218e-72, 7.8762463049180392e-74, 1.4064725544496498e-75, 2.4674957095607889e-77, 4.2543029475186016e-79, 7.2106829618959346e-81, 1.2017804936493225e-82, 1.9701319568021682e-84, 3.1776321883905938e-86, 5.0438606164930067e-88, 7.8810322132703230e-90];
p=p(1:m+1);
end

function c=coefs_expm_sastre(m)
% Provides the approximation Taylor polynomial coefficients for the
% matrix exponential function to be evaluated by means of the Sastre formulas.
% Inputs:
% - m: approximation Taylor polynomial order.
% Outputs:
% - c: vector of m+1 components with the coefficients of the polynomial 
%      ordered in descending powers.

if m == 0
    c = 1;
elseif m == 1
    c = [1, 1];
elseif m == 2
    c = [1, 1, 0.5];
elseif m == 4
    c = [1, 1, 0.5, 0.3333333333333333, 0.25];
elseif m == 8
    c = [4.980119205559973e-03, 1.992047682223989e-02, 7.665265321119147e-02, 8.765009801785554e-01, 1.225521150112075e-01, 2.974307204847627, 0.5, 1, 1];
elseif m == 15
    c = [4.018761610201036e-04, 2.945531440279683e-03, -8.709066576837676e-03, 4.017568440673568e-01, 3.230762888122312e-02, 5.768988513026145, 2.338576034271299e-02, 2.381070373870987e-01, 2.224209172496374, -5.792361707073261, -4.130276365929783e-02, 1.040801735231354e+01, -6.331712455883370e+01, 3.484665863364574e-01, 1, 1];
end
end

function [E,nProd]=polyvalm(p,pA,poleval)
% Evaluates the polynomial E = p(1)*I + p(2)*A + p(3)*A^2 + ...+ p(m+1)*A^m 
% efficiently by means of the Paterson-Stockmeyer algorithm or the Sastre
% formulas.
%
% Inputs:
% - p:       vector of length m+1 with the polynomial coefficients ordered 
%            in ascending (Paterson-Stockmeyer) or descending powers 
%            (Sastre). 
% - pA:      cell array with the powers of A nedeed, such that pA{i} 
%            contains A^i, for i=1, 2,3,...,q.
% - poleval: formulas to apply in the polynomial evaluation: 
%            Paterson-Stockmeyer (1) or Sastre (2).
%
% Outputs: 
% - E:     polynomial evaluation result.
% - nProd: number of matrix products carried out by the function.

if poleval==1
    % Polynomial evaluation by means of Paterson-Stockmeyer algorithm
    [E,nProd]=polyvalm_paterson_stockmeyer(p,pA);
elseif poleval==2
    % Polynomial evaluation by means of Sastre formulas
    [E,nProd]=polyvalm_sastre(p,pA);
end
    
end

function [E,nProd]=polyvalm_paterson_stockmeyer(p,pA)
% Evaluates the polynomial E = p(1)*I + p(2)*A + p(3)*A^2 + ...+ p(m+1)*A^m 
% efficiently by means of the Paterson-Stockmeyer algorithm.
%
% Inputs:
% - p:     vector of length m+1 with the polynomial coefficients in 
%          ascending powers.
% - pA:    cell array with the powers of A nedeed, such that pA{i} contains
%          A^i, for i=1, 2,3,...,q.
%
% Outputs: 
% - E:     polynomial evaluation result.
% - nProd: number of matrix products carried out by the function.  

n=size(pA{1},1);
I=eye(n);
m=length(p)-1;
c=m+1;
q=length(pA);
k=ceil(m/q);
mIdeal=q*k;
if m==0
    E=p(1)*I;
else
    E=zeros(n);
end
nProd=0;
for j=k:-1:1
    if j==k
        inic=q-mIdeal+m;
    else
        inic=q-1;
    end
    for i=inic:-1:1
        E=E+p(c)*pA{i};
        c=c-1;
    end
    E=E+p(c)*I;
    c=c-1;
    if j~=1
        E=E*pA{q}; 
        nProd=nProd+1;
    end
end

end

function [E,nProd]=polyvalm_sastre(c,pA)
% Evaluates the polynomial efficiently by means of the Sastre formulas.    
% 
% Inputs:
% - c:     vector of m+1 componentes with the coefficients in descending 
%          powers.
% - pA:    list with the powers of A nedeed, such that pA{i} contains
%          A^i, for i=1,2,3,...,q.     
% Outputs: 
% - E:     polynomial evaluation result.
% - nProd: number of matrix products required by the function.  

m=length(c)-1;
E=eye(size(pA{1},1),class(pA{1}));
if m==0
    nProd=0;
elseif m==1
    E=c(2)*pA{1}+E;
    nProd=0;
elseif m==2
    E=c(3)*pA{2}+c(2)*pA{1}+E;
    nProd=0;
elseif m==4
    E=((pA{2}*c(5)+pA{1})*c(4)+E)*pA{2}*c(3)+pA{1}*c(2)+E;
    nProd=1;
elseif m==8
    y0s=pA{2}*(c(1)*pA{2}+c(2)*pA{1});
    E=(y0s+c(3)*pA{2}+c(4)*pA{1})*(y0s+c(5)*pA{2})+c(6)*y0s+c(7)*pA{2}+c(8)*pA{1}+c(9)*E;
    nProd=2;
elseif m==15
    y0s=pA{2}*(c(1)*pA{2}+c(2)*pA{1});
    y1s=(y0s+c(3)*pA{2}+c(4)*pA{1})*(y0s+c(5)*pA{2})+c(6)*y0s+c(7)*pA{2};
    E=(y1s+c(8)*pA{2}+c(9)*pA{1})*(y1s+c(10)*y0s+c(11)*pA{1})+c(12)*y1s+c(13)*y0s+c(14)*pA{2}+c(15)*pA{1}+c(16)*E;
    nProd=3;
end
end
