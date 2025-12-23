import torch

def expm_flow_ps(A):    
#def expm(A):        
    """
    Computes the matrix exponential of A using the scaling and squaring 
    algorithm and the Taylor polynomial approximation. Polynomials are
    efficiently evaluated by means of the Paterson-Stockmeyer.
    
    Inputs:
      - A:       input matrix.
    
    Outputs: 
      - E:       exponential of matrix A.
      - m:       approximation polynomial order used.
      - s:       scaling parameter.
      - nProd:   number of matrix products required by the function.         
     
    Group of High Performance Scientific Computation (HiPerSC)
    Universitat Politècnica de València (Spain)
    http://hipersc.blogs.upv.es
    """
    
    E,m,s,nProd=expm_taylor_flow(A,1)
    
    #return E,m,s,nProd
    return E

def expm_flow_sastre(A):
#def expm(A):    
    """
    Computes the matrix exponential of A using the scaling and squaring 
    algorithm and the Taylor polynomial approximation. Polynomials are
    efficiently evaluated by means of Sastre formulas. 
    
    Inputs:
      - A:       input matrix.
    
    Outputs: 
      - E:       exponential of matrix A.
      - m:       approximation polynomial order used.
      - s:       scaling parameter.
      - nProd:   number of matrix products required by the function.         
     
    Group of High Performance Scientific Computation (HiPerSC)
    Universitat Politècnica de València (Spain)
    http://hipersc.blogs.upv.es
    """
    
    E,m,s,nProd=expm_taylor_flow(A,2)
    
    #return E,m,s,nProd
    return E

def expm_taylor_flow(A,poleval=2):    
    """
    Computes the matrix exponential of A using the scaling and squaring 
    algorithm and the Taylor polynomial approximation. Polynomials are
    efficiently evaluated by means of the Paterson-Stockmeyer or Sastre 
    formulas. Accuracy is fixed to 1e-8.
    
    Inputs:
      - A:       input matrix.
      - metms:   fixed (1) or variable (2) tolerance.
      - poleval: formulas to apply in the polynomial evaluation: 
                 Paterson-Stockmeyer (1) or Sastre (2).
    
    Outputs: 
      - E:       exponential of matrix A.
      - m:       approximation polynomial order used.
      - s:       scaling parameter.
      - nProd:   number of matrix products required by the function.         
     
    Group of High Performance Scientific Computation (HiPerSC)
    Universitat Politècnica de València (Spain)
    http://hipersc.blogs.upv.es
    """
    
    # Select m and s values
    m,s,pA,nProd_ms=select_m_s_expm(A,poleval)
    
    # Scaling technique
    scaling(pA,s)
    
    # Get polynomial coefficients
    c=coefs_expm(m,poleval)
    
    #Evaluate efficiently the polynomial
    fA,nProd_eval=polyvalm(c,pA,poleval)
    
    #Squaring technique
    E,nProd_sq=squaring(fA,s)
    
    #Total number of matrix products
    nProd=nProd_ms+nProd_eval+nProd_sq
    return E,m,s,nProd

def series_flow_ps(A):
#def series(A):   
    """
    Computes the matrix series sum_{k=0}^{infty}\frac{x^{k}}{(k+1)!}. 
    Polynomials are efficiently evaluated by means of the Paterson-Stockmeyer
    algorithm.
    
    Inputs:
      - A:       input matrix.
      - poleval: formulas to apply in the polynomial evaluation: 
                 Paterson-Stockmeyer (1) or Sastre (2).
    
    Outputs: 
      - S:       result of the matrix series computation.
      - m:       approximation polynomial order used.
      - s:       scaling parameter.
      - nProd:   number of matrix products required by the function.
     
    Group of High Performance Scientific Computation (HiPerSC)
    Universitat Politècnica de València (Spain)
    http://hipersc.blogs.upv.es
    """
    
    S,m,s,nProd=series_taylor_flow(A,1)
    #return S,m,s,nProd
    return S
    

#def series_flow_sastre(A):
def series(A):
    """
    Computes the matrix series sum_{k=0}^{infty}\frac{x^{k}}{(k+1)!}. 
    Polynomials are efficiently evaluated by means of the Sastre formulas. 
    
    Inputs:
      - A:       input matrix.
      - poleval: formulas to apply in the polynomial evaluation: 
                 Paterson-Stockmeyer (1) or Sastre (2).
    
    Outputs: 
      - S:       result of the matrix series computation.
      - m:       approximation polynomial order used.
      - s:       scaling parameter.
      - nProd:   number of matrix products required by the function.
     
    Group of High Performance Scientific Computation (HiPerSC)
    Universitat Politècnica de València (Spain)
    http://hipersc.blogs.upv.es
    """
    
    S,m,s,nProd=series_taylor_flow(A,2)
    #return S,m,s,nProd
    return S

def series_taylor_flow(A,poleval=2):    
    """
    Computes the matrix series sum_{k=0}^{infty}\frac{x^{k}}{(k+1)!}. 
    Polynomials are efficiently evaluated by means of the Paterson-Stockmeyer
    algorithm of the Sastre formulas. Accuracy is fixed to 1e-8.
    
    Inputs:
      - A:       input matrix.
      - poleval: formulas to apply in the polynomial evaluation: 
                 Paterson-Stockmeyer (1) or Sastre (2).
    
    Outputs: 
      - S:       result of the matrix series computation.
      - m:       approximation polynomial order used.
      - s:       scaling parameter.
      - nProd:   number of matrix products required by the function.
     
    Group of High Performance Scientific Computation (HiPerSC)
    Universitat Politècnica de València (Spain)
    http://hipersc.blogs.upv.es
    """    
    
    # Select m and s values
    m,s,pA,nProd_ms=select_m_s_series(A,poleval)
    
    # Scaling technique
    scaling(pA,s)
    
    # Get polynomial coefficients
    c=coefs_series(m,poleval)
    
    # Evaluate efficiently the series
    fA,nProd_eval=polyvalm(c,pA,poleval)
    
    #Squaring technique
    S,nProd_sq=squaring(fA,s)
    
    #Total number of matrix products
    nProd=nProd_ms+nProd_eval+nProd_sq
    
    return S,m,s,nProd

def select_m_s_expm(A,poleval=2):
    """
    Determines the polynomial order and the scaling parameter to compute the
    exponential of matrix A by means of Taylor approximation of order m<=16,
    without estimations of matrix power norms. Accuracy is fixed to 1e-8.
    
    Inputs:
      - A:       input matrix.
      - poleval: formulas to apply in the polynomial evaluation: 
                 Paterson-Stockmeyer (1) or Sastre (2).
  
    Outputs: 
      - m:       approximation polynomial order to be used.
      - s:       scaling parameter.
      - pA:      list with the powers of A nedeed, such that pA[i] contains
                 A^(i+1), for i=0,1,2,3,...,q-1.
      - nProd:   number of matrix products required by the function.
    """
    if poleval==1:
        m,s,pA,nProd=select_m_s_expm_paterson_stockmeyer(A,1e-8)
    elif poleval==2:
        m,s,pA,nProd=select_m_s_expm_sastre(A,1e-8)      
    return m,s,pA,nProd


def select_m_s_series(A,poleval=2):
    """
    Determines the polynomial order and the scaling parameter to compute the
    matrix series sum_{k=0}^{infty}\frac{x^{k}}{(k+1)!}.
    
    Inputs:
      - A:       input matrix.
      - poleval: formulas to apply in the polynomial evaluation: 
                 Paterson-Stockmeyer (1) or Sastre (2).
  
    Outputs: 
      - m:       approximation polynomial order to be used.
      - s:       scaling parameter.
      - pA:      list with the powers of A nedeed, such that pA[i] contains
                 A^(i+1), for i=0,1,2,3,...,q-1.
      - nProd:   number of matrix products required by the function.
    """

    if poleval==1:
        m,s,pA,nProd=select_m_s_series_paterson_stockmeyer(A,1e-8)
    elif poleval==2:
        m,s,pA,nProd=select_m_s_series_sastre(A,1e-8)      
    return m,s,pA,nProd

def select_m_s_expm_paterson_stockmeyer(A,tol=1e-8):
    """
    Determines the polynomial order and the scaling parameter to compute the
    exponential of matrix A by means of Taylor approximation of order m<=16. 
    The polynomial will be evaluated by means of Paterson-Stockmeyer algorithm. 
    
    Inputs:
      - A:     input matrix.
      - tol:   tolerance.
  
    Outputs: 
      - m:     approximation polynomial order used.
      - s:     scaling parameter.
      - pA:    list with the powers of A nedeed, such that pA[i] contains
               A^(i+1), for i=0,1,2,3,...,q-1.
      - nProd: number of matrix products required by the function.
    """
    pA=[]
    norm_pA=[]
    s=0
    nProd=0
 
    pA.append(A)
    norm_pA.append(torch.norm(pA[0],p=1,dim=-1).max().item())
   
    # Try with m=0
    m=0
    if norm_pA[0]==0:
        return m,s,pA,nProd
   
    M=[1, 2, 4, 6, 9, 12, 16]
    J=[1, 2, 2, 3, 3, 4, 4]
    K=[1, 1, 2, 2, 3, 3, 4]
   
    norm_pA[0]=torch.math.log2(norm_pA[0])
    tol=torch.math.log2(tol);

    L=[-1, -2.584962500721156, -2.584962500721156, -4.584962500721156, -6.906890595608519, -9.491853096329674, -1.229920801838728e+01, -1.529920801838728e+01, -2.179106111471695e+01, -2.525049273335425e+01, -3.253589495221650e+01, -3.634324987427410e+01, -4.833760331113296e+01, -5.250752831257527e+01, -6.546970134368499e+01, -6.992913296232229e+01, -8.838195332701626e+01, -9.313684082917973e+01, -1.126632636523603e+02, -1.176632636523603e+02, -1.433037819969853e+02, -1.485517095104289e+02 -1.753351740062775e+02, -1.807946056249148e+02, -2.142081380635902e+02, -2.198805634055617e+02, -2.544854157301764e+02, -2.603433967253040e+02, -3.020175117547525e+02, -3.080619058741109e+02]

    fin=False
    i=0
    error=[0, 0]
    while i<len(M) and fin==False:
        # Try with m
        m=M[i]
        j=J[i]
        k=K[i]
        
        if j>k:
            pA.append(torch.matmul(pA[-1],A))
            norm_pA.append(torch.math.log2(torch.norm(pA[-1],p=1,dim=-1).max().item()))
            nProd+=1
           
        p=2*i    
        if m==1:
            error[0]=L[p] + 2*norm_pA[0]
            error[1]=L[p+1]+ 3*norm_pA[0]
        else:
            error[0]=L[p] + k*norm_pA[j-1] + norm_pA[0]
            error[1]=L[p+1]+ k*norm_pA[j-1] + norm_pA[1]
       
        if error[0]<=tol and error[1]<=tol:
            fin=True
        else:
            i+=1
           
    if fin==False:
        for i in range(2):
            aux=error[i]-tol
            s1=torch.math.ceil(aux/(m+i+1))
            if s1>s:
                s=s1
    if s>20:
       s=20
    
    return m,s,pA,nProd


def select_m_s_expm_sastre(A,tol=1e-8):
    """
    Determines the polynomial order and the scaling parameter to compute the
    exponential of matrix A by means of Taylor approximation of order m<=15+. 
    The polynomial will be evaluated by means of Sastre formulas. 
    
    Inputs:
      - A:     input matrix.
      - tol:   tolerance.
  
    Outputs: 
      - m:     approximation polynomial order used.
      - s:     scaling parameter.
      - pA:    list with the powers of A nedeed, such that pA[i] contains
               A^(i+1), for i=0,1,2,3,...,q-1.
      - nProd: number of matrix products required by the function.
    """
    
    pA=[]
    norm_pA=[]
    s=0
    nProd=0
  
    pA.append(A)
    norm_pA.append(torch.norm(pA[0],p=1,dim=-1).max().item())
    
    # Try with m=0
    m=0
    if norm_pA[0]==0:
        return m,s,pA,nProd
      
    M=[1, 2, 4, 8, 15]
    J=[1, 2, 2, 2, 2]
    K=[1, 1, 2, 4, 8]

    norm_pA[0]=torch.math.log2(norm_pA[0])
    tol=torch.math.log2(tol);

    L=[-1, -2.584962500721156, -2.584962500721156, -4.584962500721156, -6.906890595608519, -9.491853096329674, -1.846913301982959e+01, -2.179106111471695e+01, -4.538856141351141e+01, -4.833760331113296e+01]

    fin=False
    i=0
    error=[0, 0]
    while i<len(M) and fin==False:
        # Try with m
        m=M[i]
        j=J[i]
        k=K[i]
        
        if m==2:
            pA.append(torch.matmul(pA[-1],A))
            norm_pA.append(torch.math.log2(torch.norm(pA[-1],p=1,dim=-1).max().item()))
            nProd+=1
            
        p=2*i    
        if m==1:
            error[0]=L[p] + 2*norm_pA[0]
            error[1]=L[p+1]+ 3*norm_pA[0]
        else:
            error[0]=L[p] + k*norm_pA[j-1]
            error[1]=L[p+1] + k*norm_pA[j-1]
            if j*k==m:
                error[0]+= norm_pA[0]
                error[1]+= norm_pA[1]
            else: # m=15
                error[1]+=norm_pA[0]

        if error[0]<=tol and error[1]<=tol:
            fin=True
        else:
            i+=1
            
    if fin==False:
        for i in range(2):
            aux=error[i]-tol
            s1=torch.math.ceil(aux/(m+i+1))
            if s1>s:
                s=s1
   
    return m,s,pA,nProd

def select_m_s_series_paterson_stockmeyer(A,tol):
    """
    Determines the polynomial order and the scaling parameter to compute the
    series of matrix A by means of Taylor approximation of order m<=16. The
    polynomial will be evaluated by means of Paterson-Stockmeyer algorithm. 
    
    Inputs:
      - A:     input matrix.
      - tol:   tolerance.
  
    Outputs: 
      - m:     approximation polynomial order used.
      - s:     scaling parameter.
      - pA:    list with the powers of A nedeed, such that pA[i] contains
               A^(i+1), for i=0,1,2,3,...,q-1.
      - nProd: number of matrix products required by the function.
    """
    pA=[]
    norm_pA=[]
    s=0
    nProd=0
 
    pA.append(A)
    norm_pA.append(torch.norm(pA[0],p=1,dim=-1).max().item())
   
    # Try with m=0
    m=0
    if norm_pA[0]==0:
        return m,s,pA,nProd
   
    M=[1, 2, 4, 6, 9, 12, 16, 20, 25, 30, 36, 42, 49, 56, 64]
    J=[1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8]
    K=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8]
   
    norm_pA[0]=torch.math.log2(norm_pA[0])
    tol=torch.math.log2(tol);

    L=[-2.584962500721156, -4.584962500721156, -4.584962500721156, -6.906890595608519, -9.491853096329674, -1.229920801838728e+01, -1.529920801838728e+01, -1.846913301982959e+01, -2.525049273335425e+01, -2.883545523407541e+01, -3.634324987427410e+01, -4.025014046988262e+01, -5.250752831257527e+01, -5.675545582601886e+01, -6.992913296232229e+01, -7.445269491837929e+01, -9.313684082917973e+01, -9.794419575123733e+01, -1.176632636523603e+02, -1.227076577717188e+02, -1.485517095104289e+02, -1.538371117292912e+02, -1.807946056249148e+02, -1.862864587212444e+02, -2.198805634055617e+02, -2.255810031237028e+02, -2.603433967253040e+02, -2.662260397746658e+02, -3.080619058741109e+02, -3.141279950645687e+02]
    fin=False
    i=0
    error=[0, 0]
    while i<len(M) and fin==False:
        # Try with m
        m=M[i]
        j=J[i]
        k=K[i]
        
        if j>k:
            pA.append(torch.matmul(pA[-1],A))
            norm_pA.append(torch.math.log2(torch.norm(pA[-1],p=1,dim=-1).max().item()))
            nProd+=1
           
        p=2*i    
        if m==1:
            error[0]=L[p] + 2*norm_pA[0]
            error[1]=L[p+1]+ 3*norm_pA[0]
        else:
            error[0]=L[p] + k*norm_pA[j-1] + norm_pA[0]
            error[1]=L[p+1]+ k*norm_pA[j-1] + norm_pA[1]
       
        if error[0]<=tol and error[1]<=tol:
            fin=True
        else:
            i+=1
         
    return m,s,pA,nProd


def select_m_s_series_sastre(A,tol=1e-8):
    """
    Determines the polynomial order and the scaling parameter to compute the
    exponential of matrix A by means of Taylor approximation of order m<=64. 
    The polynomial will be evaluated by means of Sastre formulas. 
    
    Inputs:
      - A:     input matrix.
      - tol:   tolerance.
  
    Outputs: 
      - m:     approximation polynomial order used.
      - s:     scaling parameter.
      - pA:    list with the powers of A nedeed, such that pA[i] contains
               A^(i+1), for i=0,1,2,3,...,q-1.
      - nProd: number of matrix products required by the function.
    """
    
    pA=[]
    norm_pA=[]
    s=0
    nProd=0
  
    pA.append(A)
    norm_pA.append(torch.norm(pA[0],p=1,dim=-1).max().item())
    
    # Try with m=0
    m=0
    if norm_pA[0]==0:
        return m,s,pA,nProd
      
    M=[1, 2, 4, 8, 15]
    J=[1, 2, 2, 2, 2]
    K=[1, 1, 2, 4, 8]

    norm_pA[0]=torch.math.log2(norm_pA[0])
    tol=torch.math.log2(tol);

    L=[-2.584962500721156, -4.584962500721156, -4.584962500721156, -6.906890595608519, -9.491853096329674, -1.229920801838728e+01, -2.179106111471695e+01, -2.525049273335425e+01, -4.429922743997088e+01, -5.250752831257527e+01]
    
    fin=False
    i=0
    error=[0, 0]
    while i<len(M) and fin==False:
        # Try with m
        m=M[i]
        j=J[i]
        k=K[i]
        
        if m==2:
            pA.append(torch.matmul(pA[-1],A))
            norm_pA.append(torch.math.log2(torch.norm(pA[-1],p=1,dim=-1).max().item()))
            nProd+=1
            
        p=2*i    
        if m==1:
            error[0]=L[p] + 2*norm_pA[0]
            error[1]=L[p+1]+ 3*norm_pA[0]
        else:
            error[0]=L[p] + k*norm_pA[j-1]
            error[1]=L[p+1] + k*norm_pA[j-1]
            if j*k==m:
                error[0]+= norm_pA[0]
                error[1]+= norm_pA[1]
            else: # m=15
                error[1]+=norm_pA[0]

        if error[0]<=tol and error[1]<=tol:
            fin=True
        else:
            i+=1
            
    if fin==False:
        M=[20, 25, 30, 36, 42, 49, 56, 64]
        J=[5, 5, 6, 6, 7, 7, 8, 8]
        K=[4, 5, 5, 6, 6, 7, 7, 8]
        
        pA.append(torch.matmul(pA[-1],A))
        norm_pA.append(torch.math.log2(torch.norm(pA[-1],p=1,dim=-1).max().item()))
        nProd+=1
        
        pA.append(torch.matmul(pA[-1],A))
        norm_pA.append(torch.math.log2(torch.norm(pA[-1],p=1,dim=-1).max().item()))
        nProd+=1
    
        L=[-6.992913296232229e+01, -7.445269491837929e+01, -9.313684082917973e+01, -9.794419575123733e+01, -1.176632636523603e+02, -1.227076577717188e+02, -1.485517095104289e+02, -1.538371117292912e+02, -1.807946056249148e+02, -1.862864587212444e+02, -2.198805634055617e+02, -2.255810031237028e+02, -2.603433967253040e+02, -2.662260397746658e+02, -3.080619058741109e+02, -3.141279950645687e+02]
        fin=False
        i=0
        error=[0, 0]
        while i<len(M) and fin==False:
            # Try with m
            m=M[i]
            j=J[i]
            k=K[i]
              
            if j>k:
                pA.append(torch.matmul(pA[-1],A))
                norm_pA.append(torch.math.log2(torch.norm(pA[-1],p=1,dim=-1).max().item()))
                nProd+=1
                 
            p=2*i    
            error[0]=L[p] + k*norm_pA[j-1] + norm_pA[0]
            error[1]=L[p+1]+ k*norm_pA[j-1] + norm_pA[1]
             
            if error[0]<=tol and error[1]<=tol:
                fin=True
            else:
                i+=1  
    return m,s,pA,nProd
       
def scaling(pA,s):    
    """
    Scaling of A powers.
    Inputs:
      - pA: list with the powers of A nedeed. pA[i] contains A^(i+1), 
            for i=0,1,2,3,...,q-1.
      - s:  scaling parameter.
    Outputs:
      - pA: list with the powers of A after scaling.
    """
    
    if s>0:
        q=len(pA)
        for k in range(q):
            pA[k]=pA[k]/(2**((k+1)*s))
   
def squaring(A,s):    
    """
    Squaring of A.
    Inputs:
      - A:     matrix resulting of the approximation polynomial evaluation.
      - s:     scaling parameter.
    
    Outputs:
      - A:     the exponential of matrix A after squaring technique.
      - nProd: number of matrix products required by the function.
    """
    
    nProd=0
    for i in range(s):
        A=torch.matmul(A,A)
        nProd+=1 
    return A,nProd

def coefs_expm(m,poleval=2):
    """
    Provides the approximation Taylor polynomial coefficients for the 
    matrix exponential function to be evaluated by means of the Paterson-
    Stockmeyer algorithm or the Sastre formulas.
    
    Inputs:
     - m:       approximation Taylor polynomial order.
     - poleval: formulas to apply in the polynomial evaluation: 
                Paterson-Stockmeyer (1) or Sastre (2).
    Outputs:
      - p:      vector of m+1 components with the coefficients of the 
                polynomial ordered in ascending (Paterson-Stockmeyer) or 
                descending (Sastre) powers.
    """

    if poleval==1:
        p=coefs_expm_paterson_stockmeyer(m)
    else:
        p=coefs_expm_sastre(m)
    return p

def coefs_expm_paterson_stockmeyer(m):  
    """
    Provides the approximation Taylor polynomial coefficients for the 
    matrix exponential function to be evaluated by means of the Paterson-
    Stockmeyer algorithm.
    Inputs:
      - m: approximation Taylor polynomial order.
    Outputs:
      - p: vector of m+1 components with the coefficients of the polynomial 
           ordered from degree zero to degree m, i.e. in ascending powers.
    """
    
    p=[1.0000000000000000e+00, 1.0000000000000000e+00, 5.0000000000000000e-01, 1.6666666666666666e-01, 4.1666666666666664e-02, 8.3333333333333332e-03, 1.3888888888888889e-03, 1.9841269841269841e-04, 2.4801587301587302e-05, 2.7557319223985893e-06, 2.7557319223985888e-07, 2.5052108385441720e-08, 2.0876756987868100e-09, 1.6059043836821613e-10, 1.1470745597729725e-11, 7.6471637318198164e-13, 4.7794773323873853e-14, 2.8114572543455206e-15, 1.5619206968586225e-16, 8.2206352466243295e-18, 4.1103176233121648e-19, 1.9572941063391263e-20, 8.8967913924505741e-22, 3.8681701706306835e-23, 1.6117375710961184e-24, 6.4469502843844736e-26, 2.4795962632247972e-27, 9.1836898637955460e-29, 3.2798892370698385e-30, 1.1309962886447718e-31, 3.7699876288159061e-33, 1.2161250415535181e-34, 3.8003907548547441e-36, 1.1516335620771951e-37, 3.3871575355211618e-39, 9.6775929586318907e-41, 2.6882202662866367e-42, 7.2654601791530724e-44, 1.9119632050402823e-45, 4.9024697565135435e-47, 1.2256174391283860e-48, 2.9893108271424051e-50, 7.1174067312914405e-52, 1.6552108677421951e-53, 3.7618428812322623e-55, 8.3596508471828045e-57, 1.8173154015614793e-58, 3.8666285139605940e-60, 8.0554760707512382e-62, 1.6439747083165791e-63, 3.2879494166331584e-65, 6.4469596404571737e-67, 1.2397999308571486e-68, 2.3392451525606576e-70, 4.3319354677049218e-72, 7.8762463049180392e-74, 1.4064725544496498e-75, 2.4674957095607889e-77, 4.2543029475186016e-79, 7.2106829618959346e-81, 1.2017804936493225e-82, 1.9701319568021682e-84, 3.1776321883905938e-86, 5.0438606164930067e-88, 7.8810322132703230e-90] 
    p=p[:m+1]
    return p

def coefs_expm_sastre(m):    
    """
    Provides the approximation Taylor polynomial coefficients for the 
    matrix exponential function to be evaluated by means of the Sastre formulas.
    Inputs:
      - m: approximation Taylor polynomial order.
    Outputs:
      - c: vector of m+1 components with the coefficients of the polynomial 
           ordered in descending powers.
    """
    
    if m==0:
        c=[1]
    elif m==1:
        c=[1, 1]
    elif m==2:
        c=[1, 1, 0.5]
    elif m==4:
        c=[1, 1, 0.5, 0.3333333333333333, 0.25]
    elif m==8:
        c=[4.980119205559973e-03, 1.992047682223989e-02, 7.665265321119147e-02, 8.765009801785554e-01, 1.225521150112075e-01, 2.974307204847627, 0.5, 1, 1]
    elif m==15:
        c=[4.018761610201036e-04, 2.945531440279683e-03, -8.709066576837676e-03, 4.017568440673568e-01, 3.230762888122312e-02, 5.768988513026145, 2.338576034271299e-02, 2.381070373870987e-01, 2.224209172496374, -5.792361707073261, -4.130276365929783e-02, 1.040801735231354e+01, -6.331712455883370e+01, 3.484665863364574e-01, 1, 1]
    return c

def coefs_series(m,poleval=2):
    """
    Provides the polynomial coefficients for the matrix series 
    sum_{k=0}^{infty}\frac{x^{k}}{(k+1)!} to be evaluated by means of the 
    Paterson-Stockmeyer algorithm of the Sastre formulas.
    
    Inputs:
     - m:       approximation Taylor polynomial order.
     - poleval: formulas to apply in the polynomial evaluation: 
                Paterson-Stockmeyer (1) or Sastre (2).
    Outputs:
      - p:      vector of m+1 components with the coefficients of the 
                polynomial ordered in ascending (Paterson-Stockmeyer) or 
                descending (Sastre) powers.
    """

    if poleval==1:
        p=coefs_series_paterson_stockmeyer(m)
    elif poleval==2:
        p=coefs_series_sastre(m)
    return p

def coefs_series_paterson_stockmeyer(m):  
    """
    Provides the polynomial coefficients for the matrix series 
    sum_{k=0}^{infty}\frac{x^{k}}{(k+1)!} to be evaluated by means of the 
    Paterson-Stockmeyer formulas.
    Inputs:
      - m: approximation polynomial order.
    Outputs:
      - p: vector of m+1 components with the coefficients of the polynomial 
           ordered from degree zero to degree m, i.e. in ascending powers.
    """    
    p=[1.0000000000000000e+00, 5.0000000000000000e-01, 1.6666666666666666e-01, 4.1666666666666664e-02, 8.3333333333333332e-03, 1.3888888888888889e-03, 1.9841269841269841e-04, 2.4801587301587302e-05, 2.7557319223985893e-06, 2.7557319223985888e-07, 2.5052108385441720e-08, 2.0876756987868100e-09, 1.6059043836821613e-10, 1.1470745597729725e-11, 7.6471637318198164e-13, 4.7794773323873853e-14, 2.8114572543455206e-15, 1.5619206968586225e-16, 8.2206352466243295e-18, 4.1103176233121648e-19, 1.9572941063391263e-20, 8.8967913924505741e-22, 3.8681701706306835e-23, 1.6117375710961184e-24, 6.4469502843844736e-26, 2.4795962632247972e-27, 9.1836898637955460e-29, 3.2798892370698385e-30, 1.1309962886447718e-31, 3.7699876288159061e-33, 1.2161250415535181e-34, 3.8003907548547441e-36, 1.1516335620771951e-37, 3.3871575355211618e-39, 9.6775929586318907e-41, 2.6882202662866367e-42, 7.2654601791530724e-44, 1.9119632050402823e-45, 4.9024697565135435e-47, 1.2256174391283860e-48, 2.9893108271424051e-50, 7.1174067312914405e-52, 1.6552108677421951e-53, 3.7618428812322623e-55, 8.3596508471828045e-57, 1.8173154015614793e-58, 3.8666285139605940e-60, 8.0554760707512382e-62, 1.6439747083165791e-63, 3.2879494166331584e-65, 6.4469596404571737e-67, 1.2397999308571486e-68, 2.3392451525606576e-70, 4.3319354677049218e-72, 7.8762463049180392e-74, 1.4064725544496498e-75, 2.4674957095607889e-77, 4.2543029475186016e-79, 7.2106829618959346e-81, 1.2017804936493225e-82, 1.9701319568021682e-84, 3.1776321883905938e-86, 5.0438606164930067e-88, 7.8810322132703230e-90] 
    p=p[:m+1]
    return p

def coefs_series_sastre(m):
    """
    Provides the polynomial coefficients for the matrix series 
    sum_{k=0}^{infty}\frac{x^{k}}{(k+1)!} to be evaluated by means of the 
    Sastre formulas.
    Inputs:
      - m: approximation polynomial order.
    Outputs:
      - c: vector of m+1 components with the coefficients of the polynomial 
           ordered in descending powers.
    """
    
    if m==0:
        c=[1]
    elif m==1:
        c=[1, 0.5]
    elif m==2:
        c=[1, 0.5, 0.16666666666666666]
    elif m==4:
        c=[1, 0.5, 0.16666666666666666, 0.25, 0.2]
    elif m==8:
        c=[1.660039735186658e-03, 7.470178808339960e-03, 2.709974334395560e-02, 4.500782732024826e-01, 5.880731295195394e-02, 2.034592904871945, 1.666666666666667e-01, 5.000000000000000e-01, 1]
    elif m==15:
        c=[1.999637069327334e-04, 1.494400064180480e-03, -4.364865621286775e-03, 2.294130909981213e-01, 1.868111448270906e-02, 3.650752775284885, 2.113847295209054e-02, 7.709174196270302e-02, 1.005952341579399, 3.134311599789341, -6.535520701550694e-02, 5.711871755550578, -1.928484398466562e+01, 1.116706435878065e-01, 5.000000000000000e-01, 1]
    return c

def polyvalm(p,pA,poleval=2):
    """
    Evaluates the polynomial E = p[0]*I + p[1]*A + p[2]*A^2 + ...+ p[m]*A^m 
    efficiently by means of the Paterson-Stockmeyer algorithm or the Sastre
    formulas.    
    
    Inputs:
      - p:      vector of length m+1 with the polynomial coefficients in 
                ascending (Paterson-Stockmeyer) or descending powers (Sastre).
      - pA:     list with the powers of A nedeed, such that pA[i] contains
                A^(i+1), for i=0,1,2,3,...,q-1
     - poleval: formulas to apply in the polynomial evaluation: 
                Paterson-Stockmeyer (1) or Sastre (2).               
    Outputs: 
      - E:      polynomial evaluation result.
      - nProd:  number of matrix products required by the function.  
    """    

    if poleval==1:
        E,nProd=polyvalm_paterson_stockmeyer(p,pA)
    else:
        E,nProd=polyvalm_sastre(p,pA)
    return E,nProd

def polyvalm_paterson_stockmeyer(p,pA):
    """
    Evaluates the polynomial E = p[0]*I + p[1]*A + p[2]*A^2 + ...+ p[m]*A^m 
    efficiently by means of the Paterson-Stockmeyer algorithm.    
    
    Inputs:
      - p:     vector of length m+1 with the polynomial coefficients in 
               ascending powers.
      - pA:    list with the powers of A nedeed, such that pA[i] contains
               A^(i+1), for i=0,1,2,3,...,q-1.    
    Outputs: 
      - E:     polynomial evaluation result.
      - nProd: number of matrix products required by the function.  
    """
    
    n=pA[0].size(-1)
    I=torch.eye(pA[0].size(-1),device=pA[0].device)
    m=len(p)-1
    nProd=0
    c=m 
    q=len(pA)
    k=torch.math.ceil(m/q)
    mIdeal=q*k
    q=q-1
    if m==0:
        E=p[0]*I
    else:
        E=torch.zeros((n,n),device=pA[0].device) 
    for j in range(k,0,-1):
        if j==k:
            inic=q-mIdeal+m
        else:
            inic=q-1
        for i in range(inic,-1,-1):
            E=E+p[c]*pA[i]
            c=c-1
        E=E+p[c]*I
        c=c-1;
        if j!=1:
            E=torch.matmul(E,pA[q])
            nProd+=1
    return E,nProd

def polyvalm_sastre(c,pA):    
    """
    Evaluates the polynomial efficiently by means of the Sastre formulas.    
    
    Inputs:
      - c:     vector of m+1 componentes with the coefficients in descending 
               powers.
      - pA:    list with the powers of A nedeed, such that pA[i] contains
               A^(i+1), for i=0,1,2,3,...,q-1.     
    Outputs: 
      - E:     polynomial evaluation result.
      - nProd: number of matrix products required by the function.  
    """ 
    m=len(c)-1
    E=torch.eye(pA[0].size(-1),device=pA[0].device)
    if m==0:
        nProd=0
    elif m==1:
        E=c[1]*pA[0]+E
        nProd=0
    elif m==2:
        E=c[2]*pA[1]+c[1]*pA[0]+E
        nProd=0
    elif m==4:
        E=torch.matmul(((pA[1]*c[4]+pA[0])*c[3]+E),pA[1]*c[2])+pA[0]*c[1]+E
        nProd=1
    elif m==8:
        y0s=torch.matmul(pA[1],(c[0]*pA[1]+c[1]*pA[0]))
        E=torch.matmul(y0s+c[2]*pA[1]+c[3]*pA[0],y0s+c[4]*pA[1])+c[5]*y0s+c[6]*pA[1]+c[7]*pA[0]+c[8]*E
        nProd=2
    elif m==15:
        y0s=torch.matmul(pA[1],(c[0]*pA[1]+c[1]*pA[0]))
        y1s=torch.matmul(y0s+c[2]*pA[1]+c[3]*pA[0],y0s+c[4]*pA[1])+c[5]*y0s+c[6]*pA[1]
        E=torch.matmul(y1s+c[7]*pA[1]+c[8]*pA[0],y1s+c[9]*y0s+c[10]*pA[0])+c[11]*y1s+c[12]*y0s+c[13]*pA[1]+c[14]*pA[0]+c[15]*E
        nProd=3        
    return E,nProd
                        