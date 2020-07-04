### partially from BICEP team
### def bandpower window function 

import camb
import numpy as np
from numpy import linalg as LA
from scipy.linalg import sqrtm

#####################################################################  
# Utility functions used to calculate the likelihood
# for a given l bin.

def Gencl(r = 0.05, raw_cl = True):
    '''
    Generate the theoretical power spectra using camb
    '''
    
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.26, ombh2=0.022, omch2=0.1199, mnu=0.06, omk=0, tau=0.078)
    pars.InitPower.set_params(As=2.19856*1e-9, ns=0.9652, r = r)
    pars.set_for_lmax(3000, lens_potential_accuracy=1)
    pars.WantTensors = True
    
    results = camb.get_results(pars)
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=raw_cl)
    
    totCL=powers['total'] ## TT EE BB TE
#     ell = np.arange(len(totCL.T[0]))
    return totCL.T


def l2(ell):
    '''
    get the l^2/np.pi
    '''
    
    return ell*(ell+1)/2/np.pi

def calc_vecp(l,C_l_hat,C_fl, C_l):

    C_fl_12 = sqrtm(C_fl[l])
    C_l_inv = LA.pinv(C_l[l])
    C_l_inv_12= sqrtm(C_l_inv)
    # the order is inverted compared to matlab hamimeche_lewis_likelihood.m line 19
    
    # line 20 of hamimeche_lewis_likelihood.m
    res = np.dot(C_l_inv_12, np.dot(C_l_hat[l], C_l_inv_12))

    [d, u] = LA.eigh(res)
    d = np.diag(d)  # noticed that python returns the eigenvalues as a vector, not a matrix
    #np. dot( u, np.dot( np.diag(d), LA.inv(u))) should be equals to res
    # real symmetric matrices are diagnalized by orthogonal matrices (M^t M = 1)  

    # this makes a diagonal matrix by applying g(x) to the eigenvalues, equation 10 in Barkats et al
    gd = np.sign(np.diag(d) - 1) * np.sqrt(2 * (np.diag(d) - np.log(np.diag(d)) - 1))
    gd = np.diag(gd);
    # Argument of vecp in equation 8; multiplying from right to left     
    X = np.dot(np.transpose(u), C_fl_12)
    X = np.dot(gd, X)
    X = np.dot(u, X)
    X = np.dot(C_fl_12, X)
    # This is the vector of equation 7  
    X = vecp(X)

    return (X)


#def g(x):
#    #  sign(x-1) \sqrt{ 2(x-ln(x) -1 }  
#    return np.sign(x-1) * np.sqrt( 2* (x - np.log(x) -1) )

def vecp(mat):
    '''
    This returns the unique elements of a symmetric matrix 
    '''

    dim = mat.shape[0]
    
    vec = np.zeros(int(dim*(dim+1)/2))
    counter = 0
    for iDiag in range(0,dim):
        vec[counter:counter+dim-iDiag] = np.diag(mat,iDiag)
        
        counter = counter + dim - iDiag

    return vec


def Marray_EEfirst(cl_f_all, nf_ind, Nmode,lbin, SamNum):
    '''
    Get the re-arranged array for Mcc'.
    -------------------------------------------
    Input
    
    cl_f_all, (SamNum, Nmode, lbin, nf, nf)
    
    -------------------------------------------
    Output
    
    marray, (Nmode*lbin*nf_ind , SamNum) for EE first.
    '''
    marray = np.zeros(((Nmode*lbin*nf_ind), SamNum)) # mode(EE, EB, BB), l-bin, nf independent corr between frequencies 
    for n in range(SamNum):

        for mode in range(Nmode):    

            cl_flat = np.zeros((lbin, nf_ind)) ## collect independent corr for each l-bin

            for ell in range(lbin):
                cl_flat[ell] = vecp(cl_f_all[n][mode][ell])# - nl_mean[2][ell] ) ##########!!!!!!!!!!!!!!!!!! need to subtract noise?? 06.27

            marray[mode*lbin*nf_ind:(mode+1)*lbin*nf_ind,n] = cl_flat.flatten()
            
    return marray

def Minv(M, lbin, nf_ind):
    '''
    Get re-organized M_inv for the calculation of Likelihood.
    M: (lbin*nf_ind, lbin*nf_ind), just BB mode for mow. 2020/06/29
    
    Output:
    M_inv, (lbin, lbin, nf_ind, nf_ind)
    '''
    cov_mat_inv = LA.inv(M)
    _M_inv = np.ones((lbin, lbin, nf_ind, nf_ind))

    for l in range(lbin):

        for lp in range(lbin):

            _M_inv[l,lp, :, :] = cov_mat_inv[l*nf_ind:(l+1)*nf_ind, lp*nf_ind:(lp+1)*nf_ind]
    
    return _M_inv


def evaluateLikelihood(C_l,C_l_hat,C_fl,M_inv, sbin = 0):
    '''
     To evaluate the likelihood itself.
     
     ------------------------------------------
     Input
     
     sbin: start-bin number
    '''
    logL = 0; lbin = C_l.shape[0]
    # Calculate X vector (Eq 8) for each l, lp
    for l in range(sbin, lbin):
        X = calc_vecp(l,C_l_hat,C_fl,C_l)
        
        for lp in range(sbin, lbin):
            Xp = calc_vecp(lp,C_l_hat,C_fl,C_l)
            M_inv_pp = M_inv[l,lp,:,:]
            # calculate loglikelihood (Eq 7)
            thislogL = (-0.5)*np.dot(X,np.dot(M_inv_pp,Xp))
            logL = logL + thislogL

    if np.isnan(logL):
        logL = -1e20

    logL = np.real(logL)
    return logL


def calc_vecp_test(cl_hat,cl_f, cl_th, Nf, Nmodes = None):
    '''
    Input
    ---------------------------
    Cl : (lbin, Nf, Nf);
    Nf : number of frequency channels;
    Nmodes: consider different modes like EE EB and BB; Only BB for now. 2020.07.04
    
    Output
    ---------------------------
    Xall : rearanged to one line,  as like lbin first, then nf_ind 
    '''
    lbin = len(cl_hat); nf_ind = int(Nf*(Nf+1)/2)
    
    Xall = np.ones(lbin*nf_ind)    
    for l in range(lbin):
    
        cl_f_12 = sqrtm(cl_f[l])
        cl_inv = LA.pinv(cl_th[l])
        cl_inv_12= sqrtm(cl_inv)
        # the order is inverted compared to matlab hamimeche_lewis_likelihood.m line 19

        # line 20 of hamimeche_lewis_likelihood.m
        res = np.dot(cl_inv_12, np.dot(cl_hat[l], cl_inv_12))

        [d, u] = LA.eigh(res)
        d = np.diag(d)  # noticed that python returns the eigenvalues as a vector, not a matrix
        #np. dot( u, np.dot( np.diag(d), LA.inv(u))) should be equals to res
        # real symmetric matrices are diagnalized by orthogonal matrices (M^t M = 1)  

        # this makes a diagonal matrix by applying g(x) to the eigenvalues, equation 10 in Barkats et al
        gd = np.sign(np.diag(d) - 1) * np.sqrt(2 * (np.diag(d) - np.log(np.diag(d)) - 1))
        gd = np.diag(gd);
        # Argument of vecp in equation 8; multiplying from right to left     
        X = np.dot(np.transpose(u), cl_f_12)
        X = np.dot(gd, X)
        X = np.dot(u, X)
        X = np.dot(cl_f_12, X)
        # This is the vector of equation 7  
        Xall[l*nf_ind:(l+1)*nf_ind] = vecp(X)

    return (Xall)

def testL(cl_hat,cl_f, cl_th, Nf, M, Nmodes = None, sbin = None):
    
    '''
    Input
    ------------------------------
    
    cl_hat,cl_f, cl_th, Nf, M, Nmodes = None, sbin = None
    
    M: covariance of all X arrays, reordered to be a line for each Xall...
    '''
    
    Xa = (calc_vecp_test(cl_hat,  cl_f,cl_th, Nf = Nf))
    
    M_inv = LA.inv(M);
    
    if sbin is not None:
        
        nf_ind = int(Nf*(Nf+1)/2)
        start = sbin*nf_ind
        
        Xa = Xa[start:]; 
        M_inv = M_inv[start:,start:]
        
    Xa = np.matrix(Xa);
    logL = -0.5*Xa*M_inv*np.transpose(Xa)  ## 1*1 matrix, use logL[0,0] to extract number 
    return (logL[0,0])