import camb
import numpy as np
from numpy import linalg as LA
from scipy.linalg import sqrtm
import cmath

def Gencl(r = 0.05, raw_cl = True, tensorBB_only = False):
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
    
    if tensorBB_only:
        totCL=powers['tensor'] ## TT EE BB TE
        
        return totCL.T[2]
    
    else: 
        
        totCL=powers['total'] ## TT EE BB TE

        return totCL.T


def vecp(mat):
    '''
    This returns the unique elements of a symmetric matrix 
    '''

    dim = mat.shape[0]
    
    vec = np.zeros(int(dim*(dim+1)/2), dtype = np.complex)
    counter = 0
    for iDiag in range(0,dim):
        vec[counter:counter+dim-iDiag] = np.diag(mat,iDiag)
        
        counter = counter + dim - iDiag

    return vec



def calc_vecp_sub(cl_hat, fl_hat, cl_f, cl_th, Nf, Nmodes = None):
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
    
    Xall = np.ones(lbin*nf_ind, dtype = np.complex)    
    for l in range(lbin):
    
        cl_f_12 = sqrtm(cl_f[l])
        cl_inv = LA.pinv(cl_th[l])
        cl_inv_12= sqrtm(cl_inv)
       
        
        res = np.dot(cl_inv_12, np.dot(cl_hat[l], cl_inv_12));
        [d, u] = LA.eigh(res); 
        
        res_fore = np.dot(cl_inv_12, np.dot(cl_hat[l] - fl_hat[l], cl_inv_12));
        D = LA.eigvals(res_fore);
               
        gd = np.ones(Nf, dtype = np.complex);
        for i in range(Nf):

            gd[i] = np.sign(d[i] - 1) * cmath.sqrt(2 * (D[i] - np.log(d[i]) - 1))

        gd = np.diag(gd);
  
        X = np.dot(np.transpose(u), cl_f_12)
        X = np.dot(gd, X)
        X = np.dot(u, X)
        X = np.dot(cl_f_12, X)
       
        Xall[l*nf_ind:(l+1)*nf_ind] = vecp(X)

    return (Xall)

def testL_sub(cl_hat,fl_hat, cl_f, cl_th, Nf, M, Nmodes = None, sbin = None):
    
    '''
    Input
    ------------------------------
    
    cl_hat, lbin*Nf*Nf
    cl_f, lbin*Nf*Nf
    cl_th, Nf, M, Nmodes = None, sbin = None
    
    M: covariance of all X arrays, reordered to be a line for each Xall...
    '''
    
    Xa = (calc_vecp_sub(cl_hat, fl_hat, cl_f,cl_th, Nf = Nf))
    
    M_inv = LA.inv(M);
    
    if sbin is not None:
        
        nf_ind = int(Nf*(Nf+1)/2)
        start = sbin*nf_ind
        
        Xa = Xa[start:]; 
        M_inv = M_inv[start:,start:]
        
    Xa = np.matrix(Xa);
    logL = -0.5*Xa*M_inv*np.conjugate(np.transpose(Xa))  ## 1*1 matrix, use logL[0,0] to extract number
    
    if np.isnan(logL[0,0]):
        logL[0,0] = -1e30
        
    return (logL[0,0])
