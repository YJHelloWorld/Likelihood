'''
Class for Likelihood analysis. Mainly from methods.ipynb
'''

### Own functions
import bandpowers

### scientific packages
import pymaster as nmt
import healpy as hp
import numpy as np
import dynesty

from scipy.linalg import sqrtm

from numpy import linalg as LA

from utils import Marray_EEfirst as Marray
from utils import Minv as Minv
from utils import testL as testL

class logLike(object):
    
    def __init__(self, Nf, lbin):
        
        self.Nf = Nf; self.nf_ind = int(Nf*(Nf+1)/2);
        
        self.lbin = lbin; 
    
    
    def M(self, cl_f_all, SamNum):
        
        marry = Marray(cl_f_all, self.nf_ind, 3, self.lbin, SamNum)
       
        self.Cov = np.cov(marry[self.lbin*self.nf_ind*2:]);
        self.Cov_inv = LA.inv(self.Cov);
        
        self.cl_f = np.mean(cl_f_all, axis = 0) ## 3,lbin, Nf, Nf
                
    
    def run(self, cl_hat, cl_th, sbin = None, ebin = None):
        
        '''
        cl_f and m_inv should be given or not given at the same time.
        sbin : start bin;
        ebin : end bin;
        
        '''
         
        # cl_hat,cl_f, cl_th, Nf, M,
        
        logL = testL(cl_hat = cl_hat, cl_f = self.cl_f[2], cl_th = cl_th, Nf = self.Nf, M_inv = self.Cov_inv, sbin = sbin, ebin = ebin)
        
        return logL
    
# class MaxL(object):
    
#     def __init__(self):

#     add_dust = True

#     if add_dust:

#         cl_hat_fore_ali = cl_hat_all_ali[10][2] + sync_dl_RJ + dust_dl_RJ

#     else: 

#         cl_hat_fore_ali = cl_hat_all_ali[10][2] + sync_dl_RJ

#     def log_likelihood(cube, dust = add_dust):
#         r_i = cube[0];
#         beta_s = cube[1];

#         if dust:
#             beta_d = cube[2]

#         bb_tensor = bb_05[0:nmtlmax+1]*r_i/0.05 

#         bb_camb = bb_tensor + lensingB[2][0:nmtlmax+1] ## tensor BB + lensing BB 

#         cl_th_i = b.bin_cell(bb_camb[0:nmtlmax+1])  ## theoretical bandpower 
#         cl_th_test = np.ones((lbin, Nf, Nf));  ## BB cross power spectra
#         for ell in range(lbin):
#             cl_th_test[ell] *= cl_th_i[ell]

#         # add Noise bias N_l to expectation values.
#         C_l = cl_th_test + nl_mean[2] 

#         if dust:
#             logL = testL((cl_hat_fore_ali - sync_ps(A_s_RJ, beta_s, 10) - dust_ps(A_d_RJ, beta_d, 10)) , cl_f_ali[2], C_l, Nf = Nf, M = cov_mat_BB_ali, sbin = 1)

#         else:
#             logL = testL((cl_hat_fore_ali - sync_ps(A_s_RJ, beta_s, 10)), cl_f_ali[2], C_l, Nf = Nf, M = cov_mat_BB_ali, sbin = 0)

#         return np.real(logL)

#     def prior(cube, dust = add_dust):
#         r = cube[0]*0.1;
#         beta_s = cube[1]*2 - 4 ## from -4 to -2

#         if dust:
#             beta_d = cube[2] + 1 ## from 1 to 2

#             return [r, beta_s, beta_d]

#         else:
#             return [r, beta_s]
    
    
#     def run(self):
        
#         sampler = dynesty.NestedSampler(self.log_likelihood, self.prior, 3, nlive=400)
#         sampler.run_nested()
#         results = sampler.results
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
