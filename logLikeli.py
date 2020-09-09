'''
Class for Likelihood analysis. Mainly from methods.ipynb
'''

### Own functions
import bandpowers
from oas import oas_est

### scientific packages
import pymaster as nmt
import healpy as hp
import numpy as np
import dynesty

from Fg_template import sync_ps as sync_ps_v0
from Fg_template import dust_ps as dust_ps_v0
from Fg_template import corre_fore_simple

from scipy.linalg import sqrtm

from numpy import linalg as LA
import utils
from utils import Marray_EEfirst as Marray
from utils import Minv as Minv
from utils import testL as testL
from utils import simple_likelihood as simple_likelihood

# from multiprocessing import Pool, cpu_count
from pathos.multiprocessing import ProcessingPool as Pool
from pathos.helpers import cpu_count


class logLike(object):
    
    def __init__(self, Nf, nside, bin_w, lmax,combine_BB = True, add_lensing = True):
        
        '''
        dynesty
        '''
        
        self.Nf = Nf; self.nf_ind = int(Nf*(Nf+1)/2); self.add_lensing = add_lensing
        
        self.lmax = lmax; self.b = nmt.NmtBin(nside, nlb=bin_w, lmax = lmax, is_Dell = True)
        
        self.lbin = len(self.b.get_effective_ells()); 
        
        if combine_BB: 
            self.bb_05 = utils.Gencl(r = 0.05, raw_cl = True, tensorBB_only=True) ##only tensor BB; ## 
            self.lensingB = utils.Gencl(r = 0, raw_cl = True) ## only lensing BB; TT EE BB TE
    
    
    def M(self, cl_f_all, SamNum, method = 'cov'):
        
        marry = Marray(cl_f_all, self.nf_ind, 3, self.lbin, SamNum)
        
        if method == 'cov':
            self.Cov = np.cov(marry[self.lbin*self.nf_ind*2:]); ### np.cov requires (Obs, SamNum)
        else:
            self.Cov = oas_est(np.transpose(marry[self.lbin*self.nf_ind*2:])); ### oas requires(SamNum, Obs)
            
        self.Cov_inv = LA.inv(self.Cov);
        
        self.cl_f = np.mean(cl_f_all, axis = 0) ## 3,lbin, Nf, Nf
                
    
    def run(self, cl_hat, cl_th, likelihood = 'HL', sbin = None, ebin = None):
        
        '''
        cl_f and m_inv should be given or not given at the same time.
        sbin : start bin;
        ebin : end bin;
        
        '''
         
        # cl_hat,cl_f, cl_th, Nf, M,
        if likelihood == 'HL':
            logL = testL(cl_hat = cl_hat, cl_f = self.cl_f[2], cl_th = cl_th, Nf = self.Nf, M_inv = self.Cov_inv, sbin = sbin, ebin = ebin)
            
        elif likelihood == 'Gauss':
            logL = simple_likelihood(cl_hat = cl_hat, cl_th = cl_th, Nf = self.Nf, M_inv = self.Cov_inv, sbin = sbin, ebin = ebin)
            
        else:
            print('Only HL or Gauss likelihood are provided!')
        
        return logL
    
    
    def combine_ps(self, r):
    
        '''
        Combine the tensor bb and lensing bb power spectrum of CMB.
        ------------------------
        lensing: bool; If False, doesn't include lensing part in the tensor BB mode. 

        '''
    
        bb_tensor = self.bb_05[0:self.lmax+1]*r/0.05;
        lensing = self.add_lensing

        if lensing:

            bb_camb = bb_tensor + self.lensingB[2][0:self.lmax+1] ## tensor BB + lensing BB 
            
        else: 
            bb_camb = bb_tensor;
                
        cl_th_i = self.b.bin_cell(bb_camb[0:self.lmax+1])  ## theoretical bandpower, use simple window function
        cl_th_test = np.ones((self.lbin, self.Nf, self.Nf));  ## BB cross power spectra
        for ell in range(self.lbin):
            cl_th_test[ell] *= cl_th_i[ell]

        return cl_th_test
    
    def prior(self, cube):
    
        r = cube[0]*0.2
        beta_s = cube[1]*4 - 5## from -4 to -1
    #     beta_d = cube[2]*0.4 + 1.4 ## from 1.4 to 1.8
        beta_d = cube[2]*2 + 1 
        epsilon = cube[3];

        return [r, beta_s, beta_d, epsilon]

# Mean = np.zeros((Nsim,npara)); Samples = np.zeros((Nsim, npara)); Weights = np.zeros(Nsim); Results = []
      
    def log_likelihood(self, cube):
        
        r_i = cube[0];
        beta_s = cube[1];
        beta_d = cube[2];
        epsilon = cube[3];

        cl_th_test = self.combine_ps(r_i)  ## combine the tensor bb and lensing bb
        
        fl_hat = sync_ps_v0(self.A_s_RJ, beta_s, self.lbin) + dust_ps_v0(self.A_d_RJ, beta_d, self.lbin) + corre_fore_simple(epsilon,self.A_d_RJ, self.A_s_RJ, beta_s, beta_d, self.lbin)
        
        # add Noise bias N_l to expectation values.########################## Noise level
        C_l = cl_th_test + self.nl + fl_hat
        
        logL =self.run(cl_hat=self.signal , cl_th = C_l, likelihood='HL',sbin=self.sbin, ebin = self.ebin); 
#         print(logL)
#         print(np.linalg.det(LogL.Cov))
        return np.real(logL)

    def dynesty_run(self, cl_hat, nl_mean, A_s_RJ, A_d_RJ, sbin, ebin):
        
        '''
        cl_hat : lbin, nf, nf; only BB mode
        nl_mean : lbin, nf, nf
        '''
        npara = 4;
        
        self.signal = cl_hat; self.nl = nl_mean; self.A_s_RJ = A_s_RJ; self.A_d_RJ = A_d_RJ; self.sbin = sbin; self.ebin = ebin;
        
#         with Pool(cpu_count()-1) as executor:
        
        sampler = dynesty.NestedSampler(self.log_likelihood, self.prior, npara, nlive=400)#, pool=executor, queue_size=cpu_count(), bootstrap = 0)
        sampler.run_nested()
        results = sampler.results

        return results

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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
