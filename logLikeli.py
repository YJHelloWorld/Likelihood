'''
Class for Likelihood analysis. Mainly from methods.ipynb
'''

### Own functions
import bandpowers

### scientific packages
import pymaster as nmt
import healpy as hp
import numpy as np

from scipy.linalg import sqrtm

from numpy import linalg as LA

from utils import Marray_EEfirst as Marray
from utils import Minv as Minv
from utils import evaluateLikelihood as evaluateL

class logLike(object):
    
    def __init__(self, Nf, lbin):
        
        self.Nf = Nf; self.nf_ind = int(Nf*(Nf+1)/2);
        
        self.lbin = lbin; 
    
    def Marray():
        pass
    
    def M_inv(self, cl_f_all, SamNum, m_inv):
        
        marry = Marray(cl_f_all, self.nf_ind, 3, self.lbin, SamNum) ## (Nmode*lbin*nf_ind, SamNum)
        
        if m_inv is None:
        
            self.cl_f = np.mean(cl_f_all, axis = 0);

        cov_mat_BB = np.cov(marry[self.lbin*self.nf_ind*2:]) ##(select BB mode)
        
        m_inv = Minv(cov_mat_BB, self.lbin, self.nf_ind);
            
        return m_inv
    
    def run(self, cl_th, cl_hat, sbin, cl_f = None, m_inv = None, cl_f_all = None, SamNum = None):
        '''
        cl_f and m_inv should be given or not given at the same time, 
        '''
        
        
        if cl_f is not None:
            self.cl_f = cl_f
            
        if m_inv is None:
            m_inv = self.M_inv(cl_f_all, SamNum, m_inv)
        
        logL = evaluateL(cl_th, cl_hat[2], self.cl_f[2], m_inv, sbin)
        
        return logL
