from logLikeli import logLike
import dynesty
from dynesty import plotting as dyplot
import utils
from utils import testL as testL
from Fg_template import sync_ps, dust_ps

import pymaster as nmt
import numpy as np
import matplotlib.pyplot as plt 

from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor

## load the data 
path = '/fnx/jianyao/Likelihood_data/Bandpowers/0710_2fre_Ali_real_fore_and_noise'

cl_f_all_ali = np.load('%s/cl_f_all.npy'%path); cl_f_ali = np.mean(cl_f_all_ali, axis = 0)

cl_hat_all_ali = np.load('/fnx/jianyao/Likelihood_data/Bandpowers/0710_2fre_Ali_real_fore_and_noise/cl_hat_r0_all.npy')

nl_all_ali = np.load('%s/nl_all.npy'%path)
nl_mean_ali = np.mean(nl_all_ali, axis = 0) ## EE EB BB

lensingB = utils.Gencl(r = 0, raw_cl = True) ## only lensing BB; TT EE BB TE
bb_05 = utils.Gencl(r = 0.05, raw_cl = True, tensorBB_only=True) ##only tensor BB; ## 

A_d_RJ = np.load('/fnx/jianyao/Likelihood_data/Bandpowers/0629_2fre_Ali/dust_template_353_BB_RJ.npy')
A_s_RJ = np.load('/fnx/jianyao/Likelihood_data/Bandpowers/0629_2fre_Ali/sync_template_30_BB_RJ.npy')

#define Hyper-parameters
nside = 128;
nmtlmax = 201;
lmin = 2;
lmax = 1521;
SamNum = 500;
Nf = 2;

b = nmt.NmtBin(nside, nlb=20, is_Dell=True, lmax=nmtlmax)
leff = b.get_effective_ells()
lbin = len(leff)

#Method

LogL = logLike(2,10) 

 ## calculate logL.cl_f; logL.Cov  

LogL.M(cl_f_all = cl_f_all_ali, SamNum=SamNum)


npara = 2; 

def prior(cube):
    
    r = cube[0]*0.2 - 0.1
#     beta_s = cube[1]*2 - 4## from -4 to -2
#     beta_d = cube[1]*0.4 + 1.4 ## from 1.4 to 1.8
    beta_d = cube[1] + 1
    return [r, beta_d]
#     return [r, beta_s, beta_d]

Mean = np.zeros((100,2)); Samples = np.zeros((100, npara)); Weights = np.zeros(100);
for n in range(1):
    
    cl_hat_fore_ali = cl_hat_all_ali[n][2]
      
    def log_likelihood(cube, subtract = False):
        r_i = cube[0];
#         beta_s = cube[1];
        beta_s = -3.0;
        beta_d = cube[1];

        bb_tensor = bb_05[0:nmtlmax+1]*r_i/0.05 

        bb_camb = bb_tensor + lensingB[2][0:nmtlmax+1] ## tensor BB + lensing BB 

        cl_th_i = b.bin_cell(bb_camb[0:nmtlmax+1])  ## theoretical bandpower 
        cl_th_test = np.ones((lbin, Nf, Nf));  ## BB cross power spectra
        for ell in range(lbin):
            cl_th_test[ell] *= cl_th_i[ell]

        # add Noise bias N_l to expectation values.########################## Noise level
        C_l = cl_th_test + nl_mean_ali[2] + sync_ps(A_s_RJ, beta_s, 10) + dust_ps(A_d_RJ, beta_d, 10);
        logL =LogL.run(cl_hat=(cl_hat_fore_ali) , cl_th = C_l, sbin=1);   

        return np.real(logL)
    
    # with ThreadPoolExecutor(max_workers=cpu_count()-1) as executor:
    with Pool(cpu_count()-1) as executor:
        
        sampler = dynesty.NestedSampler(log_likelihood, prior, 2, nlive=400, pool=executor, queue_size=cpu_count(), bootstrap = 0)
        sampler.run_nested()
        results = sampler.results

        if n == 0 : 
            samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])

        else: 
            _samples, _weights = results.samples, np.exp(results.logwt - results.logz[-1])
            samples = np.r_[samples, _samples];
            weights = np.r_[weights, _weights]
    
# np.save('./samples.npy', samples); np.save('./weights.npy', weights);