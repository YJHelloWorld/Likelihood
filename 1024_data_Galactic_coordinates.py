import numpy as np
import healpy as hp
import pymaster as nmt
import time
from BP_beam import BPE as BPE
from bandpowers import BPE as BPE_NB

import pysm 
from pysm.nominal import models
from pysm.common import convert_units

nside = 1024; bin_w = 20; lmax = 501; 

SamNum = 50;

beams = [19, 11];
fres = [95, 150]; Nf = len(fres);

# ali_ma = hp.read_map('/fnx/jianyao/Likelihood_data/ABSData/ali_mask.fits', field = None)

ali_ma = hp.read_map('/fnx/jianyao/DataChallenge/AncillaryData/AliCPTWhiteNoiseMatrix/BINARYMASK_95_C_1024.fits', field=None, verbose = False)
hit = hp.read_map('/fnx/jianyao/DataChallenge/AliCPT/HITMAP.fits',dtype=np.int,verbose=0)
extra_ratio = 0.3
extra = hit>0  # extra mask
ali_ma[hit<extra_ratio*max(hit)] = 0
extra[hit>extra_ratio*max(hit)] = 0

est_b = BPE_NB(mask_in = ali_ma, nside = nside, bin_w = bin_w, lmax = lmax)

# est_b = BPE(ali_ma, nside=nside, bin_w = bin_w, lmax = lmax, beams=[19,11])
lbin = est_b.lbin;

cpn = np.zeros((Nf, 2, 12*nside**2)); ## CMB plus noise
total = np.zeros((Nf, 2, 12*nside**2));
cl_f_all = np.ones((SamNum, 3, lbin, Nf, Nf))
nl_all = np.zeros((SamNum, 3, lbin, Nf, Nf))
Noise = np.zeros((Nf, 2, 12*nside**2))
cl_hat_all = np.ones((SamNum, 3, lbin, Nf, Nf))
#### foreground ####
sky_config = {'dust':models('d1', nside), 'synchrotron':models('s1', nside)}
sky = pysm.Sky(sky_config);
c95 = convert_units("uK_RJ", "uK_CMB", 95);
c150 = convert_units("uK_RJ", "uK_CMB", 150);

s30 = sky.synchrotron(30); 
d353 = sky.dust(353);

fore_maps = sky.signal()(np.array(fres))

fore_maps[0] *= c95;
fore_maps[1] *= c150

rotate = hp.rotator.Rotator(coord=['G','C'])

# s30_b = hp.smoothing(s30, fwhm = 52.8/60/180*np.pi, lmax = lmax, verbose = False)
# d353_b = hp.smoothing(d353, fwhm = 4.944/60/180*np.pi, lmax = lmax, verbose = False)

# s30_RJ = est_b.Auto_TEB(s30_b, 52.8)[5];
# d353_RJ = est_b.Auto_TEB(d353_b, 4.944)[5];

s30_RJ = est_b.Auto_TEB(s30)[5];
d353_RJ = est_b.Auto_TEB(d353)[5];

np.save('/fnx/jianyao/Likelihood_data/Bandpowers/0904_Ali_2fre_with_beam_new_noise_1024/No_beam/Galactic/s30_RJ.npy', s30_RJ)
np.save('/fnx/jianyao/Likelihood_data/Bandpowers/0904_Ali_2fre_with_beam_new_noise_1024/No_beam/Galactic/d353_RJ.npy', d353_RJ)

for n in range(SamNum):
    start = time.time()
    cmb_map_i = np.load('/fnx/jianyao/DataChallenge/My_simulation/CMB/cmb_maps_mc_%03d.npy'%n)
    
    for fre in range(Nf):

#         cmb_map_beamed = hp.smoothing((cmb_map_i), fwhm = beams[fre]/60/180*np.pi, lmax = lmax, verbose = False);
        cmb_map_beamed = np.array((cmb_map_i))
        assert cmb_map_beamed.shape[0] == 3;
    
        nIQU = np.load('/fnx/jianyao/DataChallenge/My_simulation/Noise/Galactic/%sGHz/Noise_%sGHz_%03d.npy'%(fres[fre], fres[fre], n)) 
        
#         fore_map_beamed = hp.smoothing((fore_maps[fre]), fwhm = beams[fre]/60/180*np.pi, lmax = lmax, verbose = False);
        fore_map_beamed = np.array((fore_maps[fre]))

        cpn[fre] = (cmb_map_beamed )[1:]  + nIQU
        total[fre] = (cmb_map_beamed+ fore_map_beamed)[1:]  + nIQU
        Noise[fre] = nIQU
        
    nl_all[n] = est_b.Cross_EB(Noise*ali_ma)
    cl_f_all[n] = est_b.Cross_EB(cpn*ali_ma)
    cl_hat_all[n] = est_b.Cross_EB(total*ali_ma)
       
    end = time.time()
    print(n)
    print('time', (end - start)/60.0)
    
np.save('/fnx/jianyao/Likelihood_data/Bandpowers/0904_Ali_2fre_with_beam_new_noise_1024/No_beam/Galactic/cl_f_all.npy',cl_f_all)
np.save('/fnx/jianyao/Likelihood_data/Bandpowers/0904_Ali_2fre_with_beam_new_noise_1024/No_beam/Galactic/nl_all.npy', nl_all)
np.save('/fnx/jianyao/Likelihood_data/Bandpowers/0904_Ali_2fre_with_beam_new_noise_1024/No_beam/Galactic/cl_hat_all.npy', cl_hat_all)


 

