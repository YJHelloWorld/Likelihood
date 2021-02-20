'''
ILC method to do the component separation.
Both ILC in harmonic space and pixel space are defined.

'''
import logging as log
import numpy as np
import matplotlib.pyplot as plt
import pymaster as nmt

from utils_ILC import Select_fre as Select_fre
### ILC in l-space

class ILC_L(object):
    
    def __init__(self, nside, lmax, bin_w, beam = None):
        
        '''
        ILC in spherical space to do the foreground removal.
        
        Parameters:
        
        singal : numpy.ndarray
        The total CROSS power-sepctrum matrix,
        with global size (N_modes, N_freq, N_freq).
        * N_freq: number of frequency bands
        * N_modes: number of angular modes
        '''
        
        self.nside = nside; self.lmax = lmax;
        
        self.b = nmt.NmtBin(self.nside, nlb=bin_w, lmax=self.lmax)
        
        self.ell_n = self.b.get_effective_ells(); self.lbin = len(self.ell_n); self.el2 = self.l2(self.ell_n)
        
    def l2(self, ell):
        '''
        get the l^2/np.pi
        '''
        
        return ell*(ell+1)/2/np.pi
    
    # calculate the weight for each multipole bin
    
    #def __call__(self, sel):
##        log.debug('@ abspipe::__call__')
        #return self.run(sel)    
    
    def run(self, signal, noise, sel, return_weights = False, return_maps = False):
        
        '''
        ILC class call function.  
        '''
        total_bin = Select_fre(signal, sel); 
        noise_bin = Select_fre(noise, sel);
        
        _nf = len(sel)
        
        e = np.matrix(np.ones(_nf)); cl_ilc = np.zeros(self.lbin); noise_ilc = np.zeros(self.lbin)
        W = np.matrix(np.zeros((self.lbin, _nf)))  # Q = lmax/bins
        
        for l in range(self.lbin):
            
            norm = e*np.linalg.pinv((total_bin[l]))*e.T
           
            W[l,:] = e*np.linalg.pinv((total_bin[l]))/norm   
        
        for i in range(self.lbin):
                    
            noise_ilc[i] = W[i,:]*(noise_bin[i])*np.transpose(W[i,:])  
            cl_ilc[i] = W[i,:]*(total_bin[i])*np.transpose(W[i,:]) - noise_ilc[i]
            
        
        if return_weights: 
            return cl_ilc, W
        
        else:
            return cl_ilc
    
        
    def ILC_maps(self, total_mask, W):
        
        '''
        Apply the ILC weights in harmonic space to the alms to get the cleaned maps' alms,
        which are then transformed back to pixels.
        
        !!! in this case, bin_width = 1 !!!
        '''
        
        m_num = int((1 + lmax)*(lmax+1 -1)/2) # the number of alm of lmax=l for m >= 0; m = 0,1 for l = 1; m = 0,1,2 for l = 2.
        alm_Q = np.zeros((Nf, m_num), dtype = 'complex128'); alm_U = np.zeros((Nf, m_num), dtype = 'complex128')
        
        for i in range(Nf):
            alm_Q[i] = hp.map2alm(total_mask[i], lmax = lmax-1)[1]; # 95GHz, 150GHz, 353GHz
            alm_U[i] = hp.map2alm(total_mask[i], lmax = lmax-1)[2];   
            
        alm_Q_clean = np.zeros(m_num,dtype = 'complex128'); alm_U_clean = np.zeros(m_num, dtype = 'complex128')
        
        for l in np.arange(lmax):
            alm_Q_clean[m_l(lmax - 1, l)] = np.dot(np.array(W[l, :]),alm_Q[:,m_l(lmax - 1, l)])[0] #np.dot(np.array(weight[l, :]),alm_Q[:,m_l(lmax, l)])[0]#
            alm_U_clean[m_l(lmax - 1, l)] = np.dot(np.array(W[l, :]),alm_U[:,m_l(lmax - 1, l)])[0]
            
        alm_Q_clean[m_l(lmax, 0)] = 0; alm_Q_clean[m_l(lmax, 1)] = 0
        alm_U_clean[m_l(lmax, 0)] = 0; alm_U_clean[m_l(lmax, 1)] = 0    
        
        almT = np.zeros_like(alm_Q_clean)
        cmb_clean = hp.alm2map(np.row_stack((almT, alm_Q_clean, alm_U_clean)), nside = nside, lmax = lmax - 1)   
        
        return cmb_clean
            
    #def plot_weight():
        #Ell = binell()
        #for i in range(Nf):
            #plt.plot(Ell, W[:, i], label = '%s'%fre[i])
   
### ILC in pixel space         
class ILC_P(object):
    
    '''
    ILC in pixel space to do the foreground removal.
    
    signal_map: ((Nf, 3, 12*nside**2))
    '''
    
    def __init__(self, signal_maps, mask_in,  nl = None):
        
        self.signal = signal_maps; self.nl =nl; # use nl to do the noise debias.nl is from the ensemble average of noise reanlizations
        
        self.mask = mask_in; pix_list = np.arange(len(mask_in)); self.nside = hp.npix2nside(len(mask_in))
        
        self.avai_index = pix_list[np.where(mask_in == 1)] # the pixel index of the remained region 
            
        self.norm = 1.0/len(self.avai_index)
        
    #def __call__(self, psbin, absbin):
##        log.debug('@ abspipe::__call__')
        #return self.run(psbin, absbin)
    
    def run(self): 
        '''
        ILC in pixel space.
        
        return: the cleaned CMB maps, in which only QU components are cleaned and I map is not considered. 
        '''
    
        Nf = len(self.signal)
        
        total_Q = self.signal[:,1,:]; total_U = self.signal[:,2,:];
    
        Cov_Q = np.zeros((Nf, Nf)); w_Q = np.zeros(Nf)
        Cov_U = np.zeros((Nf, Nf)); w_U = np.zeros(Nf)
    
        for i in range(Nf):
            for j in range(Nf):
                
                tq_i = total_Q[i][self.avai_index] - np.mean(total_Q[i][self.avai_index]);
                tq_j = total_Q[j][self.avai_index] - np.mean(total_Q[j][self.avai_index])
                
                tu_i = total_U[i][self.avai_index] - np.mean(total_U[i][self.avai_index]);
                tu_j = total_U[j][self.avai_index] - np.mean(total_U[j][self.avai_index])
                
                Cov_Q[i, j] = np.dot(tq_i, tq_j)*self.norm
                Cov_U[i, j] = np.dot(tu_i, tu_j)*self.norm
    
        Cov_Q_inv = np.linalg.pinv(Cov_Q)
        Cov_U_inv = np.linalg.pinv(Cov_U)
    
    
        for i in range(Nf):
            w_Q[i] = np.sum(Cov_Q_inv[i,:])/np.sum(Cov_Q_inv)
            w_U[i] = np.sum(Cov_U_inv[i,:])/np.sum(Cov_U_inv)
    
    
        cmb_Q = np.dot(w_Q, total_Q); cmb_U = np.dot(w_U, total_U)
    
        cmb_I = np.zeros_like(cmb_Q);
        cmb_ILC_pix = np.row_stack((cmb_I, cmb_Q, cmb_U))
    
        return cmb_ILC_pix, (w_Q, w_U)        

class ILC_BB(object):
    
    def __init__(self, signal_maps, mask_in, nl = None):
        
        '''
        ILC in **pixel space** for BB maps as input.
        signal_maps: ((Nf, 12*nside**2))
        
        '''
        
        self.signal = signal_maps; self.nl = nl; # use nl to do the noise debias.nl is from the ensemble average of noise reanlizations
   
        self.mask = mask_in; pix_list = np.arange(len(mask_in)); self.nside = hp.npix2nside(len(mask_in))
        
        self.avai_index = pix_list[np.where(mask_in == 1)] # the pixel index of the remained region 
            
        self.norm = 1.0/len(self.avai_index)
        
    def run(self): 
       
        Nf = len(self.signal)
        
        total_BB = self.signal;
    
        Cov_BB = np.zeros((Nf, Nf));  w_BB = np.zeros(Nf)

        for i in range(Nf):
            for j in range(Nf):
                
                tb_i = total_BB[i][self.avai_index] - np.mean(total_BB[i][self.avai_index]);
                tb_j = total_BB[j][self.avai_index] - np.mean(total_BB[j][self.avai_index])
                
                Cov_BB[i, j] = np.dot(tb_i, tb_j)*self.norm

        Cov_BB_inv = np.linalg.pinv(Cov_BB)  
    
        for i in range(Nf):
            w_BB[i] = np.sum(Cov_BB_inv[i,:])/np.sum(Cov_BB_inv)
           
        cmb_BB = np.dot(w_BB, total_BB); 
    
        return cmb_BB, w_BB