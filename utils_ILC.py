'''
This module contains some basic helper functions.
2019-12-20
With Yun Zheng
'''

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 8.0)

##    Only for 1-field map
def plot_ps(maps):
    _nside = int(np.sqrt(len(maps)/12))
    cls = hp.anafast(maps, lmax = 2*_nside);
    _ell = np.arange(len(cls)); _ell2 = _ell[2:]
    plt.loglog(_ell2, _ell2*(_ell2+1)/2/np.pi*cls[2:])
    
def deconv(maps, beam_in, beam_out, lmax):
    ''' 
    Beam in unit of arc-miniute.
    This function changes the value of the input map itself.
    
    '''
    
    _maps = np.copy(maps)
    for j in range(1,3): ### only for Q\U;
        _maps[j] = hp.sphtfunc.decovlving(_maps[j], fwhm = beam_in/60/180*np.pi, lmax = lmax, verbose = False)
        _maps[j] = hp.smoothing(_maps[j], fwhm = beam_out/60/180*np.pi, lmax = lmax, verbose = False)
    return _maps


def smooth(maps, beam_out, lmax):
    _maps = np.copy(maps);
    for j in range(1,3):                        ###for Q and U. Exclude I.
        _maps[j] = hp.smoothing(_maps[j], fwhm = beam_out/60/180*np.pi, lmax = lmax, verbose = False)
    return _maps


def bin_l(cl, L, Q, l2 = False, bin_scheme = None):
    ''' 
    
    cl, L, Q(bin_number); no np.mean
    
    '''
    ### some problmes exit when Q = L !!! 2020-03-14
    if len(cl.shape) > 2 :
        bin_averages = np.zeros((Q, cl.shape[1], cl.shape[1]))
    else:
        bin_averages = np.zeros(Q)
    if l2:
        for l in range(L):
            cl[l] = l*(l+1)/2/np.pi*(cl[l])    
    for q in range(Q):
        if q == 0:
            bin_averages[q] = sum(cl[2:((q+1)*L//Q)]/(L//Q))   ### l = 0,1 should be abandoned; SHOULDN'T use np.mean(), becaues it need to specify AXIS... . But for different shape, the axis is different .
        else:
            bin_averages[q] = sum(cl[q*L//Q:((q+1)*L//Q)]/(L//Q))
    return bin_averages

def get_ell(L, Q):
    ''' L, Q(bin_number) '''
    Ell = np.ones(Q)
    for i in range(0,int(Q)):
        Ell[i] = (2*i+1)*L//Q/2 
    return Ell

def Power_spectrum(maps,R, lmax):
    global cl
    ''' 
    revised for upper-triangle of the matrix.
    Input:
    maps with multi-frequencies IQU sky maps. 
    Galactic plane cut for calculating the power spectrum.
    lmax.  

    Return: EE and BB cross power spectrum. 
    Shape (2, lmax, nf, nf)
    '''
    n_f = len(maps)
    cl = np.ones((2, n_f*n_f, lmax +1)); Cl = np.zeros((2, lmax+1, n_f, n_f))
    k = 0
    for i in range(n_f):
        for j in range(n_f):
            
            if i >= j :
                cross_ps = hp.anafast(maps[i], maps[j], lmax = lmax, gal_cut=R, nspec=3) ## TT, EE, BB
            else:
                cross_ps = np.zeros((3, lmax+1)) 
            cl[0][k] = cross_ps[1]; cl[1][k] = cross_ps[2]  ## calculate the E and B_mode power spectrum 
            k += 1
            
    for l in range(lmax+1):
        Cl[0, l, : , :] = cl[0, :,l].reshape(n_f, n_f); Cl[1, l, : , :] = cl[1, :,l].reshape(n_f, n_f)
        Cl[0, l] += Cl[0, l].T - np.diag(Cl[0, l].diagonal()) ; Cl[1, l] += Cl[1, l].T - np.diag(Cl[1, l].diagonal()) 
    return Cl

def m_l(lmax, l):
    ''' 
    Return the m of corresponding l.
    
    '''
    m_id = np.ones(l+1, dtype = np.int)
    for i in range(l + 1):
        m_id[i] = hp.sphtfunc.Alm.getidx(lmax, l , i)
    return (m_id)


ali_ma = hp.read_map('/smc/jianyao/Ali_maps/ali_mask_wo_edge.fits')#, verbose=False)
def Mask(maps): 
    
    '''
    masked value = hp.unseen()
    '''
    maps_ma = hp.ma(maps)
    maps_ma.mask = np.logical_not(ali_ma)
    return maps_ma

def Mask_0(maps_raw):
    
    '''
    The masked values are equal to 0.
    
    '''
    maps = np.copy(maps_raw)
    
    index0 = np.arange(len(ali_ma));
    mask_index0 = index0[np.where(ali_ma<1)]
    
    _ndim = len(maps.shape)
    if _ndim > 2:  ### (Nf, 3, npix)
        for i in range(maps.shape[0]):
            for j in range(3):
                maps[i,j][mask_index0] = 0
    elif _ndim == 2: ### (3, npix)
        for j in range(maps.shape[0]):
            maps[j][mask_index0] = 0
    
    else: ### (npix)
        maps[mask_index0] = 0
    return maps

def l2(ell):
    '''
    get the l^2/np.pi
    '''
    
    return ell*(ell+1)/2/np.pi

def Select_fre(ps_in, sel):
        
    '''
    Take some part of the cross power spectrum matrix.
    ps_in : (Q, Nf, Nf)

    '''
    # sel = np.array((1,2,3))
    n_fre = len(sel); lbin = len(ps_in)
    ps_out = np.ones((lbin, n_fre, n_fre)); ### selected power spectra

    for q in range(lbin):
        x = 0; 
        for i in (sel):
            y = 0;
            for j in (sel):
                ps_out[q][x,y] = ps_in[q][i, j];
                y += 1;   
            x += 1;

    return ps_out

