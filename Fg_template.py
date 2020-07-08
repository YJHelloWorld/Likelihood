from pysm.common import B as pf
import numpy as np
from pysm.common import convert_units

def dust_ps(A_d, beta_d, lbin, nu = np.array([95, 150]), nu0 = 353, T_d = 19.6):
    
    '''
    Input
    -------------------------
    A_d : PS template at reference frequency at each lbin; input in uK_RJ units.
    beta_d : spectra index to be determined from likelihood analyis.
    
    Output
    -------------------------
    Cross power spectra for dust component, output in uK_CMB units.
    
    '''
    Nf = len(nu); coeff = convert_units("uK_RJ", "uK_CMB", nu);
    dl = np.ones((lbin, Nf, Nf));
    
    b0 = pf(nu0, T_d); b12 = pf(nu, T_d);    
    
    factors = np.ones((Nf, Nf))
    for i in range(Nf):
        for j in range(Nf):
            factors[i][j] = b12[i]*b12[j]/b0**2
    
    for ell in range(lbin):
        
        for i in range(Nf): 
            for j in range(Nf):
                
                dl[ell, i,j] = A_d[ell]*(nu[i]*nu[j]/nu0**2)**(beta_d - 2)*factors[i,j]*coeff[i]*coeff[j]
                
    return dl


def sync_ps(A_s, beta_s, lbin, nu = np.array([95, 150]), nu0 = 30):
    
    Nf = len(nu); coeff = convert_units("uK_RJ", "uK_CMB", nu);
    sl = np.ones((lbin, Nf, Nf));
    
    for ell in range(lbin):
        
        for i in range(Nf): 
            for j in range(Nf):
                sl[ell, i, j] = A_s[ell]*(nu[i]*nu[j]/nu0**2)**(beta_s)*coeff[i]*coeff[j] ;
                
    return sl