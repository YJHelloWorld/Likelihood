from pysm.common import B as pf
import numpy as np
from pysm.common import convert_units

def dust_ps(A_d, beta_d, lbin, nu = np.array([95, 150]), nu0 = 353, T_d = 19.6):
    
    '''
    Input
    -------------------------
    A_d : PS template at reference frequency at each lbin; Using Namaster within corresponding mask. Input in **uK_RJ** units.
    beta_d : spectra index to be determined from likelihood analyis.
    
    Output
    -------------------------
    Cross power spectra for dust component, output in **uK_CMB** units.
    
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
    
    '''
    Input
    -------------------------
    A_s : PS template at reference frequency at each lbin; input in **uK_RJ** units.
    beta_s : spectra index to be determined from likelihood analyis.
    
    Output
    -------------------------
    Cross power spectra for dust component, output in **uK_CMB** units.
    
    '''
    
    
    Nf = len(nu); coeff = convert_units("uK_RJ", "uK_CMB", nu);
    sl = np.ones((lbin, Nf, Nf));
    
    for ell in range(lbin):
        
        for i in range(Nf): 
            for j in range(Nf):
                sl[ell, i, j] = A_s[ell]*(nu[i]*nu[j]/nu0**2)**(beta_s)*coeff[i]*coeff[j] ;
                
    return sl


def corre_fore_simple(epsilon, A_d, A_s, beta_s, beta_d, lbin, nu = np.array([95, 150]), nu_s0 = 30, nu_d0 = 353):
    '''
    correlation of dust and synchrotron;
    Input
    ---------------------------
    same as above.
    
    epsilon, A_d, A_s, beta_s, beta_d, lbin, nu = np.array([95, 150]), nu_s0 = 30, nu_d0 = 353
    '''
    
    Nf = len(nu); 
    coeff = convert_units("uK_RJ", "uK_CMB", nu);
    
    fl = np.ones((lbin, Nf, Nf)); 
    
    for ell in range(lbin):
        for i in range(Nf): 
            for j in range(Nf):
                
                fl[ell, i, j] = epsilon*np.sqrt(A_s[ell]*A_d[ell])*(f_d(nu[i], beta_d)*f_s(nu[j], beta_s) + f_d(nu[j], beta_d)*f_s(nu[i], beta_s))*coeff[i]*coeff[j]  
                                                                            
    return fl

def f_d(nu, beta_d, nu0 = 353, T_d = 19.6):
    '''
    frequency dependence of dust : modified blackbody
    '''
    
    return (nu/nu0)**(beta_d - 2)*pf(nu, T_d)/pf(nu0, T_d)
    
def f_s(nu, beta_s, nu0 = 30):
    '''
    frequency dependence of synchrotro : power law.
    '''
    
    return (nu/nu0)**beta_s

def corre_fore_complex(epsilon, A_d, A_s, ells, alpha_s, alpha_d, beta_s, beta_d, nu = np.array([95, 150]), nu_s0 = 30, nu_d0 = 353):
    
    Nf = len(nu); lbin = len(ells)
    coeff = convert_units("uK_RJ", "uK_CMB", nu);
    
    fl = np.ones((lbin, Nf, Nf));
    
    f = epsilon*np.sqrt(A_s*A_d); 
    
    for ell in range(lbin):
        for i in range(Nf): 
            for j in range(Nf):
                
                fl[ell, i, j] =f*(ells[ell]/71.5)**((alpha_s + alpha_d)/2)*(f_d(nu[i], beta_d)*f_s(nu[j], beta_s) + f_d(nu[j], beta_d)*f_s(nu[i], beta_s))*coeff[i]*coeff[j]  
                                                                            
    return fl