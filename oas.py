import numpy as np

def oas_est(sample):
    """
    OAS covariance estimator.
    
    Parameters
    ----------
    sample : numpy.ndarray
        ensemble of observables, in shape (ensemble_size,data_size)
        
    Returns
    -------
    covariance matrix : numpy.ndarray
        covariance matrix in shape (data_size,data_size)
    """
    assert isinstance(sample, np.ndarray)
    _n, _p = sample.shape
    assert (_n > 0 and _p > 0)
    if _n == 1:
        return np.zeros((_p, _p))
    _m = np.mean(sample, axis=0)
    _u = sample - _m
    _s = np.dot(_u.T, _u) / _n
    _trs = np.trace(_s)
    
    # IMAGINE implementation
    #'''
    _trs2 = np.trace(np.dot(_s, _s))
    _numerator = (1 - 2. / _p) * _trs2 + _trs * _trs
    _denominator = (_n + 1. - 2. / _p) * (_trs2 - (_trs*_trs) / _p)
    #'''
    
    # skylearn implementation
    '''
    _mu = (_trs / _p)
    _alpha = np.mean(_s ** 2)
    _numerator = _alpha + _mu ** 2
    _denominator = (_n + 1.) * (_alpha - (_mu ** 2) / _p)
    '''
    
    _rho = 1. if _denominator == 0 else min(1., _numerator / _denominator)
    return (1. - _rho) * _s + np.eye(_p) * _rho * _trs / _p