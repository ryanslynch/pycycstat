"""
=====================================
Cyclostationary correlation functions
=====================================

Non-conjugate estimators
------------------------
   caf        Symmetric non-conjugate cyclic autocorrelation 
                          function
   scf          Non-conjugate spectral correlation function
   fsm_scf_estimate       Frequency-smoothing method of non-cojugate
                          spectral correlation function estimation
   tsm_scf_estimate       Time-smoothing method of non-conjugate
                          spectral correlation function estimation
   
Conjugate estimators
--------------------
   conj_cyclic_autocorr   Symmetric conjugate cyclic autocorrelation
                          function

   conj_spectral_corr     Conjugate spectral correlation function
   fsm_conj_scf_estimate  Frequency-smoothing method of conjugate
                          spectral correlation function estimation
   tsm_conj_scf_estimate  Time-smoothing method of conjugate 
                          spectral correlation function estimation
"""
import sys
import warnings
import numpy as np
from collections import defaultdict
from scipy.signal import convolve,stft,periodogram,get_window

def caf(x, nlags, cfs, conjugate=False):
    """
    Compute the cyclic autocorrelation (CAF) function at the given 
    cycle frequencies.

    Parameters
    ----------
    x : ndarray
       Input sequence.
    nlags : int
       The number of lags to use in the autocorrelation.
    cfs : ndarray
       The cycle frequencies at which to compute the CAF.
    conjugate : {bool}
       If True, return the conjugate CAF

    Returns
    -------
    out : ndarray
       An array of shape (len(cfs),nlags) containing the
       complex-valued cyclic autocorrelatioin coefficiencts.

    Notes
    -----
    The CAF is defined as 
    R_a(k) = 1/N * sum_n(x[n+k] * conj(x[n]) * exp(-2j*pi*a*n)) 
    for lag k and cycle frequency a.  The conjugate CAF is 
    defined as 
    R*_a(k) = 1/N * sum_n(x[n+k] * x[n] * exp(-2j*pi*a*n)) 
    In both cases, the symmetric autocorrelation is used, so 
    that the lag values range from -nlags/2 to nlags2/.
    """
    assert nlags <= len(x), "nlags must be <= len(x)"
    assert len(cfs) <= len(x), "len(cfs) must be <= len(x)"

    ts = np.arange(len(x))[None,None,]
    lags = np.arange(-nlags//2, nlags//2)
    x1 = np.array([np.roll(x,lag) for lag in lags])[None,]
    x2 = np.tile(x,nlags).reshape(nlags,len(x))[None,]
    if conjugate: x2 = x2.conjugate()
    caf = np.mean(
        x1*x2.conjugate()*np.exp(-2j*np.pi*cfs[:,None,None]*ts),axis=-1)

    return caf

def scf(x, window_size, cfs, conjugate=False):
    """
    Compute the spectral correlation function (SCF) at the given 
    cycle frequencies, defined as
    S_a(f) = 1/N * sum_n(1/T * X[n,f-a/2] * conj(X[n,f-a/2]))
    for cycle frequency a, where X is the sliding-window Fourier
    tranform of the input time series x, and f is the Fourier
    frequency.

    Parameters
    ----------
    x : ndarray
       Input sequence.
    window_size : int
       The number of points used in the sliding-window Fourier 
       transform of x.
    cfs : ndarray
       The cycle frequencies at which to compute the SCF.

    Returns
    -------
    out : ndarray
       An array of shape (len(cfs),window_size) containing the 
       complex-valued spectral correlation coefficients.

    Notes
    -----
       The Fourier transform frequencies are shifted so that the
       zero-frequency component is at the center of the spectrum.

    Examples
    --------
    """
    ts = np.arange(len(x))
    x1 = x*np.exp(2j*np.pi*0.5*cfs[:,None]*ts)
    x2 = x*np.exp(-2j*np.pi*0.5*cfs[:,None]*ts)
    _,_,X1 = stft(x1,window="boxcar",nperseg=window_size)
    _,_,X2 = stft(x2,window="boxcar",nperseg=window_size)
    if not conjugate: X2 = X2.conjugate()

    scf = np.fft.fftshift(np.mean(X1*X2,axis=-1)*window_size,axes=-1)
        
    return scf




def fsm_scf_estimate(x, kernel, cfs, mode="same"):
    ts = np.arange(len(x))

    assert mode in ["full","same","valid"],"unrecognized mode '%s'"%mode
    if mode == "full":
        output_size = len(x)+len(kernel)-1
    elif mode == "same":
        output_size = max(len(x),len(kernel))
    elif mode == "valid":
        output_size = max(len(x),len(kernel))-min(len(x),len(kernel))+1

    scf_est = np.empty((len(cfs),output_size),dtype=np.complex)
    for aa,cf in enumerate(cfs):
        x1 = x*np.exp(2j*np.pi*0.5*cf*ts)
        x2 = x*np.exp(-2j*np.pi*0.5*cf*ts)
        X1 = np.fft.fft(x1)
        X2 = np.fft.fft(x2)
        I = 1.0/len(x)*X1*X2.conjugate()
        scf_est[aa] = np.fft.fftshift(convolve(I,kernel,mode=mode))

    return scf_est


def fsm_conj_scf_estimate(x, kernel, cfs, mode="same"):
    ts = np.arange(len(x))
    assert mode in ["full","same","valid"],"unrecognized mode '%s'"%mode
    if mode == "full":
        output_size = len(x)+len(kernel)-1
    elif mode == "same":
        output_size = max(len(x),len(kernel))
    elif mode == "valid":
        output_size = max(len(x),len(kernel))-min(len(x),len(kernel))+1

    cscf_est = np.empty((len(cfs),output_size),dtype=np.complex)
    for aa,cf in enumerate(cfs):
        x1 = x*np.exp(2j*np.pi*0.5*cf*ts)
        x2 = np.conjugate(x*np.exp(-2j*np.pi*0.5*cf*ts))
        X1 = np.fft.fft(x1)
        X2 = np.fft.fft(x2)
        I = 1.0/len(x)*X1*X2
        cscf_est[aa] = np.fft.fftshift(convolve(I,kernel,mode=mode))

    return cscf_est


def tsm_scf_estimate(x, block_size, cfs):
    nblocks = len(x)//block_size
    if len(x) > nblocks*block_size:
        print(("Warning: Input array of length {0} does not divide evenly "
               "into blocks of size {1}.  Input will be "
               "truncated.".format(len(x),block_size)))
    x = x[:nblocks*block_size].reshape(nblocks,block_size)
    scf_est = np.empty((nblocks,block_size),dtype=np.complex)
    
    for cf in cfs:
        x1 = x*np.exp(2j*np.pi*0.5*cf*np.arange(block_size))
        x2 = x*np.exp(-2j*np.pi*0.5*cf*np.arange(block_size))
        X1 = np.fft.fft(x1,axis=1)
        X2 = np.fft.fft(x2,axis=1)
        I = 1.0/block_size*X1*X2.conjugate()
        # The CSP blog has a minus sign in this exponential but I can't
        # reproduce the results when I include that...TODO: investigate
        for ii in range(I.shape[0]): I[ii] *= np.exp(2j*np.pi*cf*jj*block_size)
        scf_est[aa] = np.fft.fftshift(np.mean(I,axis=0))

    return scf_est
        

def tsm_conj_scf_estimate(x, block_size, cfs):
    nblocks = len(x)//block_size
    if len(x) > nblocks*block_size:
        print(("Warning: Input array of length {0} does not divide evenly "
               "into blocks of size {1}.  Input will be "
               "truncated.".format(len(x),block_size)))
    x = x[:nblocks*block_size].reshape(nblocks,block_size)
    cscf_est = np.empty((nblocks,block_size),dtype=np.complex)
    
    for cf in cfs:
        x1 = x*np.exp(2j*np.pi*0.5*cf*np.arange(block_size))
        x2 = np.conjugate(x*np.exp(-2j*np.pi*0.5*cf*np.arange(block_size)))
        X1 = np.fft.fft(x1,axis=1)
        X2 = np.fft.fft(x2,axis=1)
        I = 1.0/block_size*X1*X2
        # The CSP blog has a minus sign in this exponential but I can't
        # reproduce the results when I include that...TODO: investigate
        for ii in range(I.shape[0]): I[ii] *= np.exp(2j*np.pi*cf*jj*block_size)
        cscf_est[aa] = np.fft.fftshift(np.mean(I,axis=0))

    return cscf_est


def ssca(x, nchan, nhop, fsamp=1.0, window="hamming", psd=None,
              fsm_window_size=256, conjugate=False, output="scf"):
    """
    Calculate the spectral correlation function or coherence using the 
    strip spectrum correlation analyzer.

    Parameters
    ---------
    x : ndarray
       Input sequence.
    nchan : int
       Number of points to use in the channelizer short time Fourier 
       transform.  This must be a power-of-two.
    nhop : int
       Number of points to hop between segments in the channelizer STFT.
       The overlap between segments is nchan-nhop.  This must be a 
       power-of-two.  A value of nchan/4 is recommended.
    fs : {float}
       Sampling frequency of the input data.
    window : {str or tuple or array_like}
       The windowing function to use in the channelizer STFT.  If a string
       or a tuple is given it will be passed to scipy.signal.get_window and
       must be a valid input to that function.  If an array_like object
       is given it will be used directly as the window.  See 
       scipy.signal.windows for more information on valid windows.
    psd : {ndarray}
       Side estimate of the power spectral density of X to use when 
       calculating the coherence.  If None, the frequency smoothing method
       will be used to generate a PSD estimate of the same size as x.  If
       output is 'scf' this has no effect.
    fsm_window_size : {int}
       The size of the smoothing window to use when generating the 
       frequency-smoothed PSD.  If psd is given or output is 'scf' 
       this has no effect.
    conjugate : {bool}
       If True, return the conjugate SCF or coherence.  Otherwise, return
       the non-conjugate SCF or coherence.
    output : {str}
       If 'scf', return the spectral correlation function.  If 'coherence',
       return the spectral coherence function.  If 'both', return
       the both the SCF and coherence.

    Returns
    -------
    f : ndarray
       The spectral frequencies
    alpha : ndarray
       The cycle frequencies
    scf : ndarray
       The spectral correlation function.  Omitted if output=="coherence"
    rho : ndarray
       The spectral coherence.  Omitted if output=="scf"
    
    Notes
    -----
    The strip spectrum correlation analyzer is an efficient method for 
    calculating the spectral correlation function over the cyclic bi-plane of 
    baseband frequency (f) vs cycle frequency (alpha).  This is accomplished 
    by first channelizing the input data using a short time Fourier transform, 
    correlating each segment with the input data, Fourier transforming the 
    correlation product, and then mapping the output to the f-alpha bi-plane.  
    The result will have a frequency resolution of fsamp/nchan and a cycle
    frequency resolution of fsamp/len(x).  If the spectral coherence is 
    desired, the SCF is normalized at each point by the power spectral density
    at f+alpha/2 and f-alpha/2.
    """
    npts = len(x)
    assert npts & (npts-1) == 0 and npts != 0,"len(x) must be a power of two"
    assert nchan & (nchan-1) == 0 and nchan != 0,"nchan must be a power of two"
    assert nhop & (nhop-1) == 0 and nhop != 0,"nhop must be a power of two"
    assert output in ["scf","coherence","both"],("%s it not a valid choice for "
                                                 "'output'"%output)
    nstrip = npts//nhop
    if psd is None:
        if fsm_window_size > len(x): fsm_window_size = len(x)
        fpsd,psd = periodogram(x)
        fpsd = np.fft.fftshift(fpsd)
        psd = np.fft.fftshift(psd)
        psd = np.convolve(
            1.0*np.ones(fsm_window_size)/fsm_window_size,psd,mode="same")
    fk,t,X = stft(x,fs=fsamp,window=window,nperseg=nchan,noverlap=nchan-nhop)
    fk = np.fft.fftshift(fk)
    # TODO: Normalization
    if type(window) in (str,tuple):
        win = get_window(window,nchan)
    else:
        win = window.copy()
    y1 = [np.sum(win[-nchan//2-n*nhop:])/np.sum(win) for n in range(nchan//2)]
    y2 = [np.sum(win[:nchan//2+n*nhop])/np.sum(win) for n in range(nchan//2)]
    norm = np.concatenate((y1,np.ones(len(t)-len(y1)-len(y2)),y2[::-1]))
    #X = X/np.pi/nchan*norm
    #X = X*(1.0*nchan-nhop)/npts*norm
    X = X[:,1:]
    X *= np.exp(-2j*np.pi*np.arange(nstrip)*np.arange(nchan)[:,None]*nhop/nchan)
    if conjugate:
        scf = np.fft.fft(np.repeat(X,nhop,axis=1)*x,axis=1)
    else:
        scf = np.fft.fft(np.repeat(X,nhop,axis=1)*x.conjugate(),axis=1)/nchan/np.pi
    fq = np.fft.fftshift(np.fft.fftfreq(npts,1/fsamp))    
    # Map fk,fq to f,alpha.  This algorithm was provided by user "Ethan" on
    # stackoverflow
    # https://stackoverflow.com/questions/61104413/optimizing-an-array-mapping-operation-in-python
    f = 0.5*(fk[:,np.newaxis] - fq)
    alpha = fk[:,np.newaxis] + fq
    scf = np.fft.fftshift(scf)

    if output == "coherence" or "both":
        if conjugate:
            S12 = psd[::npts//nchan,np.newaxis]*psd
        else:
            S12 = psd[::npts//nchan,np.newaxis] * psd[::-1]
        rho = scf/S12**0.5

    if output == "scf":
        return (f,alpha,scf)
    elif output == "coherence":
        return (f, alpha, rho)
    else:
        return (f, alpha, scf, rho)

def ssca_old(x, nchan, nhop, fsamp=1.0, window="hamming", psd=None,
         fsm_window_size=256, conjugate=False, output="scf"):
    """
    Calculate the spectral correlation function or coherence using the 
    strip spectrum correlation analyzer.

    Parameters
    ---------
    x : ndarray
       Input sequence.
    nchan : int
       Number of points to use in the channelizer short time Fourier 
       transform.  This must be a power-of-two.
    nhop : int
       Number of points to hop between segments in the channelizer STFT.
       The overlap between segments is nchan-nhop.  This must be a 
       power-of-two.  A value of nchan/4 is recommended.
    fs : {float}
       Sampling frequency of the input data.
    window : {str or tuple or array_like}
       The windowing function to use in the channelizer STFT.  If a string
       or a tuple is given it will be passed to scipy.signal.get_window and
       must be a valid input to that function.  If an array_like object
       is given it will be used directly as the window.  See 
       scipy.signal.windows for more information on valid windows.
    psd : {ndarray}
       Side estimate of the power spectral density of X to use when 
       calculating the coherence.  If None, the frequency smoothing method
       will be used to generate a PSD estimate of the same size as x.  If
       output is 'scf' this has no effect.
    fsm_window_size : {int}
       The size of the smoothing window to use when generating the 
       frequency-smoothed PSD.  If psd is given or output is 'scf' 
       this has no effect.
    conjugate : {bool}
       If True, return the conjugate SCF or coherence.  Otherwise, return
       the non-conjugate SCF or coherence.
    output : {str}
       If 'scf', return the spectral correlation function.  If 'coherence',
       return the spectral coherence function.  If 'both', return
       the both the SCF and coherence.

    Returns
    -------
    out : dict
       A dictionary indexed by cycle frequency.  Each entry is itself a
       dictionary with keys of 'freq', 'scf' and/or 'rho', which correspond
       to the baseband frequency, SCF, or coherence.
    
    Notes
    -----
    The strip spectrum correlation analyzer is an efficient method for 
    calculating the spectral correlation function over the cyclic bi-plane of 
    baseband frequency (f) vs cycle frequency (alpha).  This is accomplished 
    by first channelizing the input data using a short time Fourier transform, 
    correlating each segment with the input data, Fourier transforming the 
    correlation product, and then mapping the output to the f-alpha bi-plane.  
    The result will have a frequency resolution of fsamp/nchan and a cycle
    frequency resolution of fsamp/len(x).  If the spectral coherence is 
    desired, the SCF is normalized at each point by the power spectral density
    at f+alpha/2 and f-alpha/2.
    """
    npts = len(x)
    nstrip = npts/nhop
    if psd is None:
        fpsd,psd = periodogram(x)
        psd = np.convolve(
            1.0*np.ones(fsm_window_size)/fsm_window_size,psd,mode="same")
    #from scipy.signal import get_window
    fks,ts,X = stft(x,fs=fsamp,window=window,nperseg=nchan,noverlap=nchan-nhop)
    # Not 100% sure why this normalization works....
    X = X[:,1:]#/(4*np.pi*get_window(window,nchan).sum())
    X *= np.exp(-2j*np.pi*np.arange(nstrip)*np.arange(nchan)[:,None]*nhop/nchan)
    if conjugate:
        Sx = np.fft.fft(np.repeat(X,nhop,axis=1)*x,axis=1)
    else:
        Sx = np.fft.fft(np.repeat(X,nhop,axis=1)*x.conjugate(),axis=1)

    ret = {}
    for kk in range(-nchan/2,nchan/2):
        for qq in range(-npts/2,npts/2):
            alpha = 1.0*kk/nchan + 1.0*qq/npts
            f = 0.5*kk/nchan - 0.5*qq/npts
            if alpha not in ret:
                ret[alpha] = {}
                ret[alpha]["freq"] = np.array([f])
                if output == "scf" or output == "both":
                    ret[alpha]["scf"] = np.array([Sx[kk,qq]])
                if output == "coherence" or output == "both":
                    ii1 = int(round((f+0.5*alpha)*npts))
                    if not conjugate:
                        ii2 = int(round((f-0.5*alpha)*npts))
                    else:
                        ii2 = int(round((0.5*alpha-f)*npts))
                    S1 = psd[ii1]
                    S2 = psd[ii2]
                    ret[alpha]["rho"] = np.array([Sx[kk,qq]/(S1*S2)**0.5])
            else:
                ret[alpha]["freq"] = np.concatenate((ret[alpha]["freq"],[f]))
                if output == "scf" or output == "both":
                    ret[alpha]["scf"] = np.concatenate(
                        (ret[alpha]["scf"],[Sx[kk,qq]]))
                if output == "coherence" or output == "both":
                    ii1 = int(round((f+0.5*alpha)*npts))
                    ii2 = int(round((f-0.5*alpha)*npts))
                    S1 = psd[ii1]
                    S2 = psd[ii2]
                    ret[alpha]["rho"] = np.concatenate(
                        (ret[alpha]["rho"],[Sx[kk,qq]/(S1*S2)**0.5]))
    return ret


def ssca_cu(x, nchan, nhop, fsamp=1.0, window="hamming", psd=None,
            fsm_window_size=256, conjugate=False, output="scf"):
    """
    Calculate the spectral correlation function or coherence using the
    strip spectrum correlation analyzer with CUDA acceleration.

    Parameters
    ---------
    x : ndarray
       Input sequence.
    nchan : int
       Number of points to use in the channelizer short time Fourier
       transform.  This must be a power-of-two.
    nhop : int
       Number of points to hop between segments in the channelizer STFT.
       The overlap between segments is nchan-nhop.  This must be a
       power-of-two.  A value of nchan/4 is recommended.
    fs : {float}
       Sampling frequency of the input data.
    window : {str or tuple or array_like}
       The windowing function to use in the channelizer STFT.  If a string
       or a tuple is given it will be passed to scipy.signal.get_window and
       must be a valid input to that function.  If an array_like object
       is given it will be used directly as the window.  See
       scipy.signal.windows for more information on valid windows.
    psd : {ndarray}
       Side estimate of the power spectral density of X to use when
       calculating the coherence.  If None, the frequency smoothing method
       will be used to generate a PSD estimate of the same size as x.  If
       output is 'scf' this has no effect.
    fsm_window_size : {int}
       The size of the smoothing window to use when generating the
       frequency-smoothed PSD.  If psd is given or output is 'scf'
       this has no effect.
    conjugate : {bool}
       If True, return the conjugate SCF or coherence.  Otherwise, return
       the non-conjugate SCF or coherence.
    output : {str}
       If 'scf', return the spectral correlation function.  If 'coherence',
       return the spectral coherence function.  If 'both', return
       the both the SCF and coherence.

    Returns
    -------
    f : ndarray
       The spectral frequencies
    alpha : ndarray
       The cycle frequencies
    scf : ndarray
       The spectral correlation function.  Omitted if output=="coherence"
    rho : ndarray
       The spectral coherence.  Omitted if output=="scf"

    Notes
    -----
    The strip spectrum correlation analyzer is an efficient method for
    calculating the spectral correlation function over the cyclic bi-plane of
    baseband frequency (f) vs cycle frequency (alpha).  This is accomplished
    by first channelizing the input data using a short time Fourier transform,
    correlating each segment with the input data, Fourier transforming the
    correlation product, and then mapping the output to the f-alpha bi-plane.
    The result will have a frequency resolution of fsamp/nchan and a cycle
    frequency resolution of fsamp/len(x).  If the spectral coherence is
    desired, the SCF is normalized at each point by the power spectral density
    at f+alpha/2 and f-alpha/2.
    """
    try:
        import cupy as cp
    except ImportError:
        print("Error: Can't import cupy")
        sys.exit()
    try:
        import cusignal
    except ImportError:
        print("Error: Can't import cusignal")
        sys.exit()
    warnings.filterwarnings("ignore")

    x = cp.array(x)
    npts = len(x)
    assert npts & (npts-1) == 0 and npts != 0,"len(x) must be a power of two"
    assert nchan & (nchan-1) == 0 and nchan != 0,"nchan must be a power of two"
    assert nhop & (nhop-1) == 0 and nhop != 0,"nhop must be a power of two"
    assert output in ["scf","coherence","both"],("%s it not a valid choice for "
                                                 "'output'"%output)
    nstrip = npts//nhop
    if psd is None:
        if fsm_window_size > len(x): fsm_window_size = len(x)
        fpsd,psd = cusignal.periodogram(x)
        fpsd = cp.fft.fftshift(fpsd)
        psd = cp.fft.fftshift(psd)
        psd = cp.convolve(1.0*cp.ones(fsm_window_size)/fsm_window_size,psd,mode="same")

    fk,t,X = cusignal.stft(x,fs=fsamp,window=window,nperseg=nchan,noverlap=nchan-nhop)
    fk = cp.fft.fftshift(fk)
    # TODO: Normalization
    if type(window) == str:
        win = cusignal.get_window(window,nchan)
    else:
        win = window.copy()
    X = X[:,1:]
    
    X *= cp.exp(-2j*cp.pi*cp.arange(nstrip)*cp.arange(nchan)[:,None]*nhop/nchan)
    if conjugate:
        scf = cp.fft.fft(cp.repeat(X,nhop,axis=1)*x,axis=1)
    else:
        scf = cp.fft.fft(cp.repeat(X,nhop,axis=1)*x.conj(),axis=1)/nchan/cp.pi
    fq = cp.fft.fftshift(cp.fft.fftfreq(npts,1/fsamp))
    # Map fk,fq to f,alpha.  This algorithm was provided by user "Ethan" on
    # stackoverflow
    # https://stackoverflow.com/questions/61104413/optimizing-an-array-mapping-operation-in-python
    f = 0.5*(fk[:,cp.newaxis] - fq)
    alpha = fk[:,cp.newaxis] + fq
    scf = cp.fft.fftshift(scf)

    if output == "coherence" or "both":
        if conjugate:
            S12 = psd[::npts//nchan,cp.newaxis]*psd
        else:
            S12 = psd[::npts//nchan,cp.newaxis] * psd[::-1]
        rho = scf/S12**0.5
        
    if output == "scf":
        return (f,alpha,scf)
    elif output == "coherence":
        return (f, alpha, rho)
    else:
        return (f, alpha, scf, rho)

def ssca_cu2(x, nhop, fsamp=1.0, window="hamming", psd=None,
            fsm_window_size=256, conjugate=False, output="scf"):
    """
    Calculate the spectral correlation function or coherence using the
    strip spectrum correlation analyzer with CUDA acceleration.

    Parameters
    ---------
    x : ndarray
       Input sequence.
    nchan : int
       Number of points to use in the channelizer short time Fourier
       transform.  This must be a power-of-two.
    nhop : int
       Number of points to hop between segments in the channelizer STFT.
       The overlap between segments is nchan-nhop.  This must be a
       power-of-two.  A value of nchan/4 is recommended.
    fs : {float}
       Sampling frequency of the input data.
    window : {str or tuple or array_like}
       The windowing function to use in the channelizer STFT.  If a string
       or a tuple is given it will be passed to scipy.signal.get_window and
       must be a valid input to that function.  If an array_like object
       is given it will be used directly as the window.  See
       scipy.signal.windows for more information on valid windows.
    psd : {ndarray}
       Side estimate of the power spectral density of X to use when
       calculating the coherence.  If None, the frequency smoothing method
       will be used to generate a PSD estimate of the same size as x.  If
       output is 'scf' this has no effect.
    fsm_window_size : {int}
       The size of the smoothing window to use when generating the
       frequency-smoothed PSD.  If psd is given or output is 'scf'
       this has no effect.
    conjugate : {bool}
       If True, return the conjugate SCF or coherence.  Otherwise, return
       the non-conjugate SCF or coherence.
    output : {str}
       If 'scf', return the spectral correlation function.  If 'coherence',
       return the spectral coherence function.  If 'both', return
       the both the SCF and coherence.

    Returns
    -------
    f : ndarray
       The spectral frequencies
    alpha : ndarray
       The cycle frequencies
    scf : ndarray
       The spectral correlation function.  Omitted if output=="coherence"
    rho : ndarray
       The spectral coherence.  Omitted if output=="scf"

    Notes
    -----
    The strip spectrum correlation analyzer is an efficient method for
    calculating the spectral correlation function over the cyclic bi-plane of
    baseband frequency (f) vs cycle frequency (alpha).  This is accomplished
    by first channelizing the input data using a short time Fourier transform,
    correlating each segment with the input data, Fourier transforming the
    correlation product, and then mapping the output to the f-alpha bi-plane.
    The result will have a frequency resolution of fsamp/nchan and a cycle
    frequency resolution of fsamp/len(x).  If the spectral coherence is
    desired, the SCF is normalized at each point by the power spectral density
    at f+alpha/2 and f-alpha/2.
    """
    try:
        import cupy as cp
    except ImportError:
        print("Error: Can't import cupy")
        sys.exit()
    try:
        import cusignal
    except ImportError:
        print("Error: Can't import cusignal")
        sys.exit()
    warnings.filterwarnings("ignore")

    X = cp.array(x)
    nchan = X.shape[0]
    npts = X.shape[1]*nhop
    assert npts & (npts-1) == 0 and npts != 0,"len(x) must be a power of two"
    assert nchan & (nchan-1) == 0 and nchan != 0,"nchan must be a power of two"
    assert nhop & (nhop-1) == 0 and nhop != 0,"nhop must be a power of two"
    assert output in ["scf","coherence","both"],("%s it not a valid choice for "
                                                 "'output'"%output)
    nstrip = npts//nhop
    if psd is None:
        psd = np.sum(np.abs(X)**2,axis=1)

    fk = cp.fft.rfftfreq(2*nchan)
    fk = cp.fft.fftshift(fk)
    # TODO: Normalization
    if type(window) == str:
        win = cusignal.get_window(window,nchan)
    else:
        win = window.copy()
    
    #X *= cp.exp(-2j*cp.pi*cp.arange(nstrip)*cp.arange(nchan)[:,None]*nhop/nchan)
    if conjugate:
        #scf = cp.fft.fft(cp.repeat(X,nhop,axis=1)*X.mean(axis=0),axis=1)
        scf = cp.fft.fft(X*X.mean(axis=0),axis=1)
    else:
        #scf = cp.fft.fft(cp.repeat(X,nhop,axis=1)*X.mean(axis=0).conj(),axis=1)/nchan/cp.pi
        scf = cp.fft.fft(X*X.mean(axis=0).conj(),axis=1)/nchan/cp.pi
    fq = cp.fft.fftshift(cp.fft.fftfreq(npts))
    # Map fk,fq to f,alpha.  This algorithm was provided by user "Ethan" on
    # stackoverflow
    # https://stackoverflow.com/questions/61104413/optimizing-an-array-mapping-operation-in-python
    f = 0.5*(fk[:,cp.newaxis] - fq)
    alpha = fk[:,cp.newaxis] + fq
    scf = cp.fft.fftshift(scf)

    if output == "coherence" or "both":
        if conjugate:
            S12 = psd[::npts//nchan,cp.newaxis]*psd
        else:
            S12 = psd[::npts//nchan,cp.newaxis] * psd[::-1]
        rho = scf/S12**0.5
        
    if output == "scf":
        return (f,alpha,scf)
    elif output == "coherence":
        return (f, alpha, rho)
    else:
        return (f, alpha, scf, rho)
