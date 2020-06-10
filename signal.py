"""
========================================================================
Signal generators for CSP (:mod 'pycycstat.signal')
========================================================================

Contents
--------
   rp_bpsk  Rectangular pulse binary phase shift keyed signal
"""
import numpy as np
from scipy.signal import lfilter


def rp_bpsk(nbits, tbit, fc, Ebit=0.0, N0=None):
    """
    Generate a rectangular pulse binary phase shift keyed signal.

    Parameters
    ----------
    nbits : int
        Number of bits to generate.
    tbit : int
        Bit duration (seconds).  The bit rate is 1/tbit.
    fc : float
        Carrier frequency (normalized units)
    Ebit : float
        Energy per bit (dB).
    N0 : float
        Noise power spectral density (dB).  If None, do not add noise.
    
    Returns
    -------
    out : ndarray
       An array of size nbit containing the complex rectangular pulse BPSK 
       signal.

    Examples
    --------
    >> rp_bpsk(10000,10,0.05,Ebit=10,N0=-10)
    array([ 3.27208358+0.58731289j,  5.98561690+2.07007302j,
        7.60758519+5.56195842j, ..., -3.73780915+5.00217463j,
        0.02360099+0.1760474j , -0.09160716+0.18982198j])
    """

    # Create a random bipolar symbol sequence and zero pad it
    bit_sequence = np.random.randint(low=0,high=2,size=nbits)
    symbol_sequence = []
    for b in bit_sequence:
        symbol_sequence.append(2*b-1)
        for ii in range(tbit-1): symbol_sequence.append(0)
    s = lfilter(np.ones(tbit), 1, symbol_sequence)
    # Apply the carrier frequency
    Ebit_linear = 10**(Ebit/10.0)
    x = np.sqrt(Ebit_linear)*s*np.exp(2j*np.pi*fc*np.arange(len(s)))
    if N0 is not None:
        N0_linear = 10**(N0/10.0)
        n = np.random.randn(len(x)) + 1j*np.random.randn(len(x))
        n *= np.sqrt(N0_linear/np.var(n))
        x += n

    return x
