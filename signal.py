"""
========================================================================
Signal generators for CSP (:mod 'pycycstat.signal')
========================================================================

Contents
--------
   ask        Generate an amplitude-shift keyed signal with random 
              symbols.
   ook        Generate an on-off keyed signal with random symbols.
   psk        Generate a phase-shift keyed signal with random symbols.
   bsk        Generate a binary phase-shift keyed signal with random 
              symbols.
   qsk        Generate a quadrature phase-shift keyed signal with random
              symbols.
   fsk        Generate a frequency-shift keyed signal with random
              symbols.
   msk        Generate a minimum-shift keyed signal with random symbols.
   gmsk       Generate a Gaussian mignimum-shift keyed signal with random
              symbols.
   qam        Generate a quadrature amplitude modulated signal with
              random symbols.           
   specline   Generate a Gaussian-broadened spectral line
   symseq     Generate random symbols with a given number of bits per
              symbol.
   noise      Generate Gaussian random noise with given amplitude.

Notes
-----
   All functions generate random symbols.  The seed used for random
   number generation can be set by modifying the internal variable
   _seed.
"""
import numpy as np
from scipy.signal import get_window,convolve
from pycycstat.utils import vco

_seed = None # Random number generator seed used for simulated signals


def ask(nsym, tsym, nbits, fc, bias=0.0, fs=1.0, Ebit=0, N0=None, 
         smoothing_window=None, return_sym_seq=False):
    """
    Generate an amplitude-shift keyed signal with random symbols.  

    Parameters
    ----------
    nsym : int
        Number of symbols to generate.
    tsym : int
        Duration of each symbol (samples).  Inverse of the baud rate.
    nbits : int
        Number of bits per symbol.  The size of the symbol set (i.e.,
        the number of points in the corresponding constellation
        diagram) is 2**nbits.
    fc : float
        Carrier frequency (Hz).
    bias : {float}
        Bias amplitude corresponding to low bit.
    fs : {float}
        Sampling frequency (Hz).
    Ebit : {float}
        Energy per bit (dB).
    N0 : {float}
        Noise power spectral density (dB).  If None, do not add noise.
    smoothing_window : {str or tuple or array_like} The windowing
        function used to smooth the symbol sequence.  If None, no
        smoothing is used and the pulse is rectangular.  If a string
        or a tple is given it will be passed to scipy.signal.get_window
        and must be a valid input to that function.  If an array_like
        object is given it will be used directly as the window.
    return_sym_seq : bool
        Return the symbol sequence along with the signal.
    
    Returns
    -------
    signal : ndarray
       An array of size nsym*tsym containing the complex ASK signal.
    symbol_sequence : ndarray 
       An array of size nsym*tsym containing the symbols encoded in
       the signal.  Omitted if return_sym_seq == False

    Notes
    -----
    Amplitude-shift keying encodes data by modulating the amplitude of
    a carrier wave.  The signal is defined as

    sqrt(2*Eb/tbit) * (2*s/(2**nbit - 1) - 1) * exp(1j*2*pi*fc*t) + bias

    where Eb is the energy-per-bit (in linear units).  The resulting
    signal is thus varies about +/-sqrt(2*Eb/tbit) + bias.
    """
    symbol_sequence = symseq(nsym,tsym,nbits,smoothing_window)
    x = np.arange(nsym*tsym)
    Ebit_linear = 10**(Ebit/10.0)/tsym
    modulation = 2*symbol_sequence/(2**nbits - 1) - 1 + bias
    e_vec = 1j*2*np.pi*fc*x/fs
    signal = np.sqrt(Ebit_linear) * modulation * np.exp(e_vec)
    signal /= modulation.std()
    signal += bias
    if N0 is not None:
        signal += noise(len(signal),N0)
    
    if return_sym_seq:
        return signal,symbol_sequence
    else:
        return signal


def ook(nsym, tsym, fc, fs=1.0, Ebit=0, N0=None,
        smoothing_window=None, return_sym_seq=False):
    """
    Generate an on-off keyed signal with random symbols.  

    Parameters
    ----------
    nsym : int
        Number of symbols to generate.
    tsym : int
        Duration of each symbol (samples).
    fc : float
        Carrier frequency (Hz).
    fs : {float}
        Sampling frequency (Hz).
    Ebit : {float}
        Energy per bit (dB).
    N0 : {float}
        Noise power spectral density (dB).  If None, do not add noise.
    smoothing_window : {str or tuple or array_like} The windowing
        function used to smooth the symbol sequence.  If None, no
        smoothing is used and the pulse is rectangular.  If a string
        or a tple is given it will be passed to scipy.signal.get_window
        and must be a valid input to that function.  If an array_like
        object is given it will be used directly as the window.
    return_sym_seq : bool
        Return the symbol sequence along with the signal.
    
    Returns
    -------
    signal : ndarray
       An array of size nsym*tsym containing the complex OOK signal.
    symbol_sequence : ndarray 
       An array of size nsym*tsym containing the symbols encoded in
       the signal.  Omitted if return_sym_seq == False

    Notes
    -----
    On-off keying is the simplest form of amplitude modulation where a
    carrier wave is either present or not.  The signal is defined as

    sqrt(2*Eb/tbit) * s * exp(1j*2*pi*fc*t)

    where Eb is the energy-per-bit (in linear units) and s is 0 or 1.
    This implementation preserves phase across on-off cycles.
    """
    rv = ask(nsym,tsym,1,fc,1.0,fs,Ebit,N0,smoothing_window,return_sym_seq)
    
    if return_sym_seq:
        return rv[0],rv[1]
    else:
        return rv


def psk(nsym, tsym, nbits, fc, fs=1.0, Ebit=0.0, N0=None,
        smoothing_window=None, return_sym_seq=False):
    """
    Generate a phase-shift keyed signal with random symbols.  

    Parameters
    ----------
    nsym : int
        Number of symbols to generate.
    tsym : int
        Duration of each symbol (samples).
    nbits : int
        Number of bits per symbol.  The size of the symbol set (i.e.,
        the number of points in the corresponding constellation
        diagram) is 2**nbits. 
    fc : float
        Carrier frequency (Hz).
    fs : {float}
        Sampling frequency (Hz).
    Ebit : {float}
        Energy per bit (dB).
    N0 : {float}
        Noise power spectral density (dB).  If None, do not add noise.
    smoothing_window : {str or tuple or array_like} The windowing
        function used to smooth the symbol sequence.  If None, no
        smoothing is used and the pulse is rectangular.  If a string
        or a tple is given it will be passed to scipy.signal.get_window
        and must be a valid input to that function.  If an array_like
        object is given it will be used directly as the window.
    return_sym_seq : bool
        Return the symbol sequence along with the signal.
    
    Returns
    -------
    signal : ndarray
       An array of size nsym*tsym containing the complex PSK signal.
    symbol_sequence : ndarray 
       An array of size nsym*tsym containing the symbols encoded in
       the signal.  Omitted if return_sym_seq == False

    Notes
    -----
    Phase-shift keying encodes data by modulating the phase of a
    carrier wave.  When 1-bit encoding is used the result is a binary
    phase-shift keyed signal defined as

    sqrt(2*Eb/tbit) * exp(1j*pi*(2*fc*t + 1 - s)) 

    where Eb is the energy-per-bit (in linear units) and s is the
    symbol (0 or 1).  This yields two phases at 0 and pi.  When 2-bit
    or higher encoding is used the signal is defined as

    sqrt(2*Eb/tbit) * exp(1j*pi*(2*fc*t + (2*s + 1)/2**nbits))

    This yields phases at (2*n-1)*pi/2**nbits where n = 1...2**nbits.
    When 2-bit encoding is used the result is a quadrature phase-shift
    keyed signal.
    """
    symbol_sequence = symseq(nsym,tsym,nbits,smoothing_window)
    x = np.arange(nsym*tsym)
    if nbits == 1:
        e_vec = 1j*np.pi*(2*fc*x/fs + 1 - symbol_sequence)
    else:
        e_vec = 1j*np.pi*(2*fc*x/fs + (2.0*symbol_sequence + 1)/2**nbits)
    Ebit_linear = 10**(Ebit/10.0)/tsym
    signal = np.sqrt(Ebit_linear)*np.exp(e_vec)
    if N0 is not None:
        signal += noise(len(signal),N0)

    if return_sym_seq:
        return signal,symbol_sequence
    else:
        return signal


def bpsk(nsym, tsym, fc, fs=1.0, Ebit=0.0, N0=None,
        smoothing_window=None, return_sym_seq=False):
    """
    Generate a binary phase-shift keyed signal with random symbols.  

    Parameters
    ----------
    nsym : int
        Number of symbols to generate.
    tsym : int
        Duration of each symbol (samples).
    fc : float
        Carrier frequency (Hz).
    fs : {float}
        Sampling frequency (Hz).
    Ebit : {float}
        Energy per bit (dB).
    N0 : {float}
        Noise power spectral density (dB).  If None, do not add noise.
    smoothing_window : {str or tuple or array_like} The windowing
        function used to smooth the symbol sequence.  If None, no
        smoothing is used and the pulse is rectangular.  If a string
        or a tple is given it will be passed to scipy.signal.get_window
        and must be a valid input to that function.  If an array_like
        object is given it will be used directly as the window.
    return_sym_seq : bool
        Return the symbol sequence along with the signal.
    
    Returns
    -------
    signal : ndarray
       An array of size nsym*tsym containing the complex PSK signal.
    symbol_sequence : ndarray 
       An array of size nsym*tsym containing the symbols encoded in
       the signal.  Omitted if return_sym_seq == False

    Notes
    -----
    Binary Phase-shift keying encodes data by modulating the phase of
    a carrier wave between two phases.  See pycycstat.signal.psk for
    more information on phase-shift keying.
    """
    rv = psk(nsym,tsym,1,fc,fs,Ebit,N0,smoothing_window,return_sym_seq)

    return rv


def qpsk(nsym, tsym, fc, fs=1.0, Ebit=0.0, N0=None,
        smoothing_window=None, return_sym_seq=False):
    """
    Generate a quadrature phase-shift keyed signal with random symbols.  

    Parameters
    ----------
    nsym : int
        Number of symbols to generate.
    tsym : int
        Duration of each symbol (samples).
    fc : float
        Carrier frequency (Hz).
    fs : {float}
        Sampling frequency (Hz).
    Ebit : {float}
        Energy per bit (dB).
    N0 : {float}
        Noise power spectral density (dB).  If None, do not add noise.
    smoothing_window : {str or tuple or array_like} The windowing
        function used to smooth the symbol sequence.  If None, no
        smoothing is used and the pulse is rectangular.  If a string
        or a tple is given it will be passed to scipy.signal.get_window
        and must be a valid input to that function.  If an array_like
        object is given it will be used directly as the window.
    return_sym_seq : bool
        Return the symbol sequence along with the signal.
    
    Returns
    -------
    signal : ndarray
       An array of size nsym*tsym containing the complex PSK signal.
    symbol_sequence : ndarray 
       An array of size nsym*tsym containing the symbols encoded in
       the signal.  Omitted if return_sym_seq == False

    Notes
    -----
    Quadrature Phase-shift keying encodes data by modulating the phase
    of a carrier wave between four phases.  See pycycstat.signal.psk
    for more information on phase-shift keying.
    """
    rv = psk(nsym,tsym,2,fc,fs,Ebit,N0,smoothing_window,return_sym_seq)

    return rv


def fsk(nsym, tsym, nbits, Df, fc, fs=1.0, Ebit=0.0, N0=None,
        smoothing_window=None, return_sym_seq=False):
    """
    Generate a continuous-phase frequency-shift keyed signal with
    random symbols.

    Parameters
    ----------
    nsym : int
        Number of symbols to generate.
    tsym : int
        Duration of each symbol (samples).
    nbits : int
        Number of bits per symbol.  The size of the symbol set (i.e.,
        the number of points in the corresponding constellation
        diagram) is 2**nbits.
    Df : float
        Frequency deviation between symbols (Hz).
    fc : float
        Carrier frequency (Hz).
    fs : {float}
        Sampling frequency (Hz).
    Ebit : {float}
        Energy per bit (dB).
    N0 : {float}
        Noise power spectral density (dB).  If None, do not add noise.
    smoothing_window : {str or tuple or array_like} The windowing
        function used to smooth the symbol sequence.  If None, no
        smoothing is used and the pulse is rectangular.  If a string
        or a tple is given it will be passed to scipy.signal.get_window
        and must be a valid input to that function.  If an array_like
        object is given it will be used directly as the window.
    return_sym_seq : bool
        Return the symbol sequence along with the signal.
    
    Returns
    -------
    signal : ndarray
       An array of size nsym*tsym containing the complex FSK signal.
    symbol_sequence : ndarray 
       An array of size nsym*tsym containing the symbols encoded in
       the signal.  Omitted if return_sym_seq == False

    Notes
    -----
    Frequency-shift keying encodes data by modulating the frequency of
    a carrier wave.  This implementation uses a voltage-controlled
    oscillator to produce a continuous-phase signal defined as

    sqrt(2*Eb/tbit) * exp(1j*(2*pi*fc*t + phi)
    phi = Df * integral(s dt)

    where Eb is the energy-per-bit (in linear units) and s is the
    symbol sequence.  See pycycstat.utils.vco for more information on
    the implementation of the voltage-controlled-oscillator.
    """
    symbol_sequence = symseq(nsym,tsym,nbits,smoothing_window)
    x = np.arange(nsym*tsym)
    Ebit_linear = 10**(Ebit/10.0)/tsym
    signal = np.sqrt(Ebit_linear)*vco(symbol_sequence,fc,Df,tsym,fs=fs)
    if N0 is not None:
        signal += noise(len(signal),N0)

    if return_sym_seq:
        return signal,symbol_sequence
    else:
        return signal


def msk(nsym, tsym, nbits, fc, fs=1.0, Ebit=0.0, N0=None,
        smoothing_window=None, return_sym_seq=False):
    """
    Generate a minimum-shift keyed signal with random symbols.

    Parameters
    ----------
    nsym : int
        Number of symbols to generate.
    tsym : int
        Duration of each symbol (samples).
    nbits : int
        Number of bits per symbol.  The size of the symbol set (i.e.,
        the number of points in the corresponding constellation
        diagram) is 2**nbits.
    fc : float
        Carrier frequency (Hz).
    fs : {float}
        Sampling frequency (Hz).
    Ebit : {float}
        Energy per bit (dB).
    N0 : {float}
        Noise power spectral density (dB).  If None, do not add noise.
    smoothing_window : {str or tuple or array_like} The windowing
        function used to smooth the symbol sequence.  If None, no
        smoothing is used and the pulse is rectangular.  If a string
        or a tple is given it will be passed to scipy.signal.get_window
        and must be a valid input to that function.  If an array_like
        object is given it will be used directly as the window.
    return_sym_seq : bool
        Return the symbol sequence along with the signal.
    
    Returns
    -------
    signal : ndarray
       An array of size nsym*tsym containing the complex FSK signal.
    symbol_sequence : ndarray 
       An array of size nsym*tsym containing the symbols encoded in
       the signal.  Omitted if return_sym_seq == False

    Notes
    -----
    Minumum-shift keying encodes data in bits that alternate between
    in-phase and quadrature components.  This is equivalent to
    continuous-phase frequency-shift keying where the frequency
    deviation between symbols is equal to the half the symbol
    frequency (i.e. 0.5/tsym).  See pycycstat.signal.fsk for more
    information on this implrementation of continuous-phase
    frequency-shift keying.
    """
    rv = fsk(nsym,tsym,nbits,0.5*fs/tsym,fc,fs,Ebit,N0,smoothing_window,
             return_sym_seq)
    
    return rv

def gmsk(nsym, tsym, nbits, fc, BT=0.3, fs=1.0, Ebit=0.0, N0=None,
         return_sym_seq=False):
    """
    Generate a Gaussian minimum-shift keyed signal with random
    symbols.

    Parameters
    ----------
    nsym : int
        Number of symbols to generate.
    tsym : int
        Duration of each symbol (samples).
    nbits : int
        Number of bits per symbol.  The size of the symbol set (i.e.,
        the number of points in the corresponding constellation
        diagram) is 2**nbits.
    fc : float
        Carrier frequency (Hz).
    BT : {float}
        Product of the 3-dB bandwidth of the Gaussian filter and tsym.
        The standard deviation of the Gaussian filter is 
        sqrt(log(2))/(2*pi*B).
    fs : {float}
        Sampling frequency (Hz).
    Ebit : {float}
        Energy per bit (dB).
    N0 : {float}
        Noise power spectral density (dB).  If None, do not add noise.
    return_sym_seq : bool
        Return the symbol sequence along with the signal.
    
    Returns
    -------
    signal : ndarray
       An array of size nsym*tsym containing the complex FSK signal.
    symbol_sequence : ndarray 
       An array of size nsym*tsym containing the symbols encoded in
       the signal.  Omitted if return_sym_seq == False

    Notes
    -----
    Gaussian minumum-shift keying encodes data in bits that alternate
    between in-phase and quadrature components after smoothing the
    symbol sequence with a Gaussian filter.  This reduces out-of-band
    interference compared with unsmoothed minumum-shift keying, but at
    the expense of more inter-symbol interference.  See
    pycycstat.signal.msk for more information on minimum-shift keying.
    """
    B = BT/tsym
    s = np.sqrt(np.log(2))/(2*np.pi*B)
    rv = msk(nsym,tsym,nbits,fc,fs,Ebit,N0,(("gaussian",s),tsym),return_sym_seq)
    
    return rv


def qam(nsym, tsym, nbits, fc, fs=1.0, Ebit=0.0, N0=None,
        smoothing_window=None, return_sym_seq=False):
    """
    Generate a quadrature amplitude modulated signal with random
    symbols.

    Parameters
    ----------
    nsym : int
        Number of symbols to generate.
    tsym : int
        Duration of each symbol (samples).
    nbits : int
        Number of bits per symbol.  The size of the symbol set (i.e.,
        the number of points in the corresponding constellation
        diagram) is 2**nbits.
    fc : float
        Carrier frequency (Hz).
    fs : {float}
        Sampling frequency (Hz).
    Ebit : {float}
        Energy per bit (dB).
    N0 : {float}
        Noise power spectral density (dB).  If None, do not add noise.
    smoothing_window : {str or tuple or array_like} The windowing
        function used to smooth the symbol sequence.  If None, no
        smoothing is used and the pulse is rectangular.  If a string
        or a tple is given it will be passed to scipy.signal.get_window
        and must be a valid input to that function.  If an array_like
        object is given it will be used directly as the window.
    return_sym_seq : bool
        Return the symbol sequence along with the signal.
    
    Returns
    -------
    signal : ndarray
       An array of size nsym*tsym containing the complex QAM signal.
    symbol_sequence : tuple 
       An tuple of the two symbol sequences, each of size nsym*tsym.
       Omitted if return_sym_seq == False

    Notes
    -----
    Quadrature amplitude modulation encodes information by modulating
    the amplitude of two carrier waves, known as the in-phase and
    quadrature components.  The signal is defined as

    sqrt(Eb/(2*tbit)) * exp(1j*2*pi*fc*t) * (I + 1j*Q)

    where Eb is the energy-per-bit (in linear units), and I and Q are
    the two symbol sequences corresponding to the in-phase and
    quadrature components.  The total number of bits is split evenly
    between I and Q, i.e. each is a symbol sequency with nbits/2
    bits-per-symbol.  In the case of a 1-bit signal, Q = 0.  This
    produces a constellation diagram that is functionally equivalent
    to a binary phase-shift keyed signal, which is the typical
    convention.
    """
    if nbits == 1:
        rv1 = ask(nsym,tsym,nbits,fc,0,fs,Ebit,N0,smoothing_window,
                  return_sym_seq)
        rv2 = (0,None) if return_sym_seq else 0
    else:
        nbits = nbits//2
        rv1 = ask(nsym,tsym,nbits,fc,0,fs,Ebit,N0,smoothing_window,
                  return_sym_seq)
        rv2 = ask(nsym,tsym,nbits,fc,0,fs,Ebit,N0,smoothing_window,
                  return_sym_seq)
    
    if return_sym_seq:
        return 0.5*(rv1[0] + 1j*rv2[0]),rv1[1],rv2[1]
    else:
        return 0.5*(rv1 + 1j*rv2)

def specline(npts, nchan, fc, fwhm, fs=1.0, SNR=10.0, N0=None, nemitters=10):
    """
    Generate a Gaussian spectral line.

    Parameters
    ----------
    npts : int
       Length of output array
    nchan : int
       Number of frequency channels to use for Gaussian profile
    fc : float
       Center frequency of the line
    fwhm : float
       Full-width at half-maximum of the line
    fs : {float"
       Sampling frequency
    A : {float}
       Amplitude of the line
    N0 : {float}
        Noise power spectral density (dB).  If None, do not add noise.
    
    Returns
    -------
    out : ndarray
       The time series corresponding to the spectral line

    Notes
    -----
    The time series corresponding to a spectral line with a Gaussian
    profile is a continuous wave signal multiplied by the inverse
    Fourier transform of the Gaussian profile.
    """
    f = np.fft.fftfreq(nchan,1/fs)
    s = fwhm/(2*np.sqrt(2*np.log(2)))
    G = SNR*np.exp(-0.5*((f-fc)/s)**2)
    #X = np.random.normal(size=npts//2+1)# + 1j*np.random.normal(size=npts)
    g = np.fft.ifft(G)
    signal = np.ravel(np.array([np.roll(g,np.random.randint(low=0,high=len(g))) for ii in range(npts//nchan+1)]))[:npts]
    if N0 is not None:
        signal += noise(len(signal),N0)

    return signal
    

def symseq(nsym, tsym, nbits, smoothing_window=None):
    """
    Generate a random symbol sequence.

    Parameters
    ----------
    nsym : int
        Number of symbols to generate.
    tsym : int
        Duration of each symbol (samples).
    nbits : int
        Number of bits per datum. The size of the symbol set (i.e.,
        the number of points in the corresponding constellation
        diagram) is 2**nbits.
    smoothing_window : {str or tuple or array_like} 
        The windowing function used to smooth the symbol sequence.  If
        None, no smoothing is used and the pulse is rectangular.  If a
        string or a tuple is given it will be passed to
        scipy.signal.get_window and must be a valid input to that
        function.  If an array_like object is given it will be used
        directly as the window.
    
    Returns
    -------
    out : ndarray
        Random symbol sequence of length nsym*tsym
    """
    np.random.seed(_seed)
    symbols = np.random.randint(low=0,high=2**nbits,size=nsym)
    pulse = np.ones(tsym)
    symbol_sequence = np.kron(symbols,pulse)
    if smoothing_window is not None:
        if type(smoothing_window) in (str,tuple):
            window = get_window(*smoothing_window)
        else:
            window = smoothing_window.copy()
        window /= 0.5*len(window)
        symbol_sequence = convolve(symbol_sequence,window,mode="same")
    
    return symbol_sequence


def noise(npts,N0,bw=1.0):
    """
    Generate complex random Gaussian noise.  The real and imaginary
    parts are drawn separately.

    Parameters
    ----------
    npts : int
        Length of output array.
    N0 : float
        Noise power spectral density (dB/Hz).
    bw : {float}
        Bandwidth (Hz).
    
    Returns
    -------
    out : ndarray
        Complex Gaussian random noise of length npts.
    """
    np.random.seed(_seed)
    N0_linear = 10**(N0/10.0)
    x = np.random.randn(npts) + 1j*np.random.randn(npts)
    x *= np.sqrt(N0_linear*bw)/x.std()
    return x


