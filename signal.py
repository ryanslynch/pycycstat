"""
========================================================================
Signal generators for CSP (:mod 'pycycstat.signal')
========================================================================

Contents
--------
   rp_bpsk       Rectangular pulse binary phase shift keyed signal
   qpsk          Rectangular pulse quadrature phase shift keyed signal
   bfsk          Rectangular pulse binary frequency shift keyed signal
   bfsk_smooth   Hanning-smoothed binary frequency shift keyed signal
   ask_2bit      Hanning smoothed amplitude shift keyed signal with 4 levels
   noise         Add noise to signal with given amplitude. Used inside signal functions
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
    #if N0 is not None:
    #    x += noise(x,N0)

    return x



#quadrature phase shift keying
#same as binary phase shift, but information encoded is 2-bit, leading to 4 phases (45deg,135,225,315)
def qpsk(nbits,tbit,fc,Ebit=0.0,N0=None,fs=800e6):
    """
    Generate a rectangular pulse quadrature phase shift keyed signal.

    Parameters
    ----------
    nbits : int
        Number of bits to generate.
    tbit : int
        Bit duration (seconds).  The bit rate is 1/tbit.
    fc : float
        Carrier frequency (Hz)
    Ebit : float
        Energy per bit (dB).
    N0 : float
        Noise power spectral density (dB).  If None, do not add noise.
    fs : float
        Sampling frequency of ADC (Hz)
    
    Returns
    -------
    sig : 1d array
       An array of size nbit containing the complex rectangular pulse BPSK
       signal.
    sym_seq : 1d array
       An array of size nbit containing the symbol sequence used.

    Examples
    --------
    """
    #create random bit and symbol sequence
    sym_seq = np.random.randint(1,high=5,size = nbits)
    pulse = np.ones(tbit)
    sym_seq = np.kron(sym_seq,pulse)
    
    #apply carrier frequency
    ts=1/fs
    Ebit_linear = 10**(Ebit/10.0)
    x = np.arange(nbits*tbit)
    arg = 1.j*((2*np.pi*fc*x*ts) + (2*sym_seq-1)*(np.pi/4))
    sig = np.sqrt(Ebit_linear)*np.exp(arg)
    
    #if N0 is not None:
    #    sig += noise(sig,N0)
    
    return sig#,sym_seq
    
    
#===========================================
#Frequency-shift encoding
#===========================================


#binary freq-shift keying - switch between 2 freqs
def bfsk(nbits,tbit,f0,f1,Ebit=0.0,N0=None,fs=800e6):
    """
    Generate a rectangular pulse binary frequency shift keyed signal.
    NOTE: incomplete - carrier signal does not preserve phase between frequency shifts

    Parameters
    ----------
    nbits : int
        Number of bits to generate.
    tbit : int
        Bit duration (seconds).  The bit rate is 1/tbit.
    f0 : float
        Mark frequency - corresponds to symbol = 0 (Hz)
    f1 : float
        Space frequency - corresponds to symbol = 1 (Hz)
    Ebit : float
        Energy per bit (dB).
    N0 : float
        Noise power spectral density (dB).  If None, do not add noise.
    fs : float
        Sampling frequency of ADC (Hz)
    
    Returns
    -------
    sig : 1d array
       An array of size nbit containing the complex rectangular pulse BPSK
       signal.
    sym_seq : 1d array
       An array of size nbit containing the symbol sequence used.

    Examples
    --------
    """

    #make bit sequence
    bit_seq = np.random.randint(0,high=2,size=nbits)
    pulse = np.ones(tbit)
    sym_seq = np.kron(bit_seq,pulse)
    
    ts=1/fs
    Ebit_linear = 10**(Ebit/10.0)
    #apply to mark/space frequencies
    #need to make sure phase doesn't change
    #there is a vectorized way to do this im sure but im lazy
    # also this doesn't work
    sig = np.zeros(len(x),dtype=np.complex64)
    x = np.arange(nbits*tbit)
    phase = 0
    for i in range(nbits):
        if bit_seq[i] == 1:
            arg = 1.j*((2*np.pi*f0*x[i*tbit:(i+1)*tbit]*ts)+phase)
            sig[i*T_bit:(i+1)*tbit] = np.exp(arg)
            phase += fm*ts*tbit
        else:
            arg = 1.j*((2*np.pi*f1*x[i*tbit:(i+1)*tbit]*ts)+phase)
            sig[i*tbit:(i+1)*tbit] = np.exp(arg)
            phase += fm*ts*tbit
    sig *= np.sqrt(Ebit_linear)
    
    #if N0 is not None:
    #    sig += noise(sig,N0)

    return sig#,sym_seq


#binary freq-shift keying - switch between 2 freqs, with smoothing of symbol sequence
def bfsk_smoothed(nbits,tbit,f0,f1,Ebit=0.0,N0=None,fs=800e6):
    """
    Generate a rectangular pulse binary frequency shift keyed signal, with a hanning
    smoothing kernel applied to the symbol sequence so that frequency shifts are smooth.
    
    NOTE: incomplete - carrier signal does not preserve phase between frequency shifts

    Parameters
    ----------
    nbits : int
        Number of bits to generate.
    tbit : int
        Bit duration (seconds).  The bit rate is 1/tbit.
    f0 : float
        Mark frequency - corresponds to symbol = 0 (Hz)
    f1 : float
        Space frequency - corresponds to symbol = 1 (Hz)
    Ebit : float
        Energy per bit (dB).
    N0 : float
        Noise power spectral density (dB).  If None, do not add noise.
    fs : float
        Sampling frequency of ADC (Hz)
    
    Returns
    -------
    sig : 1d array
       An array of size nbit containing the complex rectangular pulse BPSK
       signal.
    sym_seq : 1d array
       An array of size nbit containing the symbol sequence used.

    Examples
    --------
    """
    #make bit sequence
    bit_seq = np.random.randint(0,high=2,size=nbits)
    pulse = np.ones(tbit)
    sym_seq = np.kron(bit_seq,pulse)

    #gaus = np.exp((-(x-(len(x)/2))**2)/(2*(tbit*0.1)**2))
    
    hann = np.hanning(int(tbit*0.1))
    sym_seq = np.convolve(sym_seq,hann,mode='same')
    sym_seq = sym_seq/np.max(sym_seq)
    
    #create frequency sequence
    f_diff = f0-f1
    freq_seq = f1 + (sym_seq)*f_diff

    #apply carrier signal
    ts=1/fs
    #Ebit_linear = 10**(Ebit/10.0)
    x = np.arange(nbits*tbit)
    arg = (2.j*np.pi*freq_seq*x*ts)
    sig = np.exp(arg)
    
    #sig *= np.sqrt(Ebit_linear)
    
    #if N0 is not None:
    #    sig += noise(sig,N0)

    return sig#,sym_seq,smoothed
    
    
    
    
    
#===========================================
#Amplitude-shift encoding
#===========================================

#4-level amplitude keying signal, with hanning smoothing
#bias: voltage bias
def ask_2bit(nbits,tbit,fc,Ebit=0.0,N0=None,fs=800e6,bias=0.5):
    """
    Generate a rectangular pulse binary 2-bit/4-level amplitude shift keyed signal, with a hanning
    smoothing kernel applied to the symbol sequence so that amplitude shifts are smooth.

    Parameters
    ----------
    nbits : int
        Number of bits to generate.
    tbit : int
        Bit duration (seconds).  The bit rate is 1/tbit.
    fc : float
        Carrier frequency (Hz)
    Ebit : float
        Energy per bit (dB).
    N0 : float
        Noise power spectral density (dB).  If None, do not add noise.
    fs : float
        Sampling frequency of ADC (Hz)
    bias : float
        symbol bias, so that symbol=00 doesn't have 0 voltage
    
    Returns
    -------
    sig : 1d array
       An array of size nbit containing the complex rectangular pulse BPSK
       signal.
    sym_seq : 1d array
       An array of size nbit containing the symbol sequence used.

    Examples
    --------
    """
    bit_seq = np.random.randint(0,5,size=nbits)+bias
    #plt.plot(bit_seq)
    pulse = np.ones(tbit)
    sym_seq = np.kron(bit_seq,pulse)

    #smooth by hanning window that is 20% the time width of one bit
    hann1 = np.hanning(int(tbit*0.2))
    #plt.plot(hann)
    sym_seq = np.convolve(sym_seq,hann1,mode='same')
    
    sym_seq = sym_seq / np.max(sym_seq)
    
    #apply carrier signal
    ts=1/fs
    Ebit_linear = 10**(Ebit/10.0)
    e_vec = np.exp(2.j*np.pi*fc*x*ts)
    
    sig = sym_seq * e_vec
    sig *= np.sqrt(Ebit_linear)
    
    #if N0 is not None:
    #    sig += noise(sig,N0)
    
    return sig#,sym_seq
    
    
    
    

    
def ask_1bit(nbits,tbit,fc,Ebit,N0=None,fs=800e6,bias=0.71):
    """
    Generate a rectangular pulse binary 1-bit/2-level amplitude shift keyed signal, with a hanning
    smoothing kernel applied to the symbol sequence so that amplitude shifts are smooth.
    
    Parameters
    ----------
    nbits : int
        Number of bits to generate.
    tbit : int
        Bit duration (seconds).  The bit rate is 1/tbit.
    fc : float
        Carrier frequency (Hz)
    Ebit : float
        Energy per bit (dB).
    N0 : float
        Noise power spectral density (dB).  If None, do not add noise.
    fs : float
        Sampling frequency of ADC (Hz)
    bias : float
        symbol bias, so that symbol=00 doesn't have 0 voltage

    Returns
    -------
    sig : 1d array
    An array of size nbit containing the complex rectangular pulse BPSK
    signal.
    """
    x = np.arange(nbits*tbit)
    bit_seq = np.random.randint(0,2,size=(int(len(x)/T_bit)+1,))+bias
    pulse = np.ones(T_bit)
    sym_seq = np.kron(bit_seq,pulse)[:len(x)]

    #smooth by hanning window that is 20% the time width of one bit
    hann1 = np.hanning(int(T_bit*0.2))
    #plt.plot(hann)
    sym_seq = np.convolve(sym_seq,hann1,mode='same')

    sym_seq = sym_seq / np.max(sym_seq)

    #take away smoothing at edges
    win_size = int(T_bit*0.2)
    sym_seq[:win_size] = sym_seq[win_size+1]
    sym_seq[-(win_size):] = sym_seq[-(win_size+1)]

    #apply carrier signal
    e_vec = np.exp(2.j*np.pi*fc*x*ts)
    #e_vec = np.exp(2.j*np.pi * fs/f_sim * x)
    
    sig = sym_seq * e_vec

    return sig#,sym_seq,bit_seq



def noise(x,N0):
    """
    Generate noise for other signal functions

    Parameters
    ----------
    x : 1d array
        input signal array
    N0 : float
        Noise power (dB)
    
    Returns
    -------
    n : 1d array
        output noise array, of same size as input signal

    Examples
    --------
    """
    N0_linear = 10**(N0/10.0)
    n = np.random.randn(len(x)) + 1j*np.random.randn(len(x))
    n *= np.sqrt(N0_linear/np.var(n))
    return n


