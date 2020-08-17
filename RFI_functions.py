


import numpy as np
import matplotlib.pyplot as plt


#known variables

#sampling frequency
fs = 800e6
ts = 1./fs





#create square wave (for duty cycle)
def sq(x,dc,dc_per):
        y = np.zeros(len(x))
        for i in range(len(x)//dc_per):
            start = int(i*dc_per)
            end = int(start + dc*dc_per)
            #y[start:end] = 1
            y[start:end] = np.hamming(end-start)
        return y
        
    
    
#dc = duty cycle
#dc_period = duty cycle period (in samples)
def RFI_sig(x,amp,freq,phase,dc,dc_period):
        #create signal
        sig = amp*np.exp(2.j*np.pi*x*freq*ts)
        #apply duty cycle
        sig = sig * sq(x,dc,dc_period)
        return sig



#bw = 800e6
#fs = bw
#ts = 1/fs



#Nchan = 4096
#Nint = 40
#N=Nchan*Nint
#x = np.arange(N)

#m = x*fs/N


#noise

#noise_power = np.var(n_of_t)

#N0_linear = 10**(N0_dB/10)
#pow_factor = np.sqrt(N0_linear / noise_power)
#n_of_t = n_of_t*pow_factor





#quadrature phase shift keying
#same as binary phase shift, but information encoded is 2-bit, leading to 4 phases (45deg,135,225,315)
def qpsk(x,dB,fc,T_bit):
    sym_seq = np.random.randint(1,high=5,size = int(len(x)/T_bit)+1)
    pulse = np.ones(T_bit)
    sym_seq = np.kron(sym_seq,pulse)[:len(x)]
    
    arg = 1.j*((2*np.pi*fc*x*ts) + (2*sym_seq-1)*(np.pi/4))
    sig = np.exp(arg)
    
    return sig



#binary freq-shift keying - switch between 2 freqs
#fs : space frequency (bit = 0)
#fm : mark frequency (bit = 1)
def bfsk(x,dB,fs,fm,T_bit):
    #make bit sequence
    bit_seq = np.random.randint(0,high=2,size=int(len(x)/T_bit))
    pulse = np.ones(T_bit)
    sym_seq = np.kron(bit_seq,pulse)
    
    #apply to mark/space frequencies
    #need to make sure phase doesn't change
    #there is a vectorized way to do this im sure but im lazy
    sig = np.zeros(len(x),dtype=np.complex64)
    phase = 0
    for i in range(int(len(x)/T_bit)):
        if bit_seq[i] == 1:
            arg = 1.j*((2*np.pi*fm*x[i*T_bit:(i+1)*T_bit]*ts)+phase)
            sig[i*T_bit:(i+1)*T_bit] = np.exp(arg)
            phase += fm*ts*T_bit
        else:
            arg = 1.j*((2*np.pi*fs*x[i*T_bit:(i+1)*T_bit]*ts)+phase)
            sig[i*T_bit:(i+1)*T_bit] = np.exp(arg)
            phase += fm*ts*T_bit

    return sig





#binary freq-shift keying - switch between 2 freqs, with gaussian smoothing of symbol sequence
#fs : space frequency (bit = 0) (lower frequency)
#fm : mark frequency (bit = 1) (higher frequency) 
def bfsk_smoothed(x,dB,fs,fm,T_bit):
    #make bit sequence
    bit_seq = np.random.randint(0,high=2,size=int(len(x)/T_bit)+1)
    pulse = np.ones(T_bit)
    sym_seq = np.kron(bit_seq,pulse)[:len(x)]

    #gaus = np.exp((-(x-(len(x)/2))**2)/(2*(T_bit*0.1)**2))
    
    hann = np.hanning(int(T_bit*0.1))
    smoothed = np.convolve(sym_seq,hann,mode='same')
    smoothed = smoothed/np.max(smoothed)
    
    f_diff = fm-fs
    freq_seq = fs + (sym_seq)*f_diff
    
    #apply to mark/space frequencies
    #need to make sure phase doesn't change
    #there is a vectorized way to do this im sure but im lazy

    arg = (2.j*np.pi*freq_seq*x*ts)
    sig = np.exp(arg)


    return sig
        





# In[281]:

#4-level amplitude keying signal, with hanning smoothing
#bias: voltage bias
def ask_2bit(x,dB,fc,T_bit,bias):
    bit_seq = np.random.randint(0,5,size=int(len(x)/T_bit)+1)+bias
    #plt.plot(bit_seq)
    pulse = np.ones(T_bit)
    sym_seq = np.kron(bit_seq,pulse)[:len(x)]
    sym_seq = sym_seq / np.max(sym_seq)
    
    #smooth by hanning window that is 10% the time width of one bit
    hann = np.hanning(int(T_bit*0.2))
    #plt.plot(hann)
    sym_seq = np.convolve(sym_seq,hann,mode='same')
    
    #apply carrier signal
    e_vec = np.exp(2.j*np.pi*fc*x*ts)
    sig = sym_seq * e_vec
    
    return sig


def ask_1bit(x,dB,fc,T_bit,bias):
    print('Creating random bit sequence...')
    bit_seq = np.random.randint(0,2,size=int(len(x)/T_bit)+1)+bias
    print('Creating symbol sequence from bits...')
    pulse = np.ones(T_bit)
    sym_seq = np.kron(bit_seq,pulse)[:len(x)]

    print('Smoothing symbol sequence...')
    #smooth by hanning window that is 20% the time width of one bit
    hann1 = np.hanning(int(T_bit*0.2))
    #plt.plot(hann)
    sym_seq = np.convolve(sym_seq,hann1,mode='same')
    
    sym_seq = sym_seq / np.max(sym_seq)
    
    #take away smoothing at edges
    win_size = int(T_bit*0.2)
    sym_seq[:win_size] = sym_seq[win_size+1]
    sym_seq[-(win_size):] = sym_seq[-(win_size+1)]
    
    print('Applying to carrier signal...')
    #apply carrier signal
    e_vec = np.exp(2.j*np.pi*fc*x*ts)
    #e_vec = np.exp(2.j*np.pi * fs/f_sim * x)
    
    sig = sym_seq * e_vec
    
    return sig,sym_seq,bit_seq





