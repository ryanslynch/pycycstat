
# coding: utf-8
"""
Simulate RFI to get ground truth measurements for comparing various RFI mitigation strats


Usage:
$ python Simulate.py new_data m

new_data : 0 or 1 (boolean)
    to either create new data set or load the previous

m : int
    Number of spectra that the SK algorithm looks at at once. This is M in the SK estimator eqn.

"""





import numpy as np
import scipy as sp
#import scipy.signal
import matplotlib.pyplot as plt
import sys


import math

sys.path.append('./')

#from SK_in_Python import *
#from simhelp import *
#from RFI_functions import *

from signalRFI import *


#get_ipython().magic('matplotlib inline')
#get_ipython().magic('matplotlib qt')






#parameters of data-taking
#simulate r0800x4096 observations

#directory to save results to
scr_dir = './'



#create new data (1) or reload current set (0)
new_data = int(sys.argv[1])

m = int(sys.argv[2])

bw = 800e6 #Hz, bandwidth
Nchan = 4096 #number of chans

#sampling frequency
fs = bw
ts = 1/fs

LO = (1.500097656)*1e9 #center frequency to tune to

print('Native time resolution: {} microsec'.format((Nchan/bw)*1e6))



n = 1
#m = 2048

d = 1

SK_int = 1

#number of integrations
Nint = m*SK_int

#number of time samples
Nsamp = Nint*Nchan

#noise power
N_dB = -10

#data rate
T_bit = int(Nchan*40)
#Nchan*10 is 20 kbps, given 800MHz sample rate, 4096 channels

#freq channels
freqs = np.arange(Nchan)*fs/Nchan


#=====================================================
# In[130]:


if new_data:

	#simulate signals

	print('Generating Signals...')
	sig = np.zeros(Nsamp)
	x = np.arange(Nsamp)

	mset = int(Nchan*m) #number of samples in one accumulated spectrum

	#make the signal from the receiver to 'tune'
	signal_loI = np.cos(2.0*np.pi * LO * x * ts)
	signal_loQ = np.sin(2.0*np.pi * LO * x * ts)


	#add = RFI_sig(x,0.5,500e6,0,0.9,20000)
	#add[:mset*10]=0
	#add[mset*100:]=0
	#sig = sig + add

	#add = RFI_sig(x,0.5,520e6,0,0.8,20000)
	#add[mset*20:mset*80]=0
	#sig = sig + add

	#add = RFI_sig(x,0.5,540e6,0,0.7,20000)
	#sig = sig + add

	#add = RFI_sig(x,0.5,580e6,0,0.6,20000)
	#sig = sig + add

	#print('Sine waves...')
	#add = RFI_sig(x,0.5,600e6,0,0.5,20000)
	#sig = sig + add


	#print('QPSK....')
	#add = qpsk(x,0,70e6,T_bit)
	#sig = sig + add

	#print('BFSK...')
	#add = bfsk(x,0,140e6,150e6,T_bit)
	#sig = sig + add


	print('Smoothed BFSK....')
	add = bfsk_smoothed(len(x),T_bit,1.622e9,1.623e9)
	sig = sig + add*0.5


	#print('2bit ASK....')
	#add = ask_2bit(x,0,700e6,T_bit,0.5)
	#sig = sig + add
	


	print('1bit ASK....')
	add,sym,bit = ask_1bit(len(x),T_bit,1.621e9,0,0)

	#data *= np.flip(np.arange(0,0.25,0.25/Nsamp))
	add *= 0.5
	sig = sig+add
	data = sig

	#mix I and Q components
	bb_I = data.real * signal_loI
	bb_Q = data.imag * signal_loQ


	bb_I = np.reshape(bb_I,(Nint,Nchan))
	bb_Q = np.reshape(bb_Q,(Nint,Nchan))

	print('Filtering baseband signal...')
	#low-pass and pos freq filter (perfect square in fourier space)
	N_cutoff = int(600)
	#600 for iridium signal at 1626MHz
	signals = [bb_I, bb_Q]
	filtered_signals = []


	for signal in signals:
		fsignal = np.fft.fft(signal,axis=1)
		fsignal[:,:N_cutoff][fsignal[:,:N_cutoff]>1] = 1
		fsignal[:,int(4096/2):][fsignal[:,int(4096/2):]>1] = 1
		fsignal[:,:N_cutoff][fsignal[:,:N_cutoff]<-1] = 1
		fsignal[:,int(4096/2):][fsignal[:,int(4096/2):]<-1] = 1
		filtered_signals.append(np.fft.ifft(fsignal,axis=1))



    #combine filtered I/Q components to make a baseband signal



	bb = filtered_signals[0] + 1.j*filtered_signals[1]

	print('Adding noise...')

	N0_dB = -10


	#adding noise after mixing down because lazy

	n_of_t = np.random.normal(0,1,size=Nsamp) + 1.j*np.random.normal(0,1,size=Nsamp)
	noise_power = np.var(n_of_t)

	N0_linear = 10**(N0_dB/10)
	pow_factor = np.sqrt(N0_linear / noise_power)
	n_of_t = n_of_t*pow_factor

	n_of_t = np.reshape(n_of_t,(Nint,Nchan))

	bb += n_of_t



	#window?
	hann = np.hanning(Nchan)
	bb *= hann




	#generating science signal
	#print('Generating science signal')
	#sci = sci_sig(fb,60,20,600)
	#np.save('sci.npy',sci)
	#sci = np.load('sci.npy')
	
	fb = np.fft.fft(bb,axis=1)# + sci

	fb = np.fft.fftshift(fb,axes=1)

	
	np.save(scr_dir+'fb.npy',fb)

else:
	print('Loading data...')
	fb = np.load(scr_dir+'fb.npy')
	Nint = fb.shape[0]#in case using old data with new m value



s = np.abs(fb)**2
s = s.T
print(s.shape)

#plt.imshow(np.log10(s),interpolation='nearest',aspect='auto',cmap='hot',vmin=2.5,vmax=4.5)
#plt.colorbar()
#plt.show()




"""
Data creation/loading done
Everything after this point is applying SK
"""




print('Applying SK...')


SK_timebins = Nint//m
print('SK_timebins: {}'.format(SK_timebins))

for i in range(SK_timebins):
	s_chunk = s[:,i*m:(i+1)*m]
	this_sk = SK_EST(s_chunk,n,m,d)
	if i==0:
		sk = this_sk
		accum_p = np.average(s_chunk,axis=1)
	else:
		sk = np.c_[sk,this_sk]
		accum_p = np.c_[accum_p,np.average(s_chunk,axis=1)]

	




np.save(scr_dir+'sk.npy',sk)
#p: accumulated power (m spectra at a time)
np.save(scr_dir+'p.npy',accum_p)

print('Applying flags...')

sigma = 3.0
SK_p = (1-scipy.special.erf(sigma/math.sqrt(2))) / 2
lt, ut = SK_thresholds(m, N = n, d = d, p = SK_p)
print('Upper Threshold: '+str(ut))
print('Lower Threshold: '+str(lt))

accum_p[sk>ut] = 1e-3
accum_p[sk<lt] = 1e-3

np.save(scr_dir+'p_flagged.npy',accum_p)

flags = np.zeros(accum_p.shape)
flags[sk>ut] = 1
flags[sk<lt] = 1
np.save(scr_dir+'flags.npy',flags)

tot_pts = flags.size
count = np.float64(np.count_nonzero(flags))
pct = 100*count/tot_pts
print('Percentage of data flagged: {}'.format(pct))

print('Program Done')














