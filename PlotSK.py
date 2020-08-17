
"""
Automatically plot and save results of simulated SK flagging

Usage:
$ python PlotSk.py figname

figname : str
    base filename to save figures under
    
This program loads in the averaged dataset from Simulate.py
(averaged every m spectra), along with the flag mask array and plots:

1) Flagged and Unflagged total average spectrum centered on the
RFI frequency band 1610-1630MHz
2) Unflagged image centered on the RFI frequency band 1610-1630MHz
3) Flagged image centered on the RFI frequency band 1610-1630MHz

    
And saves those figures using the base filename provided. At the end, the program
gives an attempted measure of RFI flagging efficacy by:
- Using a clean portion to determine noise
- Sets a noise + 3-sigma mask to determine where RFI is
- Compares this mask of to the SK flagging mask

Then outputs three percentages:
- % of data under the 3-sigma mask
- % of that data that is also flagged by SK
- false positive rate: % of data flagged by SK that is not flagged by 3-sigma mask

"""

import numpy as np
import matplotlib.pyplot as plt

import sys

figname = sys.argv[1]


scr_dir = './'

#load data + mask
p = np.load(scr_dir+'p.npy')
flags = np.load(scr_dir+'flags.npy')

#set 0's to 1e-3 for logscale
p[p==0]=1e-3

#applies SK mask
p_f = np.array(p)
p_f[flags==1]=1e-3




#define frequency channels (NOT the same as the SK m)
m = ((np.arange(4096)*800e6/4096)/1e6)+1100

#=======================================
#plot total average flagged/unflagged spectrum centered on Iridium

plt.plot(m,np.average(p,axis=1),'r',label='Unflagged')
plt.plot(m,np.average(p_f,axis=1),'b',label='Flagged')
plt.yscale('log')
plt.xlabel('Frequency (MHz)')
plt.xlim((1610,1630))
plt.ylim((140,300))
plt.legend()
plt.savefig(figname+'_spect.png')
plt.show()



#=======================================
#plot unflagged image centered on Iridium band
#plot flagged image centered on Iridium band

ext = [0,200,1900,1100]


plt.imshow(np.log10(p),aspect='auto',interpolation='nearest',cmap='hot',extent=ext,vmin=2.0,vmax=3.0)

plt.ylim((1610,1630))
plt.xlabel('Time (2.62ms)')
plt.ylabel('Frequency (MHz)')
plt.colorbar()

plt.savefig(figname+'_im_unflagged.png')
plt.show()
#=======================================
#plot flagged image centered on Iridium band

plt.imshow(np.log10(p_f),aspect='auto',interpolation='nearest',cmap='hot',extent=ext,vmin=2.0,vmax=3.0)
plt.ylim((1610,1630))
plt.xlabel('Time (2.62ms)')
plt.ylabel('Frequency (MHz)')
plt.colorbar()

plt.savefig(figname+'_im_flagged.png')
plt.show()



#=======================================
#RFI occupancy and flagging %

#rough occupancy counter by whatever power values are higher than mean+3*std
#(clean spectrum)
mu = np.mean(p[100:1400,:])
sig = np.std(p[100:1400,:])


RFI_p = np.zeros(p.shape)
RFI_p[p > (mu + 3*sig)] = 1

rfi_occ = (100.*np.count_nonzero(RFI_p))/RFI_p.size

#dirty_flag_pct
p_dirty = np.array(RFI_p)
p_dirty[flags==0]=0

dirty_flag_pct = (100.*np.count_nonzero(p_dirty))/np.count_nonzero(RFI_p)

#false positive rate
p_clean = -np.array(RFI_p)+1
p_clean[flags==0]=0

false_pos_rate = (100.*np.count_nonzero(p_clean))/np.count_nonzero(-RFI_p+1)

print('RFI Occupancy: {}%'.format(rfi_occ))
print('RFI Flagged: {}%'.format(dirty_flag_pct))
print('False positive rate: {}%'.format(false_pos_rate))










