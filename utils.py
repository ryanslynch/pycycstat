"""
=========================================
Utility functions (:mod 'pycycstat.utils')
=========================================

Contents
--------
   dB              Linear to decibel conversion
   shift_params    Parameters for discrete cycle frequency shifts
   discrete_shift  Rotate an array
   top_scf_values  Return indices of top SCF values
   top_cycle_freqs Unique cycle frequencies of top SCF values
   surfaceplot     Plot a 3-D surface
   plot_top_scf    Plot a 3-D surface for the top SCF values
"""
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import stats
from scipy.integrate import cumtrapz


def dB(x, type="power"):
    """
    Convert a value to decibels.

    Parameters
    ----------
    x : array_like
       Input value.
    type : {'power', 'root-power', 'field'}
       Type of conversion.

    Returns
    -------
    out : ndarray
       An array of the same shape as x containing the values in dB

    Notes
    -----
    If type is 'power' return 10*np.log10(x).  If type is 'root-power' 
    or 'field' return 20*np.log10(x).
    

    Examples
    --------
    >> dB(2)
    3.0102999566398121

    >> dB(2, type='root-power')
    6.0205999132796242

    >> dB([10, 100, 1000])
    array([  0.,  10.,  20.,  30.])
    """

    assert type in ['power','root-power','field'], ("Type must be 'power, "
                                                    "'root-power', or 'field'")

    return 10.0*np.log10(x) if type == "power" else 20.0*np.log10(x)


def vco(Vin,f0,K0,tbit,fs=1.0):
    """
    Implementation of a voltage-controlled oscillator.

    Parameters
    ----------
    Vin : array_like
       Input voltages.
    f0 : float
       Quiescent oscilator frequency (Hz).
    K0 : float
       Oscillator gain (Hz/V)
    tbit : int
       Bit duration (samples).
    fs : {float}
       Sampling frequency (Hz)

    Returns
    -------
    out : ndarray
       Output voltages.
    """
    Vin = np.array(Vin)
    nbits = len(Vin)//tbit
    x = np.tile(np.arange(tbit),nbits)
    f = f0 + Vin*K0
    phase = cumtrapz(f,dx=1/fs,initial=0)
    e_vec = 1j*(2*np.pi*x*f0/fs + phase)
    Vout = np.exp(e_vec)
    
    return Vout

def pfb(x, nchan, ntap, window="hann", fs=1.0, return_freqs=False, 
        force_complex=False):
    """
    Channelize data using a polyphase filterbank

    Parameters
    ----------
    x : ndarray
       The input time series.
    nchan : int
       The number of channels to form.
    ntap : int
       The number of PFB taps to use.
    window : str
       The windowing function to use for the PFB coefficients.
    fs : float
       The sampling frequency of the input data.
    return_freqs : bool
       If True, return the center frequency of each channel.
    force_complex : bool
       If True, treat input as complex even if the imaginary component is zero.

    Returns
    -------
    x_pfb : ndarray
       The channelized data
    freqs : ndarray
       The center frequency of each channel.  Omitted if 
       return_freqs == False
    
    Notes
    -----
    If the input data are real-valued then only positive frequencies
    are returned.
    """
    real = np.isreal(x).all() and not force_complex
    h = signal.firwin(ntap*nchan,cutoff=1.0/nchan,window="rectangular")
    h *= signal.get_window(window,ntap*nchan)
    nwin = x.shape[0]//ntap//nchan
    x = x[:nwin*ntap*nchan].reshape((nwin*ntap,nchan)).T
    h = h.reshape((ntap,nchan)).T
    xs = np.zeros((nchan,ntap*(nwin-1)+1),dtype=x.dtype)
    for ii in range(ntap*(nwin-1)+1):
        xw = h*x[:,ii:ii+ntap]
        xs[:,ii] = xw.sum(axis=1)
    xs = xs.T
    xpfb = np.fft.rfft(xs,nchan,axis=1) if real else np.fft.fft(xs,nchan,axis=1)
    xpfb *= np.sqrt(nchan)

    if return_freqs:
        freqs = np.fft.rfftfreq(nchan,d=1.0/fs) if real else \
                np.fft.fftfreq(nchan,d=1.0/fs)
        return xpfb,freqs
    else:
        return xpfb
    


def shift_params(cycle_freqs,npts,tolerance=1e-5):
    """
    Determine optimal parameters for discrete frequency shifts.

    Parameters
    ----------
    cycle_freqs : ndarray
       A list of cycle frequencies
    npts : int
       Nominal number of points in the planned FFT
    tolerance : {float}
       Acceptable difference between true and discrete cycle frequencies
    
    Returns
    -------
    nshift : int
       Number of frequency bins to shift
    npad : int
       Number of points to zero-pad to achieve the necessary
       frequency resolution.  The total FFT size should be
       npts+npad
    """
    npad = 0
    fs = np.fft.fftfreq(npts)
    while max([min(abs(0.5*cf-fs)) for cf in cycle_freqs]) > tolerance:
        npad += 2
        npts += 2
        fs = np.fft.fftfreq(npts)
    nshifts1 = [int(round(0.5*cf*npts)) for cf in cycle_freqs]
    nshifts2 = [int(round(-0.5*cf*npts)) for cf in cycle_freqs]
    return nshifts1,nshifts2,npad


def discrete_shift(x,n,circular=False,axis=None):
    """
    Shift an array by a discrete number of frequency bins.  If n>0
    then x is shifted to the left (this is the opposite of numpy.roll).

    Parameters
    ----------
    x : ndarray
       Input data
    n : int
       Number of points by which to shift x.  If n is positive shift
       to the left; if n is negative shift to the right.
    circular : {bool}
       If True, execute a circular shift where values from the 
       endpoints of x are shifted to the opposite end of the 
       array.  If False, replace any out-of-bounds values with
       zeros.
    axis : {int} or tuple of {ints}
       Axis or axes along which elements are shifted.  By default, the
       array is falttened before shifting, after which the 
       original shape is restored.

    Returns
    -------
    out : ndarray
       The shifted version of x
    """
    x = np.roll(x,-n,axis=axis)
    if circular or n == 0:
        return x
    elif not circular and n > 0:
        x[-n:] = 0
        return x
    elif not circular and n < 0:
        x[:-n] = 0
        return x


def top_scf_values(scf, n, magnitude=True):
    """
    Return the indices of the top values of the spectral correlation
    or coherence.

    Parameters
    ----------
    scf : ndarray
       The spectral correlation or coherence function.
    n : int
       The number of values to return.
    magnitude : {bool}
       If True, use the absolute magnitude of the SCF; if False
       use the complex value.

    Returns
    ------
    out : list
       The unique cycle frequencies corresponding to the top 'n'
       SCF values.

    Notes
    -----
    numpy.argsort is used to sort the SCF values.  It uses a
    lexographic sort for complex values, such that the order is
    determined by the real part first, and the imaginary part is only
    used when the real parts of two or more numbers are equal.
    """
    if magnitude:
        return np.unravel_index(np.argsort(np.abs(scf.ravel()))[-n:],scf.shape)
    else:
        return np.unravel_index(no.argsort(scf.ravel())[-n:],scf.shape)

def top_cycle_freqs(scf, cfs, n, magnitude=True):
    """
    Return the unique cycle frequencies corresponding to the top 'n'
    values of the spectral correlation or coherence.

    Parameters
    ----------
    scf : ndarray
       The spectral correlation or coherence function.
    cfs : ndarray
       The cycle frequencies.
    n : int
       The number of values to return.
    magnitude : {bool}
       If True, use the absolute magnitude of the SCF; if False
       use the complex value.

    Returns
    ------
    out : ndarray
       The unique cycle frequencies corresponding to the top 'n'
       SCF values.

    Notes
    -----
    numpy.argsort is used to sort the SCF values.  It uses a
    lexographic sort for complex values, such that the order is
    determined by the real part first, and the imaginary part is only
    used when the real parts of two or more numbers are equal.
    """
    assert scf.shape == cfs.shape,"scf and cfs must have the same shape"
    idx = top_scf_values(scf, n, magnitude=magnitude)

    return list(set(cfs[idx]))


def surfaceplot(x, y, z, alpha=0.8, facecolors=plt.cm.viridis_r,
                edgecolors="black", figure=None, figsize=None):
    """
    Plot a 3-D surface.

    Parameters
    ----------
    x,y,z : ndarray
       The coordinates (x,y) and values (z) to plot.  z must be of 
       shape (len(x),len(y)).
    alpha : {float}
       The alpha value used when plotting 
    facecolors : {matplotlib color or colormap}
       The face colors.  This can be a constant value or a colormap.
       It must be recognized by matplotlib.
    edgecolors : str
       The edge colors.  This can be a constant value or a color map.
       It must be recognized by matplotlib.
    figure : {matplotlib.figure.Figure}
       An existing matplotlib figure to add a subplot to.  If None, 
       create a new figure
    figsize : {tuple}
       Figure size in inches.  If None, use matplotlib defaults.
       This will be ignore if figure is not None.

    Returns
    -------
    fig : matplotlib.figure.Figure
       An instance of a matplotlib figure
    ax : matplotlib.axes._subplots.Axes3DSubplot
       An instance of a matplotlib 3-D subplot

    Notes
    -----
       This recipe is adapted from https://matplotlib.org/3.1.1/gallery/mplot3d/polys3d.html#sphx-glr-gallery-mplot3d-polys3d-py

    Examples
    --------
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.collections import PolyCollection
    
    def _polygon_under_graph(xlist,ylist):
        return [(xlist[0],0)]+[xy for xy in zip(xlist,ylist)]+[(xlist[-1],0)]

    if figure is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1,projection="3d")
    else:
        fig = figure
        naxes = len(figure.axes)
        for ii in range(naxes): fig.axes[ii].change_geometry(1,naxes+1,ii+1)
        ax = fig.add_subplot(1,naxes+1,naxes+1,projection="3d")

    verts = []
    for ii in range(len(y)):
        yval = z[ii]
        verts.append(_polygon_under_graph(x,yval))
    print(verts[0])
    poly = PolyCollection(verts, alpha=alpha, edgecolors=edgecolors,
                          facecolors=facecolors)
    ax.add_collection3d(poly, zs=y, zdir="y")
    ax.set_xlim(min(x),max(x))
    ax.set_ylim(min(y),max(y))
    ax.set_zlim(min(z.ravel()),max(z.ravel()))

    return fig,ax


def plot_top_scf(fs, cfs, scf, n, coherence=None, magnitude=True, 
                 conjugate=False, units="Hz", show=False, figure=None, 
                 figsize=None):
    """
    Plot the spectral correlation or coherence function on 
    the frequency vs cycle frequency bi-plane for the top 
    'n' SCF values using some default plotting parameters.

    Parameters
    ----------
    fs : ndarray
       The spectral frequencies.
    cfs : ndarray
       The cycle frequencies.
    scf : ndarray
       The spectral correlation or coherence values.
    n : int
       Plot the SCF at cycle frequencies corresponding to the 
       top 'n' SCF values.
    coherence : {ndarray}
       If the spectral coherence is explicitly specified, use this to 
       find the top cycle frequencies, but still plot scf.
    magnitude : {bool}
       If True, use the absolute magnitude of the SCF; if False
       use the complex value.
    conjugate : {bool}
       If True, plot cycle frequencies on the interval [-1,1] as
       appropriate for the conjugate SCF.  If False, plot cycle
       frequencies on the interval [0,1] as appropriate for the
       non-conjugate SCF.
    show : {bool}
       If True, execute matplotlib.pyplot.show() and return None.  Otherwise,
       return the Figure object.
    fig : {matplotlib.figure}
       If provided, add axes to an existing figure.  Otherwise, make a new one.
    figsize : {tuple}
       If provided, create a figure with the existing size.  If 'figure' is not
       None, then the existing figure's size will be adjusted.
    
    Returns
    -------
    matplotlib.pyplot.figure instance or None

    Notes
    -----
    numpy.argsort is used to sort the SCF values.  It uses a
    lexographic sort for complex values, such that the order is
    determined by the real part first, and the imaginary part is only
    used when the real parts of two or more numbers are equal.
    """
    if not conjugate:
        idx0 = np.where(cfs >= 0)
        fs = fs[idx0].reshape((fs.shape[0]-1,fs.shape[1]//2))
        cfs = cfs[idx0].reshape((cfs.shape[0]-1,cfs.shape[1]//2))
        scf = scf[idx0].reshape((scf.shape[0]-1,scf.shape[1]//2))
        if coherence is not None:
            coherence = coherence[idx0].reshape((coherence.shape[0]-1,
                                                 coherence.shape[1]//2))
        else: 
            pass
    if coherence is not None:
        top_cfs = top_cycle_freqs(coherence, cfs, n, magnitude=magnitude)
    else:
        top_cfs = top_cycle_freqs(scf, cfs, n, magnitude=magnitude)
    verts = []
    for cf in top_cfs:
        fs_to_plot = fs[cfs == cf]
        args = np.argsort(fs_to_plot)
        scfs_to_plot = np.abs(scf[cfs == cf])
        fs_to_plot = fs_to_plot[args]
        scfs_to_plot = scfs_to_plot[args]
        verts.append(
            [(fs_to_plot[0],0)] + \
            [xy for xy in zip(fs_to_plot,scfs_to_plot)] + \
            [(fs_to_plot[-1],0)])
    if figure is None:
        figure = plt.figure(figsize=figsize) if figsize is not None else plt.figure()
        ax = figure.gca(projection="3d",proj_type="ortho")
    else:
        naxes = len(figure.axes)
        ax=figure.add_subplot(
            naxes+1,1,naxes+1,projection="3d",proj_type="ortho")
        if figsize is not None: figure.set_size_inches(**figsize)
        
    poly_collection = PolyCollection(verts,alpha=0.8,edgecolors="k",
                                     facecolors="gray")
    ax.add_collection3d(poly_collection,zs=top_cfs,zdir="y")
    ax.set_xlim(fs.min(),fs.max())
    if conjugate:
        ax.set_ylim(cfs.max(),cfs.min())
    else:
        ax.set_ylim(cfs.max(),0)
    ax.set_zlim(0,1.15*np.abs(scf).max())
    ax.locator_params(nbins=5)
    ax.set_xlabel(r"Spectral Frequency ($\nu$; %s)"%units)
    ax.set_ylabel(r"Cycle Frequency ($\alpha$; %s)"%units)
    ax.set_zlabel("SCF (Magnitude)")
    #ax.invert_yaxis()
    if show:
        plt.show()
        return None
    else:
        return (figure,ax)

def flag_cs(x, nchan, nhop, fsamp, psd=None, fsm_window_size=256,
            conjugate=False, coherence=True, threshold=3):

    from pycycstat.correlators import ssca

    assert nchan in (32,64,128,256), "Unsupported value of nchan"
    assert len(x) in (4096,8192,16384,32768), "Unsupported input array length"
    
    if coherence:
        expected_params = {
            (4096,64): (654.6579387286482,3.5905962805952334)
            }
    else:
        expected_params = {
            (4096,32): (0.6934484348596286,0.010893650247280331),
            (8192,32): (0.6929309406742655,0.007714571376986666),
            (16384,32): (0.6924660061803586,0.005439735175162053),
            (32768,32): (0.6920923851886733,0.0038472345545220955),
            (4096,64): (0.6919309443620992,0.010893650247280331),
            (8192,64): (0.6919665849625346,0.007714571376986666),
            (16384,64): (0.6918328552571094,0.005439735175162053),
            (32768,64): (0.6916647305961944,0.0038472345545220955),
            (4096,128): (0.6899798235234413,0.010893650247280331),
            (8192,128): (0.6908462516642049,0.007714571376986666),
            (16384,128): (0.6911695554868199,0.005439735175162053),
            (32768,128): (0.6912597239091195,0.0038472345545220955),
            (4096,256): (0.6869570549645402,0.010893650247280331),
            (8192,256): (0.689224759019894,0.007714571376986666),
            (16384,256): (0.6902832207239482,0.005439735175162053),
            (32768,256): (0.6907632985449913,0.0038472345545220955),
    }

    npts = len(x)
    expected_amp = expected_params[(npts,nchan)][0]
    expected_std = expected_params[(npts,nchan)][1]
    #x -= x.mean()
    #x /= x.std()
    fs,cfs,scf = ssca(x,nchan,nhop,fsamp,window="hann",psd=psd,
                      fsm_window_size=fsm_window_size,conjugate=conjugate,
                      output="scf")
    #scf *= np.sqrt(nchan/len(x)**3)
    scf *= np.sqrt(nchan**3/len(x))
    amp = np.sum(np.abs(scf))/len(scf.ravel())
    if np.abs(amp-expected_amp)/expected_std > threshold:
        return True
    else:
        return False
    


def flag_cs_cu(x, nchan, nhop, fsamp, psd=None, fsm_window_size=256,
               conjugate=False, coherence=True, threshold=0.95):

    from pycycstat.correlators import ssca_cu

    assert nchan in (8,15,32,64,128,256,1024,2048,4096), "Unsupported value of nchan ({})".format(nchan)
    assert len(x) in (32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536), "Unsupported input array length ({})".format(len(x))
        
    if coherence:
        if not conjugate:
            expected_params = {
                (32, 8): (1.3327438591172636,0.21202843634959007),
                (32, 16): (1.4297366975599504,0.2295741052813927),
                (32, 32): (1.5702784586531502,0.2748628470674266),
                (64, 8): (1.4600752259867147,0.1984575235682888),
                (64, 16): (1.5263958638698127,0.2095854061599497),
                (64, 32): (1.6588224684901565,0.24480018444726037),
                (64, 64): (1.8344167379585878,0.29441024255055187),
                (128, 8): (1.6039649421777507,0.19203756862248983),
                (128, 16): (1.6204920430473313,0.19128602826122604),
                (128, 32): (1.734432906066085,0.21435674109966527),
                (128, 64): (1.9057346942230422,0.2553919431615478),
                (128, 128): (2.1282520829789715,0.3124828611310525),
                (256, 8): (1.8174543625949644,0.2075227926020542),
                (256, 16): (1.7132228358964274,0.17304147074131715),
                (256, 32): (1.8095687394593518,0.1896109108737185),
                (256, 64): (1.955198162228139,0.21634301193911462),
                (256, 128): (2.1726413082434104,0.26293365889617304),
                (256, 256): (2.4402060117422453,0.32608612022456535),
                (512, 8): (1.9372617160120285,0.20807455252724233),
                (512, 16): (1.6379971758392489,0.15463679029080912),
                (512, 32): (1.6984248078749429,0.16304963536209177),
                (512, 64): (1.7977833917608168,0.17634627649459167),
                (512, 128): (1.9549296191491838,0.20349615779195065),
                (512, 256): (2.181328178990525,0.24858084460913465),
                (512, 512): (2.457772163471082,0.308348229648119),
                (1024, 8): (2.5063894715260373,0.2575464203407941),
                (1024, 16): (1.623530260002306,0.1477206954244023),
                (1024, 32): (1.6354372826512553,0.14875967087687209),
                (1024, 64): (1.6989162426971534,0.15446098681367754),
                (1024, 128): (1.8074730939008232,0.16920727760253051),
                (1024, 256): (1.977078838051756,0.19483449666641842),
                (1024, 512): (2.2148553510874462,0.23841998268684916),
                (1024, 1024): (2.505615707174297,0.2967332486358813),
                (2048, 8): (3.409135341521736,0.3001690868808072),
                (2048, 16): (1.6971104417249834,0.15276876905127904),
                (2048, 32): (1.6230725708451321,0.13685939706792075),
                (2048, 64): (1.6469759153251757,0.13936512441354643),
                (2048, 128): (1.7156144533827256,0.14513446553884674),
                (2048, 256): (1.8336142104874773,0.15854377336997183),
                (2048, 512): (2.0121047547948807,0.183104675399749),
                (2048, 1024): (2.2671738431180564,0.22467817015877944),
                (2048, 2048): (2.578330340473527,0.2814012579313862),
                (4096, 8): (4.70508337156544,0.3428684799774985),
                (4096, 16): (1.9535727349571788,0.21421685530636414),
                (4096, 32): (1.6543088327833007,0.12405446470635803),
                (4096, 64): (1.648282808217122,0.12687391371753767),
                (4096, 128): (1.6754951173761143,0.13001837790073253),
                (4096, 256): (1.7488868090740992,0.13594586224357985),
                (4096, 512): (1.871268976702024,0.14757377451616419),
                (4096, 1024): (2.0605758098447673,0.17110997587931406),
                (4096, 2048): (2.3375070649898193,0.2097133198200016),
                (4096, 4096): (2.673815277645329,0.2637812548957392),
                (8192, 8): (6.54690225858675,0.3960401987847372),
                (8192, 16): (2.54466779153633,0.2732589457614715),
                (8192, 32): (1.7016163433776683,0.12201274097747461),
                (8192, 64): (1.688710790079572,0.11546169593236845),
                (8192, 128): (1.6863711738838603,0.1190819388655187),
                (8192, 256): (1.7150523393665758,0.12265010508534424),
                (8192, 512): (1.790028853095534,0.12719391648448555),
                (8192, 1024): (1.9203403400491654,0.1376948314182972),
                (8192, 2048): (2.1239862082424885,0.15921337922943934),
                (8192, 4096): (2.4235462554156606,0.19393766979588029),
                (16384, 8): (9.160918835595478,0.48489533965674336),
                (16384, 16): (3.466828755360178,0.31285934101532026),
                (16384, 32): (1.7529026215522199,0.12244203246554126),
                (16384, 64): (1.741766363023315,0.114297942729824),
                (16384, 128): (1.731882752422196,0.10789180417642957),
                (16384, 256): (1.7326959021152026,0.11332330734975349),
                (16384, 512): (1.7621720482624812,0.11694780676424435),
                (16384, 1024): (1.83850436646609,0.11934993247764969),
                (16384, 2048): (1.9770454106948332,0.1278554739633273),
                (32768, 8): (12.869207402378171,0.619961682166387),
                (32768, 16): (4.7984873137715445,0.353049674883168),
                (32768, 32): (1.8119604612204279,0.12409797083970009),
                (32768, 64): (1.796959628892229,0.11717683787083248),
                (32768, 128): (1.7902256818214632,0.10929015341130065),
                (32768, 256): (1.7821304000093048,0.10321625628285308),
                (32768, 512): (1.7817071375413112,0.10844936230113911),
                (32768, 1024): (1.8106878972166474,0.11120675956852155),
                (65536, 8): (18.125611245510214,0.8301614977455478),
                (65536, 16): (6.695678611109317,0.40885886408331334),
                (65536, 32): (1.9059942884369896,0.13953620914353135),
                (65536, 64): (1.8533491995929632,0.11850992082344775),
                (65536, 128): (1.8488409779662198,0.11366860785066499),
                (65536, 256): (1.8412417277809603,0.10491388241494598),
                (65536, 512): (1.832356513890551,0.09904618603355018),
            }
        else:
            expected_params = {
                (32,8): (-0.003577975123695543,34.272518069267406,5.641168499790139),
               (64,8): (0.014049802326233115,37.482741894157094,5.3283881366544055),
               (128,8): (0.03735208641125132,40.579676235853626,5.036280215608025),
               (256,8): (0.055104202052100225,43.59921948682623,4.728621845777214),
               (512,8): (0.04328891885898439,42.939317934918975,4.436157588694638),
               (1024,8): (0.10691890554141958,43.06317733627374,5.2866752377721475),
               (2048,8): (0.031515303666724365,43.980897082871195,3.993013037956076),
               (4096,8): (0.02814857709606699,44.793184160832524,3.921632477528695),
               (8192,8): (0.10112941220534444,44.97158985164596,4.567605878299881),
               (16384,8): (0.029524386945859793,46.806689923980215,3.71362147928524),
               (32768,8): (0.03768228644102073,48.0185987939178,3.588653494074688),
               (65536,8): (0.05247516697336835,49.34924073290628,3.495106534096924),
               (32,16): (-0.03248128234703257,76.51962035635847,13.552649642798174),
               (64,16): (-2.3839292793247453,56.75686029530458,2.7489787574021047),
               (128,16): (0.02124992014537398,90.28554494775594,12.375495228597927),
               (256,16): (0.03656404190186156,96.82510471302862,11.543841738211405),
               (512,16): (0.0463290556106804,95.23210790616204,10.845551964807939),
               (1024,16): (0.054500873153148,95.4588980627239,10.50597489993134),
               (2048,16): (0.04980925511153468,96.85160972778857,9.899725487547578),
               (4096,16): (0.035097602993239094,98.38595026759201,9.468721105761393),
               (8192,16): (0.0235974350046576,99.78462931865724,9.427230917249258),
               (16384,16): (0.01555611986903095,101.48408526733144,9.13691860506887),
               (32768,16): (0.021168727644578823,103.68778213499104,8.836945170802663),
               (65536,16): (-0.053841782416830575,106.34471007294664,8.063513736286048),
               (32,32): (-0.02504644918529573,164.6047414053465,30.860897294818898),
               (64,32): (-0.4404374723331905,173.42694267139927,33.59397383552357),
               (128,32): (0.00499510612229774,198.32629563038142,29.77935795669358),
               (256,32): (-0.3449331811747314,205.61170169898423,29.901881665171032),
               (512,32): (0.03372639586363013,206.21330696423647,25.050864322346442),
               (1024,32): (0.051182825254146844,205.39762477678738,24.35009919807913),
               (2048,32): (0.06060681877938677,208.10001457900773,23.44276171955811),
               (4096,32): (0.05878195378492328,212.57146514940723,22.213438260801585),
               (8192,32): (0.04056786693560056,215.89344674483732,21.743680358103532),
               (16384,32): (0.02995667690704895,218.40953719654536,21.635393663145848),
               (32768,32): (0.19002501780954983,159.05773069251865,59.13274227310149),
               (65536,32): (0.013889430525628015,226.09870719118922,19.938571973768862),
               (64,64): (-0.019163426209010563,384.8029123285804,67.75697735194944),
               (128,64): (-0.024143775324871733,423.6061066832623,67.86387851620248),
               (256,64): (0.008573599955821725,450.4806847466672,62.50460298106027),
               (512,64): (0.01870893519999458,431.3306938382899,54.729599822201244),
               (1024,64): (0.03080015239940993,421.71785862620357,51.46937802838346),
               (2048,64): (0.04915519332567157,421.0658196523692,50.037985733376296),
               (4096,64): (0.05641260048662783,427.90269525493625,48.096473211062246),
               (8192,64): (0.05485267042515202,437.7082885744222,45.47512933901039),
               (16384,64): (0.04042382103257729,444.64147964128193,45.15625204568934),
               (32768,64): (0.025150861094470567,449.8608104348283,44.49486845783217),
               (65536,64): (0.20534902695132673,275.53551702401603,137.15064511660694),
               (128,128): (-0.018070397664473544,888.5958223632695,143.52185654750193),
               (256,128): (-0.01165793660712773,959.4383470057179,140.64630108642763),
               (512,128): (-0.006844998919006695,903.0532185253692,121.29919125293124),
               (1024,128): (0.0999927137909701,871.6150675977726,112.92028134765599),
               (2048,128): (-4.664438233195067,599.5805704843989,1.7152289148556261),
               (4096,128): (-1.7251117266080005,602.0694548532094,10.8189687204457),
               (8192,128): (0.050658601329429884,862.6465756200014,96.26399733057428),
               (16384,128): (0.055850066994007505,883.4726633295143,91.40035359461878),
               (32768,128): (-1.3105195376746201,727.6493111481429,70.11657112569063),
               (65536,128): (0.026733623998224404,905.3190897809654,88.93682130594357),
               (256,256): (-0.005447153039166887,2031.257936837022,298.4934838201869),
               (512,256): (-0.02787665997249001,1918.9828919483707,269.2860635671202),
               (1024,256): (-0.02052020815194637,1809.946481511377,235.45783801937847),
               (2048,256): (0.0013646574962403396,1740.491829555563,213.5507776947785),
               (4096,256): (0.010227943092444826,1703.4859783906513,199.8985324249832),
               (8192,256): (0.22932918046827,1370.3838399223314,471.56873426672223),
               (16384,256): (-0.0394023490690725,1720.1508660098943,186.88057245499618),
               (32768,256): (0.25228128731289545,822.5293944990535,646.0313483188531),
               (65536,256): (0.1819845398891149,1377.9097435879512,455.5904161488754),
               (512,512): (-0.016678801103033943,4071.932304749661,565.1967348804092),
               (1024,512): (-5.712896293698233,2643.365320417809,2.2233362587521643),
               (2048,512): (0.16429101447928324,2491.3210819578107,1152.7260426609719),
               (4096,512): (-0.021031824030544016,3496.8462727607393,407.1189963954556),
               (8192,512): (-0.010948627636839613,3418.1472644238556,384.1535290834071),
               (16384,512): (0.0026887120305683285,3406.8407716742745,375.00213770914047),
               (32768,512): (0.02158051137470626,3464.9378650278322,367.6442281690231),
               (65536,512): (0.028432894342747997,3545.672818566057,351.3186260945156),
               (1024,1024): (-0.02852298531220193,8264.774575396394,1078.5249481646008),
               (2048,1024): (-0.059923523894770514,7835.7773846636,976.820470338703),
               (4096,1024): (-0.06372814014660055,7366.629649920446,848.2938340809173),
               (8192,1024): (-5.854853872815284,5302.44403192157,4.781188476259066),
               (16384,1024): (-5.966211618613139,5370.214240727538,2.421248722372125),
               (32768,1024): (-0.02239603974871643,6842.1825890241025,713.2612511459856),
               (2048,2048): (-7.56283388833734,12177.55361746508,1.2472501439218264),
               (4096,2048): (-0.07902855133200654,15978.523889044027,1811.167581540809),
               (8192,2048): (-0.08597572847879081,14960.629104583124,1572.4928820669093),
               (16384,2048): (-6.474102357666348,11317.812534771645,3.575083158970468),
               (4096,4096): (-7.897811851342611,26424.366389911655,1.7434019674848487),
               (8192,4096): (-8.23843467481241,25468.020391835293,1.7829825679234408),
            }
                
    else:
        expected_params = {
            (4096,32): (0.6934484348596286,0.010893650247280331),
            (8192,32): (0.6929309406742655,0.007714571376986666),
            (16384,32): (0.6924660061803586,0.005439735175162053),
            (32768,32): (0.6920923851886733,0.0038472345545220955),
            (4096,64): (0.6919309443620992,0.010893650247280331),
            (8192,64): (0.6919665849625346,0.007714571376986666),
            (16384,64): (0.6918328552571094,0.005439735175162053),
            (32768,64): (0.6916647305961944,0.0038472345545220955),
            (4096,128): (0.6899798235234413,0.010893650247280331),
            (8192,128): (0.6908462516642049,0.007714571376986666),
            (16384,128): (0.6911695554868199,0.005439735175162053),
            (32768,128): (0.6912597239091195,0.0038472345545220955),
            (4096,256): (0.6869570549645402,0.010893650247280331),
            (8192,256): (0.689224759019894,0.007714571376986666),
            (16384,256): (0.6902832207239482,0.005439735175162053),
            (32768,256): (0.6907632985449913,0.0038472345545220955),
    }

    npts = len(x)
    #x -= x.mean()
    #x /= x.std()/np.sqrt(2)
    output = "coherence" if coherence else "scf"
    fs,cfs,scf = ssca_cu(x,nchan,nhop,fsamp,window="hann",psd=psd,
                         fsm_window_size=fsm_window_size,conjugate=conjugate,
                         output=output)
    #scf *= np.sqrt(nchan/len(x)**3)
    #scf *= np.sqrt(nchan**3/len(x))
    if coherence:
        scf /= np.sqrt(npts/nchan**3)
        params = expected_params[(npts,nchan)]
        amp = np.max(np.abs(scf[(cfs != 0) & (fs != 0)]))
        if not conjugate:
            if amp > stats.gumbel_r.ppf(threshold,*params):
                return True
            else:
                return False
        else:
            if amp > stats.genextreme.ppf(threshold,*params):
                return True
            else:
                return False
    else:
        expected_amp = expected_params[(npts,nchan)][0]
        expected_std = expected_params[(npts,nchan)][1]
        amp = np.sum(np.abs(scf))/len(scf.ravel())
        if np.abs(amp-expected_amp)/expected_std > threshold:
            return True
        else:
            return False
    

