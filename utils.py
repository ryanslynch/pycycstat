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
                 conjugate=False, show=False, figure=None, figsize=None):
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
                                     facecolors=plt.cm.viridis_r(top_cfs))
    ax.add_collection3d(poly_collection,zs=top_cfs,zdir="y")
    ax.set_xlim(fs.min(),fs.max())
    if conjugate:
        ax.set_ylim(cfs.max(),cfs.min())
    else:
        ax.set_ylim(cfs.max(),0)
    ax.set_zlim(0,1.15*np.abs(scf).max())
    ax.locator_params(nbins=5)
    ax.set_xlabel("freq (Hz)")
    ax.set_ylabel("cycle freq (Hz)")
    ax.set_zlabel("SCF (magnitude)")
    #ax.invert_yaxis()
    if show:
        plt.show()
        return None
    else:
        return (figure,ax)
