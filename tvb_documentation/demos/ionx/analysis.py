import numpy as np
from scipy.signal import lfilter, resample, hann, convolve
from skimage.measure import block_reduce
from sklearn.decomposition import PCA
import mne
from scipy import stats

def trip(edges_arr,n):
    mat=np.zeros((n,n))
    mat[np.triu_indices(n,1)]=edges_arr
    mat=mat+mat.T+np.eye(n)
    return mat


def compute_K(r, V):
    x = 1 + np.pi**2 * r**2 + V**2
    K = np.sqrt( (x - 2*np.pi*r ) / (x + 2*np.pi*r ) )

    return K

#Kuramoto
def Kuramotize(r,V,active=1):
    Z = np.empty_like(r+1j*r)
    for i in range(len(r[0])):
        Z[:,i]=(1-np.conj(np.pi*r[:,i]+1j*V[:,i]))/(1+np.conj(np.pi*r[:,i]+1j*V[:,i]))
    if active==1:
        Z=Z*r
    return Z

def enlarger(fmri_image):
    fmri_image = np.repeat(fmri_image, repeats=2, axis=0)
    fmri_image = np.repeat(fmri_image, repeats=2, axis=1)
    larger_fmri_image = np.repeat(fmri_image, repeats=2, axis=2)
    return larger_fmri_image

def go_edge(tseries):
    nregions=tseries.shape[1]
    Blen=tseries.shape[0]
    nedges=int(nregions**2/2-nregions/2)
    iTriup= np.triu_indices(nregions,k=1) 
    gz=stats.zscore(tseries)
    Eseries = gz[:,iTriup[0]]*gz[:,iTriup[1]]
    return Eseries

def intervals(a):
    # Create an array of how long are all the intervals (zero sequences).
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return list(np.diff(ranges).flatten())

def compute_fcd(ts, win_len=30, win_sp=1):
    """
    Arguments:
        ts:      time series of shape [time,nodes]
        win_len: sliding window length in samples
        win_sp:  sliding window step in samples

    Returns:
        FCD: matrix of functional connectivity dynamics
        fcs: windowed functional connectivity matrices
    """
    n_samples, n_nodes = ts.shape
    fc_triu_ids = np.triu_indices(n_nodes, 1)
    n_fcd = len(fc_triu_ids[0])
    fc_stack = []


    for t0 in range( 0, ts.shape[0]-win_len, win_sp ):
        t1=t0+win_len
        fc = np.corrcoef(ts[t0:t1,:].T)
        fc = fc[fc_triu_ids]
        fc_stack.append(fc)

    fcs = np.array(fc_stack)
    FCD = np.corrcoef(fcs)
    return FCD, fcs
    

def resample_raw(raw, freq=256):
    """ Resample raw recordings in-place. """
    raw.resample(freq, npad="auto")

def tavg_to_rK_mne(tavg_data, resample_freq=256):
    r=tavg_data[:,0,:,0]
    V=tavg_data[:,1,:,0]
    K = compute_K(r,V)
    rK = r*K
    
    info=mne.create_info(
        ch_names=rK.shape[1],
        sfreq=1000.,
        ch_types=["eeg"]*rK.shape[1]
    )

    raw_rK = mne.io.RawArray(rK.T,info)

    if resample_freq is not None:
        resample_raw(raw_rK, resample_freq)

    return raw_rK


def generate_noise(shape, scale=0.01, iir_filter=None):
    """ Generate temporally IIR-filtered noise.

    Parameters
        shape           shape of the noise [nodes, time]
        scale           scaling constant, default aims for range +-0.5
        iir_filter      denominator coefficient vector (parameter b in
                        scipy.signal.lfilter)
        
    Returns
        noise           noise array of shape [nodes, time]
    """
    if iir_filter is None:
        iir_filter=[0.2, -0.2, 0.04] # thank you mne

    noise = np.random.normal(size=shape)
    noise = lfilter([1], iir_filter, noise, axis=-1)

    return noise * scale

def add_noise(raw, scale=0.01, iir_filter=None):
    noise = generate_noise(raw._data.shape, scale, iir_filter)
    raw._data += noise

def multiply_alpha(raw, l_freq=8, h_freq=13):
    raw_data = raw.get_data()
    noise = np.random.normal(size=raw_data.shape)
    noise = mne.filter.filter_data(
            noise, 
            sfreq=raw.info['sfreq'], 
            l_freq=l_freq, 
            h_freq=h_freq
    )
    raw._data[:] = noise*raw_data

def project_to_sensors(raw, ch_names, gain):
    eeg_data = np.dot(gain, raw.get_data())

    info=mne.create_info(
        ch_names=ch_names,
        sfreq=raw.info["sfreq"],
        ch_types=["eeg"]*eeg_data.shape[0]
    )

    raw_sensors = mne.io.RawArray(eeg_data,info)
        
    return raw_sensors


def robust_detrend_fit(x,y,deg=3,thr=1., iters=5):
    """
    Fits a low-degree polynomial to the data, while iteratively estimating the
    weights of the time points (should ignore jump glitches).

    Adapted from 10.1016/j.neuroimage.2018.01.035 (NoiseTools)
    
    Parameters:
        x       x-coordinates of shape [N,]
        y       y-values of shape [N,]
        deg     degree of the polynomial
        thr     threshold for outliers: multiples of sd from fitted signal 
        iters   fixed number of iterations
    """
    w = np.ones_like(x)
    for i in range(iters):
        f_poly = np.polynomial.polynomial.Polynomial.fit(
            x=x,
            y=y,
            deg=deg,
            w=w,
            window=[x[0],x[-1]]
        )
        d = np.abs(y - f_poly.linspace(len(x))[1])
        w[np.where( d/d.std() > 1)]=0
    return f_poly

def rescale_sensor_data_to_median(raw, med=35.):
    scale = med /np.median(raw._data.max(axis=1))
    raw._data *= scale

    return raw

def apply_baseline_drift(raw,poly_coeff):
    for i,_ in enumerate(raw.ch_names):
        n = np.random.randint(poly_coeff.shape[0])
        p = np.polynomial.polynomial.Polynomial(poly_coeff[n,:])
        _, drift = p.linspace(domain=(0,600), n=raw._data.shape[1])
        raw._data[i] += drift

    return raw

def generate_heart_beats(sfreq, t):
    N = np.int(1.1*t)
    beats = np.cumsum(np.random.normal(loc=1.0, scale=0.1,size=N))
    beats = beats[beats<t]

    return (beats * sfreq).astype(np.int)

def add_ecg(raw, ecg_mean, ch_scale, ecg_sfreq=600.614990234375, scale=2.):
    beats = generate_heart_beats(raw.info['sfreq'],raw.times[-1])

    ecg_mean_resampled = resample(
            x=ecg_mean,
            num=int(ecg_mean.size * raw.info['sfreq'] / ecg_sfreq)
    )
    
    ecg = np.zeros_like(raw._data)
    for beat_start in beats:
        beat_end = beat_start + ecg_mean_resampled.size
        beat_end = min(beat_end,ecg.shape[1]-1)
        ecg[:,beat_start:beat_end] = ecg_mean_resampled[:beat_end-beat_start]

    raw._data += ecg * ch_scale[:,np.newaxis] * scale

    return raw


def compute_gain(loc, ori, loc_sens, sigma=1.0):
    "Equation 12 of [Sarvas_1987]_"
    # r => sensor positions
    # r_0 => source positions
    # a => vector from sources_to_sensor
    # Q => source unit vectors
    r_0, Q = loc, ori
    center = np.mean(r_0, axis=0)[np.newaxis, ]
    radius = 1.05125 * max(np.sqrt(np.sum((r_0 - center)**2, axis=1)))
    loc = loc_sens.copy()
    sen_dis = np.sqrt(np.sum((loc)**2, axis=1))
    loc = loc / sen_dis[:, np.newaxis] * radius + center
    V_r = np.zeros((loc.shape[0], r_0.shape[0]))
    for sensor_k in np.arange(loc.shape[0]):
        a = loc[sensor_k, :] - r_0
        na = np.sqrt(np.sum(a**2, axis=1))[:, np.newaxis]
        V_r[sensor_k, :] = np.sum(Q * (a / na**3), axis=1 ) / (4.0 * np.pi * sigma)
    return V_r


def filt_all(ts, win_size):
    ts_filt = np.zeros_like(ts)
    win = hann(win_size)
    for i in range(ts.shape[1]):
        ts_filt[:,i] = convolve(ts[:,i], win, mode='same') / sum(win)
    return ts_filt

def bold_convolution(ts, sf, tr=2.0, win=20, nsig=0.01):
    """
    ts:     raw time series [time,nodes]
    sf:     sampling frequency of sf
    tr:     TR of the resulting BOLD
    win:    Hann window length [s]
    """
    ts_filt=filt_all(ts, int(sf*win))
    ts = block_reduce(
        ts_filt,
        block_size=(2000,1),
        func=np.mean
    )
    ts += np.random.normal(scale=nsig, size=ts.shape)

    return ts


def pca(ts,n_components=2):
    """
    Parameters:
        ts : array-like, shape (n_samples, n_features)
            Time series, for single variable is n_features ~ n_nodes.
        n_components : int
            Number of PCA components.
    Returns:
        pca: sklearn.decomposition.PCA instance
        pca_ts: transformed input time series
    """
    pca = PCA(n_components=n_components)
    pca_ts = pca.fit_transform(ts)
    return pca, pca_ts
