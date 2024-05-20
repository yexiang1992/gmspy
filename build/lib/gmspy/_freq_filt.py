import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Union

def freq_filt(dt: float,
              acc: Union[list, tuple, np.ndarray],
              ftype: str = "Butterworth",
              btype: str = "bandpass",
              order: int = 4,
              freq1: float = 0.1,
              freq2: float = 24.99,
              rp: float = 3,
              plot: bool = False
              ):
    """Filtering employed to remove unwanted frequency components from a given acceleration signal.

    .. note::
        `freq2` cannot be higher than 1/2 of the record's time-step frequency.

    Parameters
    ----------
    dt: float
        Time step size.
    acc : 1D ArrayLike
        Acceleration time-history.
    ftype : str, optional, {'Butterworth', 'Chebyshev', 'Bessel'}
        The type of IIR filter to design, by default "Butterworth"
    btype : str, optional, {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        The type of filter. Default is 'bandpass'.
    order : int, optional, recommended range [1, 8]
        The order of the filter, by default 4
    freq1 : float, default = 0.1
        Cut-off frequency (Hz) for `lowpass` and `highpass` filtering.

        * `lowpass` filtering suppresses frequencies that are higher than freq1.
        * `highpass` filtering allows frequencies that are higher than freq1 to pass through.

    freq2 : float, default = 24.99
        Cut-off frequency (Hz) required for `bandpass` and `bandstop` filtering.

        * `bandpass` filtering allows signals within a given frequency range (freq1 to freq2) bandwidth to pass through.
        * `bandstop` filtering suppresses signals within the given frequency range (freq1 to freq2)

    rp: float, default=3.0, recommended range [0.1, 5]
        Required when `btype`= 'Chebyshev',
        the maximum ripple allowed below unity gain in the passband.
        Specified in decibels (db), as a positive number.
    plot: bool, default=False
        If True, plot time histories.

    Returns
    -------
    acc_filt: 1D ArrayLike
        Filtered acceleration time-history.
    """
    # check
    acc = np.array(acc)
    if btype not in ['lowpass', 'highpass', 'bandpass', 'bandstop']:
        raise ValueError(
            f"error btype={btype}, should one of ('lowpass', 'highpass', 'bandpass', 'bandstop')!")
    if ftype.lower() not in ['butterworth', 'chebyshev', 'bessel']:
        raise ValueError(
            f"error ftype={ftype}, should one of ('Butterworth', 'Chebyshev', 'Bessel')!")
    if ftype.lower() == 'butterworth':
        ftype = 'butter'
    elif ftype.lower() == 'chebyshev':
        ftype = 'cheby1'
    elif ftype.lower() == 'bessel':
        ftype = 'bessel'
    # filter
    if btype.startswith("l") or btype.startswith("h"):
        freq = freq1
    else:
        freq = np.array([freq1, freq2])
    fs = 1 / dt
    if freq2 > (fs / 2):
        raise ValueError(
            "freq2 cannot be higher than 1/2 of the record's time-step frequency!")
    # Lowpass Butterworth Transfer Function
    wn = 2 * freq / fs
    ba = signal.iirfilter(order, wn, rp=rp,
                          btype=btype, ftype=ftype, analog=False)
    acc_filt = signal.filtfilt(*ba, acc)

    if plot:
        t = np.arange(len(acc)) * dt
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(t, acc, c='b', lw=1, label="origin")
        ax.plot(t, acc_filt, c='r', lw=1, label="filtering")
        ax.hlines(0, np.min(t), np.max(t), lw=0.5, colors='k')
        ax.set_xlim(np.min(t), np.max(t))
        ax.grid(False)
        ax.set_xlabel("Time (s)", fontsize=15)
        ax.set_ylabel('acceleration', fontsize=15)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=12)
        plt.show()
    return acc_filt
