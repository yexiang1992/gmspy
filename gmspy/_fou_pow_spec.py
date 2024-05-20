from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq


def fou_pow_spec(
        ts: Union[list, tuple, np.ndarray],
        acc: Union[list, tuple, np.ndarray],
        plot: bool = False):
    """The Fourier Amplitude Spectrum and the Power Spectrum (or Power Spectral Density Function)
    are computed by means of Fast Fourier Transformation (FFT) of the input time-history.

    * Fourier Amplitude is computed as the square root of the sum of the squares of the real and imaginary parts of the
      Fourier transform: SQRT (Re^2+Im^2);
    * Fourier Phase is computed as the angle given by the real and imaginary parts of
      the Fourier transform: ATAN (Re/Im);
    * Power Spectral Amplitude is computed as FourierAmpl^2/(Pi*duration*RmsAcc^2),
      where duration is the time length of the record, RmsAcc is the acceleration RMS and Pi is 3.14159.

    Parameters
    ----------
    ts : 1D ArrayLike
        Time.
    acc : 1D ArrayLike
        Acceleration time series.
    plot: bool, default=False
        If True, plot time histories.

    Returns
    -------
    freq: 1D ArrayLike
        Frequency.
    amp: 1D ArrayLike
        Fourier Amplitude.
    phase: 1D ArrayLike
        Fourier Phase.
    pow_amp: 1D ArrayLike
        Power Spectral Amplitude.
    """
    n = len(acc)
    dt = ts[1] - ts[0]
    af = fft(acc)[:n//2] / n
    amp = 2.0 * np.abs(af)
    freq = fftfreq(n, d=dt)[:n//2]
    df = freq[1] - freq[0]
    phase = np.angle(af)  # Fourier Phase
    # Power Spectral Amplitude
    # Arms = np.sqrt(np.trapz(acc ** 2, ts) / ts[-1])
    pow_amp = 2 * np.abs(af) ** 2 / df  # / (np.pi * ts[-1] * Arms**2)

    if plot:
        fig, axs = plt.subplots(4, 1, figsize=(10, 15))
        plot_x = [ts, freq, freq, freq]
        plot_y = [acc, amp, pow_amp, phase]
        xlabels = ['Time(s)', "frequency(Hz)",
                   "frequency(Hz)", "frequency(Hz)"]
        ylabels = ['acceleration', "Fourier Amplitude",
                   "Power Amplitude", "Phase Angle"]
        for i in range(4):
            ax = axs[i]
            if i < 3:
                ax.plot(plot_x[i], plot_y[i], c='k', lw=1)
            else:
                ax.plot(plot_x[i], plot_y[i], 'o', c='k', )
                ax.set_aspect('equal')
            ax.set_xlabel(xlabels[i], fontsize=15)
            ax.set_ylabel(ylabels[i], fontsize=15)
            ax.tick_params(labelsize=12)
            if i == 0:
                ax.set_xlim(np.min(plot_x[i]), np.max(plot_x[i]))
            else:
                ax.set_xlim(np.min(plot_x[i]), 15)
            ax.grid(False)
        plt.subplots_adjust(hspace=0.25)
        plt.show()
    return freq, amp, phase, pow_amp
