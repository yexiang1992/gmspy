import matplotlib.pyplot as plt
import numpy as np


def fou_pow_spec(ts: list, acc: list, plot: bool = False):
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
    Nfft = int(2 ** np.ceil(np.log2(n)))
    af = np.fft.fft(acc, Nfft, norm='ortho')
    freq = np.fft.fftfreq(Nfft, d=dt)
    # freq = freq[np.argsort(freq)]
    # af = af[np.argsort(freq)]
    idx = freq > 0
    af = af[idx]
    freq = freq[idx]
    # Fourier amplitudes
    amp = np.abs(af)
    # Fourier Phase
    phase = np.arctan(af.real / af.imag)
    # Power Spectral Amplitude
    Arms = np.sqrt(np.trapz(acc ** 2, ts) / ts[-1])
    pow_amp = amp ** 2 / (np.pi * ts[-1] * Arms**2)

    if plot:
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        plot_x = [ts, freq, freq]
        plot_y = [acc, amp, pow_amp]
        xlabels = ['Time(s)', "frequency(Hz)", "frequency(Hz)"]
        ylabels = ['acceleration', "Fourier Amplitude", "Power Amplitude"]
        for i in range(3):
            ax = axs[i]
            ax.plot(plot_x[i], plot_y[i], c='k', lw=1)
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
