import numpy as np
import matplotlib.pyplot as plt
from numba import jit


def lida(
    dt: float,
    acc: list,
    omega: float,
    damp_ratio: float = 0.05,
    method: str = "nigam_jennings",
    plot: bool = False,
):
    """Linear Dynamic Time-History Analysis of Single Degree of Freedom Systems via Fast Fourier Transform.

    Parameters
    ----------
    dt : float
        Time step.
    acc : 1D ArrayLike
        Acceleration time series.
    omega : float
        Natural circular frequency of an SDOF system.
    damp_ratio : float, optional
        Damping ratio, by default 0.05
    plot: bool, default=False
        If True, plot time histories.
    method: str, default="Nigam_Jennings"
        Linear Dynamic Time-History Analysis method, optional,
        one of ("FFT", "Nigam_Jennings", "Newmark0", "Newmark1"):

        * "FFT"---Fast Fourier Transform;
        * "Nigam_Jennings"---exact solution by interpolating the excitation over each time interval;
        * "Newmark0"---const acceleration Newmark-beta method, gamma=0.5, beta=0.25;
        * "Newmark1"---linear acceleration Newmark-beta method, gamma=0.5, beta=1/6.

    Returns
    -------
    tuple(u: 1D ArrayLike, v: 1D ArrayLike, a: 1D ArrayLike)
        Displacement, Velocity, Acceleration Time History.
    """
    if method.lower() == "fft":
        a, v, u = _sodf_fft(dt, acc, omega, damp_ratio)
    elif method.lower() == "nigam_jennings":
        a, v, u = _nigam_jennings(dt, acc, omega, damp_ratio)
    elif method.lower() == "newmark0":
        a, v, u = _newmark(dt, acc, omega, damp_ratio, gamma=0.5, beta=0.25)
    elif method.lower() == "newmark1":
        a, v, u = _newmark(dt, acc, omega, damp_ratio, gamma=0.5, beta=1 / 6)
    else:
        raise ValueError(f"Not supported method {method}!")

    if plot:
        fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex="all")
        plot_y = [a, v, u]
        y_labels = ["acc", "vel", "disp"]
        ts = np.arange(len(acc)) * dt
        for i in range(3):
            ax = axs[i]
            ax.plot(ts, plot_y[i], c="k", lw=1.2)
            ax.set_ylabel(y_labels[i], fontsize=15)
            ax.grid(False)
            ax.tick_params(labelsize=12)
            ax.set_xlim(np.min(ts), np.max(ts))
        axs[-1].set_xlabel("Time (s)", fontsize=15)
        plt.show()
    return u, v, a


def _sodf_fft(dt, acc, omega, damp_ratio):
    n = len(acc)
    Nfft = int(2 ** np.ceil(np.log2(n)))
    af = np.fft.fft(acc, Nfft)
    f = np.fft.fftfreq(Nfft, d=dt)
    ws = 2.0 * np.pi * f
    H = 1 / (ws**2 - 2 * ws * damp_ratio * omega * 1j - omega**2)
    u = np.fft.ifft(af * H).real
    v = np.fft.ifft(af * ws * H * 1j).real
    a = np.fft.ifft(-af * ws**2 * H).real
    u = u[:n]
    v = v[:n]
    a = a[:n]
    a = a + acc
    return a, v, u


@jit(nopython=True)
def _nigam_jennings(dt, acc, omega, damp_ratio):
    xi = damp_ratio
    wn = omega
    wd = wn * np.sqrt(1 - xi**2)
    a11 = np.exp(-xi * wn * dt) * (
        xi / np.sqrt(1 - xi**2) * np.sin(wd * dt) + np.cos(wd * dt)
    )
    a12 = np.exp(-xi * wn * dt) / wd * np.sin(wd * dt)
    a21 = -wn / np.sqrt(1 - xi**2) * np.exp(-xi * wn * dt) * np.sin(wd * dt)
    a22 = np.exp(-xi * wn * dt) * (
        np.cos(wd * dt) - xi / np.sqrt(1 - xi**2) * np.sin(wd * dt)
    )
    b11 = (
        np.exp(-xi * wn * dt)
        * (
            ((2 * xi**2 - 1) / wn**2 / dt + xi / wn) * np.sin(wd * dt) / wd
            + (2 * xi / wn**3 / dt + 1 / wn**2) * np.cos(wd * dt)
        )
        - 2 * xi / wn**3 / dt
    )
    b12 = (
        -np.exp(-xi * wn * dt)
        * (
            ((2 * xi**2 - 1) / wn**2 / dt) * np.sin(wd * dt) / wd
            + 2 * xi / wn**3 / dt * np.cos(wd * dt)
        )
        - 1 / wn**2
        + 2 * xi / wn**3 / dt
    )
    b21 = (
        -1
        / wn**2
        * (
            -1 / dt
            + np.exp(-xi * wn * dt)
            * (
                (wn / np.sqrt(1 - xi**2) + xi / dt / np.sqrt(1 - xi**2))
                * np.sin(wd * dt)
                + 1 / dt * np.cos(wd * dt)
            )
        )
    )
    b22 = (
        -1
        / wn**2
        / dt
        * (
            1
            - np.exp(-xi * wn * dt)
            * (xi / np.sqrt(1 - xi**2) * np.sin(wd * dt) + np.cos(wd * dt))
        )
    )
    d = [0]
    v = [0]
    a = [-acc[0]]
    for i in range(len(acc) - 1):
        d_ = a11 * d[i] + a12 * v[i] + b11 * acc[i] + b12 * acc[i + 1]
        v_ = a21 * d[i] + a22 * v[i] + b21 * acc[i] + b22 * acc[i + 1]
        a_ = -acc[i + 1] - wn**2 * d_ - 2 * xi * wn * v_
        d.append(d_)
        v.append(v_)
        a.append(a_)
    return np.array(a) + acc, np.array(v), np.array(d)


@jit(nopython=True)
def _newmark(dt, acc, omega, damp_ratio, gamma=0.5, beta=1 / 6):
    xi = damp_ratio
    wn = omega
    acc0 = -acc[0]
    a1 = 1 / beta / dt**2 + gamma / beta / dt * 2 * xi * wn
    a2 = 1 / beta / dt + (gamma / beta - 1) * 2 * xi * wn
    a3 = (1 / 2 / beta - 1) + dt * (gamma / 2 / beta - 1) * 2 * xi * wn
    k_hat = wn**2 + a1
    d = [0]
    v = [0]
    a = [acc0]
    for i in range(len(acc) - 1):
        acc_temp = -acc[i + 1] + a1 * d[i] + a2 * v[i] + a3 * a[i]
        d.append(acc_temp / k_hat)
        v.append(
            gamma / beta / dt * (d[i + 1] - d[i])
            + (1 - gamma / beta) * v[i]
            + dt * (1 - gamma / 2 / beta) * a[i]
        )
        a.append(
            1 / beta / dt**2 * (d[i + 1] - d[i])
            - 1 / beta / dt * v[i]
            - (1 / 2 / beta - 1) * a[i]
        )
    return np.array(a) + acc, np.array(v), np.array(d)
