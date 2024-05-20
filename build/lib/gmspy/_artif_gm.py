import matplotlib.pyplot as plt
import numpy as np


def get_s0(mag, distance, wg, xig, wf=8*np.pi, va=0.98, component='H'):
    Tg = 2 * np.pi / wg
    if component.lower() == 'h':
        a1, a2, a3, a4 = -1.555, 0.165, 0.831, 0.148
    else:
        a1, a2, a3, a4 = -1.340, 0.104, 0.982, 0.184
    Ts = 10**(a1+a2*mag+a3*np.log10(distance+30)+a4*Tg)

    if component.lower() == 'h':
        a1, a2, a3, a4 = 3.226, 0.219, -1.377, 0.100
    else:
        a1, a2, a3, a4 = 3.078, 0.306, -1.774, 0.059
    r = np.sqrt(2*np.log(va*Ts)) + 0.5772 / np.sqrt(2*np.log(va*Ts))
    xif = xig
    beta = (4*xif**2*wf+2*xig*wg+wf) / (wf**2+2*xig*wg*wf-wg**2) * (np.pi*wg*wf) / (2*xig)
    S0 = 10 ** (2*(a1+a2*mag+a3*np.log10(distance+30)+a4*Tg)-2*np.log10(r)-np.log10(beta))
    return S0 / 10000


def psf_kanai_tajimi(w, mag, distance, soil_class=2, component='H'):
    if soil_class == 1:
        wg, xig = 25.13, 0.64
    elif soil_class == 2:
        wg, xig = 15.71, 0.72
    elif soil_class == 3:
        wg, xig = 11.42, 0.80
    elif soil_class == 4:
        wg, xig = 7.39, 0.90
    s0 = get_s0(mag, distance, wg, xig, component=component)
    return (wg**4 + 4 * xig**2 * wg**2 * w**2) / ((wg**2-w**2)**2 + (2*xig*wg*w) ** 2) * s0

def psf_markov(w, mag, distance, soil_class=2, component='H'):
    if soil_class == 1:
        wg, xig = 25.13, 0.64
    elif soil_class == 2:
        wg, xig = 15.71, 0.72
    elif soil_class == 3:
        wg, xig = 11.42, 0.80
    elif soil_class == 4:
        wg, xig = 7.39, 0.90
    wf = 8 * np.pi
    s0 = get_s0(mag, distance, wg, xig, component=component)
    return ((wg**4 + 4 * xig**2 * wg**2 * w**2) / ((wg**2-w**2)**2 + (2*xig*wg*w) ** 2) *
            wf**2 / (wf**2 + w**2) * s0)

def psf_clough_penzien(w, mag, distance, soil_class=2, component='H'):
    if soil_class == 1:
        wg, xig = 8 * np.pi, 0.64
    elif soil_class == 2:
        wg, xig = 5 * np.pi, 0.72
    elif soil_class == 3:
        wg, xig = 11.42, 0.80
    elif soil_class == 4:
        wg, xig = 7.39, 0.90
    wf = 8 * np.pi
    xif = xig
    s0 = get_s0(mag, distance, wg, xig, component=component)
    return ((wg**4 + 4 * xig**2 * wg**2 * w**2) / ((wg**2-w**2)**2 + (2*xig*wg*w) ** 2) *
            w**4 / ((w**2-wf**2)**2 + (2*xif*wf*w) ** 2) * s0)

def artif_gm(
    w,
    mag, distance,
    soil_class=2, component='H',
    total_time: float = 20,
    dt: float = 0.01,
):
    n = int(total_time / dt) + 1
    npts = int(2 ** np.ceil(np.log2(n)))
    fmax = 1 / (2 * dt)
    df = fmax / npts
    dw = 2 * np.pi * df
    psf = psf_clough_penzien(w, mag=mag, distance=distance, soil_class=soil_class, component=component)
    amp = np.sqrt(psf * dw)
    np.random.rand(int())
    return amp


f = np.linspace(0, 50, 501)
w = 2 * np.pi * f
psd = artif_gm(w, 6.5, 50, 2, total_time=20.48, dt=0.01)

plt.plot(f, psd)
plt.show()
