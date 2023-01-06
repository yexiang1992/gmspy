import matplotlib.pyplot as plt
from _load_gm_examples import load_gm_examples
import os
import numpy as np
import openseespy.opensees as ops
from joblib import Parallel, delayed
from numba import jit
from numpy.typing import ArrayLike

from _lin_dyna_resp import lida_fft


def const_duct_spec(dt: float,
                    acc: ArrayLike,
                    Ts: ArrayLike,
                    harden_ratio: float = 0.02,
                    damp_ratio: float = 0.05,
                    analy_dt: float = None,
                    mu: float = 5,
                    niter: int = 100,
                    tol: float = 0.01,
                    n_jobs: int = 0
                    ):
    """Constant-ductility inelastic spectra.
    See section 7.5 in Anil K. Chopra (DYNAMICS OF STRUCTURES, Fifth Edition, 2020) and
    the section "Inelastic Response Spectra" (CEE 541. Structural Dynamics) by Henri P. Gavin.

    Parameters
    ----------
    dt : float
        _description_
    acc : ArrayLike
        _description_
    Ts : ArrayLike
        _description_
    harden_ratio : float, optional
        _description_, by default 0.02
    damp_ratio : float, optional
        _description_, by default 0.05
    analy_dt : float, default = None
        Time step for bilinear SDOF response analysis, if None, default=dt.
    mu : float, optional
        _description_, by default 5
    niter : int, optional
        _description_, by default 100
    tol : float, optional
        _description_, by default 0.01
    n_jobs : int, optional
        _description_, by default 0

    Returns
    -------
    _type_
        _description_
    """
    if analy_dt is None:
        analy_dt = dt
    Ts = np.atleast_1d(Ts)
    omegas = 2 * np.pi / Ts
    omegas = omegas[::-1]
    mass = 1.0
    # file_dir = os.path.abspath(os.path.dirname(__file__))
    # filename = 'temp_f.txt'  # file_dir.replace("\\", '/') + '/temp_f.txt'
    Fdata = -mass * acc
    # np.savetxt(filename, Fdata, fmt="%f")

    def run(omegai):
        k = mass * omegai ** 2
        ue, ve, ae = lida_fft(dt, acc, omegai, damp_ratio)
        upeak = np.max(np.abs(ue))
        fpeak = k * upeak
        # mumin
        maxuy = np.max(np.abs(ue)) * 0.999
        fy = k * maxuy
        umax, vmax, amax, _, _ = sdfResponse(
            mass, damp_ratio, k, fy, harden_ratio, Fdata, analy_dt)
        fybark = fy / fpeak
        mumin = (umax / upeak) / fybark
        # mumax
        minuy = upeak / (15 * mu)
        fy = k * minuy
        umax, vmax, amax, _, _ = sdfResponse(
            mass, damp_ratio, k, fy, harden_ratio, Fdata, analy_dt)
        fybark = fy / fpeak
        mumax = (umax / upeak) / fybark
        # step 4d
        alpha = mumin * mumax * (maxuy - minuy) / (mumax - mumin)
        beta = (maxuy * mumin - minuy * mumax) / (mumax - mumin)
        uy1 = np.max([alpha / (1.2 * mu) - beta, minuy])
        uy2 = np.min([alpha / (0.8 * mu) - beta, maxuy])
        for j in range(niter):
            # Step 5.(b) Solve for uy1
            fy = k * uy1
            umax, vmax, amax, _, _ = sdfResponse(
                mass, damp_ratio, k, fy, harden_ratio, Fdata, analy_dt)
            fybark = fy / fpeak
            mu1 = (umax / upeak) / fybark
            # Step 5.(c) Solve for uy2
            fy = k * uy2
            umax, vmax, amax, _, _ = sdfResponse(
                mass, damp_ratio, k, fy, harden_ratio, Fdata, analy_dt)
            fybark = fy / fpeak
            mu2 = (umax / upeak) / fybark
            S = (mu2 - mu1) / (uy2 - uy1)
            Duy = np.min([np.abs(mu - mu2) / S, 0.1 * (uy1 + uy2)])
            if (mu - mu2) / S > 0:
                Duy = -Duy
            if S > 0 and mu2 < mu:
                Duy = -0.5 * uy2
            if S > 0 and mu2 > mu:
                Duy = 0.1 * uy2
            if uy2 + Duy < 0:
                Duy = (minuy - uy2) * (mu - mu2) / (mumax - mu2)
            uy1 = uy2
            mu1 = mu2
            # Step 5.(k)
            uy2 = uy2 + Duy
            # Step 5.(l)
            fy = k * uy2
            umax, vmax, amax, _, _ = sdfResponse(
                mass, damp_ratio, k, fy, harden_ratio, Fdata, analy_dt)
            fybark = fy / fpeak
            mu2 = (umax / upeak) / fybark
            # Store the output values of the current iteration
            mui = mu2
            fyi = fy
            uyi = uy2
            iteri = j
            # Step 5.(m) Check for convergence
            if np.abs(uy1 - uy2) < 1e-5 * tol or np.abs(mu1 - mu2) < 1e-5 * tol or np.abs(mu2 - mu) < tol:
                # print(f"Current mui = {mui}")
                break
        # find Sd, Sv, Sa
        Sdi = umax
        Svi = vmax
        Sai = amax
        Dy = uy2
        Ry = upeak / uy2
        Cy = 1 / Ry
        print(mu2)
        print(Sdi / Dy)

        return Sai, Svi, Sdi, Dy, Ry, Cy

    if n_jobs == 0:
        output = []
        for i, wi in enumerate(omegas):
            output.append(run(wi))
        output = np.array(output)
    else:
        output = Parallel(n_jobs=n_jobs)(delayed(run)(wi) for wi in omegas)
        output = np.array(output)
    return output[::-1]


@jit(nopython=True)
def sdfResponse(m, zeta, k, Fy, alpha, F, dt, uresidual=0, umaxprev=0):
    gamma = 0.5
    beta = 1 / 6  # 0.25
    tol = 1.0e-8
    maxIter = 10
    c = zeta * 2 * np.sqrt(k * m)
    Hkin = alpha / (1.0 - alpha) * k

    p0 = 0.0
    u0 = uresidual
    v0 = 0.0
    fs0 = 0.0
    a0 = (p0 - c * v0 - fs0) / m

    a1 = m / (beta * dt * dt) + (gamma / (beta * dt)) * c
    a2 = m / (beta * dt) + (gamma / beta - 1.0) * c
    a3 = (0.5 / beta - 1.0) * m + dt * (0.5 * gamma / beta - 1.0) * c

    au = 1.0 / (beta * dt * dt)
    av = 1.0 / (beta * dt)
    aa = 0.5 / beta - 1.0

    vu = gamma / (beta * dt)
    vv = 1.0 - gamma / beta
    va = dt * (1 - 0.5 * gamma / beta)

    kT0 = k

    umax = np.abs(umaxprev)
    amax = 0.0
    vmax = 0.0
    up = uresidual
    up0 = up
    i = 0
    u = 0
    urs = []
    vrs = []
    ars = []
    fsss = []
    for ft in F:
        i += 1
        u = u0
        fs = fs0
        kT = kT0
        up = up0
        phat = ft + a1 * u0 + a2 * v0 + a3 * a0
        R = phat - fs - a1 * u
        R0 = R
        if R0 == 0.0:
            R0 = 1.0
        iter = 0
        while iter < maxIter and np.abs(R / R0) > tol:
            iter += 1
            kTeff = kT + a1
            du = R / kTeff
            u = u + du
            fs = k * (u - up0)
            zs = fs - Hkin * up0
            ftrial = np.abs(zs) - Fy
            if ftrial > 0:
                dg = ftrial / (k + Hkin)
                if (fs < 0):
                    fs = fs + dg * k
                    up = up0 - dg
                else:
                    fs = fs - dg * k
                    up = up0 + dg
                kT = k * Hkin / (k + Hkin)
            else:
                kT = k
            R = phat - fs - a1 * u

        v = vu * (u - u0) + vv * v0 + va * a0
        a = au * (u - u0) - av * v0 - aa * a0

        u0 = u
        v0 = v
        a0 = a
        fs0 = fs
        kT0 = kT
        up0 = up
        urs.append(u)
        vrs.append(v)
        ars.append(a)
        fsss.append(fs)

        if np.abs(u) > umax:
            umax = np.abs(u)
            # tumax = i * dt
        if np.abs(a) > amax:
            amax = np.abs(a)
            # tamax = i * dt
        if np.abs(v) > vmax:
            vmax = np.abs(v)
            # tvmax = i * dt
    return umax, vmax, amax, u, up, urs, vrs, ars, fsss

def ops_sdof():


ts, acc = load_gm_examples('Kobe')
dt = ts[1] - ts[0]
umax, vmax, amax, u, up, urs, vrs, ars, fsss = sdfResponse(
    1, 0.05, 10, 0.2, 0.1, -acc*9.81, dt, uresidual=0, umaxprev=0)
plt.plot(urs, fsss)
plt.show()
