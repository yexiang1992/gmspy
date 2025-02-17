from typing import Union

import numpy as np
from joblib import Parallel, delayed

from ._lin_dyna_resp import lida


def elas_resp_spec(dt: float,
                   acc: Union[list, tuple, np.ndarray],
                   Ts: Union[list, tuple, np.ndarray],
                   damp_ratio: float = 0.05,
                   method: str = "nigam_jennings",
                   n_jobs: int = 0) -> np.ndarray:
    """Computing the Elastic Response Spectrum.

    Parameters
    ----------
    dt : float
        Time step.
    acc : 1D ArrayLike
        Acceleration time series.
    Ts : ArrayLike
        Eigenperiods for which the response spectra are requested.
    damp_ratio : float, optional
        Damping ratio, by default 0.05.
    method: str, default="Nigam_Jennings"
        Linear Dynamic Time-History Analysis method, optional,
        one of ("FFT", "Nigam_Jennings", "Newmark0", "Newmark1"):

        * "FFT"---Fast Fourier Transform;
        * "Nigam_Jennings"---exact solution by interpolating the excitation over each time interval;
        * "Newmark0"---const acceleration Newmark-beta method, gamma=0.5, beta=0.25;
        * "Newmark1"---linear acceleration Newmark-beta method, gamma=0.5, beta=1/6.

        .. note::
           It is recommended to use the “Nigam_Jennings” method as this is exact for linear systems and
           will be accelerated using
           `numba.jit <https://numba.readthedocs.io/en/stable/user/jit.html>`_
           so speed of computation should not be an issue.


    n_jobs : int, optional, by default 0

        * If 0, do not use parallelism.
        * If an integer greater than 0, call ``joblib`` for parallel computing,
        * and the number of cpu cores used is `n_jobs`.
        * If -1, use all cpu cores.

    Returns
    -------
    output: (len(Ts), 5) ArrayLike.
        Each column is the *pseudo-acceleration spectrum*, *pseudo-velocity spectrum*,
        *acceleration spectrum*, *velocity spectrum* and *displacement spectrum* in turn.
    """
    acc = np.array(acc)
    Ts = np.atleast_1d(Ts)
    if np.abs(Ts[0] - 0) < 1e-8:
        Ts[0] = 1e-6
    omegas = 2 * np.pi / Ts

    def spec(wn):
        d, v, a = lida(dt, acc, wn, damp_ratio, method=method)
        sd = np.max(np.abs(d))
        sv = np.max(np.abs(v))
        sa = np.max(np.abs(a))
        psv = sd * wn
        psa = sd * wn * wn
        return [psa, psv, sa, sv, sd]

    if n_jobs == 0:
        output = np.zeros((len(Ts), 5))
        for i, wi in enumerate(omegas):
            output[i, :] = spec(wi)
    else:
        output = Parallel(n_jobs=n_jobs)(delayed(spec)(wi) for wi in omegas)
        output = np.array(output)
    return output


def _half_step(acc):
    u = np.array(acc)
    u1 = np.array([0, *u[0:-1]])
    u2 = (u1 + u) / 2
    u3 = np.array([u2, u])
    a = np.ravel(u3.T)
    a = np.delete(a, 0)
    return a
