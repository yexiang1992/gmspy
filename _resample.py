import fractions
import numpy as np
from scipy import signal
from numpy.typing import ArrayLike


def resample(dt: float, acc: ArrayLike, dti: float):
    """Resampling the signal.

    Parameters
    ----------
    dt: float
        The size of the time step of the input acceleration time history.
    acc: 1D ArrayLike
        The acceleration time history.
    dti: float, default=None
        New time step size for resampling of the input acceleration time history.

    Returns
    -------
    time: 1D ArrayLike
        New time.
    acc: 1D ArrayLike
        Resamped acceleration time history.
    """
    rat = fractions.Fraction.from_float(dt / dti).limit_denominator()
    d1, d2 = rat.numerator, rat.denominator
    # Resample the acceleration time history
    acc = signal.resample_poly(acc, d1, d2)
    NANxgtt = np.argwhere(np.isnan(acc)).ravel()
    errxgtt = np.argwhere(np.diff(NANxgtt) > 1).ravel()
    if any(errxgtt):
        raise ValueError(
            'Non consecutive NaNs in resampled acceleration time history')
    if any(NANxgtt):
        acc = acc[:NANxgtt[0] - 1]
    # Time scale
    time = np.arange(len(acc)) * dti
    return time, acc
