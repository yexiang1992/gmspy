# -*- coding: utf-8 -*-

import re
import numpy as np
import matplotlib.pyplot as plt


def loadPEER(filename: str, plot: bool = False):
    """This function is used to read ground motion data from PEER database.

    Parameters
    ----------
    filename : str
        File name of PEER database.
    plot : bool, optional
        If True, plot the time history, by default False

    Returns
    -------
    ts: 1D ArrayLike
        Time.
    tsg: 1D ArrayLike
        Time-history of data.
    RSN, int
        RSN tag.
    unit: str
        Data unit.
    """
    ends = filename[-3:].lower()
    if ends not in ('at2', 'vt2', 'dt2'):
        raise ValueError("Error! not PEER database!")
    with open(filename, 'r') as f:
        content = f.read().splitlines()
    time_histories = []
    for line in content[4:]:
        currentLine = list(map(float, line.split()))
        time_histories.extend(currentLine)
    tsg = np.array(time_histories)
    NPTS_DT = re.findall(r'-?\d*\.?\d+e?-?\d*', content[3])
    npts = len(tsg)
    dt = float(NPTS_DT[1])
    RSN = int(re.findall(r"(?<=RSN|rsn)\d+", filename)[0])
    unit = content[2].split()[-1].lower()
    ts = np.arange(0, npts * dt, dt)

    if plot:
        ylabels = dict(at2="acc", vt2="vel", dt2="disp")
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(ts, tsg, c='k', lw=1.2)
        ax.set_xlabel("Time (s)", fontsize=15)
        ax.set_ylabel(ylabels[ends] + f" ({unit})", fontsize=15)
        ax.set_title(f"RSN={RSN} DT={dt} NPTS={npts} UNIT={unit}", fontsize=15)
        ax.tick_params(labelsize=12)
        ax.set_xlim(np.min(ts), np.max(ts))
        ax.grid(False)
        plt.show()
    return ts, tsg, RSN, unit
