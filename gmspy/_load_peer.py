# -*- coding: utf-8 -*-

import re
import tkinter as tk
import numpy as np
from tkinter import filedialog
from pathlib import Path
from collections import namedtuple
from typing import Union, NamedTuple
import matplotlib.pyplot as plt


def loadPEER(filename: Union[str, Path, None] = None, plot: bool = False) -> NamedTuple:
    """This function is used to read ground motion data from PEER database.

    Parameters
    ----------
    filename : Optional[str], default None
        Path of the PEER ground motion file.
        If None, ``tkinter`` will be used.
    plot : bool, optional
        If True, plot the time-history, by default False

    Returns
    -------
    GMdata: namedtuple("GM", ["tsg", "times", "dt", "npts", "RSN", "file_name", "unit"])
    Each field is:

    * "tsg" -- Time-history data.
    * "times" -- Times data.
    * "dt" -- Time step size.
    * "npts" -- Number of points.
    * "RSN" -- RSN tag.
    * "file_name" -- File name.
    * "unit" -- data unit.

    You can call the output like this, ``tsg = GMdata.tsg``.
    See .. _collections.namedtuple: https://docs.python.org/3/library/collections.html#collections.namedtuple.
    """
    if filename is None:
        root = tk.Tk()
        root.withdraw()
        root.call("wm", "attributes", ".", "-topmost", True)
        file_path = filedialog.askopenfilename()
        p = Path(file_path)
        file_name = p.stem
    else:
        file_path = Path(filename)
        file_name = file_path.stem

    ends = file_path.suffix.lower()
    if ends not in ('.at2', '.vt2', '.dt2'):
        raise ValueError("Error! Not PEER database, only .AT2 is supported.!")

    with open(file_path, "r") as f:
        content = f.read().splitlines()

    time_histories = []
    for line in content[4:]:
        currentLine = [float(d) for d in line.split()]
        time_histories.extend(currentLine)

    NPTS_DT = re.findall(r"-?\d*\.?\d+e?-?\d*", content[3])
    npts = int(NPTS_DT[0])
    dt = float(NPTS_DT[1])

    RSNlist = re.findall(r"\d+", file_name)
    RSN = int(RSNlist[0])

    unit = content[2].split()[-1].lower()

    time = np.arange(0, npts * dt, dt)

    # PEER_GM = {'time':time,'time_histories':time_histories,'dt':dt,'npts':npts,'RSN':RSN}
    tsg = np.array(time_histories)
    GM = namedtuple("GM", ["tsg", "times", "dt", "npts", "RSN", "file_name", "unit"])

    if plot:
        ylabels = {'.at2': 'acc', '.vt2': 'vel', '.dt2': 'disp'}
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(time, tsg, c='k', lw=1.2)
        ax.set_xlabel("Time (s)", fontsize=15)
        ax.set_ylabel(ylabels[ends] + f" ({unit})", fontsize=15)
        ax.set_title(f"RSN={RSN} DT={dt} NPTS={npts} UNIT={unit}", fontsize=15)
        ax.tick_params(labelsize=12)
        ax.set_xlim(np.min(time), np.max(time))
        ax.grid(False)
        plt.show()

    return GM._make([tsg, time, dt, npts, RSN, file_name, unit])

