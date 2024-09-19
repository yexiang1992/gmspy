import numpy as np
from pathlib import Path
from rich import print
from collections import namedtuple, defaultdict
from ._load_peer import loadPEER



def _read_by_suffix(suffix: str, file_path):
    p = Path(file_path)
    files = list(p.rglob(f"*.{suffix}"))
    datas = defaultdict(list)
    for p_i in files:
        data_ = loadPEER(p_i)
        datas[data_.RSN].append(data_)
    return datas


def _make_gm_data(data, vertical_factor=0.65, print_info=True):
    GMdata = dict()
    ver_suffixs = ('UP', 'DWN', 'V', 'VER', 'UD')
    k, num = 0, len(data)
    for rsn, values in data.items():
        GM = namedtuple("GM", ["tsgH1", "tsgH2", "tsgV3", "times", "dt", "npts", "filenames"])
        length = np.min([len(v.tsg) for v in values])
        idxs = np.argsort([np.max(np.abs(v.tsg)) for v in values])[::-1]
        ver_name, names, tsg = "", [], dict()
        if len(idxs) == 3:
            i = 0
            for idx in idxs:
                if values[idx].file_name.upper().endswith(ver_suffixs):
                    tsg["V3"] = values[idx].tsg[:length]
                    ver_name = values[idx].file_name
                else:
                    if i == 0:
                        tsg["H1"] = values[idx].tsg[:length]
                    else:
                        tsg["H2"] = values[idx].tsg[:length]
                    i += 1
                    names.append(values[idx].file_name)
                names.append(ver_name)
        elif len(idxs) == 2:
            for i, idx in enumerate(idxs):
                if i == 0:
                    tsg["H1"] = values[idx].tsg[:length]
                else:
                    tsg["H2"] = values[idx].tsg[:length]
                tsg["V3"] = tsg["H1"] * vertical_factor
                names.append(values[idx].file_name)
        else:
            tsg["H1"] = values[idxs[0]].tsg[:length]
            tsg["H2"] = tsg["H1"]
            tsg["V3"] = tsg["H1"] * vertical_factor
            names.append(values[idxs[0]].file_name)
        dt = values[0].dt
        GMdata[rsn] = GM(
            tsgH1=tsg["H1"], tsgH2=tsg["H2"], tsgV3=tsg["V3"],
            times=np.arange(length) * dt,
            dt=dt, npts=length, filenames=names
        )
        if print_info:
            print(f"Info:: RSN={rsn} has been read and stored, {k + 1}/{num}, {(k+1)/num*100:.0f}%")
        k += 1
    return GMdata


def loadPEERbatch(
        file_path: str,
        vertical_factor: float = 0.65,
        print_info: bool = True,
        return_vel: bool = False,
        return_disp: bool = False
):
    """Read PEER ground motion data under a certain path in batches.
    The requirement is that the data has been decompressed into `*.AT2`, `*.VT2`, `*.DT2` files.

    Parameters
    -----------
    file_path : str or ``pathlib.Path``
        The path to the data file.
    vertical_factor : float, default=0.65
        The scaling factor used when the vertical component is missing.
        Multiply it by the horizontal component with the largest peak value to get the vertical component.
    print_info : bool, default=True
        Print information when reading data.
    return_vel: bool, default=False
        Read and return the velocity data identified by `*.VT2` if True.
    return_disp: bool, default=False
        Read and return the displacement data identified by `*.DT2` if True.

    Returns
    --------
    * If return_vel is ``False`` and return_disp is ``False``:
        return accel_data: dict
    * If return_vel is ``True`` and return_disp is ``True``:
        return (accel_data: dict, vel_data: dict, disp_data: dict)
    * If return_vel is ``True`` and return_disp is ``False``:
        return (accel_data: dict, vel_data: dict)
    * If return_vel is ``False`` and return_disp is ``True``:
        return (accel_data: dict, disp_data: dict)

    ``accel_data``, ``vel_data`` or ``disp_data`` is a ``dict`` obj with unique ``RSN`` tags as keys,
    and each value is a ``namedtuple``:

    ``namedtuple("GM", ["tsgH1", "tsgH2", "tsgV3", "times", "dt", "npts", "filenames"])``

    * tsgH1: 1D numpy array, horizontal component with largest peak.
    * tsgH2: 1D numpy array, the second horizontal component.
    * tsgV3: 1D numpy array, vertical component.
    * times: 1D numpy array, the times corresponding to the components.
    * dt: float, Sampling time step.
    * npts: int, the number of points along the components.
    * filenames: list, the file names of the components.

    You can call the output like this, ``tsg = accel_data[RSNtag].tsgH1``.
    See .. _collections.namedtuple: https://docs.python.org/3/library/collections.html#collections.namedtuple.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> accel = loadPEERbatch(file_path)
    >>> tag = 666  # A specific RSN tag in accel.keys()
    >>> print(
    >>>     np.max(np.abs(accel[tag].tsgH1)),
    >>>     np.max(np.abs(accel[tag].tsgH2)),
    >>>     np.max(np.abs(accel[tag].tsgV3))
    >>> )
    >>> plt.plot(accel[tag].times, accel[tag].tsgH1, c='b', label='H1')
    >>> plt.plot(accel[tag].times, accel[tag].tsgH2, c='r', label='H2')
    >>> plt.plot(accel[tag].times, accel[tag].tsgV3, c='g', label='V3')
    >>> plt.legend()
    >>> plt.show()
    """
    DATA = []
    accels = _read_by_suffix("AT2", file_path)
    accel_data = _make_gm_data(accels, vertical_factor, print_info)
    DATA.append(accel_data)
    if return_vel:
        vels = _read_by_suffix("VT2", file_path)
        vel_data = _make_gm_data(vels, vertical_factor, print_info)
        DATA.append(vel_data)
    if return_disp:
        disps = _read_by_suffix("DT2", file_path)
        disp_data = _make_gm_data(disps, vertical_factor, print_info)
        DATA.append(disp_data)
    if print_info:
        print(f"Info:: All {len(accels)} groups of ground motions have been read and stored!")
    if len(DATA) == 1:
        return DATA[0]
    else:
        return tuple(DATA)
