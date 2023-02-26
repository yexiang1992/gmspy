import re
import numpy as np
from pathlib import Path
from copy import deepcopy
from rich import print
from ._load_peer import loadPEER
from ._elas_resp_spec import elas_resp_spec


def _get_gm_info(p, GM, i):
    p_i = Path(GM[i])
    tsi, datai, rsni, uniti = loadPEER(str(p_i))
    dti = tsi[1] - tsi[0]
    npti = len(datai)
    GMnamei = p_i.stem
    return datai, tsi, dti, npti, rsni, uniti, GMnamei


def loadPEERbatch(path: str,
                  scale_base: str = None,
                  scale_target: float = None):
    """Read PEER ground motion records in batches, and scale the records according to PGA or Sa(T1),
    where each component uses the same scaling factor, and scales according to the largest record component of PGA.

    Batch reading method: This program can automatically read all ground movement records (e.g., .AT2 files) in a folder,
    and the output result is a list, and each element of the list is a ``dict`` object.

    Parameters
    ----------
    path : str, 
        The folder that ground motion records saved.
    scale_base : str, optional, default=None, i.e, not scale
        Scaling parameter, PGA or response spectrum value at a certain period Sa(T1).
        If use PGA, scale_base="PGA";
        If use Sa(T1), scale_base="Sa(T1)", in which T1 can be replaced by any number, such as "Sa(1.0)".
    scale_target : float, optional
        Target scaling value, if scale_base=None, it is ignored.

    Returns
    -------
    If scale_base=None, only output GMdata.

    GMdata: list,
        A list storing the original (unscaled) PEER ground motion records,
        each element is a dict object, including the following key-value pairs:

        * GMdata[i]['GMH1'] ------- Horizontal 1-component, the largest horizontal component of PGA;
        * GMdata[i]['GMH2'] ------- horizontal 2-component;
        * GMdata[i]['GMV3'] ------- Vertical component;
          if there is no vertical data in the original data, use horizontal 1-component multiply by 0.65
          as the vertical component;
        * GMdata[i]['time'] ------- time;
        * GMdata[i]['dt'] ------- sampling time step;
        * GMdata[i]['npts'] ------- The number of sampling points, that is, the number of data points;
        * GMdata[i]['RSN'] ------- RSN number of record;
        * GMdata[i]['GMname'] ----- three-direction file name of record, list.

    If scale_base!=None, output GMdata and Target_GMdata.

    Target_GMdata: list,
        The format is the same as GMdata, except that the components after scaling.

    Examples
    --------
    >>> GMdata = loadPEERbatch(path="C:\my_records")
    >>> GMdata, Target_GMdata = loadPEERbatch(path="C:\my_records", scale_base="PGA", scale_target=0.5) # 0.5g
    >>> GMdata, Target_GMdata = loadPEERbatch(path="C:\my_records", scale_base="Sa(1.0)", scale_target=0.5) # 0.5g

    """
    suffix = "AT2"
    p = Path(path)
    GM = list(p.rglob("*." + suffix))
    n = len(GM)
    data = [None] * n
    times = [None] * n
    dts = [None] * n
    npts = [None] * n
    rsns = [None] * n
    units = [None] * n
    GMnames = [None] * n

    for i in range(n):
        output = _get_gm_info(p, GM, i)
        data[i], times[i], dts[i], npts[i], rsns[i], units[i], GMnames[i] = output
    data = np.array(data, dtype=object)
    times = np.array(times, dtype=object)
    dts = np.array(dts)
    npts = np.array(npts)
    rsns = np.array(rsns)
    GMnames = np.array(GMnames, dtype=object)

    newRSN = np.unique(rsns)
    numRSN = len(newRSN)
    GMdata = [None] * numRSN

    for i in range(numRSN):
        idxRSN = np.nonzero(np.abs(rsns - newRSN[i]) <= 1E-8)
        datai = data[idxRSN]
        timei = times[idxRSN]
        rsni = rsns[idxRSN]
        dti = dts[idxRSN]
        GMnamei = GMnames[idxRSN]
        #
        minLength = np.min([len(gm) for gm in datai])

        for j in range(len(datai)):
            datai[j] = datai[j][0:minLength]
            timei[j] = timei[j][0:minLength]
        datai_new = []
        data_ver = None
        for k in range(len(datai)):
            ver_sw = (GMnamei[k].upper().endswith('UP') | GMnamei[k].upper().endswith('DWN') |
                      GMnamei[k].upper().endswith('V') | GMnamei[k].upper().endswith('VER') |
                      GMnamei[k].upper().endswith('UD'))
            if ver_sw:
                data_ver = datai[k]
            else:
                datai_new.append(datai[k])
        datai_new = np.array(datai_new, dtype=float)
        if np.max(np.abs(datai_new[0])) < np.max(np.abs(datai_new[1])):
            datai_new[[0, 1]] = datai_new[[1, 0]]
        if ver_sw:
            datai_new = np.array([*datai_new, data_ver], dtype=float)
        else:
            datai_new = np.array(
                [*datai_new, datai_new[0] * 0.65], dtype=float)

        # ground motions data
        GMdata[i] = {'GMH1': datai_new[0], 'GMH2': datai_new[1], 'GMV3': datai_new[2],
                     'time': timei[0], 'dt': dti[0],
                     'npts': minLength, 'RSN': rsni[0], 'GMname': GMnamei}

        print(
            f'[#0099e5]RSN={rsni[0]}[/#0099e5] has been read and stored, [#ff4c4c]{i+1}/{numRSN}[/#ff4c4c]')

    print(
        f'All [#34bf49]{numRSN}[/#34bf49] groups of ground motions have been read and stored!')

    # scale
    if scale_base:
        if scale_base.upper() == "PGA":
            target_GM = deepcopy(GMdata)
            for m in range(len(GMdata)):
                scal = scale_target / np.amax(np.abs(GMdata[m]['GMH1']))
                target_GM[m]['GMH1'] = GMdata[m]['GMH1'] * scal
                target_GM[m]['GMH2'] = GMdata[m]['GMH2'] * scal
                target_GM[m]['GMV3'] = GMdata[m]['GMV3'] * scal
            print(
                f'All [#0099e5]{numRSN}[/#0099e5] groups of ground motions '
                f'have been scaled to [#ff4c4c]{scale_target}g[/#ff4c4c] '
                f'according by [#34bf49]{scale_base}[/#34bf49]!')
            return GMdata, target_GM
        elif scale_base.capitalize().startswith("Sa"):
            T1 = float(re.findall(r"\d+\.?\d*", scale_base)[0])
            target_GM = deepcopy(GMdata)
            for m in range(len(GMdata)):
                sa = elas_resp_spec(
                    GMdata[m]['dt'], GMdata[m]['GMH1'], T1)[0, 2]
                scal = scale_target / sa
                target_GM[m]['GMH1'] = GMdata[m]['GMH1'] * scal
                target_GM[m]['GMH2'] = GMdata[m]['GMH2'] * scal
                target_GM[m]['GMV3'] = GMdata[m]['GMV3'] * scal
            print(
                f'All [#0099e5]{numRSN}[/#0099e5] groups of ground motions '
                f'have been scaled to [#ff4c4c]{scale_target}g[/#ff4c4c] '
                f'according by [#34bf49]{scale_base}[/#34bf49]!')
            return GMdata, target_GM
    else:
        return GMdata
