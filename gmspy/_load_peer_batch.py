# -*- coding: utf-8 -*-
"""
批量读取PEER地震波，并可以将地震波缩放至目标加速度,每个方向的地震波使用同一个缩放系数（缩放是可选的）。
本程序用来批量读取PEER地震波文件，批量读取的方法：事先将所有PEER地震波（三个方向）都解压到同一个文件夹内
（或者不同子文件夹内），本程序可自动对这些地震波(.AT2文件)进行读取，输出结果为一列表，列表的每个元素为一字典，
其中包含了三个方向的地震动时程(水平方向按PGA从大到小排列，第三列为竖向)，采样时间步，地震波的点数，RSN号等信息。
    GMdata           = BatchreadPEER(FileFolder, Scalsw = False);
    GMdata,Target_GM = ReadBatchPEER(FileFolder, Scalsw = True, TargetPGA = 0.5);     将所有水平1向地震波按PGA缩放至0.5g，其余方向与水平1方向保持比例不变
    输入：
        FileFolder------储存地震波文件的母文件夹，譬如  'E:\\GroundMotions' 。
        Scalsw ---------可选，False不缩放（默认），True为缩放，同时需输入目标 PGA(默认为0.5)。
        TargetPGA-------可选，默认为 0.5。
    输出：
        GMdata------储存原始（无缩放）地震波数据的列表，每一元素为一字典，包含如下键-值对；
        GMdata[i]['GMH1'] -------  水平1分量加速度时程，PGA最大的水平分量；   
        GMdata[i]['GMH2'] -------  水平2分量加速度时程； 
        GMdata[i]['GMV3'] -------  竖向分量加速度时程，如果原数据中没有竖向数据，则使用水平1方向时程×0.65后作为竖向时程； 
        GMdata[i]['time'] -------  时间序列；   
        GMdata[i]['dt']   -------  采样时间步；  
        GMdata[i]['npts'] -------  采样点数，即数据点数；  
        GMdata[i]['RSN']  -------  地震波的RSN编号；
        GMdata[i]['GMname'] -----  地震波的 三方向文件名,列表。
        Target_GM--------Scalsw = True 时才可输出，格式与GMdata相同，除了各方向时程为缩放后的时程。
@author: Yexiang Yan
"""
import re
import numpy as np
from pathlib import Path
from copy import deepcopy
# from ._load_peer import loadPEER
# from ._elas_resp_spec import elas_resp_spec
from gmspy import loadPEER, elas_resp_spec

def _get_gm_info(p, GM, i):
    p_i = Path(GM[i])
    tsi, datai, rsni, uniti = loadPEER(str(p_i))
    dti = tsi[1] - tsi[0]
    npti = len(datai)
    GMnamei = p_i.stem
    return datai, tsi, dti, npti, rsni, uniti, GMnamei


def BatchReadPEER(path: str,
                  suffix: str = "AT2",
                  scale_base: str = None,
                  scale_target: float = None):
    """批量读取某文件夹下的PEER地震波，并可以将地震波缩放至目标加速度。每个方向的地震波使用同一个缩放系数。"""

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
            datai_new = np.array([*datai_new, datai_new[0] * 0.65], dtype=float)

        # ground motions data
        GMdata[i] = {'GMH1': datai_new[0], 'GMH2': datai_new[1], 'GMV3': datai_new[2],
                     'time': timei[0], 'dt': dti[0],
                     'npts': minLength, 'RSN': rsni[0], 'GMname': GMnamei}

        print(f'RSN={rsni[0]} has been read and stored, {i+1}/{numRSN}')

    print(f'All {numRSN} groups of ground motions have been read and stored!')

    # 缩放地震波加速度
    if scale_base.upper() == "PGA":
        target_GM = deepcopy(GMdata)
        for m in range(len(GMdata)):
            scal = scale_target / np.amax(np.abs(GMdata[m]['GMH1']))
            target_GM[m]['GMH1'] = GMdata[m]['GMH1'] * scal
            target_GM[m]['GMH2'] = GMdata[m]['GMH2'] * scal
            target_GM[m]['GMV3'] = GMdata[m]['GMV3'] * scal
        print(f'All {numRSN} groups of ground motions have been scaled to {scale_target}!')
        return GMdata, target_GM
    elif scale_base.capitalize().startswith("Sa"):
        T1 = float(re.findall(r"\d+\.?\d*", scale_base)[0])
        target_GM = deepcopy(GMdata)
        for m in range(len(GMdata)):
            sa = elas_resp_spec(GMdata[m]['dt'], GMdata[m]['GMH1'], T1)[0, 2]
            scal = scale_target / sa
            target_GM[m]['GMH1'] = GMdata[m]['GMH1'] * scal
            target_GM[m]['GMH2'] = GMdata[m]['GMH2'] * scal
            target_GM[m]['GMV3'] = GMdata[m]['GMV3'] * scal
        print(f'All {numRSN} groups of ground motions have been scaled to {scale_target}!')
        return GMdata, target_GM
    else:
        return GMdata
    
path = r"E:\_WorkSpace\JupyterWorkSpace\地震动强度参数研究\GroundMotions"
suffix = "AT2"
BatchReadPEER(path, suffix, scale_base="sa(1)", scale_target=0.5)
