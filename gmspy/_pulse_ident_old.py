import matplotlib.pyplot as plt
import gmspy as gm
import numpy as np
from scipy.integrate import cumulative_trapezoid
#from ._elas_resp_spec import elas_resp_spec


def _get_wavelet_vel(ts, t0, a, tp, gamma, v):
    # Mavroeidis and Papageorgiou wavelet
    vel = np.zeros_like(ts)
    fp = 1 / tp
    idx = (ts >= t0 - 0.5 * gamma * tp) & (ts <= t0 + 0.5 * gamma * tp)
    ts_ = ts[idx]
    vel[idx] = 0.5 * a * (1 + np.cos(2 * np.pi * fp * (ts_ - t0) / gamma)
                          ) * np.cos(2 * np.pi * fp * (ts_ - t0) + v)
    return vel


def _get_wavelet_acc(ts, t0, a, tp, gamma, v):
    acc = np.zeros_like(ts)
    fp = 1 / tp
    idx = (ts >= t0 - 0.5 * gamma * tp) & (ts <= t0 + 0.5 * gamma * tp)
    ts_ = ts[idx]
    acc[idx] = -a * np.pi * fp / gamma * (
        np.sin(2 * np.pi * fp / gamma * (ts_ - t0)) * np.cos(
            2 * np.pi * fp * (ts_ - t0) + v) + gamma * np.sin(
                2 * np.pi * fp * (ts_ - t0) + v) * (
                    1 + np.cos(2 * np.pi * fp / gamma * (ts_ - t0))))
    return acc


def _get_wavelet_disp(ts, t0, a, tp, gamma, v):
    disp = np.zeros_like(ts)
    fp = 1 / tp
    idx = (ts >= t0 - 0.5 * gamma * tp) & (ts <= t0 + 0.5 * gamma * tp)
    ts_ = ts[idx]
    C = 0
    disp[idx] = a / (4 * np.pi * fp) * \
        (np.sin(2 * np.pi * fp * (ts_ - t0) + v) + 0.5 * gamma / (gamma - 1) *
            np.sin(2 * np.pi * fp * (gamma - 1) / gamma * (ts_ - t0) + v) +
            0.5 * gamma / (gamma + 1) * np.sin(
                2 * np.pi * fp * (gamma + 1) / gamma * (ts_ - t0) + v)) + C
    idx1 = ts < t0 - 0.5 * gamma * tp
    disp[idx1] = a / (4 * np.pi * fp) / (1 - gamma ** 2) * \
        np.sin(v - np.pi * gamma) + C
    idx2 = ts > t0 + 0.5 * gamma * tp
    disp[idx2] = a / (4 * np.pi * fp) / (1 - gamma ** 2) * \
        np.sin(v + np.pi * gamma) + C
    return disp


def _get_tp(dt, acc):
    Ts = np.arange(0, 15.01, 0.02)
    Ts[0] = 0.001
    output = gm.elas_resp_spec(dt, acc, Ts, damp_ratio=0.05)
    spec = output[:, -2] * output[:, -1]
    idx = np.argmax(spec)
    tp = Ts[idx]
    return tp


def _get_a(dt, acc, tp, gamma):
    xi = 0.05
    psv = gm.elas_resp_spec(dt, acc, tp, damp_ratio=0.05)[0, 1]
    a = 4 * xi * psv / (1 - np.exp(-2 * np.pi * gamma * xi)
                        ) / (1 + (gamma - 1) * xi)
    return a


def _get_pulse(dt, acc, gamma_max=6):
    ts = np.arange(len(acc)) * dt
    vel = cumulative_trapezoid(acc, ts, initial=0)
    disp = cumulative_trapezoid(vel, ts, initial=0)
    gamma_s = np.arange(1.1, gamma_max + 0.1, 0.1)
    v_s = np.linspace(0, 2 * np.pi, 73)
    tp = _get_tp(dt, acc)
    gamma_v = []
    for gamma in gamma_s:
        for v in v_s:
            a = _get_a(dt, acc, tp, gamma)
            vel_imp = _get_wavelet_vel(ts, 0, a, tp, gamma, v)
            acc_imp = _get_wavelet_acc(ts, 0, a, tp, gamma, v)
            disp_imp = _get_wavelet_disp(ts, 0, a, tp, gamma, v)
            if ((np.max(np.abs(vel_imp)) <= np.max(np.abs(vel))) and (
                np.max(np.abs(acc_imp)) <= np.max(np.abs(acc))) and (
                    np.max(np.abs(disp_imp)) <= np.max(np.abs(disp)))):
                gamma_v.append((gamma, v))
    factor = 0
    gamma_max = v_max = t0_max = impulse = None

    i = 0
    for gamma, v in gamma_v:
        a = _get_a(dt, acc, tp, gamma)
        t0_s = np.arange(0.5 * gamma * tp,
                         ts[-1] - 0.5 * gamma * tp + 0.1, 0.1)
        for t0 in t0_s:
            vel_imp = _get_wavelet_vel(ts, t0, a, tp, gamma, v)
            factor_ = np.correlate(vel_imp, vel)[0]
            if factor_ > factor:
                gamma_max = gamma
                v_max = v
                t0_max = t0
                impulse = vel_imp
            print(i + 1)
            i += 1
    td = t0_max - gamma_max * tp / 2
    return impulse, gamma_max, v_max, td


ts, acc = gm.load_gm_examples("Kobe")
dt = ts[1] - ts[0]
_get_tp(dt, acc)
print(_get_pulse(dt, acc, gamma_max=6))
