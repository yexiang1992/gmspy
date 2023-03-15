import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid


def baselinecorr(ts: list,
                 acc: list,
                 poly_degree: int = 1,
                 plot: bool = False):
    """Baseline Correction through regression analysis, consists in
    (1) determining, through regression analysis (least-squares-fit method), the polynomial curve that best
    fits the time-acceleration pairs of values and then
    (2) subtracting from the actual acceleration values their corresponding counterparts as obtained with the
    regression-derived equation.

    Parameters
    ----------
    ts : array_like, shape (M,)
        The time vector of the input acceleration time history acc.
    acc : array_like, shape (M,)
        Vector of the acceleration history of the excitation imposed at the base.
    poly_degree: int, default=1
        Polynomial degree to adopt. 0-constant, 1-linear, 2-quadratic and 3-cubic, and so on.
    plot: bool, default=False
        If True, plot time histories.

    Returns
    --------
    cor_acc : array_like, shape (M,)
        Time-history of acceleration.
    cor_vel : array_like, shape (M,)
        Time-history of velocity.
    cor_disp : array_like, shape (M,)
        Time-history of displacement.
    """
    ts = np.array(ts)
    acc = np.array(acc)
    # fit a polynomial trend to the acceleration time history
    cfit = np.polyfit(ts, acc, deg=poly_degree)
    accFit = np.polyval(cfit, ts)
    # Correct the time histories
    cor_acc = acc - accFit
    cor_vel = cumulative_trapezoid(cor_acc, ts, initial=0)
    cor_disp = cumulative_trapezoid(cor_vel, ts, initial=0)

    if plot:
        vel = cumulative_trapezoid(acc, ts, initial=0)
        disp = cumulative_trapezoid(vel, ts, initial=0)
        plot_obj_ori = [acc, vel, disp]
        plot_obj_corr = [cor_acc, cor_vel, cor_disp]
        titles = ['acceleration', 'velocity', 'displacement']
        fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex='all')
        for i in range(3):
            ax = axs[i]
            ax.plot(ts, plot_obj_ori[i], c='b', lw=1, label="origin")
            ax.plot(ts, plot_obj_corr[i], c='r', lw=1, label="correction")
            ax.hlines(0, np.min(ts), np.max(ts), lw=0.5, colors='k')
            ax.set_xlim(np.min(ts), np.max(ts))
            ax.grid(False)
            ax.set_ylabel(titles[i], fontsize=15)
            ax.tick_params(labelsize=12)
        axs[0].legend(fontsize=12)
        axs[-1].set_xlabel("Time (s)", fontsize=15)
        plt.show()
    return cor_acc, cor_vel, cor_disp
