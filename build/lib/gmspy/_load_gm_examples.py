import os
import numpy as np
import matplotlib.pyplot as plt


def load_gm_examples(name: str = "Kobe", plot: bool = False):
    """load built-in ground motions examples.

    Parameters
    ----------
    name : str, optional
        Record name, by default "Kobe".
        One of ("ChiChi", "Friuli", "Hollister", "Imperial_Valley", "Kobe", "Kocaeli",
        "Landers", "Loma_Prieta", "Northridge", "Trinidad")
    plot : bool, optional
        If True, plot ground motion, by default False.

    Returns
    ---------
    ts: 1D array_like
        Time.
    acc: 1D array_like
        Acceleration Time-History.
    """
    GMnames = ("ChiChi", "Friuli", "Hollister", "Imperial_Valley", "Kobe", "Kocaeli",
               "Landers", "Loma_Prieta", "Northridge", "Trinidad")
    if name not in GMnames:
        raise ValueError(f"Error {name}, must be one of"
                         "('ChiChi', 'Friuli', 'Hollister', 'Imperial_Valley', 'Kobe', 'Kocaeli', "
                         "'Landers', 'Loma_Prieta', 'Northridge', 'Trinidad')!")
    file_dir = os.path.abspath(os.path.dirname(__file__))
    GM = np.loadtxt(file_dir + f"/accelerograms/{name}.dat", skiprows=5)
    ts = GM[:, 0]
    acc = GM[:, 1]
    if plot:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ts, acc, c='k', lw=1)
        ax.set_xlabel("Time (s)", fontsize=15)
        ax.set_ylabel("acc (g)", fontsize=15)
        ax.tick_params(labelsize=12)
        ax.set_xlim(np.min(ts), np.max(ts))
        ax.grid(False)
        plt.show()
    return ts, acc
