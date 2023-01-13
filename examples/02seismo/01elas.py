r"""
.. _ref_ex_elas_spec:

Elastic response spectra
--------------------------
"""

import numpy as np
import gmspy as gm

# %%
# load examples
ts, acc = gm.load_gm_examples('Northridge')
dt = ts[1] - ts[0]

# %%
# Instantiate the class SeismoGM
GM = gm.SeismoGM(dt=dt, acc=acc, unit='g')
GM.plot_hist()

# %%
# Elastic response spectra, PSa, PSv, Sa, Sv, Sd for each column
Ts = np.arange(0.05, 4.05, 0.05)
spectra = GM.get_elas_spec(Ts=Ts, damp_ratio=0.05, plot=True)
