r"""
.. _ref_ex_cdsp:

Constant ductility response spectra
-------------------------------------
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

# %%
# Constant ductility response spectra,
# Each column corresponds to acceleration Sa, velocity Sv, displacement Sd spectra,
# yield displacement Dy, strength reduction factor Ry, and yield strength factor Cy (1/Ry)
Ts = np.arange(0.05, 4.05, 0.05)
output = GM.get_const_duct_spec(Ts=Ts, harden_ratio=0.02,
                                damp_ratio=0.05, mu=5, plot=True)
